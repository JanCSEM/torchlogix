"""Pure-PyTorch statement of the fused-tree algorithm (CPU tests, and the fallback the
`fused` backend uses where neither Triton nor CUDA is available, e.g. local pytest).

Same inputs and outputs as the Triton / CUDA kernels in `ops.py`:

    x      (B, Nsrc)       one row per sample (uint8 bits when `binary`, else fp32 / bf16)
    leaf   (K, P, 2, S)    int32 gather table, S = 2**(D-1)
    pos    (Q, WIN)        int32 window table over the P positions (-1 = none)
    w      (sum_l S_l*K*4) concatenated sigma(W) / hard weights, level 0 first

and the output (B, K, Q) = max over the window of the tree root, plus the first-max
window slot (uint8, only for WIN > 1).
"""

import torch

from .plan import level_offsets


def lut(a, b, w0, w1, w2, w3):
    """Same operation order as weighted_light_basis_sum_2 (bitwise equal in fp32)."""
    return (((w0 * ((1 - a) * (1 - b)) + w1 * ((1 - a) * b)) + w2 * (a * (1 - b)))
            + w3 * (a * b))


def _level_w(w, depth, k, lvl, dtype):
    s0 = 1 << (depth - 1)
    offs, _ = level_offsets(depth, k)
    n = s0 >> lvl
    blk = w[offs[lvl]: offs[lvl] + n * k * 4].view(n, k, 4)       # (n, K, 4)
    return blk.permute(1, 0, 2)                                    # (K, n, 4)


def tree_roots(x, leaf, pos, w, depth, binary):
    """(B, K, Q, WIN) roots (garbage where pos < 0), differentiable in x and w."""
    k, p, _, s0 = leaf.shape
    q, win = pos.shape
    idx = leaf.long()[:, pos.long().clamp(min=0)]                  # (K, Q, WIN, 2, S)
    leaves = x[:, idx]                                             # (B, K, Q, WIN, 2, S)
    a, b = leaves[..., 0, :], leaves[..., 1, :]
    w0 = _level_w(w, depth, k, 0, None)[None, :, None, None]       # (1, K, 1, 1, S, 4)
    if binary:
        ab, bb = a.bool(), b.bool()
        out = torch.where(ab, torch.where(bb, w0[..., 3], w0[..., 2]),
                          torch.where(bb, w0[..., 1], w0[..., 0]))
    else:
        a, b = a.float(), b.float()
        out = lut(a, b, w0[..., 0], w0[..., 1], w0[..., 2], w0[..., 3])
    for lvl in range(1, depth):
        wl = _level_w(w, depth, k, lvl, None)[None, :, None, None]
        v = out.unflatten(-1, (out.shape[-1] // 2, 2))
        out = lut(v[..., 0], v[..., 1], wl[..., 0], wl[..., 1], wl[..., 2], wl[..., 3])
    return out[..., 0]


def forward(x, leaf, pos, w, depth, binary, out_dtype):
    roots = tree_roots(x, leaf, pos, w, depth, binary)             # (B, K, Q, WIN)
    q, win = pos.shape
    if win == 1:
        return roots[..., 0].to(out_dtype), roots.new_zeros(0, dtype=torch.uint8)
    valid = (pos >= 0)[None, None]
    roots = torch.where(valid, roots, torch.full_like(roots, float("-inf")))
    y, arg = roots.max(dim=-1)                                     # first max on ties
    return y.to(out_dtype), arg.to(torch.uint8)


def _levels(x, leaf, pos, w, depth, binary):
    """Leaves and the output of every tree level: (idx, a0, b0, [out_0 .. out_{D-1}])."""
    k = leaf.shape[0]
    idx = leaf.long()[:, pos.long().clamp(min=0)]                  # (K, Q, WIN, 2, S)
    leaves = x[:, idx]
    a, b = leaves[..., 0, :], leaves[..., 1, :]
    w0 = _level_w(w, depth, k, 0, None)[None, :, None, None]
    if binary:
        ab, bb = a.bool(), b.bool()
        out = torch.where(ab, torch.where(bb, w0[..., 3], w0[..., 2]),
                          torch.where(bb, w0[..., 1], w0[..., 0]))
    else:
        a, b = a.float(), b.float()
        out = lut(a, b, w0[..., 0], w0[..., 1], w0[..., 2], w0[..., 3])
    outs = [out]
    for lvl in range(1, depth):
        wl = _level_w(w, depth, k, lvl, None)[None, :, None, None]
        v = out.unflatten(-1, (out.shape[-1] // 2, 2))
        out = lut(v[..., 0], v[..., 1], wl[..., 0], wl[..., 1], wl[..., 2], wl[..., 3])
        outs.append(out)
    return idx, a, b, outs


def backward(gy, x, leaf, pos, w, depth, binary, need_gx):
    """Hand-written backward with the kernels' algorithm: arg-max window slot only,
    interior recomputed, grad_W summed over samples / positions, grad_x scatter-added.
    Returns (gx (B, Nsrc) fp32 or None, gw like w). Runs without autograd (custom-op
    bodies cannot differentiate); tests check it against autograd of `forward`."""
    bsz, nsrc = x.shape
    k = leaf.shape[0]
    q, win = pos.shape
    idx, a0, b0, outs = _levels(x, leaf, pos, w, depth, binary)
    roots = outs[-1][..., 0]                                       # (B, K, Q, WIN)
    valid = (pos >= 0)[None, None]
    roots = torch.where(valid, roots, torch.full_like(roots, float("-inf")))
    sel = roots.argmax(-1) if win > 1 else torch.zeros(roots.shape[:-1], dtype=torch.long)
    onehot = torch.nn.functional.one_hot(sel, win).to(torch.float32)        # (B, K, Q, WIN)
    g = (onehot * gy.float().unsqueeze(-1)).unsqueeze(-1)                   # (B,K,Q,WIN,1)
    offs, total = level_offsets(depth, k)
    gw = torch.zeros(total, dtype=torch.float32)
    gx = torch.zeros(bsz, nsrc, dtype=torch.float32) if need_gx and not binary else None
    for lvl in range(depth - 1, -1, -1):
        n = (1 << (depth - 1)) >> lvl
        wl = _level_w(w, depth, k, lvl, None)[None, :, None, None]          # (1,K,1,1,n,4)
        if lvl == 0:
            a, b = a0, b0
        else:
            v = outs[lvl - 1].unflatten(-1, (n, 2))
            a, b = v[..., 0], v[..., 1]
        if binary and lvl == 0:
            ab, bb = a.bool(), b.bool()
            bas = [(~ab & ~bb), (~ab & bb), (ab & ~bb), (ab & bb)]
            bas = [m.to(torch.float32) for m in bas]
        else:
            bas = [(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b]
        blk = torch.stack([(g * bs).sum(dim=(0, 2, 3)) for bs in bas], dim=-1)   # (K, n, 4)
        gw[offs[lvl]: offs[lvl] + n * k * 4] = blk.permute(1, 0, 2).reshape(-1)
        if binary and lvl == 0:
            continue
        w0, w1, w2, w3 = (wl[..., e] for e in range(4))
        ga = g * ((w2 - w0) * (1 - b) + (w3 - w1) * b)
        gb = g * ((w1 - w0) * (1 - a) + (w3 - w2) * a)
        if lvl > 0:
            g = torch.stack([ga, gb], dim=-1).flatten(-2)
        elif gx is not None:
            ia, ib = idx[..., 0, :], idx[..., 1, :]                          # (K,Q,WIN,S)
            for gg, ii in ((ga, ia), (gb, ib)):
                gx.index_add_(1, ii.reshape(-1), gg.reshape(bsz, -1))
    return gx, gw


def backward_autograd(gy, x, leaf, pos, w, depth, binary, need_gx):
    """Autograd of `forward`: the independent check of `backward` and of the kernels."""
    with torch.enable_grad():
        xx = x.detach().float().requires_grad_(need_gx and not binary)
        ww = w.detach().clone().requires_grad_(True)
        y, _ = forward(xx if not binary else x, leaf, pos, ww, depth, binary, torch.float32)
        ins = [ww] + ([xx] if need_gx and not binary else [])
        g = torch.autograd.grad(y, ins, gy.float())
    return (g[1] if len(g) > 1 else None), g[0]

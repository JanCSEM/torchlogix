"""Stage 4 (`eval_backend="packed"`): bit-packed hard evaluation.

63 samples per int64 word (bits 0..62, so packing by a shifted sum cannot overflow). A
gate is (t0 & ~a & ~b) | (t1 & ~a & b) | (t2 & a & ~b) | (t3 & a & b) with t_k in {0, -1}
taken from the committed LUT (W > 0); the OR-pool is a bitwise OR over the window. Exact
by construction on exactly-binary inputs, which eval mode guarantees. Unpacked to float32
at the layer boundary (the vote and every non-logic module stay in PyTorch).
"""

import torch

from .plan import pool_table

LANES = 63


def pack(x: torch.Tensor) -> torch.Tensor:
    """(B, ...) exactly-{0,1} -> (ceil(B / 63), ...) int64."""
    b = x.shape[0]
    bw = -(-b // LANES)
    xb = (x != 0).to(torch.int64)
    if bw * LANES != b:
        xb = torch.cat([xb, xb.new_zeros((bw * LANES - b,) + tuple(xb.shape[1:]))], 0)
    xb = xb.reshape(bw, LANES, *xb.shape[1:])
    sh = torch.arange(LANES, device=x.device).view(1, LANES, *([1] * (xb.dim() - 2)))
    return (xb << sh).sum(1)


def unpack(words: torch.Tensor, b: int) -> torch.Tensor:
    sh = torch.arange(LANES, device=words.device).view(1, LANES, *([1] * (words.dim() - 1)))
    bits = (words.unsqueeze(1) >> sh) & 1
    return bits.reshape(-1, *words.shape[1:])[:b].to(torch.float32)


def gate(a, b, m):
    """m: four int64 masks (0 or -1), broadcastable against a, b."""
    na, nb = ~a, ~b
    return (m[0] & na & nb) | (m[1] & na & b) | (m[2] & a & nb) | (m[3] & a & b)


def _masks(w):                                    # w (..., 4) raw logits -> 4 x (...)
    h = (w > 0).to(torch.int64)
    return [-h[..., e] for e in range(4)]


def check_binary(st, x) -> None:
    if st.packed_checked:
        return
    if not bool(((x == 0) | (x == 1)).all()):
        raise ValueError("eval_backend=packed needs exactly-binary layer inputs (eval mode)")
    st.packed_checked = True


def conv_forward(layer, x, pool=None):
    st = layer._lut
    check_binary(st, x)
    b = x.shape[0]
    xp = layer._pad(pack(x))                                      # (Bw, C, Hp, Wp)
    g = layer.connections(xp, 0)                                  # (Bw, 2, K, P, S)
    m = [mm.t()[None, :, None, :] for mm in _masks(layer.tree_weights[0])]   # (1,K,1,S)
    out = gate(g[:, 0], g[:, 1], m)
    for level in range(1, layer.tree_depth):
        v = out.unflatten(-1, (out.shape[-1] // 2, 2))
        m = [mm.t()[None, :, None, :] for mm in _masks(layer.tree_weights[level])]
        out = gate(v[..., 0], v[..., 1], m)
    out = out[..., 0]                                             # (Bw, K, P)
    ho, wo = layer.kernel_positions
    if pool is not None:
        tab, (hq, wq) = pool_table(ho, wo, pool.kernel_size, pool.stride, pool.padding,
                                   pool.ceil_mode)
        tab = tab.to(out.device).long()
        z = torch.cat([out, out.new_zeros(out.shape[0], out.shape[1], 1)], -1)
        idx = torch.where(tab >= 0, tab, torch.full_like(tab, out.shape[-1]))
        win = z[..., idx]                                         # (Bw, K, Q, WIN)
        out = win[..., 0]
        for j in range(1, win.shape[-1]):
            out = out | win[..., j]
        return unpack(out, b).view(b, out.shape[1], hq, wq)
    return unpack(out, b).view(b, out.shape[1], ho, wo)


def dense_forward(layer, x):
    st = layer._lut
    check_binary(st, x)
    lead = x.shape[:-1]
    x = x.reshape(-1, layer.in_dim)
    b = x.shape[0]
    g = layer.connections(pack(x))                                # (Bw, 2, out)
    out = gate(g[:, 0], g[:, 1], _masks(layer.weight))
    return unpack(out, b).view(*lead, layer.out_dim)

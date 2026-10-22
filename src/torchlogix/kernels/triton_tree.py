"""Fused per-tree LUT kernels in Triton (Stage 3, bake-off candidate "triton").

One program = one kernel index k and BLOCK items (b, q) = (sample, pooled position). It
loads the 2^D leaves of the tree through the static gather table, evaluates the D levels
in registers, takes the max over the pool window (WIN positions, row-major, first max
wins like F.max_pool) and writes only the root. Backward recomputes the interior from
the re-read leaves of the arg-max position only; grad_W is reduced per program and
atomically added, grad_x is atomically added (fp32).

Compiled with enable_fp_fusion=False so the fp32 forward keeps the reference operation
order without FMA contraction (bitwise equal to weighted_light_basis_sum_2).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _lut(a, b, w0, w1, w2, w3):
    return (((w0 * ((1.0 - a) * (1.0 - b)) + w1 * ((1.0 - a) * b))
             + w2 * (a * (1.0 - b))) + w3 * (a * b))


@triton.jit
def _lw(w_ptr, k, K, lvl: tl.constexpr, S0: tl.constexpr, e: tl.constexpr):
    """Weights entry e of every node of level lvl for kernel k, shape [1, S0 >> lvl]."""
    n = tl.arange(0, S0 >> lvl)
    off = (2 * S0 - 2 * (S0 >> lvl)) * K * 4
    return tl.load(w_ptr + off + (n * K + k) * 4 + e)[None, :]


@triton.jit
def _level0(a, b, w_ptr, k, K, S0: tl.constexpr, BINARY: tl.constexpr):
    w0 = _lw(w_ptr, k, K, 0, S0, 0)
    w1 = _lw(w_ptr, k, K, 0, S0, 1)
    w2 = _lw(w_ptr, k, K, 0, S0, 2)
    w3 = _lw(w_ptr, k, K, 0, S0, 3)
    if BINARY:
        return tl.where(a != 0, tl.where(b != 0, w3, w2), tl.where(b != 0, w1, w0))
    else:
        return _lut(a, b, w0, w1, w2, w3)


@triton.jit
def _pair_level(out, w_ptr, k, K, lvl: tl.constexpr, S0: tl.constexpr, BLOCK: tl.constexpr):
    v = tl.reshape(out, [BLOCK, S0 >> lvl, 2])
    a, b = tl.split(v)
    return _lut(a, b, _lw(w_ptr, k, K, lvl, S0, 0), _lw(w_ptr, k, K, lvl, S0, 1),
                _lw(w_ptr, k, K, lvl, S0, 2), _lw(w_ptr, k, K, lvl, S0, 3))


@triton.jit
def _load_leaves(x_ptr, leaf_ptr, b, k, p, ok, P, Nsrc, S0: tl.constexpr, BINARY: tl.constexpr):
    f = tl.arange(0, S0)
    lbase = leaf_ptr + ((k * P + p) * 2) * S0
    m = ok[:, None]
    la = tl.load(lbase[:, None] + f[None, :], mask=m, other=0)
    lb = tl.load(lbase[:, None] + S0 + f[None, :], mask=m, other=0)
    xb = x_ptr + b.to(tl.int64)[:, None] * Nsrc
    a = tl.load(xb + la, mask=m, other=0)
    c = tl.load(xb + lb, mask=m, other=0)
    if not BINARY:
        a = a.to(tl.float32)
        c = c.to(tl.float32)
    return a, c, la, lb


@triton.jit
def _tree_root(x_ptr, leaf_ptr, w_ptr, b, k, p, ok, K, P, Nsrc,
               D: tl.constexpr, BINARY: tl.constexpr, BLOCK: tl.constexpr):
    S0: tl.constexpr = 1 << (D - 1)
    a, c, la, lb = _load_leaves(x_ptr, leaf_ptr, b, k, p, ok, P, Nsrc, S0, BINARY)
    out = _level0(a, c, w_ptr, k, K, S0, BINARY)
    for lvl in tl.static_range(1, D):
        out = _pair_level(out, w_ptr, k, K, lvl, S0, BLOCK)
    return tl.reshape(out, [BLOCK])


@triton.jit
def _fwd_kernel(x_ptr, leaf_ptr, pos_ptr, w_ptr, y_ptr, arg_ptr, B, K, P, Q, Nsrc,
                D: tl.constexpr, WIN: tl.constexpr, BINARY: tl.constexpr,
                OUT_BF16: tl.constexpr, STORE_ARG: tl.constexpr, BLOCK: tl.constexpr):
    k = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = i < B * Q
    b = i // Q
    q = i % Q
    best = tl.full([BLOCK], float("-inf"), tl.float32)
    arg = tl.zeros([BLOCK], dtype=tl.int32)
    for j in tl.static_range(WIN):
        p = tl.load(pos_ptr + q * WIN + j, mask=valid, other=-1)
        ok = valid & (p >= 0)
        root = _tree_root(x_ptr, leaf_ptr, w_ptr, b, k, p, ok, K, P, Nsrc, D, BINARY, BLOCK)
        upd = ok & (root > best)
        best = tl.where(upd, root, best)
        arg = tl.where(upd, j, arg)
    o = (b.to(tl.int64) * K + k) * Q + q
    if OUT_BF16:
        tl.store(y_ptr + o, best.to(tl.bfloat16), mask=valid)
    else:
        tl.store(y_ptr + o, best, mask=valid)
    if STORE_ARG:
        tl.store(arg_ptr + o, arg.to(tl.uint8), mask=valid)


@triton.jit
def _inputs_at(a0, c0, w_ptr, k, K, lvl: tl.constexpr, S0: tl.constexpr,
               BINARY: tl.constexpr, BLOCK: tl.constexpr):
    """(A, B) operand tiles of tree level lvl >= 1, recomputed from the leaves."""
    out = _level0(a0, c0, w_ptr, k, K, S0, BINARY)
    for m in tl.static_range(1, lvl):
        out = _pair_level(out, w_ptr, k, K, m, S0, BLOCK)
    v = tl.reshape(out, [BLOCK, S0 >> lvl, 2])
    a, b = tl.split(v)
    return a, b


@triton.jit
def _bwd_kernel(gy_ptr, x_ptr, leaf_ptr, pos_ptr, arg_ptr, w_ptr, gx_ptr, gw_ptr,
                B, K, P, Q, Nsrc,
                D: tl.constexpr, WIN: tl.constexpr, BINARY: tl.constexpr,
                NEED_GX: tl.constexpr, BLOCK: tl.constexpr):
    S0: tl.constexpr = 1 << (D - 1)
    k = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = i < B * Q
    b = i // Q
    q = i % Q
    o = (b.to(tl.int64) * K + k) * Q + q
    if WIN > 1:
        j = tl.load(arg_ptr + o, mask=valid, other=0).to(tl.int32)
    else:
        j = 0
    p = tl.load(pos_ptr + q * WIN + j, mask=valid, other=-1)
    ok = valid & (p >= 0)
    g = tl.load(gy_ptr + o, mask=ok, other=0.0).to(tl.float32)[:, None]      # [BLOCK, 1]
    a0, c0, la, lb = _load_leaves(x_ptr, leaf_ptr, b, k, p, ok, P, Nsrc, S0, BINARY)
    for lvl in tl.static_range(D - 1, -1, -1):
        n: tl.constexpr = S0 >> lvl
        off = (2 * S0 - 2 * n) * K * 4
        nn = tl.arange(0, n)
        gp = gw_ptr + off + (nn * K + k) * 4
        if lvl == 0:
            if BINARY:
                pa = a0 != 0
                pb = c0 != 0
                na = a0 == 0
                nb = c0 == 0
                gz = g + tl.zeros([BLOCK, n], dtype=tl.float32)
                zero = tl.zeros([BLOCK, n], dtype=tl.float32)
                tl.atomic_add(gp + 0, tl.sum(tl.where(na & nb, gz, zero), axis=0))
                tl.atomic_add(gp + 1, tl.sum(tl.where(na & pb, gz, zero), axis=0))
                tl.atomic_add(gp + 2, tl.sum(tl.where(pa & nb, gz, zero), axis=0))
                tl.atomic_add(gp + 3, tl.sum(tl.where(pa & pb, gz, zero), axis=0))
            else:
                a = a0
                c = c0
                tl.atomic_add(gp + 0, tl.sum(g * ((1.0 - a) * (1.0 - c)), axis=0))
                tl.atomic_add(gp + 1, tl.sum(g * ((1.0 - a) * c), axis=0))
                tl.atomic_add(gp + 2, tl.sum(g * (a * (1.0 - c)), axis=0))
                tl.atomic_add(gp + 3, tl.sum(g * (a * c), axis=0))
                if NEED_GX:
                    w0 = _lw(w_ptr, k, K, 0, S0, 0)
                    w1 = _lw(w_ptr, k, K, 0, S0, 1)
                    w2 = _lw(w_ptr, k, K, 0, S0, 2)
                    w3 = _lw(w_ptr, k, K, 0, S0, 3)
                    ga = g * ((w2 - w0) * (1.0 - c) + (w3 - w1) * c)
                    gb = g * ((w1 - w0) * (1.0 - a) + (w3 - w2) * a)
                    xg = gx_ptr + b.to(tl.int64)[:, None] * Nsrc
                    tl.atomic_add(xg + la, ga, mask=ok[:, None])
                    tl.atomic_add(xg + lb, gb, mask=ok[:, None])
        else:
            a, c = _inputs_at(a0, c0, w_ptr, k, K, lvl, S0, BINARY, BLOCK)
            w0 = _lw(w_ptr, k, K, lvl, S0, 0)
            w1 = _lw(w_ptr, k, K, lvl, S0, 1)
            w2 = _lw(w_ptr, k, K, lvl, S0, 2)
            w3 = _lw(w_ptr, k, K, lvl, S0, 3)
            tl.atomic_add(gp + 0, tl.sum(g * ((1.0 - a) * (1.0 - c)), axis=0))
            tl.atomic_add(gp + 1, tl.sum(g * ((1.0 - a) * c), axis=0))
            tl.atomic_add(gp + 2, tl.sum(g * (a * (1.0 - c)), axis=0))
            tl.atomic_add(gp + 3, tl.sum(g * (a * c), axis=0))
            ga = g * ((w2 - w0) * (1.0 - c) + (w3 - w1) * c)
            gb = g * ((w1 - w0) * (1.0 - a) + (w3 - w2) * a)
            g = tl.reshape(tl.join(ga, gb), [BLOCK, 2 * n])


def _block(depth: int) -> int:
    return 128 if depth <= 3 else 64


def fwd(x, leaf, pos, w, depth, binary, out_dtype, need_arg):
    b, nsrc = x.shape
    k, p = leaf.shape[0], leaf.shape[1]
    q, win = pos.shape
    y = torch.empty(b, k, q, dtype=out_dtype, device=x.device)
    store_arg = bool(need_arg and win > 1)
    arg = torch.empty(b, k, q, dtype=torch.uint8, device=x.device) if store_arg else \
        torch.empty(0, dtype=torch.uint8, device=x.device)
    block = _block(depth)
    grid = (k, triton.cdiv(b * q, block))
    _fwd_kernel[grid](x, leaf, pos, w, y, arg, b, k, p, q, nsrc,
                      D=depth, WIN=win, BINARY=binary, OUT_BF16=(out_dtype == torch.bfloat16),
                      STORE_ARG=store_arg, BLOCK=block, enable_fp_fusion=False)
    return y, arg


def bwd(gy, x, leaf, pos, arg, w, depth, binary, need_gx):
    b, nsrc = x.shape
    k, p = leaf.shape[0], leaf.shape[1]
    q, win = pos.shape
    gw = torch.zeros_like(w, dtype=torch.float32)
    gx = torch.zeros(b, nsrc, dtype=torch.float32, device=x.device) if need_gx else \
        torch.empty(0, dtype=torch.float32, device=x.device)
    block = _block(depth)
    grid = (k, triton.cdiv(b * q, block))
    _bwd_kernel[grid](gy.contiguous(), x, leaf, pos, arg, w, gx, gw, b, k, p, q, nsrc,
                      D=depth, WIN=win, BINARY=binary, NEED_GX=need_gx and not binary,
                      BLOCK=block, enable_fp_fusion=False)
    return (gx if need_gx and not binary else None), gw

"""Static tables derived from a layer's wiring (computed once, cached on the layer)."""

import math

import torch
import torch.nn.functional as F


def pool_out_size(n: int, k: int, s: int, p: int, ceil_mode: bool) -> int:
    """Output length of F.max_pool along one axis (dilation 1)."""
    num = n + 2 * p - k
    o = (-(-num // s) if ceil_mode else num // s) + 1
    if ceil_mode and (o - 1) * s >= n + p:      # last window must start inside input + left pad
        o -= 1
    return o


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


def pool_table(h: int, w: int, kernel, stride, padding, ceil_mode: bool):
    """Window table of F.max_pool2d over an (h, w) grid.

    Returns (tab, (hq, wq)): tab is int32 (hq*wq, kh*kw); entry = flat input position
    h_in*w + w_in of the j-th window element in ROW-MAJOR window order (the order
    F.max_pool2d scans, so first-max tie-breaking agrees), -1 where the element lies in
    the padding or past the edge (ceil_mode overhang).
    """
    kh, kw = _pair(kernel)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    hq = pool_out_size(h, kh, sh, ph, ceil_mode)
    wq = pool_out_size(w, kw, sw, pw, ceil_mode)
    oh = torch.arange(hq).view(hq, 1, 1, 1) * sh - ph + torch.arange(kh).view(1, 1, kh, 1)
    ow = torch.arange(wq).view(1, wq, 1, 1) * sw - pw + torch.arange(kw).view(1, 1, 1, kw)
    oh, ow = oh.expand(hq, wq, kh, kw), ow.expand(hq, wq, kh, kw)
    ok = (oh >= 0) & (oh < h) & (ow >= 0) & (ow < w)
    tab = torch.where(ok, oh * w + ow, torch.full_like(oh, -1))
    return tab.reshape(hq * wq, kh * kw).to(torch.int32), (hq, wq)


def identity_table(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.int32).view(n, 1)


def conv_leaf_table(layer) -> torch.Tensor:
    """int32 (K, P, 2, S): flat index, into one sample of the PADDED input
    (channels * Hp * Wp), of the A (slot 0) and B (slot 1) tap of every level-0 node.

    Also asserts that levels >= 1 pair neighbours (node j <- children 2j, 2j+1), the
    only structure the view/fused paths implement; anything else raises.
    """
    conn = layer.connections
    idx = conn.indices
    for lvl in range(1, layer.tree_depth):
        size = layer.lut_rank ** (layer.tree_depth - lvl)
        base = torch.arange(size).view(-1, layer.lut_rank).transpose(0, 1)
        if not torch.equal(idx[lvl].cpu(), base):
            raise ValueError(f"tree level {lvl} wiring is not the neighbour-pairing the "
                             "view/fused backends implement")
    i0 = idx[0]                                   # (2, K, P, S, nd + 1)
    if i0.shape[0] != 2 or layer.conv_dimension != 2:
        raise ValueError("lut backends support LogicConv2d with lut_rank 2 only")
    padded = [d + 2 * p for d, p in zip(layer.in_dim, layer.padding)]
    h, w, c = i0[..., 0].long(), i0[..., 1].long(), i0[..., 2].long()
    flat = (c * padded[0] + h) * padded[1] + w     # (2, K, P, S)
    return flat.permute(1, 2, 0, 3).contiguous().to(torch.int32)


def dense_leaf_table(layer) -> torch.Tensor:
    """int32 (out, 1, 2, 1) from a FixedDenseConnections `indices` (2, out)."""
    ind = layer.connections.indices
    if ind.shape[0] != 2:
        raise ValueError("lut backends support lut_rank 2 only")
    return ind.t().contiguous().to(torch.int32).view(-1, 1, 2, 1)


def level_offsets(depth: int, k: int):
    """Offsets (elements) of every level's (S_l, K, 4) block inside the concatenated
    sigma(W) buffer, and the total size."""
    s0 = 1 << (depth - 1)
    offs, o = [], 0
    for lvl in range(depth):
        offs.append(o)
        o += (s0 >> lvl) * k * 4
    return offs, o

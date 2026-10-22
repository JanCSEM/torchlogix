"""Stage 1 (`lut_backend="lookup"`): exact PyTorch rewrites of the reference path.

(1a) Tree levels >= 1: `indices[level] = arange(size).view(-1, 2).T`, so the reference
     gather is a pure reshape. A strided view replaces gather + movedim; the arithmetic
     (`weighted_light_basis_sum_2` through the layer's own parametrization) is untouched,
     so forward and backward are bitwise equal to the reference.
(1b) First logic layer, level 0, when the input is exactly {0, 1}: both inputs are bits,
     so the `light` gate output IS a table lookup, out = w[2a + b], bitwise equal to the
     reference forward. Only the 2-bit address is saved for backward; grad_W is a masked
     sum (a reordered sum, not bitwise).
"""

import torch


def sample_levels(layer):
    """sigma(W) (train) / (W > 0) (eval) of every tree level, exactly as the layer's
    parametrization computes it. Each entry is (S_l, K, 4)."""
    p = layer.parametrization
    fn = p._sample_train if layer.training else p._sample_eval
    return [fn(w) for w in layer.tree_weights]


class _BitSelect(torch.autograd.Function):
    """out = w[..., 2a + b] for exactly-binary a, b (uint8); w broadcasts against a."""

    @staticmethod
    def forward(ctx, a, b, w):
        ab, bb = a.bool(), b.bool()
        out = torch.where(ab, torch.where(bb, w[..., 3], w[..., 2]),
                          torch.where(bb, w[..., 1], w[..., 0]))
        ctx.save_for_backward(a, b)
        ctx.wshape = w[..., 0].shape
        return out

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        ab, bb = a.bool(), b.bool()
        masks = (~ab & ~bb, ~ab & bb, ab & ~bb, ab & bb)
        gw = [torch.where(m, g, torch.zeros_like(g)).sum_to_size(ctx.wshape) for m in masks]
        return None, None, torch.stack(gw, dim=-1)


def check_binary(st, x) -> None:
    """One-time check that a layer marked `binary_input` really sees {0, 1}."""
    if st.binary_checked:
        return
    if not bool(((x == 0) | (x == 1)).all()):
        raise ValueError("binary_input layer got values outside {0, 1}; the lookup/fused "
                         "level-0 shortcut is exact only for a thermometer/bit input")
    st.binary_checked = True


def _bits(x):
    return x.to(torch.uint8)


def conv_forward(layer, x):
    st = layer._lut
    x = layer._pad(x)
    if x.dtype != torch.float32 and not st.binary_input:
        x = x.float()
    if st.binary_input:
        check_binary(st, x)
        g = layer.connections(_bits(x), 0)                   # (B, 2, K, P, S) uint8
        w = sample_levels(layer)[0]                          # (S, K, 4)
        wt = w.permute(1, 0, 2)[None, :, None]               # (1, K, 1, S, 4)
        out = _BitSelect.apply(g[:, 0], g[:, 1], wt)         # (B, K, P, S)
    else:
        out = layer.parametrization.forward(
            layer.connections(x, 0), layer.tree_weights[0], layer.training,
            contraction='fc,bcsf->bcsf')
    for level in range(1, layer.tree_depth):
        v = out.unflatten(-1, (out.shape[-1] // 2, 2)).movedim(-1, 1)   # (B, 2, K, P, S/2)
        out = layer.parametrization.forward(
            v, layer.tree_weights[level], layer.training, contraction='fc,bcsf->bcsf')
    return out.view(out.shape[0], out.shape[1], *layer.kernel_positions)


def dense_forward(layer, x):
    st = layer._lut
    from ..functional import GradFactor
    if layer.grad_factor != 1.0:
        x = GradFactor.apply(x, layer.grad_factor)
    lead = x.shape[:-1]
    x = x.view(-1, layer.in_dim)
    if st.binary_input:
        check_binary(st, x)
        g = layer.connections(_bits(x))                      # (B, 2, out) uint8
        w = sample_levels_dense(layer)                       # (out, 4)
        out = _BitSelect.apply(g[:, 0], g[:, 1], w[None])
    else:
        if x.dtype != torch.float32:
            x = x.float()
        out = layer.parametrization.forward(layer.connections(x), layer.weight,
                                            layer.training, contraction='n,bn->bn')
    return out.view(*lead, layer.out_dim)


def sample_levels_dense(layer):
    p = layer.parametrization
    return (p._sample_train if layer.training else p._sample_eval)(layer.weight)

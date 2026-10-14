"""Dilated logic convolutions: geometry matches torch.nn.Conv, taps are spaced by the
dilation, and dilation=1 is bit-identical to the pre-dilation layer."""

import pytest
import torch

from torchlogix.layers import LogicConv2d, LogicConv3d


def _conv(seed=0, **kw):
    torch.manual_seed(seed)
    args = dict(in_dim=(1, 64), channels=3, num_kernels=8, tree_depth=3,
                receptive_field_size=(1, 5), parametrization="light")
    args.update(kw)
    return LogicConv2d(**args)


def test_dilation_one_is_identical_to_default():
    a, b = _conv(), _conv(dilation=1)
    for ia, ib in zip(a.connections.indices, b.connections.indices):
        assert torch.equal(ia, ib)
    x = torch.rand(2, 3, 1, 64)
    a.eval(), b.eval()
    assert torch.equal(a(x), b(x))


@pytest.mark.parametrize("k,d,pad,stride", [(3, 2, 0, 1), (5, 4, 8, 1), (3, 8, 8, 2),
                                            (2, 3, 0, 2), (5, 1, 2, 1)])
def test_output_size_matches_torch_conv(k, d, pad, stride):
    layer = _conv(receptive_field_size=(1, k), dilation=(1, d), padding=(0, pad),
                  stride=(1, stride))
    ref = torch.nn.Conv2d(3, 8, (1, k), dilation=(1, d), padding=(0, pad),
                          stride=(1, stride))
    x = torch.rand(2, 3, 1, 64)
    assert layer(x).shape == ref(x).shape
    assert layer.kernel_positions == list(ref(x).shape[2:])


def test_taps_lie_on_the_dilated_grid():
    d, k = 4, 5
    layer = _conv(receptive_field_size=(1, k), dilation=(1, d), tree_depth=4)
    idx = layer.connections.indices[0]              # (L, K, P, S, 3): (h, w, c)
    t = idx[..., 1]                                  # time coordinate
    start = torch.arange(idx.shape[2]).view(1, 1, -1, 1)
    off = t - start
    assert ((off % d) == 0).all() and off.min() >= 0 and off.max() <= d * (k - 1)
    assert (idx[..., 0] == 0).all()                  # height axis untouched


def test_output_ignores_inputs_off_the_tap_grid():
    d, k = 3, 3
    layer = _conv(receptive_field_size=(1, k), dilation=(1, d)).eval()
    x = (torch.rand(1, 3, 1, 64) > 0.5).float()
    y = layer(x)
    on_grid = {0 + d * j for j in range(k)}         # taps of output position 0
    x2 = x.clone()
    off = [t for t in range(d * (k - 1) + 1) if t not in on_grid]
    x2[..., off] = 1 - x2[..., off]                  # flip only off-grid inputs
    assert torch.equal(layer(x2)[..., 0], y[..., 0])


def test_extent_must_fit_the_input():
    with pytest.raises(AssertionError, match="extent"):
        _conv(in_dim=(1, 8), receptive_field_size=(1, 3), dilation=(1, 4))


def test_3d_dilation_shape():
    torch.manual_seed(0)
    layer = LogicConv3d(in_dim=(6, 8, 8), channels=2, num_kernels=4, tree_depth=2,
                        receptive_field_size=3, dilation=(1, 2, 2), padding=(1, 2, 2),
                        parametrization="light")
    ref = torch.nn.Conv3d(2, 4, 3, dilation=(1, 2, 2), padding=(1, 2, 2))
    x = torch.rand(1, 2, 6, 8, 8)
    assert layer(x).shape == ref(x).shape

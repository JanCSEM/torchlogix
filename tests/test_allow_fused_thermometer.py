"""The thermometer fold pairs a FixedBinarization with the conv it feeds, by adjacency in one Sequential or
by `allow_fused_thermometer` (the two may sit anywhere in the model). On CPU (the reference kernel) the fold is bitwise
the unfolded fused path; a fold that was requested and cannot apply raises."""

import copy

import pytest
import torch

from torchlogix.backend import backend_report, allow_fused_thermometer, set_backend
from torchlogix.layers import GroupSum, LogicConv2d, LogicDense, OrPooling2d
from torchlogix.layers.binarization import FixedBinarization

LL = dict(lut_rank=2, parametrization="light", parametrization_kwargs={"weight_init": "residual"})


def _parts():
    torch.manual_seed(0)
    bits = FixedBinarization(torch.tensor([[-0.5, 0.0, 0.5]]), feature_dim=1, thresh_jitter=0.0)
    conv = LogicConv2d(in_dim=(1, 16), channels=3, num_kernels=4, tree_depth=2, receptive_field_size=(1, 3),
                       padding=(0, 1), stride=1, connections_kwargs={"channel_group_size": None}, **LL)
    return bits, conv, OrPooling2d(kernel_size=(1, 2), stride=(1, 2)), LogicDense(in_dim=32, out_dim=16, **LL)


class Apart(torch.nn.Module):
    """The thermometer and the conv are children of different modules; `declared` says they are a pair."""

    def __init__(self, declared=True):
        super().__init__()
        bits, conv, pool, dense = _parts()
        self.encoder = torch.nn.Sequential(bits)
        self.block = torch.nn.Sequential(conv, pool)
        self.head = torch.nn.Sequential(torch.nn.Flatten(), dense)
        self.vote = GroupSum(k=2, tau=1.0)
        if declared:
            allow_fused_thermometer(bits, conv)

    def forward(self, x):
        return self.vote(self.head(self.block(self.encoder(x))))


def _run(model, x):
    model = copy.deepcopy(model)
    with torch.no_grad():
        hard = model.eval()(x)
    logits = model.train()(x)
    logits.sum().backward()
    return hard, logits.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None]


def _fold_of(model):
    return next(r for r in backend_report(model) if r["type"] == "LogicConv2d")["fold"]


def test_declared_pair_folds_and_is_bitwise_the_unfolded_path():
    x = torch.randn(3, 1, 1, 16)
    off, on = Apart(), Apart()
    on.load_state_dict(off.state_dict())
    set_backend(off, "fused", fold=False)
    set_backend(on, "fused", fold=True)
    assert _fold_of(on) is True and _fold_of(off) is False
    a, b = _run(off, x), _run(on, x)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert all(torch.equal(p, q) for p, q in zip(a[2], b[2])) and len(a[2]) == len(b[2])
    on.eval()(x)
    assert next(r for r in backend_report(on) if r["type"] == "LogicConv2d")["fold_ran"] is True


def test_the_declared_thermometer_returns_bits_outside_the_model_run():
    m = Apart()
    set_backend(m, "fused", fold=True)
    x = torch.randn(2, 1, 1, 16)
    assert m.encoder(x).shape == (2, 3, 1, 16)           # armed only while the whole model runs


def test_adjacent_pairs_in_one_sequential_still_fold():
    bits, conv, pool, dense = _parts()
    m = torch.nn.Sequential(bits, conv, pool, torch.nn.Flatten(), dense, GroupSum(k=2, tau=1.0))
    set_backend(m, "fused", fold=True)
    assert _fold_of(m) is True


def test_a_requested_fold_that_cannot_apply_raises_and_off_does_not():
    m = Apart(declared=False)
    set_backend(m, "fused", fold=False)
    with pytest.raises(ValueError, match="fold requested"):
        set_backend(m, "fused", fold=True)


def test_allow_fused_thermometer_checks_its_arguments():
    bits, conv, _, dense = _parts()
    with pytest.raises(TypeError):
        allow_fused_thermometer(bits, dense)
    assert allow_fused_thermometer(bits, conv) is True
    assert allow_fused_thermometer(torch.nn.Identity(), conv) is False

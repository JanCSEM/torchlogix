"""weight_init="residual-frac": a fraction of gates start as the pass-through wire,
the rest as a DEFINITE other Boolean function at the same magnitude.

⚠ This is NOT "residual init plus noise", and the difference is measurable. Writing
the non-wire share as weak Gaussian noise leaves it near |logit| ~0.8 against the
wires' 4.0 with ~1% of entries undecided; that version measures NULL. Writing it as
committed alternative gates at the SAME magnitude moved a matched CIFAR-S pair from
0.5834 to 0.5877. So "0% undecided" is the property under test, not a detail.
"""

import pytest
import torch

from torchlogix.layers import LogicDense

N = 20_000          # the reference verified its own split on a 20,000-gate layer


def make(param, frac=0.95, logit=4.0, seed=0):
    torch.manual_seed(seed)
    return LogicDense(in_dim=64, out_dim=N, parametrization=param,
                      parametrization_kwargs={"weight_init": "residual-frac",
                                              "residual_frac": frac,
                                              "residual_logit": logit})


def split(layer, param):
    """(is_wire mask, distinct truth tables) for either parametrization."""
    w = layer.weight.detach().reshape(N, -1)
    if param == "light":
        bits = (w > 0).long()
        # address = 2*A + B (light_basis is [(1-A)(1-B), (1-A)B, A(1-B), AB]), so
        # pass-through of A is the UPPER half of the table.
        wire = torch.tensor([0, 0, 1, 1])
        return (bits == wire).all(-1), {tuple(r.tolist()) for r in bits}, w
    idx = w.argmax(-1)
    return idx == 3, set(idx.tolist()), w          # gate 3 = pass-through of A


@pytest.mark.parametrize("param", ["light", "raw"])
def test_wire_fraction_matches_the_requested_frac(param):
    is_wire, _, _ = split(make(param), param)
    assert abs(is_wire.float().mean().item() - 0.95) < 0.01


@pytest.mark.parametrize("param", ["light", "raw"])
def test_nothing_starts_undecided(param):
    """THE point of the lever: every entry is committed, at one magnitude."""
    _, _, w = split(make(param), param)
    assert (w.abs() < 1e-6).sum().item() == 0
    assert torch.allclose(w.abs().unique(), torch.tensor([4.0]))


@pytest.mark.parametrize("param", ["light", "raw"])
def test_all_sixteen_gate_types_appear(param):
    """The non-wire share is uniform over the OTHER tables, not one fixed table."""
    _, types, _ = split(make(param), param)
    assert len(types) == 16


@pytest.mark.parametrize("param", ["light", "raw"])
def test_frac_one_is_all_wires(param):
    is_wire, types, _ = split(make(param, frac=1.0), param)
    assert is_wire.all()
    assert len(types) == 1


@pytest.mark.parametrize("param", ["light", "raw"])
def test_frac_zero_never_draws_the_wire(param):
    """With frac=0 the wire must be EXCLUDED, not merely improbable."""
    is_wire, types, _ = split(make(param, frac=0.0), param)
    assert not is_wire.any()
    assert len(types) == 15


@pytest.mark.parametrize("param", ["light", "raw"])
def test_logit_magnitude_is_configurable(param):
    _, _, w = split(make(param, logit=2.5), param)
    assert torch.allclose(w.abs().unique(), torch.tensor([2.5]))


@pytest.mark.parametrize("param", ["light", "raw"])
def test_wires_are_not_a_contiguous_prefix(param):
    """Drawn per gate, so the split cannot correlate with position in the layer."""
    is_wire, _, _ = split(make(param), param)
    first, last = is_wire[:N // 2].float().mean(), is_wire[N // 2:].float().mean()
    assert abs(first - last) < 0.02
    assert not is_wire[:int(0.95 * N)].all()      # would hold for a prefix layout


def test_light_wire_actually_passes_input_zero():
    """Behavioural: the wire must forward A (input 0), not B.

    Value-based checks miss this -- the MSB-first address order makes `a & 1` look
    plausible while selecting the wrong input.
    """
    torch.manual_seed(0)
    L = LogicDense(in_dim=2, out_dim=64, parametrization="light",
                   parametrization_kwargs={"weight_init": "residual-frac",
                                           "residual_frac": 1.0}).eval()
    with torch.no_grad():
        L.connections.indices.copy_(torch.zeros_like(L.connections.indices))
        L.connections.indices[1] = 1                      # in0 = A, in1 = B
        x = torch.tensor([[1.0, 0.0]])                    # A=1, B=0
        assert torch.equal(L(x), torch.ones(1, 64))       # forwards A
        x = torch.tensor([[0.0, 1.0]])
        assert torch.equal(L(x), torch.zeros(1, 64))      # not B

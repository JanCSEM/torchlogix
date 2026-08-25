"""Threshold jitter: a train-only, correlated perturbation of the thermometer grid.

The two assertions that matter most are `lambda=0 is a bitwise no-op` and `eval is
unaffected`. If either fails, every downstream A/B is measuring something other than
the regularizer.
"""

import math

import pytest
import torch

from torchlogix.layers import jitter_disabled, pinned_jitter
from torchlogix.layers.binarization import (
    Binarization,
    DummyBinarization,
    FixedBinarization,
    LearnableBinarization,
    SoftBinarization,
)

CONV_T = torch.tensor([[0.2, 0.5, 0.8], [0.1, 0.4, 0.9]])   # (C=2, K=3)
GLOBAL_T = torch.tensor([[0.25, 0.5, 0.75]])                 # (1, K=3)


def conv_x(b=4):
    torch.manual_seed(0)
    return torch.rand(b, 2, 5, 5)


def dense_x(b=4):
    torch.manual_seed(0)
    return torch.rand(b, 2)


def make(cls, thresholds=CONV_T, **kw):
    return cls(thresholds=thresholds.clone(), feature_dim=1, **kw)


# --------------------------------------------------------------------------- #
# the load-bearing pair
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cls", [FixedBinarization, SoftBinarization, LearnableBinarization])
def test_lambda_zero_is_bitwise_noop(cls):
    """lambda=0 must be indistinguishable from not passing the kwarg at all."""
    x = conv_x()
    a, b = make(cls), make(cls, thresh_jitter=0.0)
    a.train(); b.train()
    torch.manual_seed(1); out_a = a(x)
    torch.manual_seed(1); out_b = b(x)
    assert torch.equal(out_a, out_b)


@pytest.mark.parametrize("cls", [FixedBinarization, SoftBinarization, LearnableBinarization])
def test_eval_is_unaffected_by_jitter(cls):
    """Jitter is train-only: eval must be deterministic AND equal to the lambda=0 eval."""
    x = conv_x()
    jit, ref = make(cls, thresh_jitter=0.5), make(cls)
    jit.eval(); ref.eval()
    o1, o2 = jit(x), jit(x)
    assert torch.equal(o1, o2)          # deterministic
    assert torch.equal(o1, ref(x))      # and identical to no jitter


# --------------------------------------------------------------------------- #
# the mechanism
# --------------------------------------------------------------------------- #

def test_train_is_stochastic():
    m = make(FixedBinarization, thresh_jitter=0.25); m.train()
    x = conv_x()
    assert not torch.equal(m(x), m(x))


def test_gauss_sigma_is_lambda_times_mean_spacing():
    """t~ = t + eta * (lambda * spacing), spacing = mean adjacent gap per channel."""
    t = torch.tensor([[0.0, 1.0, 3.0]])            # gaps 1 and 2 -> spacing 1.5
    m = FixedBinarization(thresholds=t, thresh_jitter=0.4); m.train()
    assert torch.allclose(m._level_spacing(t), torch.tensor([[1.5]]))
    torch.manual_seed(0)
    disp = m._apply_threshold_jitter(t, 16384) - t          # displacement, not value
    assert math.isclose(disp.std().item(), 0.4 * 1.5, rel_tol=0.05)
    assert abs(disp.mean().item()) < 0.02                   # zero-mean


def test_gauss_grid_can_go_non_monotone():
    """Levels move independently and are deliberately NOT clamped or re-sorted.

    This is the characterized behaviour of the reference arm, not a defect. Use
    mode='bounded' when crossing must be impossible.
    """
    t = torch.tensor([[0.0, 1.0, 2.0]])
    m = FixedBinarization(thresholds=t, thresh_jitter=0.5); m.train()
    torch.manual_seed(0)
    jit = m._apply_threshold_jitter(t, 20000)
    frac = (jit.diff(dim=-1) < 0).any(dim=-1).float().mean().item()
    assert 0.05 < frac < 0.60, frac


def test_bounded_never_crosses():
    t = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    m = FixedBinarization(thresholds=t, thresh_jitter=10.0, thresh_jitter_mode="bounded")
    m.train()
    torch.manual_seed(0)
    jit = m._apply_threshold_jitter(t, 20000)
    # No CROSSING is the invariant: |tanh| < 1 strictly, so neighbours moving
    # maximally toward each other meet at the midpoint at worst. At lambda=10 tanh
    # saturates to exactly +-1 in float32, so `meet` really does occur -- hence >= 0,
    # not > 0. At a realistic lambda the inequality is strict, checked below.
    assert (jit.diff(dim=-1) >= 0).all()
    m2 = FixedBinarization(thresholds=t, thresh_jitter=0.5, thresh_jitter_mode="bounded")
    m2.train()
    torch.manual_seed(0)
    assert (m2._apply_threshold_jitter(t, 20000).diff(dim=-1) > 0).all()


def test_bounded_displacement_is_within_half_the_nearest_gap():
    t = torch.tensor([[0.0, 1.0, 3.0]])
    m = FixedBinarization(thresholds=t, thresh_jitter=10.0, thresh_jitter_mode="bounded")
    m.train()
    torch.manual_seed(0)
    jit = m._apply_threshold_jitter(t, 4096)
    assert ((jit - t).abs() <= torch.tensor([[0.5, 0.5, 1.0]])).all()


def test_indep_marginal_matches_the_analytic_flip_rate():
    """P(flip | d) = Phi(-|d| / (lambda*s)), the same marginal gauss induces."""
    t = torch.tensor([[0.0]])
    lam = 0.5
    m = FixedBinarization(thresholds=t, thresh_jitter=lam, thresh_jitter_mode="indep")
    m.train()
    s = m._level_spacing(t).item()
    for d in (0.05, 0.2, 0.5):
        x = torch.full((60000, 1), d)
        torch.manual_seed(0)
        bits = m(x)
        expected = 0.5 * (1.0 + math.erf(-abs(d) / (lam * s) / math.sqrt(2)))
        assert abs((1.0 - bits.mean().item()) - expected) < 0.01, (d, expected)


def test_indep_is_uncorrelated_across_positions_but_gauss_is_not():
    """The A/B that proves correlation is the mechanism, not the marginal rate."""
    x = torch.full((256, 1, 1, 64), 0.5001)      # all positions just above threshold
    t = torch.tensor([[0.5]])

    g = FixedBinarization(thresholds=t, feature_dim=1, thresh_jitter=1.0); g.train()
    i = FixedBinarization(thresholds=t, feature_dim=1, thresh_jitter=1.0,
                          thresh_jitter_mode="indep"); i.train()
    torch.manual_seed(0)
    gv = g(x).flatten(1).std(dim=1).mean()       # within-sample spread across positions
    torch.manual_seed(0)
    iv = i(x).flatten(1).std(dim=1).mean()
    assert gv.item() < 1e-6         # one shared draw -> every position agrees
    assert iv.item() > 0.2          # independent draws -> positions disagree


# --------------------------------------------------------------------------- #
# pinning and the clean pass
# --------------------------------------------------------------------------- #

def test_pin_makes_two_forwards_identical_and_release_restores():
    m = make(FixedBinarization, thresh_jitter=0.3); m.train()
    x = conv_x()
    with pinned_jitter(m):
        assert torch.equal(m(x), m(x))
    assert m._jitter_cache is None and not m._pin_jitter
    assert not torch.equal(m(x), m(x))


def test_pin_cache_is_not_in_state_dict():
    m = make(FixedBinarization, thresh_jitter=0.3); m.train()
    with pinned_jitter(m):
        m(conv_x())
    assert not any("jitter" in k for k in m.state_dict())


def test_jitter_disabled_keeps_training_true():
    m = make(FixedBinarization, thresh_jitter=0.5); m.train()
    x = conv_x()
    with jitter_disabled(m):
        assert m.training is True
        assert torch.equal(m(x), m(x))                  # clean and deterministic
        clean = m(x)
    assert m.thresh_jitter == 0.5                       # restored
    ref = make(FixedBinarization); ref.train()
    assert torch.equal(clean, ref(x))                   # equals the true lambda=0 pass


# --------------------------------------------------------------------------- #
# shapes, guards, plumbing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("thr,x_fn", [(CONV_T, conv_x), (GLOBAL_T, lambda: conv_x())])
def test_conv_shapes_survive_jitter(thr, x_fn):
    x = x_fn()
    ref = FixedBinarization(thresholds=thr.clone(), feature_dim=1); ref.train()
    jit = FixedBinarization(thresholds=thr.clone(), feature_dim=1, thresh_jitter=0.5); jit.train()
    assert jit(x).shape == ref(x).shape
    assert not torch.equal(jit(x), ref(x))              # jitter actually applied


def test_dense_shapes_survive_jitter():
    x = dense_x()
    ref = FixedBinarization(thresholds=CONV_T.clone()); ref.train()
    jit = FixedBinarization(thresholds=CONV_T.clone(), thresh_jitter=0.5); jit.train()
    assert jit(x).shape == ref(x).shape


def test_fixed_and_soft_actually_see_the_kwarg():
    """Regression: both forwards used to read self.thresholds directly.

    They bypassed get_thresholds(), so the jitter kwarg was silently inert on exactly
    the two binarizations the CIFAR models use.
    """
    x = conv_x()
    for cls in (FixedBinarization, SoftBinarization):
        a, b = make(cls), make(cls, thresh_jitter=0.5)
        a.train(); b.train()
        assert not torch.equal(a(x), b(x)), cls.__name__


def test_indep_rejected_for_soft_and_learnable():
    x = conv_x()
    for cls in (SoftBinarization, LearnableBinarization):
        m = make(cls, thresh_jitter=0.5, thresh_jitter_mode="indep"); m.train()
        with pytest.raises(ValueError, match="indep"):
            m(x)


def test_dummy_rejects_jitter():
    with pytest.raises(ValueError, match="no meaning"):
        DummyBinarization(thresh_jitter=0.25)
    DummyBinarization(thresh_jitter=0.0)        # fine


def test_bad_mode_rejected():
    with pytest.raises(ValueError, match="thresh_jitter_mode"):
        FixedBinarization(thresholds=CONV_T.clone(), thresh_jitter_mode="nope")

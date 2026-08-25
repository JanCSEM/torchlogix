from contextlib import contextmanager
from typing import Union, List
from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn.functional as F

from ..functional import sigmoid, gumbel_sigmoid


def setup_binarization(thresholds, binarization: str, **binarization_kwargs):
    bin_dict = {
        "dummy": DummyBinarization,
        "fixed": FixedBinarization,
        "soft": SoftBinarization,
        "learnable": LearnableBinarization
    }
    if binarization not in bin_dict:
        raise ValueError(
            f"Unsupported binarization method: {binarization}. "
            f"Choose from {list(bin_dict.keys())}."
        )
    bin_cls = bin_dict[binarization]
    return bin_cls(thresholds=thresholds, **binarization_kwargs)


class Binarization(torch.nn.Module, ABC):
    """Abstract base class for binarization modules."""
    def __init__(
            self,
            thresholds: Tensor | List,
            feature_dim=-2,
            thresh_jitter: float = 0.0,
            thresh_jitter_mode: str = "gauss",
            **kwargs
        ):
        super().__init__()
        if isinstance(thresholds, list):
            thresholds = torch.tensor(thresholds, dtype=torch.float32)
        self.register_buffer('thresholds', thresholds)
        self.feature_dim = feature_dim

        if thresh_jitter_mode not in ("gauss", "bounded", "indep"):
            raise ValueError(
                f"thresh_jitter_mode must be 'gauss', 'bounded' or 'indep', "
                f"got {thresh_jitter_mode!r}"
            )
        self.thresh_jitter = float(thresh_jitter)
        self.thresh_jitter_mode = str(thresh_jitter_mode)
        # Per-forward state, NOT buffers: these must never enter state_dict().
        self._pin_jitter = False
        self._jitter_cache = None

    # ------------------------------------------------------------------ #
    # thresholds
    # ------------------------------------------------------------------ #
    def _raw_thresholds(self) -> Tensor:
        """The un-jittered thresholds. THIS is the subclass override point."""
        return self.thresholds

    def get_thresholds(self, batch_size: int | None = None) -> Tensor:
        """Thresholds as the forward pass should see them. Do not override.

        Applies threshold jitter when training. Returns the raw shape when jitter
        is off, or a batched shape ``(batch, *raw.shape)`` when it is on -- the
        draw is per sample, so the leading axis is real, not broadcast.
        """
        t = self._raw_thresholds()
        if (
            self.training
            and self.thresh_jitter > 0.0
            and self.thresh_jitter_mode != "indep"   # indep is a post-compare flip
            and t is not None
        ):
            t = self._apply_threshold_jitter(t, batch_size)
        return t

    # ------------------------------------------------------------------ #
    # threshold jitter
    # ------------------------------------------------------------------ #
    @staticmethod
    def _level_spacing(t: Tensor) -> Tensor:
        """Mean gap between adjacent thermometer levels, reduced over the level axis.

        Layout-agnostic because it only ever reduces the LAST axis: ``(K,)`` gives a
        scalar spacing, ``(C, K)`` gives ``(C, 1)``. That reproduces the per-channel
        semantics of the reference implementation for channel-wise thresholds and
        degrades correctly to a single global spacing for global ones -- with global
        thresholds there is no per-channel scale to normalise by.
        """
        if t.shape[-1] > 1:
            return (t[..., 1:] - t[..., :-1]).abs().mean(dim=-1, keepdim=True)
        return t.abs().mean(dim=-1, keepdim=True) + 1e-6

    def _draw_noise(self, shape, like: Tensor) -> Tensor:
        """Standard normal on the right device/dtype, honouring the pin.

        dtype/device come from ``like`` so this does not break under autocast and
        does not allocate on CPU first. The cache is keyed by shape so a change of
        batch size cannot silently reuse a mis-shaped draw.
        """
        shape = tuple(shape)
        if not self._pin_jitter:
            return torch.randn(shape, device=like.device, dtype=like.dtype)
        key = (self.thresh_jitter_mode, shape)
        cached = self._jitter_cache
        if cached is None or cached[0] != key:
            cached = (key, torch.randn(shape, device=like.device, dtype=like.dtype))
            self._jitter_cache = cached
        return cached[1]

    def _apply_threshold_jitter(self, t: Tensor, batch_size: int | None) -> Tensor:
        """Perturb the thermometer grid. Train-only; the caller has checked that.

        The mechanism is CORRELATION, not marginal noise: one draw per
        (sample, channel, level) is SHARED across every spatial/time position, so the
        whole grid shifts coherently -- a self-consistent code, i.e. what a slightly
        mis-calibrated quantizer would actually produce. Corrupting bits independently
        at the same marginal rate instead is measurably much worse; that is the
        ``indep`` control arm, not an implementation shortcut.
        """
        lam = self.thresh_jitter
        b = 1 if batch_size is None else int(batch_size)
        spacing = self._level_spacing(t)
        noise = self._draw_noise((b, *t.shape), t)

        if self.thresh_jitter_mode == "bounded":
            # Each level moves at most halfway to its nearest neighbour, so adjacent
            # levels can never cross: |tanh| < 1 strictly, and even two neighbours
            # moving maximally toward each other only meet at the midpoint.
            if t.shape[-1] > 1:
                gaps = (t[..., 1:] - t[..., :-1]).abs()
                left = torch.cat([gaps[..., :1], gaps], dim=-1)
                right = torch.cat([gaps, gaps[..., -1:]], dim=-1)
                h = 0.5 * torch.minimum(left, right)
            else:
                h = 0.5 * t.abs() + 1e-6
            return t + h * torch.tanh(lam * noise)

        # gauss: unbounded, sigma = lambda * mean level spacing. Levels move
        # INDEPENDENTLY and are deliberately NOT clamped or re-sorted, so the grid can
        # become non-monotone. That is the characterized behaviour of the arm these
        # numbers come from -- do not "fix" it; use mode="bounded" if you want
        # crossing to be impossible.
        return t + noise * (lam * spacing)

    def _apply_indep_flip(self, x: Tensor, t: Tensor) -> Tensor:
        """The ``indep`` control arm: same marginal flip rate, zero correlation.

        Matches gauss's marginal exactly -- P(flip | d) = Phi(-|d| / (lambda*s)) --
        but draws independently per position instead of sharing one draw across the
        window. Isolating correlation this way is the point: the same signal value
        can get different bits at adjacent positions, which no threshold placement
        could ever produce.

        Returns HARD bits and has no gradient path through ``x``, so it is only valid
        with hard binarization; the callers enforce that.
        """
        spacing = self._broadcast_thresholds(self._level_spacing(t), x, batched=False)
        d = x - t
        z = -d.abs() / (self.thresh_jitter * spacing + 1e-12)
        p_flip = 0.5 * (1.0 + torch.erf(z / 1.4142135623730951))
        clean = (d > 0).to(dtype=torch.float32)
        if self._pin_jitter:
            key = ("indep", tuple(p_flip.shape))
            cached = self._jitter_cache
            if cached is None or cached[0] != key:
                cached = (key, torch.rand_like(p_flip))
                self._jitter_cache = cached
            u = cached[1]
        else:
            u = torch.rand_like(p_flip)
        flip = (u < p_flip).to(dtype=clean.dtype)
        return clean * (1.0 - flip) + (1.0 - clean) * flip      # XOR

    @staticmethod
    def _broadcast_thresholds(t: Tensor, x: Tensor, batched: bool) -> Tensor:
        """Shape thresholds to broadcast against ``x`` (already unsqueezed(-1)).

        Replaces three copies of the same reshape that lived in the three forwards.

        When ``batched`` is False the behaviour is exactly what those copies did, so
        turning jitter off is bit-for-bit identical to the pre-jitter code. When it
        is True the tensor carries a real leading batch axis and singleton dims are
        inserted BEFORE the level axis -- right-aligned broadcasting would otherwise
        line the batch axis up against a spatial one.
        """
        if not batched:
            if t.dim() == 2 and x.dim() == 5:      # conv, channel-wise thresholds
                return t.view(1, -1, 1, 1, t.shape[-1])
            return t
        while t.dim() < x.dim():
            t = t.unsqueeze(-2)
        return t

    def pin_jitter(self, on: bool = True) -> None:
        """Freeze (or release) the jitter draw across forwards.

        Any loss term or probe that compares two forwards measures the difference
        between two DRAWS unless the draw is held fixed. Releasing also DROPS the
        cache: leaving a stale draw pinned silently turns the regularizer into a
        constant for the rest of training.
        """
        self._pin_jitter = bool(on)
        if not on:
            self._jitter_cache = None
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Subclasses must implement forward."""
        pass
    
    @staticmethod
    def get_uniform_thresholds(data_set: Tensor, num_bits: int, one_per: str) -> Tensor:
        if one_per == "feature":
            min_value = data_set.min(dim=0)[0]
            max_value = data_set.max(dim=0)[0]
        elif one_per == "channel":
            # For conv: (batch_size, num_channels, h, w)
            # We want min/max per channel across all other dimensions
            batch_size, num_channels = data_set.shape[:2]
            # Reshape to (num_channels, -1) to flatten all non-channel dims
            reshaped = data_set.transpose(0, 1).reshape(num_channels, -1)
            min_value = reshaped.min(dim=1)[0]  # (num_channels,)
            max_value = reshaped.max(dim=1)[0]  # (num_channels,)
        elif one_per == "global":
            min_value = data_set.min()
            max_value = data_set.max()
        else:
            raise ValueError(f"one_per must be 'feature', 'channel', or 'global'. Got {one_per}.")
        threshs = min_value.unsqueeze(-1) + torch.arange(1, num_bits+1).unsqueeze(0) * (
            (max_value - min_value) / (num_bits + 1)).unsqueeze(-1)
        return threshs
        
    @staticmethod
    def get_distributive_thresholds(
        data_set: Tensor,
        num_bits: int,
        one_per: str
    ) -> Tensor:
        """
        Compute distributive (quantile-based) thresholds.

        one_per:
            - "global":   one set of thresholds for entire tensor
            - "feature":  per-feature thresholds (last dimension)
            - "channel":  per-channel thresholds (dim=1, conv tensors)
        """
        if one_per == "global":
            # Flatten everything
            data = torch.sort(data_set.flatten())[0]  # (N,)
            indices = torch.tensor(
                [int(data.numel() * i / (num_bits + 1)) for i in range(1, num_bits + 1)],
                device=data.device
            )
            thresholds = data[indices]  # (num_bits,)
            return thresholds

        elif one_per == "feature":
            # Feature = last dimension
            # Shape: (..., F)
            data = torch.sort(data_set, dim=0)[0]  # sort along batch dimension
            n = data.shape[0]

            indices = torch.tensor(
                [int(n * i / (num_bits + 1)) for i in range(1, num_bits + 1)],
                device=data.device
            )

            thresholds = data[indices, ...]  # (num_bits, F)
            return thresholds.permute(1, 0)  # (F, num_bits)

        elif one_per == "channel":
            # Expected shape: (batch, channels, ...)
            batch_size, num_channels = data_set.shape[:2]

            # Move channels first and flatten everything else
            reshaped = data_set.transpose(0, 1).reshape(num_channels, -1)
            sorted_data = torch.sort(reshaped, dim=1)[0]  # (C, N)

            n = sorted_data.shape[1]
            indices = torch.tensor(
                [int(n * i / (num_bits + 1)) for i in range(1, num_bits + 1)],
                device=sorted_data.device
            )

            thresholds = sorted_data[:, indices]  # (C, num_bits)
            return thresholds

        else:
            raise ValueError(
                f"one_per must be 'global', 'feature', or 'channel'. Got {one_per}."
            )

    
    @staticmethod
    def get_initial_thresholds(data_set: Tensor, num_bits: int, one_per: str, method: str = "uniform") -> Tensor:
        assert one_per in ["feature", "channel", "global"], "one_per must be 'feature', 'channel', or 'global'."
        assert method in ["uniform", "distributive"], "method must be 'uniform' or 'distributive'"
        if method == "uniform":
            return Binarization.get_uniform_thresholds(data_set, num_bits, one_per)
        elif method == "distributive":
            return Binarization.get_distributive_thresholds(data_set, num_bits, one_per)
        else:
            raise ValueError(f"Unknown threshold initialization method: {method}.")


class FixedBinarization(Binarization):
    """Binarization with fixed (non-learnable) thresholds."""
    def __init__(self, thresholds: Tensor, feature_dim=-2, **kwargs):
        super().__init__(thresholds, feature_dim, **kwargs)
    
    def forward(self, x: Tensor) -> Tensor:
        if self.thresholds is None:
            raise ValueError('Need to fit before calling apply')
        batch = x.shape[0]
        x = x.unsqueeze(-1)

        # `indep` is a post-comparison flip, not a threshold transform, so it takes
        # the raw grid and its own branch.
        if self.training and self.thresh_jitter > 0.0 and self.thresh_jitter_mode == "indep":
            raw = self._broadcast_thresholds(self._raw_thresholds(), x, batched=False)
            return merge_dim_with_last(self._apply_indep_flip(x, raw), self.feature_dim)

        t = self.get_thresholds(batch_size=batch)
        jittered = t.dim() > self._raw_thresholds().dim()
        t = self._broadcast_thresholds(t, x, batched=jittered)
        x = (x > t).float()
        return merge_dim_with_last(x, self.feature_dim)
        

class DummyBinarization(Binarization):
    """ Dummy binarization module that does nothing."""
    def __init__(self, **kwargs):
        if float(kwargs.get("thresh_jitter", 0.0)) > 0.0:
            raise ValueError(
                "thresh_jitter has no meaning for DummyBinarization: there is no "
                "threshold grid to perturb. Silently ignoring it would make a jitter "
                "sweep on such a model look like a null result."
            )
        super().__init__(None)
    
    def forward(self, x: Tensor) -> Tensor:
        return x.float()
    

class SoftBinarization(Binarization):
    """ Soft binarization with fixed thresholds using sigmoid."""
    def __init__(self, thresholds: Tensor, temperature=0.1, feature_dim=-2, **kwargs):
        super().__init__(thresholds, feature_dim, **kwargs)
        self.temperature_sampling = temperature
    
    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.thresh_jitter > 0.0 and self.thresh_jitter_mode == "indep":
            raise ValueError(
                "thresh_jitter_mode='indep' returns hard bits and has no gradient path "
                "through x, so it cannot be combined with SoftBinarization. It is an "
                "A/B control for the correlation hypothesis, not a training mode."
            )
        batch = x.shape[0]
        x = x.unsqueeze(-1)
        t = self.get_thresholds(batch_size=batch)
        jittered = t.dim() > self._raw_thresholds().dim()
        thresholds = self._broadcast_thresholds(t, x, batched=jittered)
        if self.training:
            x = sigmoid((x - thresholds) / self.temperature_sampling)
        else:
            x = (x > thresholds).to(dtype=torch.float32)
        return merge_dim_with_last(x, self.feature_dim)
    

class LearnableBinarization(Binarization):
    def __init__(
        self, 
        thresholds: Tensor | List,
        feature_dim=-2,
        temperature_sampling=0.1,
        temperature_softplus=0.1,
        forward_sampling="soft", 
        max_grad_norm=0.001,
        **kwargs
    ):
        self.forward_sampling = forward_sampling
        # NOTE: **kwargs was previously dropped here, so any base-class option
        # (thresh_jitter among them) was silently ignored for this class only.
        super().__init__(thresholds, feature_dim, **kwargs)
        self.temperature_sampling = temperature_sampling
        self.temperature_softplus = temperature_softplus
        diffs = torch.diff(self.thresholds, 
                           prepend=self.thresholds.new_zeros(*self.thresholds.shape[:-1], 1), dim=-1)
        self.raw_diffs = torch.nn.Parameter(diffs)
        self.raw_diffs.register_hook(self._clip_grad)
        self._max_grad_norm = max_grad_norm

    def _clip_grad(self, grad):
        norm = grad.norm()
        if norm > self._max_grad_norm:
            grad = grad * (self._max_grad_norm / (norm + 1e-6))
        return grad
            
    def _raw_thresholds(self):
        # Renamed from get_thresholds: the base class now owns get_thresholds() and
        # applies jitter on top of whatever this returns. Body unchanged.
        if self.training:
            # first diff can be negative: global shift
            first_diff = self.raw_diffs[..., :1]  # unconstrained

            # remaining diffs are positive
            if self.raw_diffs.shape[-1] > 1:
                rest_diffs = (self.temperature_softplus + 1e-6) * F.softplus(
                    self.raw_diffs[..., 1:] / (self.temperature_softplus + 1e-6)
                )
                diffs_pos = torch.cat([first_diff, rest_diffs], dim=-1)
            else:
                diffs_pos = first_diff

            thresholds = torch.cumsum(diffs_pos, dim=-1)

        else:
            thresholds = torch.cumsum(self.raw_diffs, dim=-1)

        return thresholds        

    def _sample_train(self, x: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid/gumbel_sigmoid based on forward_sampling mode."""
        if self.forward_sampling == "soft":
            return sigmoid(x - thresholds, tau=self.temperature_sampling, hard=False)
        elif self.forward_sampling == "hard":
            return sigmoid(x - thresholds, tau=self.temperature_sampling, hard=True)
        elif self.forward_sampling == "gumbel_soft":
            return gumbel_sigmoid(x - thresholds, tau=self.temperature_sampling, hard=False)
        elif self.forward_sampling == "gumbel_hard":
            return gumbel_sigmoid(x - thresholds, tau=self.temperature_sampling, hard=True)

    def _sample_eval(self, x: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
        """Threshold for discrete output."""
        return (x > thresholds).to(dtype=torch.float32)

    def forward(self, x):
        if self.training and self.thresh_jitter > 0.0 and self.thresh_jitter_mode == "indep":
            raise ValueError(
                "thresh_jitter_mode='indep' returns hard bits, cutting the gradient to "
                "raw_diffs, so it cannot be combined with LearnableBinarization."
            )
        batch = x.shape[0]
        x = x.unsqueeze(-1)
        t = self.get_thresholds(batch_size=batch)
        jittered = t.dim() > self._raw_thresholds().dim()
        thresholds = self._broadcast_thresholds(t, x, batched=jittered)
        if self.training:
            # Hard thermometer encoding
            outputs = self._sample_train(x, thresholds)
        else:
            outputs = self._sample_eval(x, thresholds)
        return merge_dim_with_last(outputs, self.feature_dim)
    

def merge_dim_with_last(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Merge dimension k with the last dimension of x.

    Input shape:  (d0, d1, ..., d{k}, ..., d{n-2}, d{n-1})
    Output shape: (d0, ..., d{k}*d{n-1}, ..., d{n-2})
    i.e., last dim is folded into dim k, and the last dim disappears.

    The order of the other dimensions is preserved.
    """
    n = x.ndim
    if n < 2:
        raise ValueError("Need at least 2 dimensions to merge with last.")
    if k < 0:
        k += n
    if not (0 <= k < n - 1):
        raise ValueError(f"k must be in [0, {n-2}] (cannot be the last dim). Got {k}.")

    last = n - 1

    # Permute so dims become: [0..k-1, k, last, k+1..last-1]
    perm = list(range(n))
    perm.pop(last)          # remove last
    perm.insert(k + 1, last)  # insert last right after k

    y = x.permute(*perm)

    # Now k and k+1 are adjacent: (.., d_k, d_last, ..)
    shape = list(y.shape)
    shape[k] = shape[k] * shape[k + 1]   # merge
    shape.pop(k + 1)                     # drop the extra dim
    y = y.reshape(*shape)

    return y


# ---------------------------------------------------------------------------- #
# jitter control, model-wide
# ---------------------------------------------------------------------------- #

@contextmanager
def pinned_jitter(model):
    """Freeze the jitter draw across every Binarization in ``model``.

    Two forwards inside this block differ by exactly the thing you changed, not by
    the draw. Without it, any consistency loss or perturbation probe measures
    draw-to-draw noise and reports a null result.
    """
    mods = [m for m in model.modules() if isinstance(m, Binarization)]
    for m in mods:
        m.pin_jitter(True)
    try:
        yield model
    finally:
        for m in mods:
            m.pin_jitter(False)          # also drops the cache


@contextmanager
def jitter_disabled(model):
    """Set lambda = 0 WITHOUT leaving train mode.

    The clean half of a consistency loss must come from the same network. Using
    ``model.eval()`` would additionally switch every LUT to a hard lookup and every
    soft binarization to a step, so the "clean" logits would be a different model's.
    This changes one number and nothing else; ``.training`` is untouched.
    """
    mods = [m for m in model.modules() if isinstance(m, Binarization)]
    saved = [m.thresh_jitter for m in mods]
    for m in mods:
        m.thresh_jitter = 0.0
    try:
        yield model
    finally:
        for m, v in zip(mods, saved):
            m.thresh_jitter = v

from typing import Union
from abc import ABC, abstractmethod
import itertools

import torch
from torch.nn.common_types import _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _triple

from .functional import softmax, take_tuples, get_combination_indices
    

def setup_connections(
    connections: str,
    structure: str,
    lut_rank: int,
    device: str = None,
    **connections_kwargs
):
    """Factory method to create connection modules."""
    if structure == "dense":
        if connections == "fixed":
            return FixedDenseConnections(
                lut_rank=lut_rank,
                device=device,
                **connections_kwargs
            )
        elif connections == "learnable":
            return LearnableDenseConnections(
                lut_rank=lut_rank,
                device=device,
                **connections_kwargs
            )
        else:
            raise ValueError(f"Unknown connections method: {connections}")
    elif structure == "conv":
        if connections == "fixed":
            return FixedConvConnections(
                lut_rank=lut_rank,
                device=device,
                **connections_kwargs
            )
        else:
            raise ValueError(f"Unknown connections method: {connections}")
    else:
        raise ValueError(f"Unknown structure method: {structure}")
    


def repair_duplicate_inputs(slots: torch.Tensor, max_rounds: int = 1000) -> torch.Tensor:
    """Make every gate read ``lut_rank`` DISTINCT inputs, in place, by swapping slots.

    ``slots`` is ``(n_gates, lut_rank, *coord)``: one input coordinate (an index, or a
    tap tuple) per gate input. A gate whose inputs coincide computes a function of fewer
    inputs than it pays for. Each offending slot is swapped with a uniformly random other
    slot, so the MULTISET of slots is unchanged -- per-channel leaf counts and input
    coverage survive exactly -- and only the pairing moves. Raises if the slots cannot be
    made distinct (e.g. fewer distinct values than ``lut_rank``).
    """
    n, L = slots.shape[0], slots.shape[1]
    flat = slots.reshape(n * L, -1)
    eye = torch.eye(L, dtype=torch.bool, device=slots.device)
    for _ in range(max_rounds):
        g = flat.view(n, L, -1)
        same = (g.unsqueeze(1) == g.unsqueeze(2)).all(-1) & ~eye        # (n, L, L)
        dup = same.any(-1).any(-1)
        if not dup.any():
            return slots
        for j in torch.nonzero(dup).flatten().tolist():
            a, b = torch.nonzero(torch.triu(same[j], diagonal=1))[0].tolist()
            s = j * L + b
            r = int(torch.randint(0, n * L, (1,), device=slots.device))
            tmp = flat[s].clone()
            flat[s] = flat[r]
            flat[r] = tmp
    raise ValueError("could not give every gate distinct inputs: too few distinct "
                     "candidates for lut_rank")


# Default connection draw for layers that do not pass `init_method` explicitly.
# "random" is the historical draw (inputs with replacement). A caller can switch every
# layer built inside a block, without threading an argument through each builder:
#     with default_init_method("random-distinct"):
#         model = build(...)
_DEFAULT_INIT_METHOD = "random"


class default_init_method:
    """Context manager: the connection init method used when a layer passes None."""

    def __init__(self, method: str):
        self.method = method

    def __enter__(self):
        global _DEFAULT_INIT_METHOD
        self._prev, _DEFAULT_INIT_METHOD = _DEFAULT_INIT_METHOD, self.method
        return self

    def __exit__(self, *exc):
        global _DEFAULT_INIT_METHOD
        _DEFAULT_INIT_METHOD = self._prev
        return False


# Default leaf-tap draw of the conv trees (FixedConvConnections). "replace" is the
# historical draw (taps with replacement); "no-replace" draws the 2^d leaf taps of a tree
# without replacement from the tree's candidate pool (see FixedConvConnections).
_DEFAULT_TREE_TAP_DRAW = "replace"
_TREE_TAP_DRAWS = ("replace", "no-replace")


class default_tree_tap_draw:
    """Context manager: the conv-tree leaf-tap draw used when a layer passes None."""

    def __init__(self, mode: str):
        if mode not in _TREE_TAP_DRAWS:
            raise ValueError(f"tree_tap_draw must be one of {_TREE_TAP_DRAWS}, got {mode!r}")
        self.mode = mode

    def __enter__(self):
        global _DEFAULT_TREE_TAP_DRAW
        self._prev, _DEFAULT_TREE_TAP_DRAW = _DEFAULT_TREE_TAP_DRAW, self.mode
        return self

    def __exit__(self, *exc):
        global _DEFAULT_TREE_TAP_DRAW
        _DEFAULT_TREE_TAP_DRAW = self._prev
        return False


def draw_without_replacement(n_pool: int, n_draw: int, device=None) -> torch.Tensor:
    """``n_draw`` indices into a pool of ``n_pool``, uniformly random order.

    n_draw <= n_pool: one without-replacement pass. n_draw > n_pool (refill): complete
    passes over the pool, then the remainder by one more pass, so every index appears
    floor(n_draw / n_pool) or ceil(n_draw / n_pool) times.
    """
    full, rem = divmod(n_draw, n_pool)
    parts = [torch.randperm(n_pool, device=device) for _ in range(full)]
    if rem:
        parts.append(torch.randperm(n_pool, device=device)[:rem])
    return torch.cat(parts)


class Connections(torch.nn.Module, ABC):
    """Abstract base class for connection strategies."""
    def __init__(
            self,
            lut_rank=2,
            device=None,
            init_method=None,
            **kwargs
        ):
        super().__init__()
        self.lut_rank = lut_rank
        self.device = device
        self.init_method = init_method if init_method is not None else _DEFAULT_INIT_METHOD

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def _init_connections(self):
        pass

    def update_temperature(self, temperature: float):
        pass


class FixedDenseConnections(Connections):
    """Fixed dense connections implementation.
    Each neuron connects to ``lut_rank`` input features chosen from the input dimension.
    The connections are fixed after initialization.
    
    Args:
        in_dim: Input feature dimension.
        out_dim: Number of neurons (output dimension).
        lut_rank: Number of input features each neuron connects to.
        device: Device to store the connection indices tensor.
        init_method: Method to initialize connections. Options are:
            - "random": Randomly select input features (with replacement).
            - "random-unique": Randomly select unique input features (without replacement).
            - "random-distinct": like "random" (balanced coverage), but the ``lut_rank``
              inputs of each neuron are distinct.
        relabel: Randomly permute input ids after the draw. ``randperm(k*n) % in_dim``
            gives the ``(k*n) % in_dim`` extra reads to inputs ``0..r-1``, i.e. always the
            FIRST inputs (for a flattened conv output: the first channels). Relabelling
            spreads them over random inputs (SilentWear ``get_connections``).
    """
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            lut_rank=2, 
            device=None,
            init_method=None,
            relabel: bool = False,
            **kwargs
        ):
        super().__init__(
            lut_rank=lut_rank,
            device=device,
            init_method=init_method,
            **kwargs
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relabel = bool(relabel)
        self.register_buffer('indices', self._init_connections())

    def _init_connections(self):
        """Constructs possible input–neuron connection indices.

        Each neuron takes ``lut_rank`` input features chosen out of ``lut_rank * num_candidates``
        possibilities. This function returns a tensor encoding which input indices are connected 
        to which neuron.

        Returns:
            A tensor of shape ``(num_candidates, lut_rank, out_dim)`` with integer indices into
            the last dimension of the input.
        """
        assert self.in_dim >= self.lut_rank, (
            f"Cannot have num_candidates * lut_rank > in_dim "
            f"({self.lut_rank} > {self.in_dim})"
        )
        assert self.out_dim * self.lut_rank >= self.in_dim, (
                f"Need out_dim * lut_rank >= in_dim to cover all inputs "
                f"({self.out_dim} * {self.lut_rank} < {self.in_dim})."
                )

        if self.init_method in ("random", "random-distinct"):
            # "random": both inputs can stem from the same input feature
            c = torch.randperm(self.lut_rank * self.out_dim, 
                               device=self.device) % self.in_dim
            c = c.reshape(self.lut_rank, self.out_dim)
            if self.init_method == "random-distinct":
                per_gate = c.t().contiguous().view(self.out_dim, self.lut_rank, 1)
                repair_duplicate_inputs(per_gate)
                c = per_gate.view(self.out_dim, self.lut_rank).t()
            if self.relabel:
                c = torch.randperm(self.in_dim, device=self.device)[c]
        elif self.init_method == "random-unique":
            c = get_random_unique_connections(
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                n=self.lut_rank
            )
        else:
            raise ValueError(self.init_method)
        if self.relabel and self.init_method == "random-unique":
            raise ValueError("relabel is already part of random-unique; pass relabel=False")
        c = c.contiguous().to(torch.int64).to(self.device)
        return c
    
    def forward(self, x):
        return x[:, self.indices]
    

class LearnableConnectionFunction(torch.autograd.Function):
    """Autograd function for learnable connections.
    Implements the forward and backward pass for learnable connections
    using Gumbel-Softmax for differentiable sampling.
    """
    @staticmethod
    def forward(ctx, x, weights, tau, gumbel, indices):
        if gumbel:
            u = torch.rand_like(weights)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        else:
            g = torch.zeros_like(weights)
        connections = (weights + g).argmax(dim=0)
        l = torch.arange(weights.shape[1], device=x.device).unsqueeze(1)
        o = torch.arange(weights.shape[2], device=x.device).unsqueeze(0)
        output = x[:, indices[connections, l, o]]
        ctx.save_for_backward(x, weights, tau, g, indices)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        x, weights, tau, g, indices = ctx.saved_tensors
        # compute gradient w.r.t. to learnable weights
        weights_grad = torch.einsum("bclo,blo->clo", 2*x[:,indices]-1, output_grad)
        # compute gradient w.r.t. to input with sparse scatter_add method
        input_grad = torch.zeros_like(x)
        temp = softmax((weights + g)/tau, dim=0).unsqueeze(0) * output_grad.unsqueeze(1)
        input_grad.scatter_add_(dim=1, 
                                index=indices.expand(x.shape[0], -1, -1, -1).reshape(x.shape[0], -1),
                                src=temp.reshape(x.shape[0], -1))
        return input_grad, weights_grad, None, None, None
    

class LearnableDenseConnections(Connections):
    """Learnable dense connections implementation.
    Each neuron connects to ``lut_rank`` input features chosen from a set of candidates.
    The connections are learnable parameters optimized during training.

    Args:
        in_dim: Input feature dimension.
        out_dim: Number of neurons (output dimension).
        lut_rank: Number of input features each neuron connects to.
        temperature: Temperature parameter for Gumbel-Softmax.
        num_candidates: Number of candidate input features per neuron.
            If -1, all input features are candidates.
        gumbel: Whether to use Gumbel noise for sampling.
        device: Device to store the connection indices tensor.
        init_method: Method to initialize connections. Options are:
            - "random": Randomly select input features (with replacement).
            - "random-unique": Randomly select unique input features (without replacement).
    """
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            lut_rank=2, 
            temperature=0.001,
            num_candidates=-1, 
            gumbel=False,
            device=None,
            init_method=None,
            **kwargs
        ):
        super().__init__(
            lut_rank=lut_rank,
            device=device,
            init_method=init_method,
            **kwargs
        )
        self.temperature = temperature
        self.num_candidates = num_candidates
        self.lut_rank = lut_rank
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.gumbel = gumbel
        if num_candidates == -1:
            num_candidates = in_dim
            self.num_candidates = num_candidates
            self.register_buffer('indices', torch.arange(in_dim, device=self.device).view(
                in_dim, 1, 1).expand(in_dim, lut_rank, out_dim).contiguous())
        else:
            assert num_candidates > 0, "num_candidates must be bigger than 0"
            self.num_candidates = num_candidates
            self.register_buffer('indices', self._init_connections())
        self.weights = torch.nn.Parameter(torch.rand(
            num_candidates, lut_rank, out_dim, dtype=torch.float32), requires_grad=True)
        
    def update_temperature(self, temperature: float):
        self.temperature = temperature
        
    def forward(self, x):
        return LearnableConnectionFunction.apply(x, self.weights, torch.tensor(self.temperature), 
                                                 self.gumbel, self.indices)
    
    def _init_connections(self):
        """Constructs possible input–neuron connection indices.

        Each neuron takes ``lut_rank`` input features chosen out of ``lut_rank * num_candidates``
        possibilities. This function returns a tensor encoding which input indices are connected 
        to which neuron.

        Returns:
            A tensor of shape ``(num_candidates, lut_rank, out_dim)`` with integer indices into
            the last dimension of the input.
        """
        assert self.in_dim >= self.num_candidates * self.lut_rank, (
            f"Cannot have num_candidates * lut_rank > in_dim "
            f"({self.num_candidates * self.lut_rank} > {self.in_dim})"
        )
        assert self.out_dim * self.lut_rank >= self.in_dim, (
                f"Need out_dim * lut_rank >= in_dim to cover all inputs "
                f"({self.out_dim} * {self.lut_rank} < {self.in_dim})."
                )

        if self.init_method in ("random", "random-distinct"):
            # "random": both inputs can stem from the same input feature
            c = torch.randperm(self.lut_rank * self.out_dim * self.num_candidates, 
                               device=self.device) % self.in_dim
            c = c.reshape(self.num_candidates, self.lut_rank, self.out_dim)
            if self.init_method == "random-distinct":
                # each (candidate, neuron) tuple reads lut_rank distinct inputs
                per_gate = c.permute(0, 2, 1).reshape(-1, self.lut_rank, 1).contiguous()
                repair_duplicate_inputs(per_gate)
                c = per_gate.view(self.num_candidates, self.out_dim,
                                  self.lut_rank).permute(0, 2, 1)
        elif self.init_method == "random-unique":
            c = get_random_unique_connections(
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                n=self.lut_rank*self.num_candidates
            )
            c = c.reshape(self.num_candidates, self.lut_rank, self.out_dim)
        else:
            raise ValueError(self.connections)
        c = c.contiguous().to(torch.int64).to(self.device)
        return c
    

def get_random_unique_connections(in_dim, out_dim, n):
    # Feasibility check
    n_max = int(in_dim * (in_dim // (n - 1) - 1) / 2)
    assert out_dim <= n_max, (
        "The number of neurons ({}) must not be greater than the number of pair-wise combinations "
        "of the inputs ({})".format(out_dim, n_max)
    )
    x = torch.arange(in_dim)
    c = take_tuples(x, tuple_size=n, stride_within=1)
    offset = 2
    while c.shape[-1] < out_dim:
        c_ = take_tuples(x, tuple_size=n, stride_within=offset)
        c = torch.cat([c, c_], dim=-1)
        offset += 1
    c = c[:, :out_dim]
    perm_out = torch.randperm(out_dim)
    perm_in = torch.randperm(in_dim)
    c = c[:, perm_out]
    c = perm_in[c]
    return c


class FixedConvConnections(Connections):
    """Fixed convolutional connections implementation.
    Each convolutional kernel connects to input features within its receptive field.
    The connections are fixed after initialization."""
    def __init__(
            self, 
            in_dim: Union[_size_2_t, _size_3_t, int],
            channels: int = 1,
            num_kernels: int = 16,
            tree_depth: int = None,
            receptive_field_size: Union[_size_2_t, _size_3_t, int] = 2,
            stride: int = 1,
            padding: int = 0,
            conv_dimension: int = 2,
            lut_rank=2, 
            device=None,
            dilation: int = 1,
            init_method=None,  # | "random" | "random-unique" | "random-distinct"
            channel_group_size: int = None,
            channel_groups: int = None,
            tree_tap_draw: str = None,  # None -> default_tree_tap_draw | "replace" | "no-replace"
            **kwargs
        ):
        super().__init__(
            lut_rank=lut_rank,
            device=device,
            init_method=init_method,
            **kwargs
        )
        self.tree_tap_draw = (tree_tap_draw if tree_tap_draw is not None
                              else _DEFAULT_TREE_TAP_DRAW)
        if self.tree_tap_draw not in _TREE_TAP_DRAWS:
            raise ValueError(f"tree_tap_draw must be one of {_TREE_TAP_DRAWS}, "
                             f"got {self.tree_tap_draw!r}")
        self.num_kernels = num_kernels
        self.tree_depth = tree_depth
        self.channels = channels
        self.conv_dimension = conv_dimension
        assert conv_dimension in [2, 3], "conv_dimension must be 2 or 3"
        if conv_dimension == 2:
            self.receptive_field_size = _pair(receptive_field_size)
            self.in_dim = _pair(in_dim)
        else:
            self.receptive_field_size = _triple(receptive_field_size)
            self.in_dim = _triple(in_dim)
        _n = _pair if self.conv_dimension == 2 else _triple
        assert all(st <= rf for st, rf in zip(_n(stride), self.receptive_field_size)), (
            f"Stride ({stride}) cannot be larger than "
            f"receptive field size ({receptive_field_size}) on any axis"
        )        
        self.stride = _n(stride)
        self.padding = _n(padding)
        # Dilation spaces the receptive-field taps `dilation` apart along each axis, as in
        # torch.nn.Conv: a kernel of size k then spans dilation * (k - 1) + 1 positions.
        # Leaves are still SAMPLED over the k taps per axis, so dilation=1 draws exactly
        # the same random connections as before.
        self.dilation = _n(dilation)
        assert all(d >= 1 for d in self.dilation), f"dilation must be >= 1, got {dilation}"
        self.extent = tuple(d * (k - 1) + 1
                            for d, k in zip(self.dilation, self.receptive_field_size))
        # channel_groups=G: PRIVATE kernels per input block, like torch.nn.Conv's
        # `groups`. Input channels split into G contiguous blocks and kernel k belongs
        # to block k // (num_kernels // G); every leaf of that kernel reads only its
        # block. Weights are per kernel already, so this alone makes kernels private,
        # e.g. per electrode in a folded (B, electrodes*bits, 1, T) layout.
        # channel_group_size (if set) then picks a contiguous window INSIDE the block.
        self.channel_groups = int(channel_groups) if channel_groups else None
        if self.channel_groups:
            assert channels % self.channel_groups == 0, (
                f"channels ({channels}) must be divisible by channel_groups "
                f"({self.channel_groups})")
            assert num_kernels % self.channel_groups == 0, (
                f"num_kernels ({num_kernels}) must be divisible by channel_groups "
                f"({self.channel_groups})")
        self.channel_group_size = channel_group_size
        if channel_group_size is not None:
            per_block = channels // (self.channel_groups or 1)
            assert per_block > channel_group_size, (
                "channel_group_size must be smaller than the number of channels "
                "(per block when channel_groups is set)"
            )
        for i, tensor in enumerate(self._init_connections()):
            self.register_buffer(f'_indices_L{i}', tensor)

    @property
    def indices(self):
        return [getattr(self, f'_indices_L{i}') for i in range(self.tree_depth)]

    @indices.setter
    def indices(self, value):
        for i, tensor in enumerate(value):
            self.register_buffer(f'_indices_L{i}', tensor)

    def _init_connections(self):
        # Setup connections
        if self.init_method in ("random", "random-distinct"):
            kernels = self._get_random_receptive_field_tensor()
        elif self.init_method == "random-unique":
            if self.tree_tap_draw != "replace":
                raise ValueError("tree_tap_draw=no-replace is not implemented for random-unique")
            if self.channel_groups:
                raise ValueError("channel_groups is not implemented for random-unique")
            kernels = self._get_random_unique_receptive_field_tensor()
        else:
            raise ValueError(f"Unknown connections type: {self.init_method}")
        if any(d > 1 for d in self.dilation):
            # tap index -> position offset; the trailing channel coordinate is unscaled
            scale = torch.tensor([*self.dilation, 1], device=kernels.device)
            kernels = kernels * scale
        # Build tree indices
        return self._get_indices_from_kernel_tensor(kernels)


    def pool_report(self):
        """Candidate pool of one tree, for the dry-run (tree_tap_draw rule).

        pool = ke * kt * (channels the tree may read); windowed layers also report the
        per-channel sub-pool (ke * kt) and leaves per channel.
        """
        rf = 1
        for d in self.receptive_field_size:
            rf *= d
        leaves = self.lut_rank ** self.tree_depth
        c_blk = self.channels // (self.channel_groups or 1)
        g = self.channel_group_size
        if g is None:
            pool, sub, per_ch = rf * c_blk, None, None
        else:
            pool, sub, per_ch = rf * g, rf, leaves // g
        return {"tree_tap_draw": self.tree_tap_draw, "pool": pool, "leaves": leaves,
                "window": g, "sub_pool": sub, "leaves_per_channel": per_ch,
                "refill": leaves > pool}

    def _get_random_receptive_field_tensor(self):
        """
        Random sampling (with replacement unless tree_tap_draw="no-replace").

        Returns:
            coords: (lut_rank, num_kernels, sample_size, 3)
        """

        c = self.channels
        g = self.channel_group_size
        device = self.device

        sample_size = self.lut_rank ** (self.tree_depth - 1)
        total_inputs = self.lut_rank * sample_size

        # ---------------------------
        # Precompute spatial grid
        # ---------------------------
        rf_axes = [
            torch.arange(0, dim, device=device)
            for dim in self.receptive_field_size
        ]

        spatial_grid = torch.meshgrid(*rf_axes, indexing="ij")
        spatial_positions = torch.stack(
            [grid.flatten() for grid in spatial_grid], dim=1
        )
        num_spatial = spatial_positions.shape[0]

        # ---------------------------
        # Channel group setup
        # ---------------------------
        # Private blocks (channel_groups): kernel k lives in block k // kpb and sees only
        # that block's channels; everything below is then the ungrouped rule applied to
        # the block, with the kernel's LOCAL index. Without blocks this is exactly the
        # historical code path (block 0 = all channels, local index = k).
        G = self.channel_groups or 1
        c_blk = c // G
        kpb = self.num_kernels // G
        distinct = self.init_method == "random-distinct"
        no_replace = self.tree_tap_draw == "no-replace"
        if g is None:
            starts = None
        else:
            starts = torch.arange(0, c_blk - g + 1, device=device)
            num_groups = starts.numel()

        coords_per_kernel = []

        for k in range(self.num_kernels):
            base = (k // kpb) * c_blk
            k_loc = k % kpb

            if g is None:
                c_rf = base + torch.arange(0, c_blk, device=device)

                # full 3D position space
                grid = torch.meshgrid(*rf_axes, c_rf, indexing="ij")
                all_positions = torch.stack(
                    [grid_i.flatten() for grid_i in grid], dim=1
                )
                num_positions = all_positions.shape[0]

                if no_replace:
                    # tree rule: the sample_size * lut_rank leaf taps are distinct
                    # positions of the pool (refill only if the pool is smaller)
                    idx = draw_without_replacement(
                        num_positions, total_inputs, device).view(sample_size, self.lut_rank)
                elif distinct:
                    if num_positions < self.lut_rank:
                        raise ValueError(
                            f"random-distinct needs >= {self.lut_rank} candidate inputs "
                            f"per kernel, got {num_positions}")
                    # uniform over ordered tuples of DISTINCT positions
                    idx = torch.rand(sample_size, num_positions,
                                     device=device).argsort(dim=1)[:, :self.lut_rank]
                else:
                    idx = torch.randint(
                        0, num_positions,
                        (sample_size, self.lut_rank),
                        device=device,
                    )

                coords_k = all_positions[idx]

            else:
                start = base + starts[k_loc % num_groups]
                c_rf = start + torch.arange(g, device=device)

                if total_inputs % g != 0:
                    raise ValueError(
                        f"Cannot evenly distribute {total_inputs} across {g} channels."
                    )

                inputs_per_channel = total_inputs // g
                channel_chunks = []

                for channel in c_rf:
                    if no_replace:
                        # tree rule: without replacement within this channel's sub-pool
                        idx = draw_without_replacement(
                            num_spatial, inputs_per_channel, device)
                    else:
                        idx = torch.randint(
                            0, num_spatial,
                            (inputs_per_channel,),
                            device=device,
                        )

                    chosen = spatial_positions[idx]

                    ch_col = torch.full(
                        (inputs_per_channel, 1),
                        channel,
                        device=device,
                    )

                    channel_chunks.append(
                        torch.cat([chosen, ch_col], dim=1)
                    )

                coords_k = torch.cat(channel_chunks, dim=0)

                perm = torch.randperm(total_inputs, device=device)
                coords_k = coords_k[perm]

                coords_k = coords_k.view(sample_size, self.lut_rank, coords_k.shape[-1])
                if distinct:
                    # same per-channel leaf counts; only the pairing changes
                    repair_duplicate_inputs(coords_k)

            coords_per_kernel.append(coords_k)

        coords = torch.stack(coords_per_kernel, dim=0)
        coords = coords.permute(2, 0, 1, 3)

        return coords
    
    
    def _get_random_unique_receptive_field_tensor(self):
        """
        Random unique sampling (without replacement across tuples).

        Returns:
            coords: (lut_rank, num_kernels, sample_size, 3)
        """

        c = self.channels
        g = self.channel_group_size
        device = self.device

        sample_size = self.lut_rank ** (self.tree_depth - 1)
        total_inputs = self.lut_rank * sample_size

        # ---------------------------
        # Precompute spatial grid
        # ---------------------------
        rf_axes = [
            torch.arange(0, dim, device=device)
            for dim in self.receptive_field_size
        ]

        spatial_grid = torch.meshgrid(*rf_axes, indexing="ij")
        spatial_positions = torch.stack(
            [grid.flatten() for grid in spatial_grid], dim=1
        )
        num_spatial = spatial_positions.shape[0]

        # ---------------------------
        # Channel group setup
        # ---------------------------
        if g is None:
            starts = None
        else:
            starts = torch.arange(0, c - g + 1, device=device)
            num_groups = starts.numel()

        coords_per_kernel = []

        for k in range(self.num_kernels):

            if g is None:
                c_rf = torch.arange(0, c, device=device)

                grid = torch.meshgrid(*rf_axes, c_rf, indexing="ij")
                all_positions = torch.stack(
                    [grid_i.flatten() for grid_i in grid], dim=1
                )

                # all unique lut_rank combinations
                all_indices = list(
                    itertools.combinations(
                        range(all_positions.shape[0]),
                        self.lut_rank,
                    )
                )

                if len(all_indices) < sample_size:
                    raise ValueError("Not enough unique combinations.")

                chosen = torch.randperm(
                    len(all_indices),
                    device=device
                )[:sample_size]

                selected = [
                    torch.tensor(all_indices[i], device=device)
                    for i in chosen
                ]

                idx = torch.stack(selected, dim=0)
                coords_k = all_positions[idx]

            else:
                start = starts[k % num_groups]
                c_rf = start + torch.arange(g, device=device)

                if total_inputs % g != 0:
                    raise ValueError(
                        f"Cannot evenly distribute {total_inputs} across {g} channels."
                    )

                inputs_per_channel = total_inputs // g

                if num_spatial < inputs_per_channel:
                    raise ValueError(
                        "Not enough spatial positions for balanced per-channel sampling."
                    )

                channel_chunks = []

                for channel in c_rf:
                    idx = torch.randperm(
                        num_spatial,
                        device=device
                    )[:inputs_per_channel]

                    chosen = spatial_positions[idx]

                    ch_col = torch.full(
                        (inputs_per_channel, 1),
                        channel,
                        device=device,
                    )

                    channel_chunks.append(
                        torch.cat([chosen, ch_col], dim=1)
                    )

                coords_k = torch.cat(channel_chunks, dim=0)

                perm = torch.randperm(total_inputs, device=device)
                coords_k = coords_k[perm]

                coords_k = coords_k.view(sample_size, self.lut_rank, 3)

            coords_per_kernel.append(coords_k)

        coords = torch.stack(coords_per_kernel, dim=0)
        coords = coords.permute(2, 0, 1, 3)

        return coords


    def _apply_sliding_window_tensor(self, tensor):
        """Apply sliding window offsets to receptive field tensor.

        Args:
            tensor: torch.Tensor of shape (lut_rank, num_kernels, sample_size, 3)
                where last dim is (h, w, c).

        Returns:
            out: torch.Tensor of shape (lut_rank, num_kernels, num_positions, sample_size, 3),
                with the sliding-window offsets applied.
        """
        #h, w = self.in_dim
        #h_k, w_k = self.receptive_field_size

        # Account for padding
        padded = [d + 2 * pad for d, pad in zip(self.in_dim, self.padding)]
        #h_padded = h + 2 * self.padding
        #w_padded = w + 2 * self.padding

        assert all(ext <= p for ext, p in zip(self.extent, padded)), (
            f"Receptive field size {self.receptive_field_size} (extent {self.extent} at "
            f"dilation {self.dilation}) must fit within input dimensions {padded} after "
            f"padding."
        )

        # Sliding positions
        starts = [torch.arange(0, p - ext + 1, st, device=self.device)
                  for p, ext, st in zip(padded, self.extent, self.stride)]
        #h_starts = torch.arange(0, padded[0] - self.receptive_field_size[0] + 1, self.stride, device=self.device)
        #w_starts = torch.arange(0, padded[1] - self.receptive_field_size[1] + 1, self.stride, device=self.device)

        # Meshgrid for all receptive-field start positions
        grid = torch.meshgrid(*starts, indexing="ij")
        offsets = [g.flatten() for g in grid]
        num_positions = [o.numel() for o in offsets]

        # tensor: (L, K, S, 3) → (K, L, S, 3)
        pairs_all = tensor.permute(1, 0, 2, 3)
        # K, L, S, _ = pairs_all.shape

        # Split h, w, c coordinates: (K, L, S)
        base = [pairs_all[..., i] for i in range(len(offsets))]
        #h_base = pairs_all[..., 0]
        #w_base = pairs_all[..., 1]
        c_base = pairs_all[..., -1]

        # Add sliding-window offsets (broadcasted) → (K, P, L, S)
        idx = [b.unsqueeze(1) + o.view(1, num_positions[0], 1, 1) 
               for b, o in zip(base, offsets)]
        c_idx = c_base.unsqueeze(1).expand(-1, num_positions[0], -1, -1)

        # Combine back into indices: (K, P, L, S, 3)
        all_indices = torch.stack([*idx, c_idx], dim=-1)

        # Reorder so first axis is L: (L, K, P, S, 3)
        out = all_indices.permute(2, 0, 1, 3, 4)

        return out

    def _get_indices_from_kernel_tensor(self, tensor):
        """Build index tensors for all tree levels."""
        indices = [
            self._apply_sliding_window_tensor(tensor)
        ]
        for level in range(1, self.tree_depth):
            size = self.lut_rank ** (self.tree_depth - level)
            base = torch.arange(size, device=self.device).view(-1, self.lut_rank).transpose(0, 1)
            indices.append(base)
        return indices
    
    def forward(self, x, tree_level):
        if tree_level == 0:
            # unbind is used instead of * unpacking, which torch.compile cannot trace correctly
            index_tensors = self.indices[0].unbind(-1)  # (h, w[, d], c) — each (L, K, P, S)
            c_idx = index_tensors[-1]
            spatial_idx = index_tensors[:-1]
            return x[(slice(None), c_idx) + spatial_idx]
        else:
            return x[..., self.indices[tree_level]]

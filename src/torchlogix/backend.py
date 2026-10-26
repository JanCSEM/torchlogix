"""H027 backend selection for the LUT layers.

    set_backend(model, lut_backend="torch" | "lookup" | "fused",
                lut_compile=False, eval_backend="torch" | "packed",
                precision="fp32" | "bf16")

attaches a `LutState` (`layer._lut`) to every `LogicConv2d` / `LogicDense` and marks the
pools. With all defaults every layer runs the reference code (`_forward_torch`), bit for
bit. Anything else is validated here and raises on a setting it does not implement; no
backend changes a model, wiring, parameter or LUT.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from .layers.binarization import FixedBinarization
from .layers.conv import LogicConv2d, LogicConv3d
from .layers.dense import LogicDense
from .layers.pool import OrPooling2d
from .connections import FixedConvConnections, FixedDenseConnections
from .parametrization import LightLUTParametrization

LUT_BACKENDS = ("torch", "lookup", "fused")
EVAL_BACKENDS = ("torch", "packed")
PRECISIONS = ("fp32", "bf16")

# classes whose output is exactly {0, 1} in training and eval (thermometer fronts)
BIT_FRONTS = ("FixedBinarization", "FeatureThermometer")

# H030 fold: compute the first logic layer's leaf bits (x > t) inside its fused tree kernel
# instead of materialising the thermometer. A speed switch, not part of the run config: it
# changes neither the function nor the run id. FOLD_DEFAULT stays False until the H030
# decision rule (measured faster, §11) turns it on; `TORCHLOGIX_FOLD=1|0` overrides.
FOLD_ENV = "TORCHLOGIX_FOLD"
FOLD_DEFAULT = False


def fold_requested(fold: Optional[bool]) -> bool:
    if fold is not None:
        return bool(fold)
    env = os.environ.get(FOLD_ENV)
    return FOLD_DEFAULT if env is None else env == "1"


@dataclass
class LutState:
    backend: str = "torch"
    compile: bool = False
    eval_backend: str = "torch"
    bf16: bool = False                 # store this layer's output in bf16
    head: bool = False                 # last logic layer: output stays fp32 (vote input)
    binary_input: bool = False         # input is exactly {0, 1} (first layer after a bit front)
    fused_pool: Optional[object] = None
    ran: Optional[str] = None
    ran_train: Optional[str] = None    # backend of the last training-mode forward
    ran_eval: Optional[str] = None     # backend of the last eval-mode forward
    lang: Optional[str] = None         # fused: kernel language resolved on the last call
    plan: Optional[dict] = field(default=None, repr=False)
    binary_checked: bool = False
    packed_checked: bool = False
    fold: bool = False                 # H030: this (first) layer takes raw input from fold_src
    fold_src: Optional[object] = field(default=None, repr=False)   # the FixedBinarization
    fold_plan: Optional[dict] = field(default=None, repr=False)
    fold_ran: Optional[bool] = None    # did the last forward run folded
    fold_note: Optional[str] = None    # why the fold is not applied (None = applied / not asked)


def logic_layers(model):
    return [m for m in model.modules() if isinstance(m, (LogicConv2d, LogicDense))]


def _validate(m, lut_backend):
    if isinstance(m, LogicConv3d):
        raise ValueError("lut backends do not support LogicConv3d (thermo3d)")
    if not isinstance(m.parametrization, LightLUTParametrization):
        raise ValueError(f"lut backends need parametrization='light', got "
                         f"{type(m.parametrization).__name__}")
    p = m.parametrization
    # H038: lut_backend="torch" trains through the reference forward, which implements every
    # sampling mode; only lookup / fused need soft. (The packed eval is the committed function.)
    if p.forward_sampling != "soft" and lut_backend != "torch":
        raise ValueError(f"lut backends need forward_sampling='soft', got {p.forward_sampling!r}")
    if p.materialize_basis:
        raise ValueError("lut backends do not support materialize_basis")
    if m.lut_rank != 2:
        raise ValueError(f"lut backends need lut_rank=2, got {m.lut_rank}")
    if m.export_mode:
        raise ValueError("lut backends do not support export mode")
    want = FixedConvConnections if isinstance(m, LogicConv2d) else FixedDenseConnections
    if type(m.connections) is not want:
        raise ValueError(f"lut backends need fixed connections, got "
                         f"{type(m.connections).__name__}")


def _pair_pools(model):
    """conv -> pool pairs that are adjacent children of one Sequential."""
    pairs = []
    for mod in model.modules():
        if isinstance(mod, torch.nn.Sequential):
            kids = list(mod.children())
            for a, b in zip(kids, kids[1:]):
                if isinstance(a, LogicConv2d) and isinstance(b, OrPooling2d):
                    pairs.append((a, b))
    return pairs


def _binary_first(model):
    """The first logic layer in module order, if a bit front precedes it."""
    seen_front = False
    for m in model.modules():
        if type(m).__name__ in BIT_FRONTS:
            seen_front = True
        elif isinstance(m, (LogicConv2d, LogicDense, LogicConv3d)):
            return m if seen_front else None
    return None


def _arm_pre(mod, args):
    for b in mod._fold_bins:
        object.__setattr__(b, "_fold_armed", True)


def _arm_post(mod, args, out):
    for b in mod._fold_bins:
        object.__setattr__(b, "_fold_armed", False)


def _arm_with(seq, binarizer):
    """The binarizer folds only while its Sequential is running (internal forward hooks that
    arm / disarm it): a standalone `binarizer(x)` call returns the bits, never the raw input.
    Module-level functions + a plain list attribute, so a deepcopy of the model stays correct."""
    if getattr(seq, "_fold_bins", None) is None:
        object.__setattr__(seq, "_fold_bins", [])
        seq.register_forward_pre_hook(_arm_pre)
        seq.register_forward_hook(_arm_post)
    seq._fold_bins.append(binarizer)


def _pair_folds(model):
    """(binarizer, conv) pairs for the H030 fold: a FixedBinarization directly followed by
    the first logic conv (a binary-input LogicConv2d) in one Sequential. Returns
    (pairs, notes) where notes maps id(first logic layer) -> why it is not folded."""
    pairs, paired = [], set()
    for mod in model.modules():
        if isinstance(mod, torch.nn.Sequential):
            kids = list(mod.children())
            for a, b in zip(kids, kids[1:]):
                if (isinstance(a, FixedBinarization) and isinstance(b, LogicConv2d)
                        and b._lut is not None and b._lut.binary_input):
                    if a.thresh_jitter > 0.0 and a.thresh_jitter_mode == "indep":
                        continue                         # a post-compare flip: no fold
                    pairs.append((a, b))
                    paired.add(id(b))
                    _arm_with(mod, a)
    notes = {}
    for m in logic_layers(model)[:1]:                    # the first logic layer is the candidate
        if id(m) not in paired:
            notes[id(m)] = (
                "not applied: the first logic layer is not a LogicConv2d directly behind a "
                "FixedBinarization (or the jitter mode is indep)" if m._lut.binary_input else
                "not applied: the first logic layer does not take a thermometer front")
    return pairs, notes


def prepare_plans(model) -> int:
    """Build every fused layer's plan eagerly on its current device (H030 (1): a compiled
    region then reuses it instead of tracing the data_ptr cache key). Returns the number of
    plans built; call after the model is on its final device."""
    from .kernels.dispatch import _plan
    n = 0
    for m in logic_layers(model):
        if m._lut is not None and m._lut.backend == "fused":
            _plan(m, next(m.parameters()).device)
            n += 1
    return n


def set_backend(model, lut_backend="torch", lut_compile=False, eval_backend="torch",
                precision="fp32", binary_first: Optional[bool] = None,
                fold: Optional[bool] = None) -> list:
    """Attach the backend to every logic layer; returns `backend_report(model)`.

    `binary_first`: None = auto (the first logic layer is marked when a thermometer front
    precedes it; its input is then verified exact on the first call), True/False force.
    `fold` (H030): None = `fold_requested` (env `TORCHLOGIX_FOLD`, default off); with
    lut_backend="fused" the first conv takes the raw input of the FixedBinarization in front
    of it and computes the thermometer bits in its kernel. Pairs that cannot fold stay on the
    materialised path and say why in `backend_report`.
    """
    if lut_backend not in LUT_BACKENDS:
        raise ValueError(f"lut_backend must be one of {LUT_BACKENDS}, got {lut_backend!r}")
    if eval_backend not in EVAL_BACKENDS:
        raise ValueError(f"eval_backend must be one of {EVAL_BACKENDS}, got {eval_backend!r}")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    default = (lut_backend == "torch" and not lut_compile and eval_backend == "torch"
               and precision == "fp32")
    if precision == "bf16" and not (lut_backend == "fused"
                                    or (lut_backend == "lookup" and lut_compile)):
        raise ValueError("precision must be fp32 here: bf16 is supported with "
                         "lut_backend=fused, or lut_backend=lookup + lut_compile (uncompiled "
                         "torch/lookup would compute in bf16 throughout, a different "
                         "numerics class)")
    layers = logic_layers(model)
    if not default:
        for m in layers:
            _validate(m, lut_backend)
        if any(isinstance(m, LogicConv3d) for m in model.modules()):
            raise ValueError("lut backends do not support LogicConv3d (thermo3d)")
        if not layers:
            raise ValueError("no logic layer found to apply the backend to")
    first = _binary_first(model) if binary_first is None else None
    if binary_first and layers:
        first = layers[0]
    for pool in (m for m in model.modules() if isinstance(m, OrPooling2d)):
        object.__setattr__(pool, "_absorbed_by", None)
        pool._lut_req = lut_backend if eval_backend == "torch" else f"{lut_backend}+{eval_backend}"
        pool._lut_ran = None
    for i, m in enumerate(layers):
        m._lut = LutState(backend=lut_backend, compile=lut_compile, eval_backend=eval_backend,
                          bf16=(precision == "bf16"), head=(i == len(layers) - 1),
                          binary_input=(m is first and not isinstance(m, LogicConv3d)))
    if not hasattr(model, "forward_all"):          # early-exit models tap pre-pool activations
        for conv, pool in _pair_pools(model):
            if conv._lut is not None:
                conv._lut.fused_pool = pool
                object.__setattr__(pool, "_absorbed_by", conv)
    for b in (m for m in model.modules() if isinstance(m, FixedBinarization)):
        object.__setattr__(b, "_fold_target", None)       # re-setting a backend resets folds
        object.__setattr__(b, "_fold_armed", False)
    for q in (m for m in model.modules() if getattr(m, "_fold_bins", None) is not None):
        object.__setattr__(q, "_fold_bins", [])           # (the arming hooks then do nothing)
    if fold_requested(fold) and lut_backend == "fused" and not hasattr(model, "forward_all"):
        pairs, notes = _pair_folds(model)
        for b, conv in pairs:
            object.__setattr__(b, "_fold_target", conv)
            conv._lut.fold, conv._lut.fold_src = True, b
        for m in layers:
            if id(m) in notes:
                m._lut.fold_note = notes[id(m)]
    elif fold_requested(fold):
        for m in layers[:1]:
            m._lut.fold_note = (f"not applied: lut_backend={lut_backend!r} (fold needs fused)"
                                if lut_backend != "fused" else
                                "not applied: early-exit model (taps intermediate activations)")
    return backend_report(model)


def apply_compile(model, lut_backend: str, backend: str = "inductor", mode=None) -> int:
    """lut_compile=true. fused: compile the model (kernels are custom ops, so Inductor
    fuses the glue around them); torch / lookup: compile every logic layer. In place
    (`nn.Module.compile`), so state_dict keys are unchanged. Returns the number of modules
    compiled."""
    if lut_backend == "fused":
        prepare_plans(model)          # H030 (1): plans built eagerly, outside the compiled region
        model.compile(backend=backend, mode=mode)
        return 1
    n = 0
    for m in logic_layers(model):
        m.compile(backend=backend, mode=mode)
        n += 1
    return n


def backend_report(model) -> list:
    """Per module: the backend requested and the one that ran on the last forward."""
    out = []
    for name, m in model.named_modules():
        if isinstance(m, (LogicConv2d, LogicDense)) and m._lut is not None:
            st = m._lut
            out.append({"module": name, "type": type(m).__name__, "lut_backend": st.backend,
                        "lut_compile": st.compile, "eval_backend": st.eval_backend,
                        "bf16": st.bf16 and not st.head, "binary_input": st.binary_input,
                        "pool_fused": st.fused_pool is not None,
                        "fold": st.fold, "fold_ran": st.fold_ran, "fold_note": st.fold_note,
                        "ran": st.ran,
                        "ran_train": st.ran_train, "ran_eval": st.ran_eval,
                        "lang": st.lang})
        elif isinstance(m, OrPooling2d) and m._lut_req is not None:
            out.append({"module": name, "type": "OrPooling2d", "lut_backend": m._lut_req,
                        "absorbed": m._absorbed_by is not None, "ran": m._lut_ran})
    return out

"""H027 backend selection for the LUT layers.

    set_backend(model, lut_backend="torch" | "lookup" | "fused",
                lut_compile=False, eval_backend="torch" | "packed",
                precision="fp32" | "bf16")

attaches a `LutState` (`layer._lut`) to every `LogicConv2d` / `LogicDense` and marks the
pools. With all defaults every layer runs the reference code (`_forward_torch`), bit for
bit. Anything else is validated here and raises on a setting it does not implement; no
backend changes a model, wiring, parameter or LUT.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

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


def logic_layers(model):
    return [m for m in model.modules() if isinstance(m, (LogicConv2d, LogicDense))]


def _validate(m):
    if isinstance(m, LogicConv3d):
        raise ValueError("lut backends do not support LogicConv3d (thermo3d)")
    if not isinstance(m.parametrization, LightLUTParametrization):
        raise ValueError(f"lut backends need parametrization='light', got "
                         f"{type(m.parametrization).__name__}")
    p = m.parametrization
    if p.forward_sampling != "soft":
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


def set_backend(model, lut_backend="torch", lut_compile=False, eval_backend="torch",
                precision="fp32", binary_first: Optional[bool] = None) -> list:
    """Attach the backend to every logic layer; returns `backend_report(model)`.

    `binary_first`: None = auto (the first logic layer is marked when a thermometer front
    precedes it; its input is then verified exact on the first call), True/False force.
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
            _validate(m)
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
    return backend_report(model)


def apply_compile(model, lut_backend: str, backend: str = "inductor", mode=None) -> int:
    """lut_compile=true. fused: compile the model (kernels are custom ops, so Inductor
    fuses the glue around them); torch / lookup: compile every logic layer. In place
    (`nn.Module.compile`), so state_dict keys are unchanged. Returns the number of modules
    compiled."""
    if lut_backend == "fused":
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
                        "pool_fused": st.fused_pool is not None, "ran": st.ran,
                        "ran_train": st.ran_train, "ran_eval": st.ran_eval,
                        "lang": st.lang})
        elif isinstance(m, OrPooling2d) and m._lut_req is not None:
            out.append({"module": name, "type": "OrPooling2d", "lut_backend": m._lut_req,
                        "absorbed": m._absorbed_by is not None, "ran": m._lut_ran})
    return out

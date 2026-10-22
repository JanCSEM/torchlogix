"""Layer-level dispatch for `lut_backend` / `eval_backend` (set by `torchlogix.backend`).

The reference arithmetic stays in the layers (`_forward_torch`); this module routes to the
alternatives and records the backend that ran in `layer._lut.ran` (+ `ran_train`, `ran_eval`, `lang`).
"""

import torch

from . import lookup, ops, packed
from .plan import (conv_leaf_table, dense_leaf_table, fold_tables, identity_table, pool_table)


def fuses_pool(layer) -> bool:
    """Whether conv `layer` evaluates its following OrPooling2d in this call. A pure
    function of static state and the train/eval flag, so the pool module can ask too."""
    st = layer._lut
    if st is None or st.fused_pool is None or layer.export_mode or st.fused_pool.export_mode:
        return False
    if layer._forward_hooks or layer._forward_pre_hooks:
        return False
    if not layer.training and st.eval_backend == "packed":
        return True
    # bf16 storage: the pool must see the fp32 values (rounding BEFORE the max creates ties
    # and re-routes the pool gradient), so the lookup path pools inside the conv too
    return st.backend == "fused" or (st.backend == "lookup" and st.bf16 and not st.head)


def _get_plan(layer, device):
    """H030 (1): the traceable accessor. Inside a compiled region the plan that
    `backend.prepare_plans` built eagerly is returned as is (no graph break); the builder
    `_plan` is dynamo-opaque (it is `torch.compiler.disable`d, a graph break per layer) and
    is called when no plan for this device exists yet or outside compilation."""
    st = layer._lut
    if st.plan is not None and torch.compiler.is_compiling() and st.plan["key"][0] == str(device):
        return st.plan
    return _plan(layer, device)


@torch.compiler.disable
def _plan(layer, device):
    # static tables built eagerly on first use: dynamo must not trace the CPU table
    # construction (job 5123806: Inductor tried to compile it for the CPU), so this is a
    # deliberate, single graph break per layer
    st = layer._lut
    ref = layer.connections._indices_L0 if hasattr(layer.connections, "_indices_L0") \
        else layer.connections.indices
    key = (str(device), ref.data_ptr(), ref._version,
           None if st.fused_pool is None else (st.fused_pool.kernel_size, st.fused_pool.stride,
                                               st.fused_pool.padding, st.fused_pool.ceil_mode))
    if st.plan is not None and st.plan["key"] == key:
        return st.plan
    if hasattr(layer, "kernel_positions"):
        leaf = conv_leaf_table(layer).to(device)
        ho, wo = layer.kernel_positions
        plan = {"key": key, "leaf": leaf, "id_pos": identity_table(ho * wo).to(device),
                "hw": (ho, wo)}
        if st.fused_pool is not None:
            p = st.fused_pool
            tab, hwq = pool_table(ho, wo, p.kernel_size, p.stride, p.padding, p.ceil_mode)
            plan["pool_pos"], plan["pool_hw"] = tab.to(device), hwq
    else:
        plan = {"key": key, "leaf": dense_leaf_table(layer).to(device),
                "id_pos": identity_table(1).to(device)}
    st.plan = plan
    return plan


def _to_kernel_input(st, x):
    """(B, ...) -> contiguous (B, N): uint8 bits for a binary-input layer, else
    float32 / bfloat16."""
    x = x.reshape(x.shape[0], -1)
    if st.binary_input:
        if not torch.compiler.is_compiling():
            # (a host read; skipped inside a compiled region, where it is a graph break.
            # The eager --correctness stage and the first eager call verify the front.)
            lookup.check_binary(st, x)
        return x.to(torch.uint8).contiguous()
    if x.dtype not in (torch.float32, torch.bfloat16):
        x = x.float()
    return x.contiguous()


def _out_dtype(st):
    return st.bf16 and not st.head


def _mark(layer, what, lang=None):
    """Record what ran: last call, last training-mode call, last eval-mode call, and for
    the fused backend the kernel language that was resolved (`ops.resolve_lang`)."""
    st = layer._lut
    st.ran = what
    if layer.training:
        st.ran_train = what
    else:
        st.ran_eval = what
    if lang is not None:
        st.lang = lang


def fold_ready(binarizer, conv, x) -> bool:
    """Whether `binarizer` (a FixedBinarization) hands its raw input and thresholds to `conv`
    on this forward (H030 fold): a function of static state and the train/eval flag. Anything
    else leaves the binarizer materialising the thermometer, as before."""
    st = conv._lut
    if st is None or not st.fold or st.backend != "fused" or conv.export_mode:
        return False
    if conv._forward_hooks or conv._forward_pre_hooks:        # hooks expect the bit tensor
        return False
    if not conv.training and st.eval_backend == "packed":     # packed eval keeps the bits
        return False
    if binarizer._forward_hooks or binarizer._forward_pre_hooks:   # hooks expect the bits too
        return False
    return x.dtype == torch.float32


def _fold_plan(layer, plan, raw_shape, jittered, thr):
    st = layer._lut
    key = (plan["key"], tuple(raw_shape), bool(jittered),
           tuple(thr.shape[1:] if jittered else thr.shape))      # the batch size is not in the tables
    fp = st.fold_plan
    if fp is not None and fp["key"] == key:
        return fp
    return _build_fold_plan(layer, plan, raw_shape, jittered, thr, key)


@torch.compiler.disable
def _build_fold_plan(layer, plan, raw_shape, jittered, thr, key):
    """The (raw element, threshold element) tables of the fold; built eagerly on first use
    (dynamo-opaque like `_plan`: a compiled model is warmed up eagerly first, see
    `dlgn.train.bench.warm_eager`)."""
    st = layer._lut
    raw, tidx, tstride = fold_tables(st.fold_src, layer, tuple(raw_shape), jittered,
                                     tuple(thr.shape))
    leaf = plan["leaf"]                                       # (K, P, 2, S), on the device
    dev = leaf.device
    raw, tidx = raw.to(dev), tidx.to(dev)
    fp = {"key": key, "leaf_r": raw[leaf.long()].to(torch.int32).contiguous(),
          "leaf_t": tidx[leaf.long()].to(torch.int32).contiguous(), "tstride": tstride}
    st.fold_plan = fp
    return fp


def _fold_conv_forward(layer, x, handed):
    """The fused first conv on the RAW input: leaf bits are computed in the tree kernel."""
    st = layer._lut
    thr, jittered = handed
    if thr.dtype != torch.float32:
        raise TypeError(f"fold: thresholds must be fp32, got {thr.dtype}")
    _mark(layer, "fused", ops.resolve_lang(x.device.type))
    st.fold_ran = True
    plan = _get_plan(layer, x.device)
    fp = _fold_plan(layer, plan, x.shape[1:], jittered, thr)
    xin = x.reshape(x.shape[0], -1)
    if xin.dtype != torch.float32:
        raise TypeError(f"fold: the raw input must be fp32, got {xin.dtype}")
    xin = xin.contiguous()
    w = torch.cat([s.reshape(-1) for s in lookup.sample_levels(layer)])
    pooled = fuses_pool(layer)
    pos, (hq, wq) = (plan["pool_pos"], plan["pool_hw"]) if pooled else \
        (plan["id_pos"], plan["hw"])
    y = ops.tree_fold(xin, fp["leaf_r"], fp["leaf_t"], thr.reshape(-1).contiguous(),
                      fp["tstride"], pos, w, layer.tree_depth, _out_dtype(st))
    return y.view(x.shape[0], layer.num_kernels, hq, wq)


def conv_forward(layer, x):
    st = layer._lut
    handed = layer._fold_thr
    if handed is not None:                    # H030: the binarizer returned its raw input
        layer._fold_thr = None                # consumed: never reused by a later forward
        return _fold_conv_forward(layer, x, handed)
    if st.fold_src is not None:
        st.fold_ran = False                   # the binarizer materialised the bits this time
    if st.backend == "torch" and (layer.training or st.eval_backend != "packed"):
        _mark(layer, "torch")
        return layer._forward_torch(x)
    if not layer.training and st.eval_backend == "packed":
        pool = st.fused_pool if fuses_pool(layer) else None
        _mark(layer, "packed")
        return packed.conv_forward(layer, x, pool)
    if st.backend == "lookup":
        _mark(layer, "lookup")
        out = lookup.conv_forward(layer, x)
        if fuses_pool(layer):
            p = st.fused_pool
            out = torch.nn.functional.max_pool2d(
                out.float(), kernel_size=p.kernel_size, stride=p.stride, padding=p.padding,
                ceil_mode=p.ceil_mode)
        return out.to(torch.bfloat16) if _out_dtype(st) else out
    # fused
    _mark(layer, "fused", ops.resolve_lang(x.device.type))
    plan = _get_plan(layer, x.device)
    xin = _to_kernel_input(st, layer._pad(x))
    w = torch.cat([s.reshape(-1) for s in lookup.sample_levels(layer)])
    pooled = fuses_pool(layer)
    pos, (hq, wq) = (plan["pool_pos"], plan["pool_hw"]) if pooled else \
        (plan["id_pos"], plan["hw"])
    y = ops.tree(xin, plan["leaf"], pos, w, layer.tree_depth, st.binary_input, _out_dtype(st))
    return y.view(x.shape[0], layer.num_kernels, hq, wq)


def dense_forward(layer, x):
    st = layer._lut
    if st.backend == "torch" and (layer.training or st.eval_backend != "packed"):
        _mark(layer, "torch")
        return layer._forward_torch(x)
    if not layer.training and st.eval_backend == "packed":
        _mark(layer, "packed")
        return packed.dense_forward(layer, x)
    if st.backend == "lookup":
        _mark(layer, "lookup")
        out = lookup.dense_forward(layer, x)
        return out.to(torch.bfloat16) if _out_dtype(st) else out
    _mark(layer, "fused", ops.resolve_lang(x.device.type))
    from ..functional import GradFactor
    if layer.grad_factor != 1.0:
        x = GradFactor.apply(x, layer.grad_factor)
    lead = x.shape[:-1]
    x = x.reshape(-1, layer.in_dim)
    plan = _get_plan(layer, x.device)
    xin = _to_kernel_input(st, x)
    w = lookup.sample_levels_dense(layer).reshape(-1)
    y = ops.tree(xin, plan["leaf"], plan["id_pos"], w, 1, st.binary_input, _out_dtype(st))
    return y.view(*lead, layer.out_dim)


def pool_absorbed(pool) -> bool:
    conv = pool._absorbed_by
    return conv is not None and fuses_pool(conv)

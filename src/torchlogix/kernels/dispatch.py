"""Layer-level dispatch for `lut_backend` / `eval_backend` (set by `torchlogix.backend`).

The reference arithmetic stays in the layers (`_forward_torch`); this module routes to the
alternatives and records the backend that ran in `layer._lut.ran`.
"""

import torch

from . import lookup, ops, packed
from .plan import (conv_leaf_table, dense_leaf_table, identity_table, pool_table)


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
    return st.backend == "fused"


def _plan(layer, device):
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
        lookup.check_binary(st, x)
        return x.to(torch.uint8).contiguous()
    if x.dtype not in (torch.float32, torch.bfloat16):
        x = x.float()
    return x.contiguous()


def _out_dtype(st):
    return st.bf16 and not st.head


def conv_forward(layer, x):
    st = layer._lut
    if st.backend == "torch" and (layer.training or st.eval_backend != "packed"):
        st.ran = "torch"
        return layer._forward_torch(x)
    if not layer.training and st.eval_backend == "packed":
        pool = st.fused_pool if fuses_pool(layer) else None
        st.ran = "packed"
        return packed.conv_forward(layer, x, pool)
    if st.backend == "lookup":
        st.ran = "lookup"
        out = lookup.conv_forward(layer, x)
        return out.to(torch.bfloat16) if _out_dtype(st) else out
    # fused
    st.ran = "fused"
    plan = _plan(layer, x.device)
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
        st.ran = "torch"
        return layer._forward_torch(x)
    if not layer.training and st.eval_backend == "packed":
        st.ran = "packed"
        return packed.dense_forward(layer, x)
    if st.backend == "lookup":
        st.ran = "lookup"
        out = lookup.dense_forward(layer, x)
        return out.to(torch.bfloat16) if _out_dtype(st) else out
    st.ran = "fused"
    from ..functional import GradFactor
    if layer.grad_factor != 1.0:
        x = GradFactor.apply(x, layer.grad_factor)
    lead = x.shape[:-1]
    x = x.reshape(-1, layer.in_dim)
    plan = _plan(layer, x.device)
    xin = _to_kernel_input(st, x)
    w = lookup.sample_levels_dense(layer).reshape(-1)
    y = ops.tree(xin, plan["leaf"], plan["id_pos"], w, 1, st.binary_input, _out_dtype(st))
    return y.view(*lead, layer.out_dim)


def pool_absorbed(pool) -> bool:
    conv = pool._absorbed_by
    return conv is not None and fuses_pool(conv)

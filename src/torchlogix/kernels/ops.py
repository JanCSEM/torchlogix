"""The fused tree op: `torch.library` custom ops with autograd, so eager and
`torch.compile` (Inductor fuses the glue around them) see the same op.

    tree_fwd(x, leaf, pos, w, depth, binary, out_bf16, need_arg) -> (y, arg)
    tree_bwd(gy, x, leaf, pos, arg, w, depth, binary, need_gx)   -> (gx, gw)

Implementation choice per call: CUDA tensors -> `FUSED_LANG` ("triton" | "cuda" | "auto";
auto = Triton when importable, else CUDA C++); CPU tensors -> `CPU_IMPL` ("reference" =
pure PyTorch, "triton" = the Triton CPU interpreter, TRITON_INTERPRET=1, "cuda_emu" is
test-only and installed by the tests).
"""

import os
from typing import Optional, Tuple

import torch

from . import reference

FUSED_LANG = os.environ.get("TORCHLOGIX_FUSED_LANG", "auto")
CPU_IMPL = "reference"
_CPU_EMU = None          # tests: module with fwd/bwd emulating the CUDA kernels on CPU


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def resolve_lang(device_type: str) -> str:
    if device_type == "cuda":
        if FUSED_LANG in ("triton", "cuda"):
            return FUSED_LANG
        return "triton" if triton_available() else "cuda"
    return CPU_IMPL


def _impl(lang):
    if lang == "triton":
        from . import triton_tree
        return triton_tree
    if lang == "cuda":
        from . import cuda_tree
        return cuda_tree
    if lang == "cuda_emu":
        return _CPU_EMU
    return None


@torch.library.custom_op("torchlogix::tree_fwd", mutates_args=())
def tree_fwd(x: torch.Tensor, leaf: torch.Tensor, pos: torch.Tensor, w: torch.Tensor,
             depth: int, binary: bool, out_bf16: bool, need_arg: bool
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    out_dtype = torch.bfloat16 if out_bf16 else torch.float32
    impl = _impl(resolve_lang(x.device.type))
    if impl is None:
        y, arg = reference.forward(x, leaf, pos, w, depth, binary, out_dtype)
        if not need_arg or pos.shape[1] == 1:
            arg = arg.new_empty(0)
        return y.contiguous(), arg.contiguous()
    return impl.fwd(x, leaf, pos, w, depth, binary, out_dtype, need_arg)


@tree_fwd.register_fake
def _(x, leaf, pos, w, depth, binary, out_bf16, need_arg):
    b, k, (q, win) = x.shape[0], leaf.shape[0], pos.shape
    y = x.new_empty((b, k, q), dtype=torch.bfloat16 if out_bf16 else torch.float32)
    arg = x.new_empty((b, k, q) if (need_arg and win > 1) else (0,), dtype=torch.uint8)
    return y, arg


@torch.library.custom_op("torchlogix::tree_bwd", mutates_args=())
def tree_bwd(gy: torch.Tensor, x: torch.Tensor, leaf: torch.Tensor, pos: torch.Tensor,
             arg: torch.Tensor, w: torch.Tensor, depth: int, binary: bool, need_gx: bool
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    need_gx = bool(need_gx and not binary)
    impl = _impl(resolve_lang(x.device.type))
    if impl is None:
        gx, gw = reference.backward(gy, x, leaf, pos, w, depth, binary, need_gx)
    else:
        gx, gw = impl.bwd(gy, x, leaf, pos, arg, w, depth, binary, need_gx)
    if gx is None:
        gx = gw.new_empty(0)
    return gx, gw


@tree_bwd.register_fake
def _(gy, x, leaf, pos, arg, w, depth, binary, need_gx):
    gx = x.new_empty(x.shape if (need_gx and not binary) else (0,), dtype=torch.float32)
    return gx, torch.empty_like(w)


def _setup(ctx, inputs, output):
    x, leaf, pos, w, depth, binary, out_bf16, need_arg = inputs
    _, arg = output
    ctx.save_for_backward(x, leaf, pos, arg, w)
    ctx.depth, ctx.binary = depth, binary


def _backward(ctx, gy, _garg):
    x, leaf, pos, arg, w = ctx.saved_tensors
    need_gx = bool(ctx.needs_input_grad[0])
    gx, gw = tree_bwd(gy, x, leaf, pos, arg, w, ctx.depth, ctx.binary, need_gx)
    gx = gx.to(x.dtype) if need_gx else None
    return gx, None, None, gw, None, None, None, None


tree_fwd.register_autograd(_backward, setup_context=_setup)


def tree(x: torch.Tensor, leaf, pos, w, depth: int, binary: bool, out_bf16: bool) -> torch.Tensor:
    """Run the fused tree op; returns y (B, K, Q)."""
    need_arg = bool(torch.is_grad_enabled() and w.requires_grad)
    y, _ = tree_fwd(x, leaf, pos, w, depth, binary, out_bf16, need_arg)
    return y

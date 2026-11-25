# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Per-device FFI calls for the EP/GEMM overlap (FusedMoE) path.

Each function wraps one ``nvte_ep_*_gemm_*`` call of ``ep_gemm.h``: an NCCL EP dispatch or combine
overlapped with the CuTeDSL grouped GEMMs through device readiness signals, on XLA's stream.
They take and return *per-device* arrays and must be called inside ``shard_map`` (all mesh axes
manual), one call per device, in the same order on every rank. BF16 only.

Shapes (per device): T local tokens, K top-k, R recv capacity, E local experts, H hidden, F FFN.
FC1 weights are ``[E, 2F, H]`` with gate/up in alternating 32-row blocks
(:func:`transformer_engine.jax.cpp_extensions.pack_gated_activation_pair` on ``[E, H, 2F]`` then
transpose), FC2 weights are ``[E, H, F]``.
"""

import functools
import os

import jax
import jax.numpy as jnp
from jax import ffi

import transformer_engine_jax

from .misc import jax_dtype_to_te_dtype

__all__ = [
    "ep_overlap_built",
    "ep_overlap_env_reason",
    "ep_overlap_workspace_size",
    "ep_dispatch_gemm_fwd",
    "ep_combine_gemm_fwd",
    "ep_combine_gemm_bwd",
    "ep_dispatch_gemm_bwd",
]


def ep_overlap_built() -> bool:
    """True when the JAX extension has the overlap FFI targets."""
    return hasattr(transformer_engine_jax, "ep_overlap_workspace_size")


def ep_overlap_env_reason():
    """None when the runtime environment enables the overlap GEMMs, else the reason."""
    if os.environ.get("NVTE_ENABLE_CUTEDSL_BACKEND", "0") != "1":
        return "needs NVTE_ENABLE_CUTEDSL_BACKEND=1"
    path = os.environ.get("NVTE_MOE_OVERLAP", "")
    if not path or not os.path.isdir(path):
        return "needs NVTE_MOE_OVERLAP pointing at the EPCompOverlap gemm/ tree"
    for var in (
        "NCCL_EP_COMM_SMS",
        "NCCL_EP_DISPATCH_SMS",
        "NCCL_EP_COMBINE_SMS",
        "NCCL_EP_DISPATCH_BWD_SMS",
        "NCCL_EP_COMBINE_BWD_SMS",
    ):
        if os.environ.get(var):
            return f"{var} must be unset (use ep_bootstrap(max_num_sms=...))"
    return None


@functools.lru_cache(maxsize=None)
def ep_overlap_workspace_size(
    hidden, ffn, num_local_experts, recv_capacity, top_k, alignment, topk_idx_dtype=jnp.int32
):
    """Workspace bytes for every overlap call of a layer.

    Also compiles, ahead of time, every CuTeDSL GEMM the layer's four calls will launch
    (``nvte_ep_overlap_compile``): the calls themselves would compile on first use, re-entering
    Python from an XLA execution thread (or a CUDA-graph capture). Runs at trace time, once per
    layer configuration, after ep_bootstrap (it needs the EP group)."""
    args = dict(
        hidden_dim=int(hidden),
        ffn_dim=int(ffn),
        num_local_experts=int(num_local_experts),
        recv_capacity=int(recv_capacity),
        with_weight_grad=True,
        top_k=int(top_k),
        dispatch_output_per_expert_alignment=int(alignment),
        topk_idx_dtype=int(jax_dtype_to_te_dtype(jnp.dtype(topk_idx_dtype))),
    )
    transformer_engine_jax.ep_overlap_compile(fp32_weight_grad=False, **args)
    return int(transformer_engine_jax.ep_overlap_workspace_size(mxfp8=False, **args))


def _attrs(top_k, alignment, topk_idx_dtype):
    return dict(
        top_k=int(top_k),
        dispatch_output_per_expert_alignment=int(alignment),
        topk_idx_dtype=int(jax_dtype_to_te_dtype(jnp.dtype(topk_idx_dtype))),
    )


def _sds(shape, dtype):
    return jax.ShapeDtypeStruct(tuple(int(s) for s in shape), dtype)


def ep_dispatch_gemm_fwd(topk_idx, tokens, topk_weights, w1, *, recv_capacity, alignment):
    """Dispatch || FC1 + SwiGLU. Returns (recv_tokens [R,H], recv_w [R] f32, gemm_out [R,2F],
    act_out [R,F] (= silu(gate)*up*prob), counts [E] i32, offsets [E+4] i32 (use [3:]),
    handle_mem)."""
    H = tokens.shape[-1]
    K = topk_idx.shape[-1]
    E, F2, Hw = w1.shape
    assert Hw == H and F2 % 2 == 0, (w1.shape, tokens.shape)
    F = F2 // 2
    R = recv_capacity
    hm = int(transformer_engine_jax.ep_handle_mem_size(int(K), int(alignment)))
    ws = ep_overlap_workspace_size(H, F, E, R, K, alignment, topk_idx.dtype)
    outs = (
        _sds((R, H), tokens.dtype),
        _sds((R,), jnp.float32),
        _sds((R, F2), jnp.bfloat16),
        _sds((R, F), tokens.dtype),
        _sds((E,), jnp.int32),
        _sds((E + 4,), jnp.int32),
        _sds((hm,), jnp.uint8),
        _sds((ws,), jnp.uint8),
    )
    res = ffi.ffi_call("te_ep_dispatch_gemm_fwd_ffi", outs)(
        topk_idx, tokens, topk_weights, w1, **_attrs(K, alignment, topk_idx.dtype)
    )
    return tuple(res[:7])


def ep_combine_gemm_fwd(
    handle_mem, act_out, offsets, w2, *, num_tokens, top_k, alignment, topk_idx_dtype=jnp.int32
):
    """FC2 || combine. Returns result [T, H] (sum over top-k of prob * FFN(x))."""
    R, F = act_out.shape
    E, H, Fw = w2.shape
    assert Fw == F, (w2.shape, act_out.shape)
    ws = ep_overlap_workspace_size(H, F, E, R, top_k, alignment, topk_idx_dtype)
    outs = (
        _sds((R, H), act_out.dtype),
        _sds((num_tokens, H), act_out.dtype),
        _sds((ws,), jnp.uint8),
    )
    _expert_out, result, _ws = ffi.ffi_call("te_ep_combine_gemm_fwd_ffi", outs)(
        handle_mem, act_out, offsets, w2, **_attrs(top_k, alignment, topk_idx_dtype)
    )
    return result


def ep_combine_gemm_bwd(
    handle_mem,
    grad_result,
    gemm_out,
    recv_w,
    counts,
    w2,
    act_out,
    *,
    top_k,
    alignment,
    topk_idx_dtype=jnp.int32,
):
    """Combine bwd || FC2 dgrad + SwiGLU bwd, + FC2 wgrad. Returns
    (grad_gemm_out [R,2F], grad_recv_w [R] f32, d_w2 [E,H,F])."""
    R, F2 = gemm_out.shape
    E, H, F = w2.shape
    ws = ep_overlap_workspace_size(H, F, E, R, top_k, alignment, topk_idx_dtype)
    outs = (
        _sds((R, H), grad_result.dtype),
        _sds((R, F2), gemm_out.dtype),
        _sds((R,), jnp.float32),
        _sds((E, H, F), w2.dtype),
        _sds((ws,), jnp.uint8),
    )
    _gx, grad_gemm_out, grad_recv_w, d_w2, _ws = ffi.ffi_call("te_ep_combine_gemm_bwd_ffi", outs)(
        handle_mem,
        grad_result,
        gemm_out,
        recv_w,
        counts,
        w2,
        act_out,
        **_attrs(top_k, alignment, topk_idx_dtype),
    )
    return grad_gemm_out, grad_recv_w, d_w2


def ep_dispatch_gemm_bwd(
    handle_mem,
    grad_gemm_out,
    grad_recv_w,
    counts,
    w1,
    recv_tokens,
    *,
    num_tokens,
    top_k,
    alignment,
    topk_idx_dtype=jnp.int32,
):
    """FC1 dgrad + FC1 wgrad || dispatch bwd. Returns
    (grad_tokens [T,H], grad_topk_weights [T,K] f32, d_w1 [E,2F,H])."""
    R, F2 = grad_gemm_out.shape
    E, _, H = w1.shape
    ws = ep_overlap_workspace_size(H, F2 // 2, E, R, top_k, alignment, topk_idx_dtype)
    outs = (
        _sds((R, H), recv_tokens.dtype),
        _sds((num_tokens, H), recv_tokens.dtype),
        _sds((num_tokens, top_k), jnp.float32),
        _sds((E, F2, H), w1.dtype),
        _sds((ws,), jnp.uint8),
    )
    _grt, grad_tokens, grad_topk, d_w1, _ws = ffi.ffi_call("te_ep_dispatch_gemm_bwd_ffi", outs)(
        handle_mem,
        grad_gemm_out,
        grad_recv_w,
        counts,
        w1,
        recv_tokens,
        **_attrs(top_k, alignment, topk_idx_dtype),
    )
    return grad_tokens, grad_topk, d_w1

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
    "ep_dispatch_gemm_fwd_mxfp8",
    "ep_combine_gemm_fwd_mxfp8",
    "ep_combine_gemm_bwd_mxfp8",
    "ep_dispatch_gemm_bwd_mxfp8",
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
    hidden,
    ffn,
    num_local_experts,
    recv_capacity,
    top_k,
    alignment,
    topk_idx_dtype=jnp.int32,
    mxfp8=False,
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
    transformer_engine_jax.ep_overlap_compile(fp32_weight_grad=False, mxfp8=bool(mxfp8), **args)
    return int(transformer_engine_jax.ep_overlap_workspace_size(mxfp8=bool(mxfp8), **args))


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


# ---------------------------------------------------------------------------------------- MXFP8
# E4M3 data with E8M0 scales per 32 elements. "compact": row-major [rows, cols / 32] (tokens,
# received rows, dY, grad_expert_out); "swizzled": GEMM-swizzled, flat rows * cols / 32 (weights,
# act_out, grad_gemm_out, columnwise copies).

_E4M3 = jnp.float8_e4m3fn
_E8M0 = jnp.float8_e8m0fnu


def ep_dispatch_gemm_fwd_mxfp8(
    topk_idx, tokens, tokens_scale, topk_weights, w1, w1_scale, *, recv_capacity, alignment
):
    """MXFP8 dispatch || FC1 + SwiGLU. tokens [T,H] E4M3 with compact scales [T,H/32]; w1
    [E,2F,H] E4M3 with swizzled scales. Returns (recv_tokens [R,H] E4M3, recv_scale [R,H/32],
    recv_w [R], gemm_out [R,2F] bf16, act_out [R,F] E4M3 (silu(gate)*up, no prob), act_scale
    (swizzled), counts, offsets [E+4], handle_mem)."""
    H = tokens.shape[-1]
    K = topk_idx.shape[-1]
    E, F2, _ = w1.shape
    F = F2 // 2
    R = recv_capacity
    hm = int(transformer_engine_jax.ep_handle_mem_size(int(K), int(alignment)))
    ws = ep_overlap_workspace_size(H, F, E, R, K, alignment, topk_idx.dtype, mxfp8=True)
    outs = (
        _sds((R, H), _E4M3),
        _sds((R, H // 32), _E8M0),
        _sds((R,), jnp.float32),
        _sds((R, F2), jnp.bfloat16),
        _sds((R, F), _E4M3),
        _sds((R * F // 32,), _E8M0),
        _sds((E,), jnp.int32),
        _sds((E + 4,), jnp.int32),
        _sds((hm,), jnp.uint8),
        _sds((ws,), jnp.uint8),
    )
    res = ffi.ffi_call("te_ep_dispatch_gemm_fwd_mxfp8_ffi", outs)(
        topk_idx,
        tokens,
        tokens_scale,
        topk_weights,
        w1,
        w1_scale,
        **_attrs(K, alignment, topk_idx.dtype),
    )
    return tuple(res[:9])


def ep_combine_gemm_fwd_mxfp8(
    handle_mem,
    act_out,
    act_scale,
    recv_w,
    offsets,
    w2,
    w2_scale,
    *,
    num_tokens,
    top_k,
    alignment,
    topk_idx_dtype=jnp.int32,
):
    """MXFP8 FC2 || combine; the routing weights are applied to FC2's output. Returns [T, H]."""
    R, F = act_out.shape
    E, H, _ = w2.shape
    ws = ep_overlap_workspace_size(H, F, E, R, top_k, alignment, topk_idx_dtype, mxfp8=True)
    outs = (
        _sds((R, H), jnp.bfloat16),
        _sds((num_tokens, H), jnp.bfloat16),
        _sds((ws,), jnp.uint8),
    )
    _eo, result, _ws = ffi.ffi_call("te_ep_combine_gemm_fwd_mxfp8_ffi", outs)(
        handle_mem,
        act_out,
        act_scale,
        recv_w,
        offsets,
        w2,
        w2_scale,
        **_attrs(top_k, alignment, topk_idx_dtype),
    )
    return result


def ep_combine_gemm_bwd_mxfp8(
    handle_mem,
    dy,
    dy_scale,
    gemm_out,
    recv_w,
    counts,
    w2_col,
    w2_col_scale,
    *,
    top_k,
    alignment,
    weight_grad_dtype=jnp.bfloat16,
    topk_idx_dtype=jnp.int32,
):
    """MXFP8 combine bwd || FC2 dgrad + dSwiGLU, + FC2 wgrad. dy [T,H] E4M3 compact scales; w2_col
    the columnwise [E,H,F] weight. Returns (grad_gemm_out, its scales, its columnwise copy and
    scales (all E4M3/swizzled), grad_recv_w [R] f32, d_w2 [E,H,F])."""
    R, F2 = gemm_out.shape
    F = F2 // 2
    E, H, _ = w2_col.shape
    ws = ep_overlap_workspace_size(H, F, E, R, top_k, alignment, topk_idx_dtype, mxfp8=True)
    outs = (
        _sds((R, H), _E4M3),
        _sds((R, H // 32), _E8M0),
        _sds((R, H), _E4M3),
        _sds((R * H // 32,), _E8M0),
        _sds((R, F2), _E4M3),
        _sds((R * F2 // 32,), _E8M0),
        _sds((R, F2), _E4M3),
        _sds((R * F2 // 32,), _E8M0),
        _sds((R,), jnp.float32),
        _sds((R, F), _E4M3),
        _sds((R * F // 32,), _E8M0),
        _sds((E, H, F), weight_grad_dtype),
        _sds((ws,), jnp.uint8),
    )
    res = ffi.ffi_call("te_ep_combine_gemm_bwd_mxfp8_ffi", outs)(
        handle_mem,
        dy,
        dy_scale,
        gemm_out,
        recv_w,
        counts,
        w2_col,
        w2_col_scale,
        **_attrs(top_k, alignment, topk_idx_dtype),
    )
    _geo, _geo_s, _geo_c, _geo_cs, ggo, ggo_s, ggo_c, ggo_cs, grad_recv_w, _ac, _acs, d_w2, _ = res
    return ggo, ggo_s, ggo_c, ggo_cs, grad_recv_w, d_w2


def ep_dispatch_gemm_bwd_mxfp8(
    handle_mem,
    ggo,
    ggo_s,
    ggo_c,
    ggo_cs,
    grad_recv_w,
    counts,
    w1_col,
    w1_col_scale,
    recv_tokens,
    recv_scale,
    *,
    num_tokens,
    top_k,
    alignment,
    weight_grad_dtype=jnp.bfloat16,
    topk_idx_dtype=jnp.int32,
):
    """MXFP8 FC1 dgrad + FC1 wgrad || dispatch bwd. Returns (grad_tokens [T,H] bf16,
    grad_topk_weights [T,K] f32, d_w1 [E,2F,H])."""
    R, F2 = ggo.shape
    E, _, H = w1_col.shape
    ws = ep_overlap_workspace_size(H, F2 // 2, E, R, top_k, alignment, topk_idx_dtype, mxfp8=True)
    outs = (
        _sds((R, H), jnp.bfloat16),
        _sds((num_tokens, H), jnp.bfloat16),
        _sds((num_tokens, top_k), jnp.float32),
        _sds((E, F2, H), weight_grad_dtype),
        _sds((R, H), _E4M3),
        _sds((R * H // 32,), _E8M0),
        _sds((ws,), jnp.uint8),
    )
    res = ffi.ffi_call("te_ep_dispatch_gemm_bwd_mxfp8_ffi", outs)(
        handle_mem,
        ggo,
        ggo_s,
        ggo_c,
        ggo_cs,
        grad_recv_w,
        counts,
        w1_col,
        w1_col_scale,
        recv_tokens,
        recv_scale,
        **_attrs(top_k, alignment, topk_idx_dtype),
    )
    _grt, grad_tokens, grad_topk, d_w1, _rc, _rcs, _ws = res
    return grad_tokens, grad_topk, d_w1

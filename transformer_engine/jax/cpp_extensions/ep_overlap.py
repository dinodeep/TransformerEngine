# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Per-device FFI calls for the EP/GEMM overlap (FusedMoE) path.

Each function wraps one ``nvte_ep_*_gemm_*`` call of ``ep_gemm.h``: an NCCL EP dispatch or combine
overlapped with the CuTeDSL grouped GEMMs through device readiness signals, on XLA's stream.
They take and return *per-device* arrays and must be called inside ``shard_map`` (all mesh axes
manual), one call per device, in the same order on every rank.

Each call takes a ``scaling_mode``: ``ScalingMode.NO_SCALING`` (BF16) or
``ScalingMode.MXFP8_1D_SCALING`` (E4M3 data, E8M0 scales per 32 elements). Scale arguments are
optional and default to zero-size arrays, which is how BF16 calls pass them; the MXFP8-only
outputs are zero-size under BF16. MXFP8 operand forms follow ``ep_gemm.h``: tokens, received rows
and dY carry compact rowwise scales ``[rows, cols / 32]``; weights and GEMM inputs/outputs carry
GEMM-swizzled scales (flat, ``rows * cols / 32``); the backward reads the weights columnwise.

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
from ..quantize import ScalingMode

__all__ = [
    "check_ep_overlap_sms_env",
    "ep_overlap_workspace_size",
    "ep_dispatch_gemm_fwd",
    "ep_combine_gemm_fwd",
    "ep_combine_gemm_bwd",
    "ep_dispatch_gemm_bwd",
]


# NCCL EP's SM overrides; the overlap path sizes its GEMM/communication SM split from
# ep_bootstrap(max_num_sms=...) and cannot honour them.
NCCL_EP_SMS_ENV_VARS = (
    "NCCL_EP_COMM_SMS",
    "NCCL_EP_DISPATCH_SMS",
    "NCCL_EP_COMBINE_SMS",
    "NCCL_EP_DISPATCH_BWD_SMS",
    "NCCL_EP_COMBINE_BWD_SMS",
)


def check_ep_overlap_sms_env():
    """Raise when an NCCL EP SM override is set: the overlapped GEMMs run on the SMs NCCL EP
    leaves free, so the split must come from ep_bootstrap(max_num_sms=...) alone."""
    overrides = [var for var in NCCL_EP_SMS_ENV_VARS if os.environ.get(var)]
    if overrides:
        raise ValueError(
            f"enable_ep_overlap=True cannot run with {', '.join(overrides)} set: the overlapped"
            " GEMMs use the SMs NCCL EP leaves free, so the SM split must come from"
            " ep_bootstrap(max_num_sms=...). Unset these variables or disable the overlap."
        )


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


_E4M3 = jnp.float8_e4m3fn
_E8M0 = jnp.float8_e8m0fnu


def _attrs(top_k, alignment, topk_idx_dtype, scaling_mode):
    """The overlap handlers' attributes (TE/JAX GEMM style: ``scaling_mode`` as its int value)."""
    if scaling_mode not in (ScalingMode.NO_SCALING, ScalingMode.MXFP8_1D_SCALING):
        raise ValueError(f"EP overlap supports NO_SCALING or MXFP8_1D_SCALING, got {scaling_mode}")
    return dict(
        top_k=int(top_k),
        dispatch_output_per_expert_alignment=int(alignment),
        topk_idx_dtype=int(jax_dtype_to_te_dtype(jnp.dtype(topk_idx_dtype))),
        scaling_mode=int(scaling_mode.value),
    )


def _sds(shape, dtype):
    return jax.ShapeDtypeStruct(tuple(int(s) for s in shape), dtype)


def _empty(dtype=_E8M0):
    """Zero-size placeholder for an operand the scaling mode does not use."""
    return jnp.zeros((0,), dtype)


def _mxfp8(scaling_mode):
    return scaling_mode == ScalingMode.MXFP8_1D_SCALING


def ep_dispatch_gemm_fwd(
    topk_idx,
    tokens,
    topk_weights,
    w1,
    *,
    recv_capacity,
    alignment,
    tokens_scale_inv=None,
    w1_scale_inv=None,
    scaling_mode=ScalingMode.NO_SCALING,
):
    """Dispatch || FC1 + SwiGLU. tokens [T,H] (MXFP8: compact scales [T,H/32]); w1 [E,2F,H] (MXFP8:
    swizzled scales). Returns (recv_tokens [R,H], recv_scale_inv, recv_w [R] f32, gemm_out [R,2F]
    bf16, act_out [R,F], act_out_scale_inv, counts [E] i32, offsets [E+4] i32 (use [3:]),
    handle_mem). BF16: act_out = silu(gate)*up*prob and the scale outputs are zero-size. MXFP8:
    act_out = silu(gate)*up (the routing weight is applied to FC2's output)."""
    mx = _mxfp8(scaling_mode)
    H = tokens.shape[-1]
    K = topk_idx.shape[-1]
    E, F2, Hw = w1.shape
    assert Hw == H and F2 % 2 == 0, (w1.shape, tokens.shape)
    F = F2 // 2
    R = recv_capacity
    hm = int(transformer_engine_jax.ep_handle_mem_size(int(K), int(alignment)))
    ws = ep_overlap_workspace_size(H, F, E, R, K, alignment, topk_idx.dtype, mxfp8=mx)
    data = _E4M3 if mx else tokens.dtype
    outs = (
        _sds((R, H), data),
        _sds((R, H // 32) if mx else (0,), _E8M0),
        _sds((R,), jnp.float32),
        _sds((R, F2), jnp.bfloat16),
        _sds((R, F), data),
        _sds((R * F // 32,) if mx else (0,), _E8M0),
        _sds((E,), jnp.int32),
        _sds((E + 4,), jnp.int32),
        _sds((hm,), jnp.uint8),
        _sds((ws,), jnp.uint8),
    )
    res = ffi.ffi_call("te_ep_dispatch_gemm_fwd_ffi", outs)(
        topk_idx,
        tokens,
        _empty() if tokens_scale_inv is None else tokens_scale_inv,
        topk_weights,
        w1,
        _empty() if w1_scale_inv is None else w1_scale_inv,
        **_attrs(K, alignment, topk_idx.dtype, scaling_mode),
    )
    return tuple(res[:9])


def ep_combine_gemm_fwd(
    handle_mem,
    act_out,
    recv_w,
    offsets,
    w2,
    *,
    num_tokens,
    top_k,
    alignment,
    act_out_scale_inv=None,
    w2_scale_inv=None,
    scaling_mode=ScalingMode.NO_SCALING,
    topk_idx_dtype=jnp.int32,
):
    """FC2 || combine. Returns result [T, H] bf16 (sum over top-k of prob * FFN(x)). BF16 reads
    the routing weights from act_out; MXFP8 applies recv_w to FC2's output."""
    mx = _mxfp8(scaling_mode)
    R, F = act_out.shape
    E, H, Fw = w2.shape
    assert Fw == F, (w2.shape, act_out.shape)
    ws = ep_overlap_workspace_size(H, F, E, R, top_k, alignment, topk_idx_dtype, mxfp8=mx)
    outs = (
        _sds((R, H), jnp.bfloat16),
        _sds((num_tokens, H), jnp.bfloat16),
        _sds((ws,), jnp.uint8),
    )
    _expert_out, result, _ws = ffi.ffi_call("te_ep_combine_gemm_fwd_ffi", outs)(
        handle_mem,
        act_out,
        _empty() if act_out_scale_inv is None else act_out_scale_inv,
        recv_w,
        offsets,
        w2,
        _empty() if w2_scale_inv is None else w2_scale_inv,
        **_attrs(top_k, alignment, topk_idx_dtype, scaling_mode),
    )
    return result


def ep_combine_gemm_bwd(
    handle_mem,
    dy,
    gemm_out,
    recv_w,
    counts,
    w2,
    *,
    top_k,
    alignment,
    wgrad_input=None,
    dy_scale_inv=None,
    w2_scale_inv=None,
    weight_grad_dtype=None,
    scaling_mode=ScalingMode.NO_SCALING,
    topk_idx_dtype=jnp.int32,
):
    """Combine bwd || FC2 dgrad + dSwiGLU, + FC2 wgrad. BF16: w2 [E,H,F] and wgrad_input = the
    forward act_out. MXFP8: dy [T,H] E4M3 with compact scales, w2 the columnwise [E,H,F] copy with
    swizzled scales; the call writes the columnwise activation the wgrad reads. Returns
    (grad_gemm_out [R,2F], its scales, its columnwise copy, its scales (MXFP8; zero-size for BF16),
    grad_recv_w [R] f32, d_w2 [E,H,F])."""
    mx = _mxfp8(scaling_mode)
    R, F2 = gemm_out.shape
    F = F2 // 2
    E, H, _ = w2.shape
    assert mx or wgrad_input is not None, "BF16 needs wgrad_input (the forward act_out)"
    ws = ep_overlap_workspace_size(H, F, E, R, top_k, alignment, topk_idx_dtype, mxfp8=mx)
    data = _E4M3 if mx else dy.dtype

    def mx_only(shape, dtype):
        return _sds(shape if mx else (0,), dtype)

    outs = (
        _sds((R, H), data),
        mx_only((R, H // 32), _E8M0),
        mx_only((R, H), _E4M3),
        mx_only((R * H // 32,), _E8M0),
        _sds((R, F2), data if mx else gemm_out.dtype),
        mx_only((R * F2 // 32,), _E8M0),
        mx_only((R, F2), _E4M3),
        mx_only((R * F2 // 32,), _E8M0),
        _sds((R,), jnp.float32),
        mx_only((R, F), _E4M3),
        mx_only((R * F // 32,), _E8M0),
        _sds((E, H, F), weight_grad_dtype or (jnp.bfloat16 if mx else w2.dtype)),
        _sds((ws,), jnp.uint8),
    )
    res = ffi.ffi_call("te_ep_combine_gemm_bwd_ffi", outs)(
        handle_mem,
        dy,
        _empty() if dy_scale_inv is None else dy_scale_inv,
        gemm_out,
        recv_w,
        counts,
        w2,
        _empty() if w2_scale_inv is None else w2_scale_inv,
        _empty(data) if wgrad_input is None else wgrad_input,
        **_attrs(top_k, alignment, topk_idx_dtype, scaling_mode),
    )
    _geo, _geo_s, _geo_c, _geo_cs, ggo, ggo_s, ggo_c, ggo_cs, grad_recv_w, _ac, _acs, d_w2, _ = res
    return ggo, ggo_s, ggo_c, ggo_cs, grad_recv_w, d_w2


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
    grad_gemm_out_scale_inv=None,
    grad_gemm_out_col=None,
    grad_gemm_out_col_scale_inv=None,
    w1_scale_inv=None,
    recv_scale_inv=None,
    weight_grad_dtype=None,
    scaling_mode=ScalingMode.NO_SCALING,
    topk_idx_dtype=jnp.int32,
):
    """FC1 dgrad + FC1 wgrad || dispatch bwd. BF16: w1 [E,2F,H]. MXFP8: grad_gemm_out rowwise and
    columnwise with swizzled scales, w1 the columnwise copy, recv_tokens with the forward's compact
    scales. Returns (grad_tokens [T,H] bf16, grad_topk_weights [T,K] f32, d_w1 [E,2F,H])."""
    mx = _mxfp8(scaling_mode)
    R, F2 = grad_gemm_out.shape
    E, _, H = w1.shape
    ws = ep_overlap_workspace_size(H, F2 // 2, E, R, top_k, alignment, topk_idx_dtype, mxfp8=mx)
    outs = (
        _sds((R, H), jnp.bfloat16),
        _sds((num_tokens, H), jnp.bfloat16),
        _sds((num_tokens, top_k), jnp.float32),
        _sds((E, F2, H), weight_grad_dtype or (jnp.bfloat16 if mx else w1.dtype)),
        _sds((R, H) if mx else (0,), _E4M3),
        _sds((R * H // 32,) if mx else (0,), _E8M0),
        _sds((ws,), jnp.uint8),
    )
    res = ffi.ffi_call("te_ep_dispatch_gemm_bwd_ffi", outs)(
        handle_mem,
        grad_gemm_out,
        _empty() if grad_gemm_out_scale_inv is None else grad_gemm_out_scale_inv,
        _empty(_E4M3) if grad_gemm_out_col is None else grad_gemm_out_col,
        _empty() if grad_gemm_out_col_scale_inv is None else grad_gemm_out_col_scale_inv,
        grad_recv_w,
        counts,
        w1,
        _empty() if w1_scale_inv is None else w1_scale_inv,
        recv_tokens,
        _empty() if recv_scale_inv is None else recv_scale_inv,
        **_attrs(top_k, alignment, topk_idx_dtype, scaling_mode),
    )
    _grt, grad_tokens, grad_topk, d_w1, _rc, _rcs, _ws = res
    return grad_tokens, grad_topk, d_w1

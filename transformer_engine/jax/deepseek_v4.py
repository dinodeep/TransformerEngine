# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Differentiable per-kernel JAX APIs for DeepSeek-V4 CSA and HCA.

The cuDNN Frontend package owns only the raw CuTeDSL forward/backward calls.
Transformer Engine owns automatic differentiation and the public JAX contract.
These functions intentionally do not compose a higher-level attention module.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from .sharding import _get_mesh


def _batch_axes_spec(batch_axes: tuple[str, ...], rank: int) -> P:
    """Return a leading-dimension shard_map spec for active batch axes."""
    mesh = _get_mesh()
    active_axes = tuple(
        axis for axis in batch_axes if axis in mesh.axis_names and mesh.shape[axis] > 1
    )
    if not active_axes:
        return P(*((None,) * rank))
    leading_axis: str | tuple[str, ...] = active_axes[0] if len(active_axes) == 1 else active_axes
    return P(leading_axis, *((None,) * (rank - 1)))


def _batched_shard_map(fn, in_specs, out_spec, batch_axes):
    """Map a batched CSA kernel locally over MaxText DP/FSDP/EP axes.

    ``shard_map`` marks these axes manual inside the body.  Thus the opaque
    CuTe calls receive local packed tensors rather than inducing an all-gather
    to recover the global batch.
    """
    mesh = _get_mesh()
    if (
        mesh is None
        or mesh.empty
        or not any(axis in mesh.axis_names and mesh.shape[axis] > 1 for axis in batch_axes)
    ):
        return fn
    return shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_spec, check_rep=False)


def _cudnn_compressor_forward(
    kv, score, ape, cu_seqlens, cu_seqlens_comp, *, total_comp, ratio, coff
):
    from cudnn import csa_compressor_forward_jax_sm100

    return csa_compressor_forward_jax_sm100(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        total_comp=total_comp,
        ratio=ratio,
        coff=coff,
    )


def _cudnn_compressor_backward(
    kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_out, *, ratio, coff
):
    from cudnn import csa_compressor_backward_jax_sm100

    return csa_compressor_backward_jax_sm100(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        grad_out,
        ratio=ratio,
        coff=coff,
    )


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def _csa_compressor(kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, coff):
    return _cudnn_compressor_forward(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        total_comp=total_comp,
        ratio=ratio,
        coff=coff,
    )


def _csa_compressor_fwd(kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, coff):
    out = _cudnn_compressor_forward(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        total_comp=total_comp,
        ratio=ratio,
        coff=coff,
    )
    return out, (kv, score, ape, cu_seqlens, cu_seqlens_comp)


def _csa_compressor_bwd(total_comp, ratio, coff, residual, grad_out):
    del total_comp
    kv, score, ape, cu_seqlens, cu_seqlens_comp = residual
    grad_kv, grad_score, grad_ape = _cudnn_compressor_backward(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        grad_out.astype(jnp.bfloat16),
        ratio=ratio,
        coff=coff,
    )
    return grad_kv, grad_score, grad_ape, None, None


_csa_compressor.defvjp(_csa_compressor_fwd, _csa_compressor_bwd)


def csa_compressor(
    kv: Any,
    score: Any,
    ape: Any,
    cu_seqlens: Any,
    cu_seqlens_comp: Any,
    *,
    total_comp: int,
    ratio: int = 4,
    coff: int = 2,
) -> Any:
    """Differentiable SM100 CSA/HCA compressor kernel.

    Inputs use the raw packed kernel layout. Sequence metadata is treated as
    non-differentiable; gradients are returned for ``kv``, ``score``, and ``ape``.
    ``ratio=4, coff=2`` is DSv4 CSA; ``ratio=128, coff=1`` is DSv4 HCA.
    """
    return _csa_compressor(kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, coff)


def csa_compressor_batched(
    kv: Any,
    score: Any,
    ape: Any,
    *,
    batch_axes: tuple[str, ...] = (),
    ratio: int = 4,
    coff: int = 2,
) -> Any:
    """Run the differentiable compressor independently on local batch shards.

    ``batch_axes`` may contain several mesh axes.  MaxText passes
    ``("data", "fsdp", "expert")`` so FSDP and EP both act as data
    parallelism for this attention-only region.

    Each sequence yields ``sequence // ratio`` blocks; a trailing partial block
    is dropped (and receives zero gradient), matching the DSv4 reference.
    """
    if kv.ndim != 3 or score.shape != kv.shape:
        raise ValueError("kv and score must have shape [batch, sequence, width]")
    if kv.shape[2] % coff:
        raise ValueError(f"compressor width {kv.shape[2]} is not divisible by coff={coff}")
    if kv.shape[1] < ratio:
        raise ValueError(f"compressor requires sequence >= ratio ({ratio})")
    batch_spec = _batch_axes_spec(batch_axes, 3)

    def body(local_kv, local_score, local_ape):
        batch, sequence, width = local_kv.shape
        blocks = sequence // ratio
        cu_seqlens = jnp.arange(batch + 1, dtype=jnp.int32) * sequence
        cu_seqlens_comp = jnp.arange(batch + 1, dtype=jnp.int32) * blocks
        out = csa_compressor(
            local_kv.reshape(batch * sequence, width),
            local_score.reshape(batch * sequence, width),
            local_ape,
            cu_seqlens,
            cu_seqlens_comp,
            total_comp=batch * blocks,
            ratio=ratio,
            coff=coff,
        )
        return out.reshape(batch, blocks, width // coff)

    return _batched_shard_map(
        body,
        (batch_spec, batch_spec, P(None, None)),
        _batch_axes_spec(batch_axes, 3),
        batch_axes,
    )(kv, score, ape)


def hca_compressor_batched(
    kv: Any,
    score: Any,
    ape: Any,
    *,
    batch_axes: tuple[str, ...] = (),
    ratio: int = 128,
) -> Any:
    """DSv4 HCA compressor: non-overlapping softmax pooling over ``ratio`` tokens.

    ``kv``/``score`` are ``[batch, sequence, head_dim]`` and ``ape`` is
    ``[ratio, head_dim]``; returns ``[batch, sequence // ratio, head_dim]``.
    """
    return csa_compressor_batched(kv, score, ape, batch_axes=batch_axes, ratio=ratio, coff=1)


def _cudnn_indexer_forward(q, k, weights, *, ratio, sm_scale, return_lse=False):
    from cudnn import indexer_forward_jax_sm100

    return indexer_forward_jax_sm100(
        q, k, weights, ratio=ratio, sm_scale=sm_scale, return_lse=return_lse
    )


def _cudnn_dense_indexer_backward(
    q, k, weights, attn_score, attn_l1norm, index_score, index_lse, grad_loss, *, ratio, sm_scale
):
    from cudnn import dense_indexer_backward_jax_sm100

    return dense_indexer_backward_jax_sm100(
        q,
        k,
        weights,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
        grad_loss,
        ratio=ratio,
        sm_scale=sm_scale,
    )


def _indexer_reference(q, k, weights, ratio, sm_scale):
    dots = jnp.einsum("bshd,bwkd->bhsw", q.astype(jnp.float32), k.astype(jnp.float32))
    scores = jnp.einsum(
        "bhsw,bsh->bsw",
        jax.nn.relu(dots) * sm_scale,
        weights.astype(jnp.bfloat16).astype(jnp.float32),
    )
    q_positions = jnp.arange(q.shape[1])[:, None]
    block_ends = ratio * (jnp.arange(k.shape[1])[None, :] + 1)
    return jnp.where(block_ends <= q_positions + 1, scores, -jnp.inf)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _dsa_indexer(q, k, weights, ratio, sm_scale):
    return _cudnn_indexer_forward(q, k, weights, ratio=ratio, sm_scale=sm_scale)


def _dsa_indexer_fwd(q, k, weights, ratio, sm_scale):
    out = _cudnn_indexer_forward(q, k, weights, ratio=ratio, sm_scale=sm_scale)
    return out, (q, k, weights)


def _dsa_indexer_bwd(ratio, sm_scale, residual, grad_out):
    q, k, weights = residual
    _, pullback = jax.vjp(
        lambda q_, k_, weights_: _indexer_reference(q_, k_, weights_, ratio, sm_scale),
        q,
        k,
        weights,
    )
    return pullback(grad_out)


_dsa_indexer.defvjp(_dsa_indexer_fwd, _dsa_indexer_bwd)


def dsa_indexer(
    q: Any,
    k: Any,
    weights: Any,
    *,
    ratio: int = 4,
    sm_scale: float = 1.0,
) -> Any:
    """Differentiable SM100 DSv4 indexer score kernel.

    The CuTeDSL kernel implements the forward pass. Until cuDNN exposes an
    equivalent compact JAX backward call, TE computes its exact mathematical
    VJP with JAX operations.
    """
    return _dsa_indexer(q, k, weights, ratio, sm_scale)


def dsa_indexer_batched(
    q: Any,
    k: Any,
    weights: Any,
    *,
    batch_axes: tuple[str, ...] = (),
    ratio: int = 4,
    sm_scale: float = 1.0,
) -> Any:
    """Run the differentiable indexer locally on each batch shard."""
    batch_spec_q = _batch_axes_spec(batch_axes, q.ndim)
    batch_spec_k = _batch_axes_spec(batch_axes, k.ndim)
    batch_spec_w = _batch_axes_spec(batch_axes, weights.ndim)
    return _batched_shard_map(
        lambda local_q, local_k, local_weights: dsa_indexer(
            local_q, local_k, local_weights, ratio=ratio, sm_scale=sm_scale
        ),
        (batch_spec_q, batch_spec_k, batch_spec_w),
        _batch_axes_spec(batch_axes, 3),
        batch_axes,
    )(q, k, weights)


def _cudnn_indexer_top_k(scores, seq_lens, *, top_k, tie_break):
    from cudnn import indexer_top_k_jax_sm100

    return indexer_top_k_jax_sm100(scores, seq_lens, top_k=top_k, tie_break=tie_break)


def dsa_indexer_top_k(
    scores: Any,
    *,
    top_k: int,
    ratio: int = 4,
    seq_lens: Any = None,
    tie_break: int = 1,
) -> Any:
    """Forward-only per-row top-``top_k`` block selection over DSv4 indexer scores.

    ``scores`` is ``[batch, query_length, num_blocks]``. Query ``t`` may only select its ratio-causal prefix of
    ``min((t + 1) // ratio, num_blocks)`` blocks unless ``seq_lens`` (``[batch, query_length]`` int32, the
    per-row valid prefix length) is given. Only that prefix is read, so no ``-inf`` masking of ``scores`` is
    needed. Returns ``[batch, query_length, top_k]`` int32 block ids: **unsorted**, ``-1`` where a row has
    fewer than ``top_k`` valid blocks. ``tie_break=1`` prefers the smaller block id among equal scores, as
    ``jax.lax.top_k`` does. The result is an integer selection and carries no gradient.
    """
    if scores.ndim != 3:
        raise ValueError("expected scores=[B,S,W]")
    batch, seqlen_q, num_blocks = scores.shape
    if top_k > num_blocks:
        raise ValueError(f"top_k ({top_k}) exceeds the number of blocks ({num_blocks})")
    if seq_lens is None:
        seq_lens = jnp.minimum((jnp.arange(seqlen_q, dtype=jnp.int32) + 1) // ratio, num_blocks)
        seq_lens = jnp.broadcast_to(seq_lens, (batch, seqlen_q))
    elif seq_lens.shape != (batch, seqlen_q):
        raise ValueError("seq_lens must have shape [batch, query_length]")
    scores = jax.lax.stop_gradient(scores)
    indices = _cudnn_indexer_top_k(
        scores.reshape(batch * seqlen_q, num_blocks),
        seq_lens.astype(jnp.int32).reshape(batch * seqlen_q),
        top_k=top_k,
        tie_break=tie_break,
    )
    return indices.reshape(batch, seqlen_q, top_k)


def dsa_indexer_top_k_batched(
    scores: Any,
    *,
    batch_axes: tuple[str, ...] = (),
    top_k: int,
    ratio: int = 4,
    tie_break: int = 1,
) -> Any:
    """Run :func:`dsa_indexer_top_k` locally on each batch shard (ratio-causal prefixes)."""
    return _batched_shard_map(
        lambda local_scores: dsa_indexer_top_k(
            local_scores, top_k=top_k, ratio=ratio, tie_break=tie_break
        ),
        (_batch_axes_spec(batch_axes, 3),),
        _batch_axes_spec(batch_axes, 3),
        batch_axes,
    )(scores)


def _ratio_causal_valid(batch, seqlen_q, seqlen_k, ratio):
    """Blocks a query may read: block ``w`` is complete once ``(w + 1) * ratio <= position + 1``."""
    q_positions = jnp.arange(seqlen_q)[:, None]
    block_ends = ratio * (jnp.arange(seqlen_k)[None, :] + 1)
    return jnp.broadcast_to(block_ends <= q_positions + 1, (batch, seqlen_q, seqlen_k))


def _indexer_kl_sum(scores, lse, target, ratio, eps):
    """Sum over tokens of ``KL(target || softmax(scores))`` restricted to ratio-causal blocks."""
    valid = _ratio_causal_valid(scores.shape[0], scores.shape[1], scores.shape[2], ratio)
    safe_lse = jnp.where(jnp.isfinite(lse), lse, 0.0)
    log_probs = jnp.where(valid, scores - safe_lse[..., None], 0.0)
    kl = jnp.where(valid & (target > 0.0), target * (jnp.log(target + eps) - log_probs), 0.0)
    return jnp.sum(kl)


def dsa_indexer_scores(
    q: Any,
    k: Any,
    weights: Any,
    *,
    ratio: int = 4,
    sm_scale: float = 1.0,
) -> tuple[Any, Any]:
    """Forward-only SM100 DSv4 indexer scores and their row log-sum-exp.

    Returns ``(scores, lse)`` with ``scores`` as in :func:`dsa_indexer` and ``lse`` the ``[batch,
    query_length]`` FP32 log-sum-exp over each row's ratio-causal blocks (``-inf`` for rows without
    one). Inputs are stop-gradient'ed: gradients for the indexer come from
    :func:`dsa_indexer_kl_loss`, which consumes ``scores`` and ``lse`` as residuals.
    """
    q, k, weights = jax.lax.stop_gradient((q, k, weights))
    return _cudnn_indexer_forward(q, k, weights, ratio=ratio, sm_scale=sm_scale, return_lse=True)


def dsa_indexer_scores_batched(
    q: Any,
    k: Any,
    weights: Any,
    *,
    batch_axes: tuple[str, ...] = (),
    ratio: int = 4,
    sm_scale: float = 1.0,
) -> tuple[Any, Any]:
    """Run :func:`dsa_indexer_scores` locally on each batch shard."""
    specs = (
        _batch_axes_spec(batch_axes, q.ndim),
        _batch_axes_spec(batch_axes, k.ndim),
        _batch_axes_spec(batch_axes, weights.ndim),
    )
    return _batched_shard_map(
        lambda local_q, local_k, local_weights: dsa_indexer_scores(
            local_q, local_k, local_weights, ratio=ratio, sm_scale=sm_scale
        ),
        specs,
        (_batch_axes_spec(batch_axes, 3), _batch_axes_spec(batch_axes, 2)),
        batch_axes,
    )(q, k, weights)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def _dsa_indexer_kl_loss(q, k, weights, scores, lse, target, ratio, sm_scale, eps):
    del q, k, weights
    return _indexer_kl_sum(scores, lse, target, ratio, eps)


def _dsa_indexer_kl_loss_fwd(q, k, weights, scores, lse, target, ratio, sm_scale, eps):
    loss = _indexer_kl_sum(scores, lse, target, ratio, eps)
    return loss, (q, k, weights, scores, lse, target)


def _dsa_indexer_kl_loss_bwd(ratio, sm_scale, eps, residual, grad_kl_sum):
    del eps
    q, k, weights, scores, lse, target = residual
    batch, seqlen_q = q.shape[:2]
    # The cuDNN kernel scales every token's KL gradient by ``grad_loss / (B * S_q)``; a unit weight per
    # token (the cotangent of a plain sum) therefore needs ``grad_loss = cotangent * B * S_q``.
    grad_loss = (grad_kl_sum.astype(jnp.float32) * (batch * seqlen_q)).reshape(1)
    grad_q, grad_weights, grad_k = _cudnn_dense_indexer_backward(
        q,
        k,
        weights,
        target.astype(jnp.float32),
        # ``target`` is already normalized, so the kernel's ``attn_score / attn_l1norm`` is ``target``.
        jnp.ones(lse.shape, jnp.float32),
        scores,
        lse,
        grad_loss,
        ratio=ratio,
        sm_scale=sm_scale,
    )
    return grad_q, grad_k.astype(k.dtype).reshape(k.shape), grad_weights, None, None, None


_dsa_indexer_kl_loss.defvjp(_dsa_indexer_kl_loss_fwd, _dsa_indexer_kl_loss_bwd)


def dsa_indexer_kl_loss(
    q: Any,
    k: Any,
    weights: Any,
    scores: Any,
    lse: Any,
    target: Any,
    *,
    ratio: int = 4,
    sm_scale: float = 1.0,
    eps: float = 1e-8,
) -> Any:
    """Summed dense DSv4 indexer KL loss with a fused cuDNN backward.

    Returns ``sum_t KL(target_t || softmax(scores_t))`` over every query token and ratio-causal
    block, with ``KL = sum_w target * (log(target + eps) - log_softmax(scores))``. ``scores`` and
    ``lse`` are the outputs of :func:`dsa_indexer_scores` for the same ``q``, ``k`` and ``weights``;
    ``target`` is a constant ``[batch, query_length, blocks]`` FP32 distribution (zero outside the
    ratio-causal blocks).

    Only ``q``, ``k`` and ``weights`` receive gradients. The VJP calls the cuDNN dense indexer-backward
    kernel, which fuses the KL gradient with the three indexer GEMMs, instead of differentiating
    through the scores. Dense (full-block) loss only; the query position is its index, i.e. no
    packed-sequence position resets.
    """
    return _dsa_indexer_kl_loss(q, k, weights, scores, lse, target, ratio, sm_scale, eps)


def dsa_indexer_kl_loss_batched(
    q: Any,
    k: Any,
    weights: Any,
    scores: Any,
    lse: Any,
    target: Any,
    *,
    batch_axes: tuple[str, ...] = (),
    ratio: int = 4,
    sm_scale: float = 1.0,
    eps: float = 1e-8,
) -> Any:
    """Run :func:`dsa_indexer_kl_loss` locally on each batch shard.

    Returns one partial sum per batch shard (shape ``[1]`` without sharding). The caller adds them and
    applies its own token normalization, which keeps the VJP exact without a collective inside the
    manual region.
    """
    arrays = (q, k, weights, scores, lse, target)
    specs = tuple(_batch_axes_spec(batch_axes, array.ndim) for array in arrays)
    return _batched_shard_map(
        lambda *local: dsa_indexer_kl_loss(*local, ratio=ratio, sm_scale=sm_scale, eps=eps).reshape(
            1
        ),
        specs,
        _batch_axes_spec(batch_axes, 1),
        batch_axes,
    )(*arrays)


def _cudnn_sparse_indexer_backward(
    q, k, weights, attn_score, index_score, topk_indices, grad_loss, *, sm_scale
):
    from cudnn import sparse_indexer_backward_jax_sm100

    return sparse_indexer_backward_jax_sm100(
        q, k, weights, attn_score, index_score, topk_indices, grad_loss, sm_scale=sm_scale
    )


def _sparse_student_log_probs(scores, topk_indices):
    """Log-softmax of the dense ``scores`` over each row's selected, valid (``>= 0``) slots.

    Returns ``(log_probs, valid)`` of shape ``[batch, query_length, selected]``; invalid slots (and every slot
    of a row without a valid one) get ``0``.
    """
    valid = topk_indices >= 0
    logits = jnp.take_along_axis(scores, jnp.maximum(topk_indices, 0), axis=-1)
    any_valid = jnp.any(valid, axis=-1, keepdims=True)
    logits = jnp.where(any_valid & valid, logits, jnp.where(any_valid, -jnp.inf, 0.0))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.where(valid, log_probs, 0.0), valid


def _sparse_indexer_kl_sum(log_probs, valid, target, eps):
    kl = jnp.where(valid & (target > 0.0), target * (jnp.log(target + eps) - log_probs), 0.0)
    return jnp.sum(kl)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _dsa_indexer_sparse_kl_loss(q, k, weights, scores, topk_indices, target, sm_scale, eps):
    del q, k, weights
    log_probs, valid = _sparse_student_log_probs(scores, topk_indices)
    return _sparse_indexer_kl_sum(log_probs, valid, target, eps)


def _dsa_indexer_sparse_kl_loss_fwd(q, k, weights, scores, topk_indices, target, sm_scale, eps):
    log_probs, valid = _sparse_student_log_probs(scores, topk_indices)
    loss = _sparse_indexer_kl_sum(log_probs, valid, target, eps)
    probs = jnp.where(valid, jnp.exp(log_probs), 0.0)
    return loss, (q, k, weights, probs, topk_indices, target)


def _dsa_indexer_sparse_kl_loss_bwd(sm_scale, eps, residual, grad_kl_sum):
    del eps
    q, k, weights, probs, topk_indices, target = residual
    batch, seqlen_q = q.shape[:2]
    # The cuDNN kernel scales every token's KL gradient by ``grad_loss / (B * S_q)``; a unit weight per
    # token (the cotangent of a plain sum) therefore needs ``grad_loss = cotangent * B * S_q``.
    grad_loss = (grad_kl_sum.astype(jnp.float32) * (batch * seqlen_q)).reshape(1)
    grad_q, grad_weights, grad_k = _cudnn_sparse_indexer_backward(
        q,
        k,
        weights,
        target.astype(jnp.float32),
        probs.astype(jnp.float32),
        topk_indices.astype(jnp.int32),
        grad_loss,
        sm_scale=sm_scale,
    )
    return grad_q, grad_k.astype(k.dtype).reshape(k.shape), grad_weights, None, None, None


_dsa_indexer_sparse_kl_loss.defvjp(_dsa_indexer_sparse_kl_loss_fwd, _dsa_indexer_sparse_kl_loss_bwd)


def dsa_indexer_sparse_kl_loss(
    q: Any,
    k: Any,
    weights: Any,
    scores: Any,
    topk_indices: Any,
    target: Any,
    *,
    sm_scale: float = 1.0,
    eps: float = 1e-8,
) -> Any:
    """Summed sparse DSv4 indexer KL loss with a fused cuDNN backward.

    Returns ``sum_t KL(target_t || softmax(scores_t[selected_t]))`` where the softmax runs over each query's
    selected blocks only. ``scores`` is the dense ``[batch, query_length, blocks]`` output of
    :func:`dsa_indexer_scores` (or :func:`dsa_indexer`) for the same ``q``, ``k`` and ``weights``;
    ``topk_indices`` is ``[batch, query_length, selected]`` int32 block ids local to each batch element
    (``-1`` marks an invalid slot; ``selected`` must be 512); ``target`` is the constant
    FP32 teacher over the same slots, L1-normalized over the valid ones.

    Only ``q``, ``k`` and ``weights`` receive gradients, computed by the cuDNN sparse indexer-backward
    kernel. The query position is its index, i.e. no packed-sequence position resets.
    """
    return _dsa_indexer_sparse_kl_loss(q, k, weights, scores, topk_indices, target, sm_scale, eps)


def dsa_indexer_sparse_kl_loss_batched(
    q: Any,
    k: Any,
    weights: Any,
    scores: Any,
    topk_indices: Any,
    target: Any,
    *,
    batch_axes: tuple[str, ...] = (),
    sm_scale: float = 1.0,
    eps: float = 1e-8,
) -> Any:
    """Run :func:`dsa_indexer_sparse_kl_loss` locally on each batch shard.

    Returns one partial sum per batch shard (shape ``[1]`` without sharding); see
    :func:`dsa_indexer_kl_loss_batched` for why the caller normalizes.
    """
    arrays = (q, k, weights, scores, topk_indices, target)
    specs = tuple(_batch_axes_spec(batch_axes, array.ndim) for array in arrays)
    return _batched_shard_map(
        lambda *local: dsa_indexer_sparse_kl_loss(*local, sm_scale=sm_scale, eps=eps).reshape(1),
        specs,
        _batch_axes_spec(batch_axes, 1),
        batch_axes,
    )(*arrays)


def _cudnn_sparse_attention_forward(
    q, kv, topk_indices, topk_length, attn_sink, *, indexer_topk, softmax_scale
):
    from cudnn import sparse_attention_forward_jax_sm100

    return sparse_attention_forward_jax_sm100(
        q,
        kv,
        topk_indices,
        topk_length,
        attn_sink,
        indexer_topk=indexer_topk,
        softmax_scale=softmax_scale,
    )


def _cudnn_sparse_attention_backward(
    q,
    kv,
    out,
    grad_out,
    lse,
    attn_sink,
    topk_indices,
    topk_length,
    *,
    softmax_scale,
):
    from cudnn import sparse_attention_backward_jax_sm100

    return sparse_attention_backward_jax_sm100(
        q,
        kv,
        out,
        grad_out,
        lse,
        attn_sink,
        topk_indices,
        topk_length,
        softmax_scale=softmax_scale,
    )


@partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _dsa_sparse_attention(q, kv, topk_indices, topk_length, attn_sink, indexer_topk, softmax_scale):
    return _cudnn_sparse_attention_forward(
        q,
        kv,
        topk_indices,
        topk_length,
        attn_sink,
        indexer_topk=indexer_topk,
        softmax_scale=softmax_scale,
    )[0]


def _dsa_sparse_attention_fwd(
    q, kv, topk_indices, topk_length, attn_sink, indexer_topk, softmax_scale
):
    out, _, lse, _ = _cudnn_sparse_attention_forward(
        q,
        kv,
        topk_indices,
        topk_length,
        attn_sink,
        indexer_topk=indexer_topk,
        softmax_scale=softmax_scale,
    )
    return out, (q, kv, out, lse, attn_sink, topk_indices, topk_length)


def _dsa_sparse_attention_bwd(indexer_topk, softmax_scale, residual, grad_out):
    del indexer_topk
    q, kv, out, lse, attn_sink, topk_indices, topk_length = residual
    grad_q, grad_kv, grad_sink = _cudnn_sparse_attention_backward(
        q,
        kv,
        out,
        grad_out.astype(jnp.bfloat16),
        lse,
        attn_sink,
        topk_indices,
        topk_length,
        softmax_scale=softmax_scale,
    )
    return grad_q, grad_kv, None, None, grad_sink


_dsa_sparse_attention.defvjp(_dsa_sparse_attention_fwd, _dsa_sparse_attention_bwd)


def dsa_sparse_attention(
    q: Any,
    kv: Any,
    topk_indices: Any,
    topk_length: Any,
    attn_sink: Any,
    *,
    indexer_topk: int = 512,
    softmax_scale: float = 1.0,
) -> Any:
    """Differentiable SM100 DSv4 sparse-attention kernel.

    Gradients are returned for Q, KV, and the per-head attention sink. Sparse
    indices and lengths are non-differentiable metadata. ``indexer_topk=0``
    runs without an indexer prefix (HCA's fixed local + compressed index list).
    """
    return _dsa_sparse_attention(
        q,
        kv,
        topk_indices,
        topk_length,
        attn_sink,
        indexer_topk,
        softmax_scale,
    )


def dsa_sparse_attention_batched(
    q: Any,
    kv: Any,
    topk_indices: Any,
    topk_length: Any,
    attn_sink: Any,
    *,
    batch_axes: tuple[str, ...] = (),
    indexer_topk: int = 512,
    softmax_scale: float = 1.0,
) -> Any:
    """Run sparse attention on local batch shards using local KV indices.

    ``topk_indices`` has shape ``[batch, query_length, selected_kv]`` and
    must be relative to that batch element's KV row.  This deliberate contract
    prevents a global-batch all-gather before the raw CuTe kernel.
    """
    if q.ndim != 4 or kv.ndim != 3 or topk_indices.ndim != 3:
        raise ValueError("expected q=[B,S,H,D], kv=[B,K,D], and indices=[B,S,N]")
    if topk_length.shape != q.shape[:2]:
        raise ValueError("topk_length must have shape [batch, query_length]")
    q_spec = _batch_axes_spec(batch_axes, 4)
    kv_spec = _batch_axes_spec(batch_axes, 3)
    indices_spec = _batch_axes_spec(batch_axes, 3)
    lengths_spec = _batch_axes_spec(batch_axes, 2)

    def body(local_q, local_kv, local_indices, local_lengths, local_sink):
        batch, query_length, heads, head_dim = local_q.shape
        kv_length = local_kv.shape[1]
        # The kernel indexes the flattened [batch * kv_length] KV; shift each
        # row's valid (non-negative) indices into its batch element's KV rows.
        row_offset = (jnp.arange(batch, dtype=jnp.int32) * kv_length)[:, None, None]
        global_indices = jnp.where(local_indices >= 0, local_indices + row_offset, local_indices)
        out = dsa_sparse_attention(
            local_q.reshape(batch * query_length, heads, head_dim),
            local_kv.reshape(batch * kv_length, head_dim),
            global_indices.reshape(batch * query_length, local_indices.shape[-1]),
            local_lengths.reshape(batch * query_length),
            local_sink,
            indexer_topk=indexer_topk,
            softmax_scale=softmax_scale,
        )
        return out.reshape(batch, query_length, heads, head_dim)

    return _batched_shard_map(
        body,
        (q_spec, kv_spec, indices_spec, lengths_spec, P(None)),
        _batch_axes_spec(batch_axes, 4),
        batch_axes,
    )(q, kv, topk_indices, topk_length, attn_sink)


__all__ = [
    "csa_compressor",
    "csa_compressor_batched",
    "dsa_indexer",
    "dsa_indexer_batched",
    "dsa_indexer_kl_loss",
    "dsa_indexer_kl_loss_batched",
    "dsa_indexer_sparse_kl_loss",
    "dsa_indexer_sparse_kl_loss_batched",
    "dsa_indexer_scores",
    "dsa_indexer_scores_batched",
    "dsa_indexer_top_k",
    "dsa_indexer_top_k_batched",
    "dsa_sparse_attention",
    "dsa_sparse_attention_batched",
    "hca_compressor_batched",
]

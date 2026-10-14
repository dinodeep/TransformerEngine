# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Parity checks for TE-owned DeepSeek-V4 CSA per-kernel VJPs."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax.sharding import Mesh

from transformer_engine.jax.deepseek_v4 import (
    csa_compressor,
    csa_compressor_batched,
    dsa_indexer,
    dsa_indexer_batched,
    dsa_indexer_kl_loss,
    dsa_indexer_kl_loss_batched,
    dsa_indexer_scores,
    dsa_indexer_scores_batched,
    dsa_indexer_sparse_kl_loss,
    dsa_indexer_sparse_kl_loss_batched,
    dsa_indexer_top_k,
    dsa_indexer_top_k_batched,
    dsa_sparse_attention,
    dsa_sparse_attention_batched,
)


def _require_sm100():
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("JAX has no CUDA device")
    from cudnn.tensor_adapter import get_compute_capability

    if get_compute_capability()[0] < 10:
        pytest.skip("DSv4 CuTeDSL kernels require compute capability >= 10")


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [128, 512])
def test_csa_compressor_vjp(head_dim):
    _require_sm100()
    key = jax.random.key(head_dim)
    kv = jax.random.normal(key, (16, 2 * head_dim), jnp.bfloat16)
    score = jax.random.normal(jax.random.fold_in(key, 1), kv.shape, jnp.bfloat16)
    ape = jax.random.normal(jax.random.fold_in(key, 2), (4, 2 * head_dim), jnp.float32)
    cu = jnp.array([0, 16], jnp.int32)
    cu_comp = jnp.array([0, 4], jnp.int32)

    def reference(x, scores, bias):
        a_kv, b_kv = jnp.split(x.reshape(4, 4, 2 * head_dim), 2, axis=-1)
        a_score, b_score = jnp.split(scores.reshape(4, 4, 2 * head_dim) + bias, 2, axis=-1)
        shifted_kv = jnp.concatenate([jnp.zeros_like(a_kv[:1]), a_kv[:-1]], axis=0)
        shifted_score = jnp.concatenate(
            [jnp.full_like(a_score[:1], -jnp.inf), a_score[:-1]], axis=0
        )
        values = jnp.concatenate([shifted_kv, b_kv], axis=1)
        logits = jnp.concatenate([shifted_score, b_score], axis=1)
        return jnp.sum(
            values * jax.nn.softmax(logits.astype(jnp.float32), axis=1).astype(values.dtype), axis=1
        )

    got = jax.jit(lambda x, s, a: csa_compressor(x, s, a, cu, cu_comp, total_comp=4))(
        kv, score, ape
    )
    np.testing.assert_allclose(
        np.asarray(got, np.float32),
        np.asarray(reference(kv, score, ape), np.float32),
        atol=8e-3,
        rtol=8e-3,
    )

    grads = jax.grad(
        lambda x, s, a: csa_compressor(x, s, a, cu, cu_comp, total_comp=4)
        .astype(jnp.float32)
        .sum(),
        argnums=(0, 1, 2),
    )(kv, score, ape)
    reference_grads = jax.grad(
        lambda x, s, a: reference(x, s, a).astype(jnp.float32).sum(), argnums=(0, 1, 2)
    )(kv, score, ape)
    for actual, expected in zip(grads, reference_grads, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=8e-3, rtol=8e-3)


@pytest.mark.L0
def test_dsa_indexer_vjp():
    _require_sm100()
    key = jax.random.key(4)
    q = jax.random.normal(key, (1, 16, 64, 128), jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (1, 4, 1, 128), jnp.bfloat16)
    weights = jax.random.normal(jax.random.fold_in(key, 2), (1, 16, 64), jnp.bfloat16)
    scale = 128**-0.5

    def reference(q_, k_, weights_):
        dots = jnp.einsum("bshd,bwkd->bhsw", q_.astype(jnp.float32), k_.astype(jnp.float32))
        scores = jnp.einsum(
            "bhsw,bsh->bsw", jax.nn.relu(dots) * scale, weights_.astype(jnp.float32)
        )
        q_pos = jnp.arange(q_.shape[1])[:, None]
        block_ends = 4 * (jnp.arange(k_.shape[1])[None, :] + 1)
        return jnp.where(block_ends <= q_pos + 1, scores, -jnp.inf)

    got = dsa_indexer(q, k, weights, ratio=4, sm_scale=scale)
    expected = reference(q, k, weights)
    valid = jnp.isfinite(got)
    np.testing.assert_allclose(
        np.asarray(got[valid]), np.asarray(expected[valid]), atol=2e-5, rtol=2e-5
    )

    loss = lambda q_, k_, w_: jnp.where(
        jnp.isfinite(dsa_indexer(q_, k_, w_, ratio=4, sm_scale=scale)),
        dsa_indexer(q_, k_, w_, ratio=4, sm_scale=scale),
        0.0,
    ).sum()
    reference_loss = lambda q_, k_, w_: jnp.where(
        jnp.isfinite(reference(q_, k_, w_)), reference(q_, k_, w_), 0.0
    ).sum()
    grads = jax.grad(loss, argnums=(0, 1, 2))(q, k, weights)
    expected_grads = jax.grad(reference_loss, argnums=(0, 1, 2))(q, k, weights)
    for actual, expected in zip(grads, expected_grads, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5, rtol=2e-5)


@pytest.mark.L0
def test_dsa_sparse_attention_vjp():
    _require_sm100()
    key = jax.random.key(5)
    q = jax.random.normal(key, (1, 64, 512), jnp.bfloat16) / jnp.sqrt(512)
    kv = jax.random.normal(jax.random.fold_in(key, 1), (640, 512), jnp.bfloat16)
    indices = jnp.arange(640, dtype=jnp.int32)[None]
    lengths = jnp.array([640], jnp.int32)
    sinks = jnp.zeros((64,), jnp.float32)
    got = dsa_sparse_attention(q, kv, indices, lengths, sinks)

    def reference(x, y, s):
        logits = jnp.einsum("thd,kd->thk", x.astype(jnp.float32), y.astype(jnp.float32))
        maximum = jnp.maximum(jnp.max(logits, axis=-1), s)
        probabilities = jnp.exp(logits - maximum[..., None])
        return (
            jnp.einsum("thk,kd->thd", probabilities, y.astype(jnp.float32))
            / (jnp.sum(probabilities, axis=-1) + jnp.exp(s - maximum))[..., None]
        )

    np.testing.assert_allclose(
        np.asarray(got, np.float32), np.asarray(reference(q, kv, sinks)), atol=1e-2, rtol=2e-2
    )
    grads = jax.grad(
        lambda x, y, s: dsa_sparse_attention(x, y, indices, lengths, s).astype(jnp.float32).sum(),
        argnums=(0, 1, 2),
    )(q, kv, sinks)
    expected_grads = jax.grad(lambda x, y, s: reference(x, y, s).sum(), argnums=(0, 1, 2))(
        q, kv, sinks
    )
    for actual, expected in zip(grads, expected_grads, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=4e-2, rtol=2e-2)


@pytest.mark.L0
def test_batched_csa_kernels_compound_fsdp_ep_batch_sharding():
    """FSDP and EP together form the local CSA batch dimension."""
    _require_sm100()
    if len(jax.devices()) < 4:
        pytest.skip("requires four local CUDA devices")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "expert"))
    batch_axes = ("fsdp", "expert")
    key = jax.random.key(7)
    kv = jax.random.normal(key, (4, 16, 256), jnp.bfloat16)
    score = jax.random.normal(jax.random.fold_in(key, 1), kv.shape, jnp.bfloat16)
    ape = jax.random.normal(jax.random.fold_in(key, 2), (4, 256), jnp.float32)
    cu = jnp.arange(5, dtype=jnp.int32) * 16
    cu_comp = jnp.arange(5, dtype=jnp.int32) * 4

    with jax.set_mesh(mesh):
        got_compressor = jax.jit(
            lambda x, s, a: csa_compressor_batched(x, s, a, batch_axes=batch_axes)
        )(kv, score, ape)
        got_indexer = jax.jit(
            lambda q, k, w: dsa_indexer_batched(q, k, w, batch_axes=batch_axes, ratio=4)
        )(
            jax.random.normal(jax.random.fold_in(key, 3), (4, 16, 64, 128), jnp.bfloat16),
            jax.random.normal(jax.random.fold_in(key, 4), (4, 4, 1, 128), jnp.bfloat16),
            jax.random.normal(jax.random.fold_in(key, 5), (4, 16, 64), jnp.bfloat16),
        )

    expected_compressor = csa_compressor(
        kv.reshape(64, 256), score.reshape(64, 256), ape, cu, cu_comp, total_comp=16
    ).reshape(4, 4, 128)
    np.testing.assert_allclose(
        np.asarray(got_compressor, np.float32),
        np.asarray(expected_compressor, np.float32),
        atol=8e-3,
        rtol=8e-3,
    )
    assert got_compressor.sharding.spec[0] == ("fsdp", "expert")
    assert got_indexer.sharding.spec[0] == ("fsdp", "expert")


def _kl_loss_inputs(batch, seqlen, ratio=4):
    key = jax.random.key(11)
    blocks = seqlen // ratio
    q = jax.random.normal(key, (batch, seqlen, 64, 128), jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (batch, blocks, 1, 128), jnp.bfloat16)
    weights = jax.random.normal(jax.random.fold_in(key, 2), (batch, seqlen, 64), jnp.bfloat16)
    valid = (jnp.arange(blocks)[None, :] + 1) * ratio <= jnp.arange(seqlen)[:, None] + 1
    raw = jax.random.uniform(jax.random.fold_in(key, 3), (batch, seqlen, blocks)) ** 3 + 1e-3
    raw = jnp.where(valid[None], raw, 0.0)
    target = raw / jnp.maximum(raw.sum(-1, keepdims=True), 1e-30)
    return q, k, weights, target, valid


def _kl_loss_reference(q, k, weights, target, valid, scale, eps=1e-8):
    dots = jnp.einsum("bshd,bwkd->bhsw", q.astype(jnp.float32), k.astype(jnp.float32))
    scores = jnp.einsum("bhsw,bsh->bsw", jax.nn.relu(dots) * scale, weights.astype(jnp.float32))
    valid = jnp.broadcast_to(valid[None], scores.shape)
    masked = jnp.where(valid, scores, -jnp.inf)
    log_probs = jax.nn.log_softmax(jnp.where(valid.any(-1, keepdims=True), masked, 0.0), axis=-1)
    kl = jnp.where(valid & (target > 0), target * (jnp.log(target + eps) - log_probs), 0.0)
    return scores, kl.sum()


def _assert_close_gradient(actual, expected, rel_l2):
    actual = np.asarray(actual, np.float32).ravel()
    expected = np.asarray(expected, np.float32).ravel()
    assert np.isfinite(actual).all()
    error = np.linalg.norm(actual - expected) / (np.linalg.norm(expected) + 1e-30)
    cosine = actual @ expected / (np.linalg.norm(actual) * np.linalg.norm(expected) + 1e-30)
    assert error < rel_l2, f"relative L2 error {error}"
    assert cosine > 1 - 1e-4, f"cosine {cosine}"


@pytest.mark.L0
@pytest.mark.parametrize("seqlen", [64, 512])
def test_dsa_indexer_kl_loss_vjp(seqlen):
    """Loss value and fused-backward gradients match differentiating the exact KL in FP32."""
    _require_sm100()
    q, k, weights, target, valid = _kl_loss_inputs(2, seqlen)
    scale = 128**-0.5

    scores, lse = jax.jit(lambda *a: dsa_indexer_scores(*a, ratio=4, sm_scale=scale))(q, k, weights)
    expected_scores, expected_loss = _kl_loss_reference(q, k, weights, target, valid, scale)
    mask = np.asarray(jnp.broadcast_to(valid[None], scores.shape))
    np.testing.assert_allclose(
        np.asarray(scores)[mask], np.asarray(expected_scores)[mask], atol=2e-5, rtol=2e-5
    )

    def loss_fn(q_, k_, w_):
        return dsa_indexer_kl_loss(q_, k_, w_, scores, lse, target, ratio=4, sm_scale=scale)

    np.testing.assert_allclose(
        float(jax.jit(loss_fn)(q, k, weights)), float(expected_loss), rtol=1e-4
    )

    # A non-unit cotangent exercises the grad_loss scaling.
    weight = 0.37
    grads = jax.jit(jax.grad(lambda *a: weight * loss_fn(*a), argnums=(0, 1, 2)))(q, k, weights)
    expected = jax.grad(
        lambda *a: weight * _kl_loss_reference(*a, target, valid, scale)[1], argnums=(0, 1, 2)
    )(q, k, weights)
    for actual, ref, tolerance in zip(grads, expected, (6e-3, 6e-3, 2e-3), strict=True):
        _assert_close_gradient(actual, ref, tolerance)


@pytest.mark.L0
def test_dsa_indexer_kl_loss_batched_matches_unsharded():
    """FSDP x EP batch sharding gives the same loss and gradients as one local call."""
    _require_sm100()
    if len(jax.devices()) < 4:
        pytest.skip("requires four local CUDA devices")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "expert"))
    q, k, weights, target, _ = _kl_loss_inputs(4, 128)
    scale = 128**-0.5
    batch_axes = ("fsdp", "expert")

    def sharded_loss(q_, k_, w_):
        scores, lse = dsa_indexer_scores_batched(
            q_, k_, w_, batch_axes=batch_axes, ratio=4, sm_scale=scale
        )
        shard_sums = dsa_indexer_kl_loss_batched(
            q_, k_, w_, scores, lse, target, batch_axes=batch_axes, ratio=4, sm_scale=scale
        )
        return shard_sums.sum() / 7.0, scores

    with jax.set_mesh(mesh):
        (loss, scores), grads = jax.jit(
            jax.value_and_grad(sharded_loss, argnums=(0, 1, 2), has_aux=True)
        )(q, k, weights)
    assert scores.sharding.spec[0] == ("fsdp", "expert")

    expected_scores, expected_lse = dsa_indexer_scores(q, k, weights, ratio=4, sm_scale=scale)
    expected_loss_fn = (
        lambda *a: dsa_indexer_kl_loss(
            *a, expected_scores, expected_lse, target, ratio=4, sm_scale=scale
        )
        / 7.0
    )
    np.testing.assert_allclose(float(loss), float(expected_loss_fn(q, k, weights)), rtol=1e-5)
    finite = np.isfinite(np.asarray(expected_scores))
    np.testing.assert_allclose(
        np.asarray(scores)[finite], np.asarray(expected_scores)[finite], atol=1e-6
    )
    for actual, ref in zip(
        grads, jax.grad(expected_loss_fn, argnums=(0, 1, 2))(q, k, weights), strict=True
    ):
        _assert_close_gradient(actual, ref, 1e-3)


def _top_k_reference(scores, top_k, ratio):
    """``jax.lax.top_k`` over ratio-causally masked scores, ``-1`` for invalid slots."""
    batch, seqlen_q, num_blocks = scores.shape
    valid = (jnp.arange(num_blocks)[None, :] + 1) * ratio <= jnp.arange(seqlen_q)[:, None] + 1
    valid = jnp.broadcast_to(valid, scores.shape)
    ids = jax.lax.top_k(jnp.where(valid, scores, -jnp.inf), top_k)[1]
    return jnp.where(jnp.take_along_axis(valid, ids, axis=-1), ids, -1), valid


def _assert_same_selection(actual, expected, scores):
    """Same set of selected blocks per row (order is unspecified); ties are resolved to lower ids."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    for b in range(actual.shape[0]):
        for t in range(actual.shape[1]):
            assert set(actual[b, t][actual[b, t] >= 0].tolist()) == set(
                expected[b, t][expected[b, t] >= 0].tolist()
            ), (b, t)
            assert (actual[b, t] >= 0).sum() == (expected[b, t] >= 0).sum()


@pytest.mark.L0
@pytest.mark.parametrize("quantize", [False, True], ids=["random", "ties"])
@pytest.mark.parametrize("num_blocks,top_k", [(1024, 512), (256, 256)])
def test_dsa_indexer_top_k_matches_lax_top_k(num_blocks, top_k, quantize):
    """Selected sets equal ``lax.top_k``'s, incl. rows with fewer than ``top_k`` valid blocks and ties."""
    _require_sm100()
    seqlen = num_blocks * 4
    scores = jax.random.normal(jax.random.PRNGKey(0), (2, seqlen, num_blocks), jnp.float32)
    if quantize:
        scores = jnp.round(scores * 2) / 2
    actual = jax.jit(lambda x: dsa_indexer_top_k(x, top_k=top_k, ratio=4))(scores)
    expected, _ = _top_k_reference(scores, top_k, 4)
    assert actual.dtype == jnp.int32
    _assert_same_selection(actual, expected, scores)


@pytest.mark.L0
def test_dsa_indexer_top_k_batched_matches_unsharded():
    """FSDP x EP batch sharding selects the same blocks as one local call and keeps the batch sharding."""
    _require_sm100()
    if len(jax.devices()) < 4:
        pytest.skip("requires four local CUDA devices")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "expert"))
    scores = jax.random.normal(jax.random.PRNGKey(1), (4, 4096, 1024), jnp.float32)
    batch_axes = ("fsdp", "expert")
    with jax.set_mesh(mesh):
        actual = jax.jit(
            lambda x: dsa_indexer_top_k_batched(x, batch_axes=batch_axes, top_k=512, ratio=4)
        )(scores)
    assert actual.sharding.spec[0] == ("fsdp", "expert")
    expected, _ = _top_k_reference(scores, 512, 4)
    _assert_same_selection(actual, expected, scores)


def _sparse_kl_inputs(batch, seqlen, selected, ratio=4):
    """Indexer inputs plus a top-k selection (``-1`` for invalid slots) and a teacher over the selected slots."""
    q, k, weights, _, valid = _kl_loss_inputs(batch, seqlen, ratio)
    blocks = k.shape[1]
    scale = 128**-0.5
    dots = jnp.einsum("bshd,bwkd->bhsw", q.astype(jnp.float32), k.astype(jnp.float32))
    scores = jnp.einsum("bhsw,bsh->bsw", jax.nn.relu(dots) * scale, weights.astype(jnp.float32))
    valid_b = jnp.broadcast_to(valid[None], scores.shape)
    ids = jax.lax.top_k(jnp.where(valid_b, scores, -jnp.inf), min(selected, blocks))[1]
    indices = jnp.where(jnp.take_along_axis(valid_b, ids, axis=-1), ids, -1)
    indices = jnp.pad(
        indices, ((0, 0), (0, 0), (0, selected - indices.shape[-1])), constant_values=-1
    )
    raw = jax.random.uniform(jax.random.key(5), indices.shape) ** 3 + 1e-3
    raw = jnp.where(indices >= 0, raw, 0.0)
    target = raw / jnp.maximum(raw.sum(-1, keepdims=True), 1e-30)
    return q, k, weights, indices.astype(jnp.int32), target, scale


def _sparse_kl_reference(q, k, weights, indices, target, scale, eps=1e-8):
    """Exact FP32 KL(target || softmax over the selected valid slots), summed over tokens."""
    dots = jnp.einsum("bshd,bwkd->bhsw", q.astype(jnp.float32), k.astype(jnp.float32))
    scores = jnp.einsum("bhsw,bsh->bsw", jax.nn.relu(dots) * scale, weights.astype(jnp.float32))
    valid = indices >= 0
    logits = jnp.take_along_axis(scores, jnp.maximum(indices, 0), axis=-1)
    any_valid = valid.any(-1, keepdims=True)
    logp = jax.nn.log_softmax(
        jnp.where(any_valid, jnp.where(valid, logits, -jnp.inf), 0.0), axis=-1
    )
    kl = jnp.where(valid & (target > 0), target * (jnp.log(target + eps) - logp), 0.0)
    return scores, kl.sum()


@pytest.mark.L0
@pytest.mark.parametrize("seqlen,selected", [(4096, 512)])
def test_dsa_indexer_sparse_kl_loss_vjp(seqlen, selected):
    """Loss value and fused sparse-backward gradients match differentiating the exact KL in FP32.

    Early rows have fewer than ``selected`` valid slots (and the first rows none), so the invalid-slot and
    empty-row handling is exercised too.
    """
    _require_sm100()
    q, k, weights, indices, target, scale = _sparse_kl_inputs(2, seqlen, selected)
    assert bool((indices < 0).any()) and bool((~(indices >= 0).any(-1)).any())
    scores, _ = jax.jit(lambda *a: dsa_indexer_scores(*a, ratio=4, sm_scale=scale))(q, k, weights)

    def loss_fn(q_, k_, w_):
        return dsa_indexer_sparse_kl_loss(q_, k_, w_, scores, indices, target, sm_scale=scale)

    expected_loss_fn = lambda *a: _sparse_kl_reference(*a, indices, target, scale)[1]
    np.testing.assert_allclose(
        float(jax.jit(loss_fn)(q, k, weights)), float(expected_loss_fn(q, k, weights)), rtol=1e-4
    )
    # A non-unit cotangent exercises the grad_loss scaling.
    weight = 0.37
    grads = jax.jit(jax.grad(lambda *a: weight * loss_fn(*a), argnums=(0, 1, 2)))(q, k, weights)
    expected = jax.grad(lambda *a: weight * expected_loss_fn(*a), argnums=(0, 1, 2))(q, k, weights)
    for actual, ref, tolerance in zip(grads, expected, (6e-3, 6e-3, 2e-3), strict=True):
        _assert_close_gradient(actual, ref, tolerance)


@pytest.mark.L0
def test_dsa_indexer_sparse_kl_loss_batched_matches_unsharded():
    """FSDP x EP batch sharding gives the same sparse loss and gradients as one local call."""
    _require_sm100()
    if len(jax.devices()) < 4:
        pytest.skip("requires four local CUDA devices")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "expert"))
    q, k, weights, indices, target, scale = _sparse_kl_inputs(4, 4096, 512)
    batch_axes = ("fsdp", "expert")

    def sharded_loss(q_, k_, w_):
        scores, _ = dsa_indexer_scores_batched(
            q_, k_, w_, batch_axes=batch_axes, ratio=4, sm_scale=scale
        )
        shard_sums = dsa_indexer_sparse_kl_loss_batched(
            q_, k_, w_, scores, indices, target, batch_axes=batch_axes, sm_scale=scale
        )
        return shard_sums.sum() / 7.0

    with jax.set_mesh(mesh):
        loss, grads = jax.jit(jax.value_and_grad(sharded_loss, argnums=(0, 1, 2)))(q, k, weights)

    scores, _ = dsa_indexer_scores(q, k, weights, ratio=4, sm_scale=scale)
    expected_loss_fn = (
        lambda *a: dsa_indexer_sparse_kl_loss(*a[:3], scores, indices, target, sm_scale=scale) / 7.0
    )
    np.testing.assert_allclose(float(loss), float(expected_loss_fn(q, k, weights)), rtol=1e-5)
    for actual, ref in zip(
        grads, jax.grad(expected_loss_fn, argnums=(0, 1, 2))(q, k, weights), strict=True
    ):
        _assert_close_gradient(actual, ref, 1e-3)

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the mHC (manifold Hyper-Connection) JAX API against MaxText's mHC layer."""

import sys

import jax
import jax.numpy as jnp
import pytest

from utils import assert_allclose, pytest_parametrize_wrapper


@pytest.fixture(autouse=True, scope="function")
def _inject_mhc(request):
    """Lazy-load the mHC API only for tests marked 'triton'."""
    if not request.node.get_closest_marker("triton"):
        yield
        return
    from transformer_engine.jax.mhc import MHCWeights, mhc

    mod = sys.modules[__name__]
    mod.MHCWeights = MHCWeights
    mod.mhc = mhc
    yield


K_STREAMS = 4
NORM_EPSILON = 1e-5
PRE_MAPPING_EPSILON = 1e-6

# (batch, seq, dim)
ALL_SHAPES = [
    (8, 32, 32),
    (8, 128, 1024),
    (2, 2048, 3072),
]
SHAPES = {
    "L0": ALL_SHAPES[0:2],
    "L2": ALL_SHAPES,
}
DTYPES = [jnp.float32, jnp.bfloat16]
SINKHORN_ITERATIONS = [1, 20]

# The reference follows MaxText and computes the mappings in `dtype`, while the kernels use fp32
# and a log-space Sinkhorn without MaxText's 1e-6 stabilizers.
FWD_TOLS = {
    jnp.float32: {"rtol": 1e-3, "atol": 1e-3},
    jnp.bfloat16: {"rtol": 5e-2, "atol": 5e-2},
}
GRAD_TOLS = {
    jnp.float32: {"rtol": 5e-3, "atol": 5e-3},
    jnp.bfloat16: {"rtol": 1e-1, "atol": 1e-1},
}


def _maxtext_rms_norm(x, scale, epsilon, dtype):
    """maxtext.layers.normalizations.RMSNorm.__call__ (scale_offset=0)."""
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * jax.lax.rsqrt(mean2 + epsilon), dtype)
    return jnp.einsum("...k,k->...k", y, jnp.asarray(scale, dtype))


def _maxtext_sinkhorn(t, iters):
    """maxtext.layers.mhc.sinkhorn."""
    initial_dtype = t.dtype
    t = t.astype(jnp.float32)
    eps = 1e-6
    t = jax.nn.softmax(t, axis=-1) + eps
    t = t / (jnp.sum(t, axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        t = t / (jnp.sum(t, axis=-1, keepdims=True) + eps)
        t = t / (jnp.sum(t, axis=-2, keepdims=True) + eps)
    return t.astype(initial_dtype)


def _maxtext_mapping(h, alpha_scale, beta, scale, dtype, eps=0.0):
    """ManifoldConstrainedHyperConnections.mapping."""
    beta = jnp.asarray(beta, dtype)
    alpha_scale = jnp.asarray(alpha_scale, dtype)
    intermediate = alpha_scale * h + beta[None, None, :]
    return scale * jax.nn.sigmoid(intermediate) + eps


def _maxtext_mhc(x, weights, norm_fn, branch_fn, sinkhorn_iterations, dtype):
    """ManifoldConstrainedHyperConnections.__call__ without mHC-lite or the Pallas kernel."""
    precision = jax.lax.Precision.HIGHEST
    b, s, k, d = x.shape
    w = jax.tree.map(lambda p: jnp.asarray(p, dtype), weights)

    norm_x = _maxtext_rms_norm(jnp.reshape(x, (b, s, k * d)), w.norm_scale, NORM_EPSILON, dtype)
    alpha_concat = jnp.concatenate([w.pre_alpha, w.post_alpha, w.res_alpha], axis=-1)
    h_concat = jnp.einsum("bsm,mn -> bsn", norm_x, alpha_concat, precision=precision)
    h_pre = h_concat[..., :k]
    h_post = h_concat[..., k : 2 * k]
    h_res = h_concat[..., 2 * k :]

    pre_mapping = _maxtext_mapping(
        h_pre, w.pre_scale, w.pre_bias, 1.0, dtype, eps=PRE_MAPPING_EPSILON
    )
    layer_input = jnp.einsum("bsk,bskd->bsd", pre_mapping, x, precision=precision)
    layer_out = branch_fn(norm_fn(layer_input))

    post_mapping = _maxtext_mapping(h_post, w.post_scale, w.post_bias, 2.0, dtype)
    post_out = jnp.expand_dims(layer_out, axis=2) * jnp.expand_dims(post_mapping, axis=3)

    h_res = jnp.reshape(h_res, (b, s, k, k))
    res_mapping = _maxtext_sinkhorn(
        w.res_scale * h_res + w.res_bias[None, None, :, :], sinkhorn_iterations
    )
    res_out = jnp.einsum("bskm,bskd->bsmd", res_mapping, x, precision=precision)
    return res_out + post_out


def _make_inputs(batch, seq, dim, dtype, key):
    """Random layer input, MaxText-layout weights, branch weight and output cotangent."""
    k = K_STREAMS
    keys = jax.random.split(key, 14)
    kd = k * dim

    def normal(i, shape, scale=1.0):
        return scale * jax.random.normal(keys[i], shape, jnp.float32)

    weights = MHCWeights(
        norm_scale=1.0 + normal(0, (kd,), 0.1),
        pre_alpha=normal(1, (kd, k), kd**-0.5),
        pre_bias=normal(2, (k,), 0.5),
        pre_scale=1.0 + normal(3, (1,), 0.1),
        post_alpha=normal(4, (kd, k), kd**-0.5),
        post_bias=normal(5, (k,), 0.5),
        post_scale=1.0 + normal(6, (1,), 0.1),
        res_alpha=normal(7, (kd, k * k), kd**-0.5),
        res_bias=normal(8, (k, k), 0.5),
        res_scale=1.0 + normal(9, (1,), 0.1),
    )
    x = normal(10, (batch, seq, k, dim)).astype(dtype)
    branch_weight = normal(11, (dim, dim), dim**-0.5)
    cotangent = normal(12, (batch, seq, k, dim))
    return x, weights, branch_weight, cotangent


def _norm_fn(y):
    y32 = y.astype(jnp.float32)
    return (y32 * jax.lax.rsqrt(jnp.mean(y32 * y32, axis=-1, keepdims=True) + 1e-6)).astype(y.dtype)


def _branch_fn(branch_weight, dtype):
    def branch(y):
        return jnp.tanh(
            jnp.einsum(
                "bsd,de->bse",
                y,
                branch_weight.astype(dtype),
                precision=jax.lax.Precision.HIGHEST,
            )
        )

    return branch


@pytest.mark.triton
class TestMHC:
    """Compare `transformer_engine.jax.mhc.mhc` with MaxText's mHC layer."""

    @staticmethod
    def _te_mhc(x, weights, branch_weight, dtype, sinkhorn_iterations):
        w = jax.tree.map(lambda p: jnp.asarray(p, dtype), weights)
        return mhc(
            x,
            w,
            _norm_fn,
            _branch_fn(branch_weight, dtype),
            norm_epsilon=NORM_EPSILON,
            pre_mapping_epsilon=PRE_MAPPING_EPSILON,
            sinkhorn_iterations=sinkhorn_iterations,
            use_tf32=False,
        )

    @staticmethod
    def _ref_mhc(x, weights, branch_weight, dtype, sinkhorn_iterations):
        return _maxtext_mhc(
            x, weights, _norm_fn, _branch_fn(branch_weight, dtype), sinkhorn_iterations, dtype
        )

    @pytest_parametrize_wrapper("batch,seq,dim", SHAPES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("sinkhorn_iterations", SINKHORN_ITERATIONS)
    def test_forward(self, batch, seq, dim, dtype, sinkhorn_iterations):
        """mhc output matches MaxText's layer output."""
        x, weights, branch_weight, _ = _make_inputs(batch, seq, dim, dtype, jax.random.PRNGKey(0))

        out = jax.jit(self._te_mhc, static_argnums=(3, 4))(
            x, weights, branch_weight, dtype, sinkhorn_iterations
        )
        ref = jax.jit(self._ref_mhc, static_argnums=(3, 4))(
            x, weights, branch_weight, dtype, sinkhorn_iterations
        )

        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert_allclose(out, ref, **FWD_TOLS[dtype])

    @pytest_parametrize_wrapper("batch,seq,dim", SHAPES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("sinkhorn_iterations", SINKHORN_ITERATIONS)
    def test_backward(self, batch, seq, dim, dtype, sinkhorn_iterations):
        """Gradients w.r.t. x, every mHC weight and the branch weight match MaxText's."""
        x, weights, branch_weight, cotangent = _make_inputs(
            batch, seq, dim, dtype, jax.random.PRNGKey(1)
        )

        def make_loss(fn):
            def loss(x, weights, branch_weight):
                out = fn(x, weights, branch_weight, dtype, sinkhorn_iterations)
                return jnp.sum(out.astype(jnp.float32) * cotangent)

            return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))

        loss, (dx, dweights, dbranch) = make_loss(self._te_mhc)(x, weights, branch_weight)
        ref_loss, (ref_dx, ref_dweights, ref_dbranch) = make_loss(self._ref_mhc)(
            x, weights, branch_weight
        )

        tols = GRAD_TOLS[dtype]
        assert_allclose(loss, ref_loss, dtype=jnp.float32, **tols)
        assert dx.dtype == x.dtype
        assert_allclose(dx, ref_dx, **tols)
        assert_allclose(dbranch, ref_dbranch, **tols)
        for name, grad, ref_grad in zip(weights._fields, dweights, ref_dweights):
            assert grad.shape == ref_grad.shape, name
            assert_allclose(grad, ref_grad, err_msg=name, **tols)

    @pytest_parametrize_wrapper("batch,seq,dim", ALL_SHAPES[:1])
    def test_streams_last(self, batch, seq, dim):
        """streams_last=True on the transposed input yields the transposed output."""
        dtype = jnp.float32
        x, weights, branch_weight, _ = _make_inputs(batch, seq, dim, dtype, jax.random.PRNGKey(2))
        branch = _branch_fn(branch_weight, dtype)

        out = jax.jit(lambda x, w: mhc(x, w, _norm_fn, branch, use_tf32=False))(x, weights)
        out_streams_last = jax.jit(
            lambda x, w: mhc(x, w, _norm_fn, branch, streams_last=True, use_tf32=False)
        )(jnp.swapaxes(x, -1, -2), weights)

        assert out_streams_last.shape == (batch, seq, dim, K_STREAMS)
        assert_allclose(jnp.swapaxes(out_streams_last, -1, -2), out, rtol=1e-5, atol=1e-5)

    @pytest_parametrize_wrapper("batch,seq,dim", ALL_SHAPES[:1])
    def test_has_aux(self, batch, seq, dim):
        """has_aux=True returns the branch's auxiliary output unchanged."""
        dtype = jnp.float32
        x, weights, branch_weight, _ = _make_inputs(batch, seq, dim, dtype, jax.random.PRNGKey(3))
        branch = _branch_fn(branch_weight, dtype)

        def branch_with_aux(y):
            out = branch(y)
            return out, {"mean": jnp.mean(out)}

        out, aux = jax.jit(
            lambda x, w: mhc(x, w, _norm_fn, branch_with_aux, has_aux=True, use_tf32=False)
        )(x, weights)
        ref = jax.jit(lambda x, w: mhc(x, w, _norm_fn, branch, use_tf32=False))(x, weights)

        assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
        assert set(aux) == {"mean"}

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""mHC (manifold Hyper-Connection) API for JAX.

This module ties the forward and backward Triton kernels of
``transformer_engine.jax.triton_extensions.mhc`` together with ``jax.custom_vjp``.

Per-stage APIs operate on the TE layout, where the hyper-connection input ``x`` has shape
(..., C, n), i.e. the n streams are the innermost dimension:

    layer_input, H_post, H_res = mhc_generate_mix_and_aggregate(x, phi, alpha, beta)
    layer_output = layer(layer_input)  # Attention / FFN
    x = mhc_expand_combine(layer_output, None, H_post, x, H_res)

``mhc`` (and its halves ``mhc_pre`` / ``mhc_post``) instead takes the parameters of MaxText's
``ManifoldConstrainedHyperConnections`` layer and its (..., k, d) stream layout, and computes
the same result.
"""

from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from transformer_engine.jax.triton_extensions.mhc import (
    DEFAULT_NORM_EPS,
    mhc_projection_fwd,
    mhc_projection_bwd,
    mhc_scale_fwd,
    mhc_scale_bwd,
    mhc_sinkhorn_fwd,
    mhc_sinkhorn_bwd,
    mhc_aggregate_fwd,
    mhc_aggregate_bwd,
    mhc_expand_combine_fwd,
    mhc_expand_combine_bwd,
)

__all__ = [
    "mhc_projection",
    "mhc_scale",
    "mhc_sinkhorn",
    "mhc_aggregate",
    "mhc_expand_combine",
    "mhc_generate_mix_and_aggregate",
    "MHCWeights",
    "mhc_weights_to_params",
    "mhc_pre",
    "mhc_post",
    "mhc",
]


def mhc_projection(
    x: jnp.ndarray,
    phi: jnp.ndarray,
    norm_weight: Optional[jnp.ndarray] = None,
    use_tf32: bool = True,
    use_split_k: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fused projection: H = x @ (phi * norm_weight)^T and ms = mean(x^2, dim=-1).

    Parameters
    ----------
    x : jnp.ndarray
        Input of shape (..., K), where K = n * C is the flattened (C, n) hyper-connection input.
    phi : jnp.ndarray
        Projection matrix of shape (N, K), where N = 2n + n*n (= 24 for n = 4).
    norm_weight : Optional[jnp.ndarray]
        RMSNorm weight of shape (K,), absorbed into phi.
    use_tf32 : bool
        Whether to use TF32 for the matmuls.
    use_split_k : bool
        Whether to use split-K (projection) and split-M (phi gradient) reductions with atomic
        adds. Faster for large K but non-deterministic.

    Returns
    -------
    H : jnp.ndarray
        fp32 array of shape (..., 32), where only the first N columns are valid.
    ms : jnp.ndarray
        fp32 mean square of shape (...,).
    """
    return _mhc_projection(x, phi, norm_weight, use_tf32, use_split_k)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _mhc_projection(x, phi, norm_weight, use_tf32, use_split_k):
    """Internal mhc_projection with custom VJP."""
    outputs, _ = _mhc_projection_fwd_rule(x, phi, norm_weight, use_tf32, use_split_k)
    return outputs


def _mhc_projection_fwd_rule(x, phi, norm_weight, use_tf32, use_split_k):
    """Forward pass rule for mhc_projection."""
    h, ms = mhc_projection_fwd(x, phi, norm_weight, use_tf32, use_split_k)
    return (h, ms), (x, phi, norm_weight)


def _mhc_projection_bwd_rule(use_tf32, use_split_k, residuals, g):
    """Backward pass rule for mhc_projection."""
    x, phi, norm_weight = residuals
    grad_h, grad_ms = g
    return mhc_projection_bwd(grad_h, grad_ms, x, phi, norm_weight, use_tf32, use_split_k)


_mhc_projection.defvjp(_mhc_projection_fwd_rule, _mhc_projection_bwd_rule)


def mhc_scale(
    h: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    ms: jnp.ndarray,
    n: int = 4,
    eps: float = DEFAULT_NORM_EPS,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fused scale producing the mixing coefficients:

    H_pre  = sigmoid(H[..., 0:n] * alpha[0] / sqrt(ms + eps) + beta[0:n])
    H_post = 2 * sigmoid(H[..., n:2n] * alpha[1] / sqrt(ms + eps) + beta[n:2n])
    H_res  = H[..., 2n:2n+n*n] * alpha[2] / sqrt(ms + eps) + beta[2n:2n+n*n]

    Parameters
    ----------
    h, ms : jnp.ndarray
        Outputs of `mhc_projection`, of shapes (..., 32) and (...,).
    alpha : jnp.ndarray
        Scaling factors of shape (3,).
    beta : jnp.ndarray
        Bias with 2n + n*n elements, e.g. of shape (1, 2n + n*n).
    n : int
        Number of hyper connections (only n=4 is supported).
    eps : float
        RMSNorm epsilon. Defaults to the fp32 machine epsilon used by the PyTorch API.

    Returns
    -------
    H_pre (..., n), H_post (..., n) and H_res (..., n*n), in h's dtype.
    """
    out = _mhc_scale(h, alpha, beta, ms, n, eps)
    return out[..., :n], out[..., n : 2 * n], out[..., 2 * n : 2 * n + n * n]


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _mhc_scale(h, alpha, beta, ms, n, eps):
    """Internal mhc_scale with custom VJP, returning the padded (..., 32) output."""
    out, _ = _mhc_scale_fwd_rule(h, alpha, beta, ms, n, eps)
    return out


def _mhc_scale_fwd_rule(h, alpha, beta, ms, n, eps):
    """Forward pass rule for mhc_scale."""
    out = mhc_scale_fwd(h, alpha, beta, ms, n, eps)
    return out, (h, alpha, beta, ms, out)


def _mhc_scale_bwd_rule(n, eps, residuals, grad_out):
    """Backward pass rule for mhc_scale."""
    h, alpha, beta, ms, out = residuals
    grad_h, grad_alpha, grad_beta, grad_ms = mhc_scale_bwd(grad_out, h, alpha, ms, out, n, eps)
    return (
        grad_h.astype(h.dtype),
        grad_alpha,
        grad_beta.reshape(beta.shape).astype(beta.dtype),
        grad_ms.astype(ms.dtype),
    )


_mhc_scale.defvjp(_mhc_scale_fwd_rule, _mhc_scale_bwd_rule)


def mhc_sinkhorn(
    h_res: jnp.ndarray,
    n: int = 4,
    recompute_hist: bool = True,
    iters: int = 20,
) -> jnp.ndarray:
    """
    Log-space Sinkhorn normalization of H_res into a doubly stochastic matrix.

    Parameters
    ----------
    h_res : jnp.ndarray
        Input of shape (..., n, n).
    n : int
        Number of hyper connections (only n=4 is supported).
    recompute_hist : bool
        Whether to recompute the f/g history in the backward pass instead of saving
        2 * (iters + 1) * n floats per token.
    iters : int
        Number of Sinkhorn iterations.

    Returns
    -------
    Doubly stochastic matrix of shape (..., n, n) in h_res's dtype.
    """
    return _mhc_sinkhorn(h_res, n, recompute_hist, iters)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _mhc_sinkhorn(h_res, n, recompute_hist, iters):
    """Internal mhc_sinkhorn with custom VJP."""
    out, _ = _mhc_sinkhorn_fwd_rule(h_res, n, recompute_hist, iters)
    return out


def _mhc_sinkhorn_fwd_rule(h_res, n, recompute_hist, iters):
    """Forward pass rule for mhc_sinkhorn."""
    out, hist = mhc_sinkhorn_fwd(h_res, n, recompute_hist, iters)
    return out, (h_res, out, hist)


def _mhc_sinkhorn_bwd_rule(n, recompute_hist, iters, residuals, grad_out):
    """Backward pass rule for mhc_sinkhorn."""
    del recompute_hist
    h_res, out, hist = residuals
    return (mhc_sinkhorn_bwd(grad_out, h_res, out, hist, n, iters),)


_mhc_sinkhorn.defvjp(_mhc_sinkhorn_fwd_rule, _mhc_sinkhorn_bwd_rule)


def mhc_aggregate(
    x: jnp.ndarray,
    h_pre: jnp.ndarray,
    use_tf32: bool = True,
) -> jnp.ndarray:
    """
    Aggregate the n streams: out = x @ H_pre, (..., C, n) @ (..., n, 1) -> (..., C).

    Returns an array of shape (..., C) in x's dtype.
    """
    return _mhc_aggregate(x, h_pre, use_tf32)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _mhc_aggregate(x, h_pre, use_tf32):
    """Internal mhc_aggregate with custom VJP."""
    out, _ = _mhc_aggregate_fwd_rule(x, h_pre, use_tf32)
    return out


def _mhc_aggregate_fwd_rule(x, h_pre, use_tf32):
    """Forward pass rule for mhc_aggregate."""
    del use_tf32
    return mhc_aggregate_fwd(x, h_pre), (x, h_pre)


def _mhc_aggregate_bwd_rule(use_tf32, residuals, grad_out):
    """Backward pass rule for mhc_aggregate."""
    x, h_pre = residuals
    return mhc_aggregate_bwd(grad_out, x, h_pre, use_tf32)


_mhc_aggregate.defvjp(_mhc_aggregate_fwd_rule, _mhc_aggregate_bwd_rule)


def mhc_expand_combine(
    f: jnp.ndarray,
    bias: Optional[jnp.ndarray],
    h_post: jnp.ndarray,
    x: jnp.ndarray,
    h_res: jnp.ndarray,
    use_tf32: bool = True,
) -> jnp.ndarray:
    """
    Expand the sub-layer output back to n streams and mix the residual streams:

    out = (f [+ bias]) @ H_post + x @ H_res: (..., C, 1) @ (..., 1, n) + (..., C, n) @ (..., n, n)

    Parameters
    ----------
    f : jnp.ndarray
        Sub-layer (attention / FFN) output of shape (..., C).
    bias : Optional[jnp.ndarray]
        Optional bias of shape (C,) of the sub-layer's last linear layer, fused here.
    h_post : jnp.ndarray
        H_post of shape (..., n).
    x : jnp.ndarray
        Hyper-connection input of shape (..., C, n).
    h_res : jnp.ndarray
        Sinkhorn-normalized H_res of shape (..., n, n).
    use_tf32 : bool
        Whether to use TF32 for the backward matmuls.

    Returns
    -------
    Array of shape (..., C, n) in x's dtype.
    """
    return _mhc_expand_combine(f, bias, h_post, x, h_res, use_tf32)


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _mhc_expand_combine(f, bias, h_post, x, h_res, use_tf32):
    """Internal mhc_expand_combine with custom VJP."""
    out, _ = _mhc_expand_combine_fwd_rule(f, bias, h_post, x, h_res, use_tf32)
    return out


def _mhc_expand_combine_fwd_rule(f, bias, h_post, x, h_res, use_tf32):
    """Forward pass rule for mhc_expand_combine."""
    del use_tf32
    out = mhc_expand_combine_fwd(f, bias, h_post, x, h_res)
    return out, (f, bias, h_post, x, h_res)


def _mhc_expand_combine_bwd_rule(use_tf32, residuals, grad_out):
    """Backward pass rule for mhc_expand_combine."""
    f, bias, h_post, x, h_res = residuals
    return mhc_expand_combine_bwd(grad_out, f, bias, h_post, x, h_res, use_tf32)


_mhc_expand_combine.defvjp(_mhc_expand_combine_fwd_rule, _mhc_expand_combine_bwd_rule)


def mhc_generate_mix_and_aggregate(
    x: jnp.ndarray,
    phi: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    norm_weight: Optional[jnp.ndarray] = None,
    use_tf32: bool = True,
    use_split_k: bool = False,
    norm_eps: float = DEFAULT_NORM_EPS,
    pre_eps: float = 0.0,
    sinkhorn_iters: int = 20,
    recompute_sinkhorn_hist: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate H_pre, H_post and H_res and aggregate the n streams of x with H_pre.

    This chains projection, scale, Sinkhorn and aggregate. The defaults match
    `mhc_generate_mix_and_aggregate` of the PyTorch API.

    Parameters
    ----------
    x : jnp.ndarray
        Hyper-connection input of shape (..., C, n).
    phi : jnp.ndarray
        Projection matrix of shape (N, n*C), with columns in the flattened (C, n) order of x.
    alpha : jnp.ndarray
        Scaling factors of shape (3,) for H_pre, H_post and H_res.
    beta : jnp.ndarray
        Bias with N = 2n + n*n elements for [H_pre, H_post, H_res].
    norm_weight : Optional[jnp.ndarray]
        RMSNorm weight of shape (n*C,), in the flattened (C, n) order of x.
    use_tf32, use_split_k :
        See `mhc_projection`.
    norm_eps : float
        RMSNorm epsilon.
    pre_eps : float
        Constant added to H_pre after its sigmoid.
    sinkhorn_iters, recompute_sinkhorn_hist :
        See `mhc_sinkhorn`.

    Returns
    -------
    layer_input : jnp.ndarray
        Aggregated input of shape (..., C) for the attention / FFN sub-layer, in x's dtype.
    H_post : jnp.ndarray
        fp32 array of shape (..., n) for `mhc_expand_combine`.
    H_res : jnp.ndarray
        fp32 array of shape (..., n, n) for `mhc_expand_combine`.
    """
    tokens = x.shape[:-2]
    C, n = x.shape[-2:]
    assert n == 4, "Only n=4 is supported in this implementation"
    h, ms = mhc_projection(x.reshape(tokens + (C * n,)), phi, norm_weight, use_tf32, use_split_k)
    h_pre, h_post, h_res = mhc_scale(h, alpha, beta, ms, n, norm_eps)
    if pre_eps:
        h_pre = h_pre + pre_eps
    h_res = mhc_sinkhorn(h_res.reshape(tokens + (n, n)), n, recompute_sinkhorn_hist, sinkhorn_iters)
    layer_input = mhc_aggregate(x, h_pre, use_tf32)
    return layer_input, h_post, h_res


class MHCWeights(NamedTuple):
    """
    Parameters of MaxText's `ManifoldConstrainedHyperConnections` layer (mirrors MaxText's
    `MhcWeights`), with k streams of dimension d.

    norm_scale : (k*d,) RMSNorm scale.
    pre_alpha, post_alpha : (k*d, k) projections for the pre and post mappings.
    res_alpha : (k*d, k*k) projection for the residual mapping.
    pre_bias, post_bias : (k,) biases of the pre and post mappings.
    res_bias : (k, k) bias of the residual mapping.
    pre_scale, post_scale, res_scale : (1,) scales of the pre, post and residual mappings.
    """

    norm_scale: jnp.ndarray
    pre_alpha: jnp.ndarray
    pre_bias: jnp.ndarray
    pre_scale: jnp.ndarray
    post_alpha: jnp.ndarray
    post_bias: jnp.ndarray
    post_scale: jnp.ndarray
    res_alpha: jnp.ndarray
    res_bias: jnp.ndarray
    res_scale: jnp.ndarray


def mhc_weights_to_params(
    weights: MHCWeights,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert MaxText-layout weights into (phi, norm_weight, alpha, beta) of
    `mhc_generate_mix_and_aggregate`.

    MaxText flattens the streams as (k, d) while the TE kernels flatten them as (d, k), so the
    projection and RMSNorm weights are permuted accordingly.
    """
    kd, k = weights.pre_alpha.shape
    d = kd // k
    assert weights.res_bias.shape == (
        k,
        k,
    ), "Only the Sinkhorn residual mapping is supported; mHC-lite (enable_mhc_lite=True) is not"
    projection = jnp.concatenate(
        [weights.pre_alpha, weights.post_alpha, weights.res_alpha], axis=-1
    )  # (k*d, N), rows in (k, d) order
    N = projection.shape[-1]
    phi = projection.reshape(k, d, N).transpose(2, 1, 0).reshape(N, d * k)
    norm_weight = weights.norm_scale.reshape(k, d).T.reshape(d * k)
    alpha = jnp.concatenate(
        [
            weights.pre_scale.reshape(-1),
            weights.post_scale.reshape(-1),
            weights.res_scale.reshape(-1),
        ]
    )
    beta = jnp.concatenate(
        [weights.pre_bias.reshape(-1), weights.post_bias.reshape(-1), weights.res_bias.reshape(-1)]
    )
    return phi, norm_weight, alpha, beta


def mhc_pre(
    x: jnp.ndarray,
    weights: MHCWeights,
    *,
    norm_epsilon: float = 1e-5,
    pre_mapping_epsilon: float = 1e-6,
    sinkhorn_iterations: int = 20,
    streams_last: bool = False,
    use_tf32: bool = True,
    use_split_k: bool = False,
    recompute_sinkhorn_hist: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute the branch input of an mHC-wrapped branch, like the part of MaxText's
    `ManifoldConstrainedHyperConnections.__call__` before `norm_fn`.

    Parameters
    ----------
    x : jnp.ndarray
        Input streams of shape (..., k, d), or (..., d, k) if `streams_last`.
    weights : MHCWeights
        Layer parameters.
    norm_epsilon : float
        RMSNorm epsilon (MaxText's `normalization_layer_epsilon`).
    pre_mapping_epsilon : float
        Constant added to the pre mapping after its sigmoid.
    sinkhorn_iterations : int
        Number of Sinkhorn iterations (MaxText's `sinkhorn_iterations`).
    streams_last : bool
        Whether x already has the TE (..., d, k) layout. Otherwise x is transposed here and
        the output transposed back in `mhc_post`.
    use_tf32, use_split_k, recompute_sinkhorn_hist :
        See `mhc_generate_mix_and_aggregate`.

    Returns
    -------
    layer_input : jnp.ndarray
        Branch input of shape (..., d) in x's dtype.
    h_post : jnp.ndarray
        Post mapping of shape (..., k), to pass to `mhc_post`.
    h_res : jnp.ndarray
        Residual mapping of shape (..., k, k), to pass to `mhc_post`.
    """
    if not streams_last:
        x = jnp.swapaxes(x, -1, -2)
    phi, norm_weight, alpha, beta = mhc_weights_to_params(weights)
    layer_input, h_post, h_res = mhc_generate_mix_and_aggregate(
        x,
        phi,
        alpha,
        beta,
        norm_weight,
        use_tf32=use_tf32,
        use_split_k=use_split_k,
        norm_eps=norm_epsilon,
        pre_eps=pre_mapping_epsilon,
        sinkhorn_iters=sinkhorn_iterations,
        recompute_sinkhorn_hist=recompute_sinkhorn_hist,
    )
    return layer_input, h_post, h_res


def mhc_post(
    layer_output: jnp.ndarray,
    x: jnp.ndarray,
    h_post: jnp.ndarray,
    h_res: jnp.ndarray,
    *,
    bias: Optional[jnp.ndarray] = None,
    streams_last: bool = False,
    use_tf32: bool = True,
) -> jnp.ndarray:
    """
    Apply the post mapping and residual stream mixing, like the part of MaxText's
    `ManifoldConstrainedHyperConnections.__call__` after `branch_fn`.

    Parameters
    ----------
    layer_output : jnp.ndarray
        Branch output of shape (..., d).
    x : jnp.ndarray
        The input streams passed to `mhc_pre`, in the same layout.
    h_post, h_res : jnp.ndarray
        Mappings returned by `mhc_pre`.
    bias : Optional[jnp.ndarray]
        Optional bias of shape (d,) of the branch's last linear layer, fused here.
    streams_last : bool
        Whether x has, and the output uses, the TE (..., d, k) layout instead of (..., k, d).
    use_tf32 : bool
        Whether to use TF32 for the backward matmuls.

    Returns
    -------
    Mixed streams of shape (..., k, d), or (..., d, k) if `streams_last`, in x's dtype.
    """
    if not streams_last:
        x = jnp.swapaxes(x, -1, -2)
    out = mhc_expand_combine(layer_output, bias, h_post, x, h_res, use_tf32)
    return out if streams_last else jnp.swapaxes(out, -1, -2)


def mhc(
    x: jnp.ndarray,
    weights: MHCWeights,
    norm_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]],
    branch_fn: Callable[[jnp.ndarray], Any],
    *,
    has_aux: bool = False,
    norm_epsilon: float = 1e-5,
    pre_mapping_epsilon: float = 1e-6,
    sinkhorn_iterations: int = 20,
    streams_last: bool = False,
    use_tf32: bool = True,
    use_split_k: bool = False,
    recompute_sinkhorn_hist: bool = True,
) -> Any:
    """
    Manifold-constrained hyper connection around `branch_fn`, computing the same result as
    MaxText's `ManifoldConstrainedHyperConnections.__call__` (non-lite variant):

        layer_input = norm_fn(sum_k pre[k] * x[k])
        layer_out = branch_fn(layer_input)
        out[m] = post[m] * layer_out + sum_k res[k, m] * x[k]

    with pre = sigmoid(.) + pre_mapping_epsilon, post = 2 * sigmoid(.) and
    res = sinkhorn(.), all generated from RMSNorm(x) by one fused projection.

    Results agree with MaxText up to floating-point differences: the kernels compute the
    mappings in fp32, and the log-space Sinkhorn omits the 1e-6 stabilizers of MaxText's
    `sinkhorn`. Only k = 4 streams are supported.

    Parameters
    ----------
    x : jnp.ndarray
        Input streams of shape (..., k, d), or (..., d, k) if `streams_last`.
    weights : MHCWeights
        Layer parameters.
    norm_fn : Optional[Callable]
        Pre-normalization applied to the branch input, or None.
    branch_fn : Callable
        The wrapped branch (attention / MLP), called as `branch_fn(layer_input)`. MaxText's
        mhc_type dispatch maps to e.g. `lambda y: branch(inputs_q=y, inputs_kv=y, **kwargs)[0]`.
    has_aux : bool
        Whether `branch_fn` returns `(layer_out, aux)`, in which case `(out, aux)` is returned.
    norm_epsilon, pre_mapping_epsilon, sinkhorn_iterations, streams_last, use_tf32,
    use_split_k, recompute_sinkhorn_hist :
        See `mhc_pre` and `mhc_post`.

    Returns
    -------
    Output streams with the layout of x, or `(output, aux)` if `has_aux`.
    """
    layer_input, h_post, h_res = mhc_pre(
        x,
        weights,
        norm_epsilon=norm_epsilon,
        pre_mapping_epsilon=pre_mapping_epsilon,
        sinkhorn_iterations=sinkhorn_iterations,
        streams_last=streams_last,
        use_tf32=use_tf32,
        use_split_k=use_split_k,
        recompute_sinkhorn_hist=recompute_sinkhorn_hist,
    )
    if norm_fn is not None:
        layer_input = norm_fn(layer_input)
    if has_aux:
        layer_out, aux = branch_fn(layer_input)
    else:
        layer_out, aux = branch_fn(layer_input), None
    out = mhc_post(layer_out, x, h_post, h_res, streams_last=streams_last, use_tf32=use_tf32)
    return (out, aux) if has_aux else out

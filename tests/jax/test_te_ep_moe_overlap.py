# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Flag-on vs flag-off tests for the MoE block's EP/GEMM overlap path (``enable_ep_overlap``).

Multiprocess, one GPU per process; launch with the TE-EP MoE launcher::

    NVTE_ENABLE_CUTEDSL_BACKEND=1 NVTE_MOE_OVERLAP=<EPCompOverlap>/gemm \\
    TEST_FILE=tests/jax/test_te_ep_moe_overlap.py bash tests/jax/run_te_ep_moe.sh

Each mesh layout (EP4, FSDP2 x EP2 on 4 GPUs) bootstraps EP once. For every layout the test
compares the overlapped block against the same block with the flag off (same parameters and
inputs): the output, and the gradients of the input, both expert weights and the gate kernel
(the gate kernel gradient carries the routing-weight gradient). It also asserts that the
overlap custom calls are in the compiled program and that no fallback warning was raised.
"""

import os
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax.linen import partitioning as nn_partitioning
import flax.linen as nn


def _read_mp_options():
    num = int(os.environ.get("MP_NUM_PROCESS", "0") or "0")
    pid = int(os.environ.get("MP_PROCESS_ID", "0") or "0")
    for a in sys.argv:
        if a.startswith("--num-process="):
            num = int(a.split("=", 1)[1])
        elif a.startswith("--process-id="):
            pid = int(a.split("=", 1)[1])
    return num, pid


_NUM, _PID = _read_mp_options()
if _NUM <= 1:
    pytest.skip("needs the multiprocess launcher (run_te_ep_moe.sh)", allow_module_level=True)
jax.distributed.initialize(
    coordinator_address=os.environ.get("TE_EP_MOE_COORDINATOR_ADDRESS", "127.0.0.1:13457"),
    num_processes=_NUM,
    process_id=_PID,
    local_device_ids=_PID,
)

from transformer_engine_jax import get_device_compute_capability  # noqa: E402

if get_device_compute_capability(0) not in (100, 103, 107):
    pytest.skip("EP overlap needs SM 10.0, 10.3 or 10.7", allow_module_level=True)
if not (os.environ.get("NVTE_ENABLE_CUTEDSL_BACKEND") == "1" and os.environ.get("NVTE_MOE_OVERLAP")):
    pytest.skip(
        "EP overlap needs NVTE_ENABLE_CUTEDSL_BACKEND=1 and NVTE_MOE_OVERLAP",
        allow_module_level=True,
    )

from transformer_engine.jax.flax import _MoEBlock as MoEBlock  # noqa: E402
from transformer_engine.jax.moe import (  # noqa: E402
    get_moe_recv_capacity_per_rank,
    record_ep_bootstrap_signature_for_moe,
)
from transformer_engine.jax.ep import ep_bootstrap, ep_finalize  # noqa: E402
from transformer_engine.jax.sharding import MeshResource, global_shard_guard  # noqa: E402

DTYPE = jnp.bfloat16
HIDDEN = 512  # multiple of 256
INTER = 256  # multiple of 64
NUM_EXPERTS = 8
TOPK = 2
SEQ = 128
BATCH_PER_RANK = 2

# The overlap applies the routing weight to the SwiGLU output before FC2 (BF16), the unfused
# path to FC2's output; both accumulate in FP32 but round at different points, so outputs and
# gradients agree to BF16 rounding, not bitwise. Errors are measured relative to the largest
# reference magnitude of each tensor.
REL_TOL = {"out": 2e-2, "x": 2e-2, "wi": 2e-2, "wo": 2e-2, "gate_kernel": 3e-2}

OVERLAP_TARGETS = (
    "te_ep_dispatch_gemm_fwd_ffi",
    "te_ep_combine_gemm_fwd_ffi",
    "te_ep_combine_gemm_bwd_ffi",
    "te_ep_dispatch_gemm_bwd_ffi",
)

LAYOUTS = {"ep4": 1, "fsdp2_ep2": 2}


@pytest.fixture(scope="module", params=list(LAYOUTS), ids=list(LAYOUTS))
def layout(request):
    fsdp = LAYOUTS[request.param]
    n = jax.device_count()
    if n % fsdp or n // fsdp < 2:
        pytest.skip(f"layout {request.param} needs a multiple of {2 * fsdp} devices")
    ep = n // fsdp
    if fsdp > 1:
        mesh = Mesh(np.asarray(jax.devices()).reshape(fsdp, ep), ("fsdp", "ep"))
        mr = MeshResource(ep_resource="ep", fsdp_resource="fsdp")
        rules = (("exp", "ep"), ("embed", "fsdp"), ("mlp", None), ("batch", ("fsdp", "ep")))
        batch_axes = ("fsdp", "ep")
    else:
        mesh = Mesh(np.asarray(jax.devices()).reshape(ep), ("ep",))
        mr = MeshResource(ep_resource="ep")
        rules = (("exp", "ep"), ("embed", None), ("mlp", None), ("batch", "ep"))
        batch_axes = "ep"
    max_tokens = BATCH_PER_RANK * SEQ
    recv_cap = get_moe_recv_capacity_per_rank(
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        max_tokens_per_rank=max_tokens,
        ep_size=ep,
        use_cudnn_fusion=False,
    )
    with mesh, global_shard_guard(mr):
        ep_bootstrap(
            world_size=n,
            rank=jax.process_index(),
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=max_tokens,
            recv_capacity_per_rank=recv_cap,
            hidden_dim=HIDDEN,
            max_token_dtype=DTYPE,
            num_topk=int(os.environ.get("OVERLAP_TEST_NUM_TOPK", TOPK)),
        )
    record_ep_bootstrap_signature_for_moe(
        num_experts=NUM_EXPERTS,
        max_tokens_per_rank=max_tokens,
        recv_capacity_per_rank=recv_cap,
        hidden_dim=HIDDEN,
        ep_size=ep,
    )
    try:
        yield dict(name=request.param, mesh=mesh, mr=mr, rules=rules, batch_axes=batch_axes)
    finally:
        jax.effects_barrier()
        multihost_utils.sync_global_devices(f"overlap_{request.param}_done")
        ep_finalize()
        multihost_utils.sync_global_devices(f"overlap_{request.param}_finalized")


def _block(enable, **kw):
    return MoEBlock(
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        intermediate_size=INTER,
        dtype=DTYPE,
        input_axes=("batch", None, None),
        use_cudnn_fusion=False,
        collect_expert_counts=False,
        enable_ep_overlap=enable,
        **kw,
    )


def _setup(layout, key=0, **kw):
    mesh, rules = layout["mesh"], layout["rules"]
    n = jax.device_count()
    x = jax.random.normal(jax.random.PRNGKey(key), (n * BATCH_PER_RANK, SEQ, HIDDEN), DTYPE)
    x = jax.device_put(x, NamedSharding(mesh, P(layout["batch_axes"], None, None)))
    block = _block(False, **kw)
    abstract = jax.eval_shape(block.init, jax.random.PRNGKey(1), x)
    shardings = nn.logical_to_mesh_sharding(nn.get_partition_spec(abstract), mesh, rules)
    variables = nn.unbox(
        jax.jit(block.init, out_shardings=shardings)(jax.random.PRNGKey(key + 1), x)
    )
    return x, variables


def _ctx(layout):
    class _C:
        def __enter__(self):
            self.m = layout["mesh"].__enter__()
            self.g = global_shard_guard(layout["mr"])
            self.g.__enter__()
            self.r = nn_partitioning.axis_rules(layout["rules"])
            self.r.__enter__()

        def __exit__(self, *a):
            self.r.__exit__(*a)
            self.g.__exit__(*a)
            layout["mesh"].__exit__(*a)

    return _C()


def _loss_and_grads(block, variables, x, dy):
    def loss(v, x):
        out = block.apply(v, x)[0]
        return jnp.sum(out.astype(jnp.float32) * dy.astype(jnp.float32)), out

    fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))
    return fn, fn(variables, x)


def _global(a):
    return np.asarray(multihost_utils.process_allgather(a, tiled=True), np.float32)


def _rel_err(test, ref):
    test, ref = _global(test), _global(ref)
    return float(np.max(np.abs(test - ref)) / max(np.max(np.abs(ref)), 1e-30))


@pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
def test_overlap_matches_flag_off(layout, score_function):
    """Output and input/weight/routing gradients of flag-on match flag-off."""
    with _ctx(layout):
        x, variables = _setup(layout, score_function=score_function)
        dy = jax.random.normal(jax.random.PRNGKey(7), x.shape, DTYPE)
        dy = jax.device_put(dy, x.sharding)
        ref_fn, ((_, ref_out), (ref_gv, ref_gx)) = _loss_and_grads(
            _block(False, score_function=score_function), variables, x, dy
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*EP overlap skipped.*")
            on_block = _block(True, score_function=score_function)
            on_fn, ((_, out), (gv, gx)) = _loss_and_grads(on_block, variables, x, dy)
        hlo = on_fn.lower(variables, x).as_text()
        for target in OVERLAP_TARGETS:
            assert target in hlo, f"{target} missing: the overlap path did not run"
        assert "te_ep_dispatch_gemm_fwd_ffi" not in ref_fn.lower(variables, x).as_text()

    errs = {
        "out": _rel_err(out, ref_out),
        "x": _rel_err(gx, ref_gx),
        "wi": _rel_err(gv["params"]["wi"], ref_gv["params"]["wi"]),
        "wo": _rel_err(gv["params"]["wo"], ref_gv["params"]["wo"]),
        "gate_kernel": _rel_err(gv["params"]["gate_kernel"], ref_gv["params"]["gate_kernel"]),
    }
    if jax.process_index() == 0:
        print(f"[{layout['name']} {score_function}] max rel err vs flag-off: {errs}", flush=True)
    for k, e in errs.items():
        assert np.isfinite(e) and e <= REL_TOL[k], f"{k}: rel err {e} > {REL_TOL[k]}"


def test_ineligible_layer_falls_back(layout):
    """A layer the overlap cannot run warns and matches the flag-off block bitwise.

    The ineligibility used is a dispatch checkpoint name (unsupported by the overlap path), which
    leaves the regular path's math unchanged.
    """
    kw = dict(dispatch_checkpoint_name="moe_dispatch")
    with _ctx(layout):
        x, variables = _setup(layout, key=3, **kw)
        ref = jax.jit(_block(False, **kw).apply)(variables, x)[0]
        with pytest.warns(UserWarning, match="EP overlap skipped.*checkpoint names"):
            fn = jax.jit(_block(True, **kw).apply)
            out = fn(variables, x)[0]
        assert "te_ep_dispatch_gemm_fwd_ffi" not in fn.lower(variables, x).as_text()
    np.testing.assert_array_equal(_global(out), _global(ref))

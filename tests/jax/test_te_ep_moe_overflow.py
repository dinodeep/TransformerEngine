# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Multi-process MoE overflow test: ``drop_on_overflow=True`` with a receive capacity below the demand.

Run it with the TE-EP MoE launcher, which forks one pytest process per GPU:

    TEST_FILE=$PWD/tests/jax/test_te_ep_moe_overflow.py bash tests/jax/run_te_ep_moe.sh

It needs its own module because ``test_te_ep_moe.py`` bootstraps EP once per module, dropless and at
the worst-case capacity; this one bootstraps with drops and a small capacity. ``moe()`` documents
``total_recv_tokens`` as the pre-drop receive total, which callers (e.g. MaxText's overflow check)
compare to the capacity. The overlap path (``enable_ep_overlap=True``) must report the same total as
the non-overlap path, including the rows it drops, or an overflow goes unnoticed.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax.linen import partitioning as nn_partitioning


def _read_mp_options():
    num, pid = 0, 0
    for i, a in enumerate(sys.argv):
        if a.startswith("--num-process="):
            num = int(a.split("=", 1)[1])
        elif a == "--num-process" and i + 1 < len(sys.argv):
            num = int(sys.argv[i + 1])
        elif a.startswith("--process-id="):
            pid = int(a.split("=", 1)[1])
        elif a == "--process-id" and i + 1 < len(sys.argv):
            pid = int(sys.argv[i + 1])
    return num, pid


_MP_NUM_PROCESS, _MP_PROCESS_ID = _read_mp_options()
if _MP_NUM_PROCESS <= 1:
    pytest.skip(
        "test_te_ep_moe_overflow.py requires the multiprocess launcher (run_te_ep_moe.sh).",
        allow_module_level=True,
    )
jax.distributed.initialize(
    coordinator_address=os.environ.get("TE_EP_MOE_COORDINATOR_ADDRESS", "127.0.0.1:13457"),
    num_processes=_MP_NUM_PROCESS,
    process_id=_MP_PROCESS_ID,
    local_device_ids=_MP_PROCESS_ID,
)

from transformer_engine_jax import get_device_compute_capability

if get_device_compute_capability(0) < 100:
    pytest.skip("MoE TE EP tests require Blackwell (sm_100+)", allow_module_level=True)

import transformer_engine_jax
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.jax.ep import ep_bootstrap, ep_finalize
from transformer_engine.jax.flax import _MoEBlock as MoEBlock
from transformer_engine.jax.moe import record_ep_bootstrap_signature_for_moe
from transformer_engine.jax.sharding import MeshResource, global_shard_guard

EP_AXIS, FSDP_AXIS = "ep", "fsdp"
EP_SIZE = 2
FSDP_SIZE = jax.device_count() // EP_SIZE
LOGICAL_AXIS_RULES = (
    ("exp", EP_AXIS),
    ("expert_weight_fsdp", (EP_AXIS, FSDP_AXIS)),
    ("gated_weight_fsdp", FSDP_AXIS),
    ("embed", FSDP_AXIS),
    ("mlp", None),
    ("batch", (FSDP_AXIS, EP_AXIS)),
)
BATCH = EP_SIZE * FSDP_SIZE * 2
SEQ = 32
HIDDEN = 512
INTER = 768  # an FFN size the MXFP8 overlap supports (multiple of 256)
NUM_EXPERTS = 8
TOPK = 2
COMM_SMS = 24
# The routing bias steers every token's top-k onto experts 0..3, which EP rank 0 owns. Each EP group
# then sends rank 0 its 2 * SEQ * TOPK = 128 routed slots, padded per expert to the 128-row
# alignment (4 experts: >= 384 rows), while the capacity is 256 rows (a multiple of 256 for the
# overlap, and >= the per-rank dispatch count of 64 tokens).
RECV_CAPACITY = 256


def _expert_bias_init(key, shape, dtype):
    del key
    n = shape[0]
    return jnp.concatenate([jnp.full((n // 2,), 5.0, dtype), jnp.full((n - n // 2,), -5.0, dtype)])


@pytest.fixture(scope="module")
def mesh():
    if jax.device_count() < 2 * EP_SIZE:
        pytest.skip(f"needs >= {2 * EP_SIZE} devices")
    if not hasattr(transformer_engine_jax, "ep_overlap_workspace_size"):
        pytest.skip("TE was built without the EP/GEMM overlap")
    if os.environ.get("NVTE_ENABLE_CUTEDSL_BACKEND") != "1" or not os.path.isdir(
        os.environ.get("NVTE_MOE_OVERLAP", "")
    ):
        pytest.skip("the EP/GEMM overlap needs NVTE_ENABLE_CUTEDSL_BACKEND=1 and NVTE_MOE_OVERLAP")
    mesh_obj = Mesh(mesh_utils.create_device_mesh((FSDP_SIZE, EP_SIZE)), (FSDP_AXIS, EP_AXIS))
    max_tokens_per_rank = (BATCH // jax.process_count()) * SEQ
    with mesh_obj, global_shard_guard(MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)):
        ep_bootstrap(
            world_size=jax.process_count(),
            rank=jax.process_index(),
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=RECV_CAPACITY,
            hidden_dim=HIDDEN,
            max_token_dtype=jnp.bfloat16,
            max_num_sms=COMM_SMS,
            drop_on_overflow=True,
            num_topk=TOPK,
        )
    record_ep_bootstrap_signature_for_moe(
        num_experts=NUM_EXPERTS,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=RECV_CAPACITY,
        hidden_dim=HIDDEN,
        ep_size=EP_SIZE,
    )
    try:
        yield mesh_obj
    finally:
        jax.effects_barrier()
        multihost_utils.sync_global_devices("te_moe_overflow_finished")
        ep_finalize()
        multihost_utils.sync_global_devices("te_moe_overflow_finalized")


@pytest.mark.parametrize("quantization", ["bf16", "mxfp8"])
def test_overlap_reports_pre_drop_total(mesh, quantization):
    """With drops, the overlap and non-overlap paths report the same pre-drop receive total, and
    it exceeds the receive capacity on the overloaded rank (the overflow callers must see)."""
    x = jax.random.normal(jax.random.PRNGKey(0), (BATCH, SEQ, HIDDEN), dtype=jnp.bfloat16)
    totals, outputs = {}, {}
    for overlap in (False, True):
        block = MoEBlock(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            intermediate_size=INTER,
            score_function="sigmoid",
            use_expert_routing_bias=True,
            expert_bias_init=_expert_bias_init,
            input_axes=("batch", None, None),
            dtype=jnp.bfloat16,
            use_cudnn_fusion=False,  # the overlap replaces the unfused path
            recv_capacity_per_rank=RECV_CAPACITY,
            enable_ep_overlap=overlap,
            quantization_recipe=MXFP8BlockScaling() if quantization == "mxfp8" else None,
        )
        with mesh, global_shard_guard(
            MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
        ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
            x_sh = jax.lax.with_sharding_constraint(
                x, NamedSharding(mesh, P((FSDP_AXIS, EP_AXIS), None, None))
            )
            variables = jax.jit(block.init)(jax.random.PRNGKey(1), x_sh)
            output, _aux, total, _counts = jax.jit(block.apply)(variables, x_sh)
            outputs[overlap] = np.asarray(output.addressable_data(0)).astype(np.float32)
        totals[overlap] = np.asarray(multihost_utils.process_allgather(total, tiled=True))
    assert np.all(np.isfinite(outputs[True])), "overlap output has NaN/Inf after the drop"
    # The rank owning experts 0..3 is asked for more rows than it can receive ...
    assert int(totals[False].max()) > RECV_CAPACITY, totals[False]
    # ... and the overlap reports that same pre-drop total, not the kept rows (<= the capacity).
    np.testing.assert_array_equal(totals[True], totals[False])

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Flag-on vs flag-off tests for the MoE block's EP/GEMM overlap path (``enable_ep_overlap``).

Multiprocess, one GPU per process; launch with the TE-EP MoE launcher::

    NVTE_ENABLE_CUTEDSL_BACKEND=1 NVTE_MOE_OVERLAP=<EPCompOverlap>/gemm \\
    TEST_FILE=tests/jax/test_te_ep_moe_overlap.py bash tests/jax/run_te_ep_moe.sh

Each mesh layout (EP4; FSDP2 x EP2 with the expert weights' hidden dim over fsdp; FSDP2 x EP2 with
the expert dim over (ep, fsdp)) bootstraps EP once. The functional ``moe()`` is tested first:
flag-on vs flag-off for the output and the gradients of the input, both expert weights and the
gate kernel (which carries the routing-weight gradient), in the default and the native weight
layouts, with a check that all four overlap custom calls are in the compiled program and that no
fallback happened; the fallback path; and that no CuTeDSL kernel is compiled at execution. Then the
Flax ``_MoEBlock`` is checked to be bitwise identical to ``moe()`` for the same parameters.
"""

import os
import re
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
if not (
    os.environ.get("NVTE_ENABLE_CUTEDSL_BACKEND") == "1" and os.environ.get("NVTE_MOE_OVERLAP")
):
    pytest.skip(
        "EP overlap needs NVTE_ENABLE_CUTEDSL_BACKEND=1 and NVTE_MOE_OVERLAP",
        allow_module_level=True,
    )

from transformer_engine.jax.flax import _MoEBlock as MoEBlock  # noqa: E402
from transformer_engine.jax.cpp_extensions import (  # noqa: E402
    pack_gated_activation_pair,
    unpack_gated_activation_pair,
)
from transformer_engine.jax.moe import (  # noqa: E402
    RouterComputationInfo,
    get_moe_recv_capacity_per_rank,
    moe,
    record_ep_bootstrap_signature_for_moe,
)
from transformer_engine.jax.ep import ep_bootstrap, ep_finalize  # noqa: E402
from transformer_engine.jax.sharding import MeshResource, global_shard_guard  # noqa: E402

DTYPE = jnp.bfloat16
HIDDEN = 512  # multiple of 256
INTER = 256  # multiple of 64
NUM_EXPERTS = 8
TOPK = 2
# Communication SMs per NCCL EP call; the overlap path needs an explicit ep_bootstrap budget.
OVERLAP_COMM_SMS = 24
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

# name -> (fsdp size, expert-weight FSDP mode). "hidden": the hidden dim of the expert weights is
# sharded over fsdp (TE's default logical rules); "expert": the expert dim over (ep, fsdp).
LAYOUTS = {"ep4": (1, None), "fsdp2_ep2": (2, "hidden"), "fsdp2_ep2_expert": (2, "expert")}
WEIGHT_LAYOUTS = ("default", "native")


@pytest.fixture(scope="module", params=list(LAYOUTS), ids=list(LAYOUTS))
def layout(request):
    fsdp, fsdp_mode = LAYOUTS[request.param]
    n = jax.device_count()
    if n % fsdp or n // fsdp < 2:
        pytest.skip(f"layout {request.param} needs a multiple of {2 * fsdp} devices")
    ep = n // fsdp
    if fsdp > 1:
        mesh = Mesh(np.asarray(jax.devices()).reshape(fsdp, ep), ("fsdp", "ep"))
        mr = MeshResource(ep_resource="ep", fsdp_resource="fsdp")
        if fsdp_mode == "expert":
            rules = (
                ("exp", ("ep", "fsdp")),
                ("embed", None),
                ("mlp", None),
                ("batch", ("fsdp", "ep")),
            )
        else:
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
            max_num_sms=OVERLAP_COMM_SMS,
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


def _install_compile_counter():
    """Wrap the CuTeDSL EP GEMM entry points that C++ calls on a kernel-cache miss.

    Returns a dict whose "calls" counts every entry (compile, register or layout query)."""
    import tvm_ffi
    from transformer_engine.common.CuTeDSL.ep_gemm import (
        dispatch_gemm,
        combine_gemm,
        requantize,
        wgrad_gemm,
    )

    counter = {"calls": 0}
    for fn in (
        dispatch_gemm.get_ep_dispatch_gemm_function,
        dispatch_gemm.get_ep_dispatch_gemm_workspace_layout,
        combine_gemm.get_ep_combine_gemm_function,
        combine_gemm.get_ep_combine_gemm_workspace_layout,
        wgrad_gemm.get_ep_wgrad_function,
        wgrad_gemm.get_ep_wgrad_workspace_layout,
        wgrad_gemm.get_ep_wgrad_reads_descriptors,
        requantize.get_ep_requant_function,
    ):

        def wrapped(*args, _fn=fn):
            counter["calls"] += 1
            counter.setdefault("log", []).append((_fn.__name__, args))
            return _fn(*args)

        tvm_ffi.register_global_func(fn.__name__, wrapped, override=True)
    return counter


_COMPILE_COUNTER = _install_compile_counter()


WI_AXES = ("exp", "embed", "mlp")
WO_AXES = ("exp", "mlp", "embed")


def _fc1_to_native(wi):
    gate, up = jnp.split(wi, 2, axis=-1)
    return pack_gated_activation_pair(gate, up).transpose(0, 2, 1)


def _fc1_from_native(wi):
    gate, up = unpack_gated_activation_pair(wi.transpose(0, 2, 1))
    return jnp.concatenate((gate, up), axis=-1)


def _to_layout(params, weight_layout):
    if weight_layout == "default":
        return params
    p = dict(params)
    p["wi"] = _fc1_to_native(p["wi"])
    p["wo"] = p["wo"].transpose(0, 2, 1)
    return p


def _from_layout(params, weight_layout):
    if weight_layout == "default":
        return params
    p = dict(params)
    p["wi"] = _fc1_from_native(p["wi"])
    p["wo"] = p["wo"].transpose(0, 2, 1)
    return p


def _axes(weight_layout):
    if weight_layout == "default":
        return WI_AXES, WO_AXES
    return (WI_AXES[0], WI_AXES[2], WI_AXES[1]), (WO_AXES[0], WO_AXES[2], WO_AXES[1])


def _shard_params(layout, params, weight_layout):
    wi_axes, wo_axes = _axes(weight_layout)
    specs = {"gate_kernel": (None, None), "wi": wi_axes, "wo": wo_axes}
    rules = dict(layout["rules"])
    return {
        k: jax.device_put(
            v,
            NamedSharding(layout["mesh"], P(*(rules.get(a) if a else None for a in specs[k]))),
        )
        for k, v in params.items()
    }


def _make_params(layout, key, inter=INTER):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(key), 3)
    params = {
        "gate_kernel": jax.random.normal(k1, (HIDDEN, NUM_EXPERTS), DTYPE) * HIDDEN**-0.5,
        "wi": jax.random.normal(k2, (NUM_EXPERTS, HIDDEN, 2 * inter), DTYPE) * HIDDEN**-0.5,
        "wo": jax.random.normal(k3, (NUM_EXPERTS, inter, HIDDEN), DTYPE) * inter**-0.5,
    }
    return params


def _make_x(layout, key):
    n = jax.device_count()
    x = jax.random.normal(jax.random.PRNGKey(key), (n * BATCH_PER_RANK, SEQ, HIDDEN), DTYPE)
    return jax.device_put(x, NamedSharding(layout["mesh"], P(layout["batch_axes"], None, None)))


def _moe_apply(layout, params, x, *, enable, weight_layout, score_function="softmax", **kw):
    wi_axes, wo_axes = _axes(weight_layout)
    native = weight_layout == "native"
    router = RouterComputationInfo(
        gate_kernel=params["gate_kernel"],
        num_experts_per_tok=TOPK,
        score_function=score_function,
    )
    return moe(
        x,
        router,
        params["wi"],
        params["wo"],
        num_experts=NUM_EXPERTS,
        mesh_resource=layout["mr"],
        input_axes=("batch", None, None),
        wi_kernel_axes=wi_axes,
        wo_kernel_axes=wo_axes,
        dtype=DTYPE,
        use_cudnn_fusion=False,
        collect_expert_counts=False,
        cudnn_native_weight_layout=native,
        fc2_native_weight_layout=native,
        enable_ep_overlap=enable,
        **kw,
    )[0]


def _value_and_grads(apply_fn, params, x, dy):
    def loss(p, x):
        out = apply_fn(p, x)
        return jnp.sum(out.astype(jnp.float32) * dy.astype(jnp.float32)), out

    fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))
    (_, out), (gp, gx) = fn(params, x)
    return fn, out, gp, gx


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


def _global(a):
    return np.asarray(multihost_utils.process_allgather(a, tiled=True), np.float32)


def _rel_err(test, ref):
    test, ref = _global(test), _global(ref)
    return float(np.max(np.abs(test - ref)) / max(np.max(np.abs(ref)), 1e-30))


# The expert weight gradients are not bitwise reproducible run to run on either path (flag-off:
# TE's grouped-GEMM weight gradient; flag-on: the CuTeDSL weight-gradient GEMMs split tiles across
# two launches), while the output is. With the flag on, the routing-weight gradient is also reduced
# with float32 atomics (as in PyTorch FusedMoE), so dx (through the gate) and the gate gradient are
# reproducible only to float32 reduction order, which can flip the last bit of a BF16 value. "Bitwise"
# checks therefore compare these within bounds (max |a - b| / max |b|): weight gradients measured
# run-to-run at up to ~3e-3, the routing-dependent BF16 gradients at up to ~3.6e-3 (1 BF16 ulp).
WGRAD_REPRO_TOL = 1e-2
ROUTING_GRAD_REPRO_TOL = 1e-2


def _assert_same(test, ref, key, enable=False):
    if key in ("wi", "wo"):
        e = _rel_err(test, ref)
        assert e <= WGRAD_REPRO_TOL, f"{key}: rel err {e} > {WGRAD_REPRO_TOL}"
    elif enable and key in ("x", "gate_kernel"):
        e = _rel_err(test, ref)
        assert e <= ROUTING_GRAD_REPRO_TOL, f"{key}: rel err {e} > {ROUTING_GRAD_REPRO_TOL}"
    else:
        np.testing.assert_array_equal(_global(test), _global(ref), err_msg=key)


MXFP8_SCALING_MODE = 2  # ScalingMode.MXFP8_1D_SCALING.value
NO_SCALING_MODE = 0  # ScalingMode.NO_SCALING.value


def _overlap_scaling_modes(hlo):
    """{overlap FFI target: set of its scaling_mode attribute values} in lowered StableHLO text."""
    modes = {}
    for line in hlo.splitlines():
        for target in OVERLAP_TARGETS:
            if f"@{target}(" in line:
                m = re.search(r"scaling_mode = (\d+)", line)
                modes.setdefault(target, set()).add(int(m.group(1)) if m else None)
    return modes


def _hlo(fn, *args):
    return fn.lower(*args).as_text()


# ----------------------------------------------------------------------------- moe()


@pytest.mark.parametrize("weight_layout", WEIGHT_LAYOUTS)
@pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
def test_moe_overlap_matches_flag_off(layout, score_function, weight_layout):
    """moe(): flag-on output and dx/dwi/dwo/dgate match the flag-off default-layout reference."""
    with _ctx(layout):
        std = _make_params(layout, 0)
        x = _make_x(layout, 1)
        dy = jax.device_put(jax.random.normal(jax.random.PRNGKey(2), x.shape, DTYPE), x.sharding)
        ref_params = _shard_params(layout, std, "default")
        ref_fn, ref_out, ref_gp, ref_gx = _value_and_grads(
            lambda p, x: _moe_apply(
                layout, p, x, enable=False, weight_layout="default", score_function=score_function
            ),
            ref_params,
            x,
            dy,
        )
        params = _shard_params(layout, _to_layout(std, weight_layout), weight_layout)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            on_fn, out, gp, gx = _value_and_grads(
                lambda p, x: _moe_apply(
                    layout,
                    p,
                    x,
                    enable=True,
                    weight_layout=weight_layout,
                    score_function=score_function,
                ),
                params,
                x,
                dy,
            )
        messages = [str(w.message) for w in caught]
        assert not any("EP overlap skipped" in m for m in messages), messages
        assert any("default layout" in m for m in messages) == (weight_layout == "default")
        hlo = _hlo(on_fn, params, x)
        modes = _overlap_scaling_modes(hlo)
        for target in OVERLAP_TARGETS:
            assert modes.get(target) == {
                NO_SCALING_MODE
            }, f"{target}: scaling modes {modes.get(target)}; the BF16 overlap path did not run"
        assert "te_ep_dispatch_gemm_fwd_ffi" not in _hlo(ref_fn, ref_params, x)
        gp = _from_layout(gp, weight_layout)

    errs = {
        "out": _rel_err(out, ref_out),
        "x": _rel_err(gx, ref_gx),
        "wi": _rel_err(gp["wi"], ref_gp["wi"]),
        "wo": _rel_err(gp["wo"], ref_gp["wo"]),
        "gate_kernel": _rel_err(gp["gate_kernel"], ref_gp["gate_kernel"]),
    }
    if jax.process_index() == 0:
        print(
            f"[moe {layout['name']} {score_function} {weight_layout}] max |on-off|/max|off|:"
            f" {errs}",
            flush=True,
        )
    for k, e in errs.items():
        assert np.isfinite(e) and e <= REL_TOL[k], f"{k}: rel err {e} > {REL_TOL[k]}"


def test_moe_native_layout_flag_off_is_bitwise_default(layout):
    """moe(): with the flag off, the native layouts only change storage (bitwise same results)."""
    with _ctx(layout):
        std = _make_params(layout, 3)
        x = _make_x(layout, 4)
        dy = jax.device_put(jax.random.normal(jax.random.PRNGKey(5), x.shape, DTYPE), x.sharding)
        _, ref_out, ref_gp, ref_gx = _value_and_grads(
            lambda p, x: _moe_apply(layout, p, x, enable=False, weight_layout="default"),
            _shard_params(layout, std, "default"),
            x,
            dy,
        )
        _, out, gp, gx = _value_and_grads(
            lambda p, x: _moe_apply(layout, p, x, enable=False, weight_layout="native"),
            _shard_params(layout, _to_layout(std, "native"), "native"),
            x,
            dy,
        )
        gp = _from_layout(gp, "native")
    np.testing.assert_array_equal(_global(out), _global(ref_out))
    np.testing.assert_array_equal(_global(gx), _global(ref_gx))
    for k in ("wi", "wo", "gate_kernel"):
        _assert_same(gp[k], ref_gp[k], k)


def test_moe_ineligible_falls_back(layout):
    """moe(): an ineligible call warns and is bitwise the flag-off result (here: a dispatch
    checkpoint name, which the overlap path does not support)."""
    kw = dict(dispatch_checkpoint_name="moe_dispatch")
    with _ctx(layout):
        params = _shard_params(layout, _to_layout(_make_params(layout, 6), "native"), "native")
        x = _make_x(layout, 7)
        ref = jax.jit(
            lambda p, x: _moe_apply(layout, p, x, enable=False, weight_layout="native", **kw)
        )(params, x)
        with pytest.warns(UserWarning, match="EP overlap skipped.*checkpoint names"):
            fn = jax.jit(
                lambda p, x: _moe_apply(layout, p, x, enable=True, weight_layout="native", **kw)
            )
            out = fn(params, x)
        assert "te_ep_dispatch_gemm_fwd_ffi" not in _hlo(fn, params, x)
    np.testing.assert_array_equal(_global(out), _global(ref))


def test_no_kernel_compile_at_execution(layout):
    """The CuTeDSL GEMMs compile while jit traces (nvte_ep_overlap_compile); executing the
    compiled step, first and later, never re-enters the Python compile/register entry points."""
    # An FFN size no other test uses, so this layer's GEMM configurations are new.
    inter = {"ep4": 384, "fsdp2_ep2": 448, "fsdp2_ep2_expert": 512}[layout["name"]]
    with _ctx(layout):
        params = _shard_params(
            layout, _to_layout(_make_params(layout, 8, inter=inter), "native"), "native"
        )
        x = _make_x(layout, 9)
        dy = jax.device_put(jax.random.normal(jax.random.PRNGKey(10), x.shape, DTYPE), x.sharding)

        def loss(p, x):
            out = _moe_apply(layout, p, x, enable=True, weight_layout="native")
            return jnp.sum(out.astype(jnp.float32) * dy.astype(jnp.float32))

        before_trace = _COMPILE_COUNTER["calls"]
        compiled = jax.jit(jax.grad(loss, argnums=(0, 1))).lower(params, x).compile()
        after_trace = _COMPILE_COUNTER["calls"]
        for _ in range(3):
            jax.block_until_ready(compiled(params, x))
        after_exec = _COMPILE_COUNTER["calls"]
    if jax.process_index() == 0:
        print(
            f"[{layout['name']}] CuTeDSL entry calls: during trace+compile"
            f" {after_trace - before_trace}, during 3 executions {after_exec - after_trace}",
            flush=True,
        )
    assert after_trace > before_trace, "the new GEMM configuration was not compiled at trace time"
    assert after_exec == after_trace, "a CuTeDSL GEMM was compiled/looked up at execution time"


# ----------------------------------------------------------------------------- _MoEBlock


@pytest.mark.parametrize("weight_layout", WEIGHT_LAYOUTS)
@pytest.mark.parametrize("enable", [False, True], ids=["flag_off", "flag_on"])
def test_block_is_bitwise_moe(layout, enable, weight_layout):
    """_MoEBlock is a thin wrapper: for its own parameters it gives bitwise the outputs and
    gradients of moe(); native layouts initialize the same values as the default layout."""
    native = weight_layout == "native"
    block = MoEBlock(
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        intermediate_size=INTER,
        dtype=DTYPE,
        input_axes=("batch", None, None),
        use_cudnn_fusion=False,
        collect_expert_counts=False,
        enable_ep_overlap=enable,
        cudnn_native_weight_layout=native,
        fc2_native_weight_layout=native,
        mesh_resource=layout["mr"],
    )
    with _ctx(layout):
        x = _make_x(layout, 12)
        dy = jax.device_put(jax.random.normal(jax.random.PRNGKey(13), x.shape, DTYPE), x.sharding)
        abstract = jax.eval_shape(block.init, jax.random.PRNGKey(14), x)
        shardings = nn.logical_to_mesh_sharding(
            nn.get_partition_spec(abstract), layout["mesh"], layout["rules"]
        )
        variables = nn.unbox(
            jax.jit(block.init, out_shardings=shardings)(jax.random.PRNGKey(14), x)
        )
        if native:
            default_block = block.clone(
                cudnn_native_weight_layout=False, fc2_native_weight_layout=False
            )
            d_abstract = jax.eval_shape(default_block.init, jax.random.PRNGKey(14), x)
            d_shardings = nn.logical_to_mesh_sharding(
                nn.get_partition_spec(d_abstract), layout["mesh"], layout["rules"]
            )
            default_vars = nn.unbox(
                jax.jit(default_block.init, out_shardings=d_shardings)(jax.random.PRNGKey(14), x)
            )
            got = _from_layout(dict(variables["params"]), "native")
            for k in ("wi", "wo", "gate_kernel"):
                np.testing.assert_array_equal(_global(got[k]), _global(default_vars["params"][k]))
        _, b_out, b_gp, b_gx = _value_and_grads(
            lambda p, x: block.apply({"params": p}, x)[0], variables["params"], x, dy
        )
        _, m_out, m_gp, m_gx = _value_and_grads(
            lambda p, x: _moe_apply(layout, p, x, enable=enable, weight_layout=weight_layout),
            variables["params"],
            x,
            dy,
        )
    np.testing.assert_array_equal(_global(b_out), _global(m_out))
    _assert_same(b_gx, m_gx, "x", enable)
    for k in ("wi", "wo", "gate_kernel"):
        _assert_same(b_gp[k], m_gp[k], k, enable)


@pytest.mark.parametrize("enable", [False, True], ids=["flag_off", "flag_on"])
def test_run_to_run_reproducibility(layout, enable):
    """The output is bitwise reproducible; dx and the gate gradient are bitwise with the flag off
    and within ROUTING_GRAD_REPRO_TOL (BF16 last-bit flips) with it on; expert weight gradients within WGRAD_REPRO_TOL.
    """
    with _ctx(layout):
        params = _shard_params(layout, _make_params(layout, 3), "default")
        x = _make_x(layout, 4)
        dy = jax.device_put(jax.random.normal(jax.random.PRNGKey(5), x.shape, DTYPE), x.sharding)
        f = lambda p, x: _moe_apply(layout, p, x, enable=enable, weight_layout="default")
        _, out1, gp1, gx1 = _value_and_grads(f, params, x, dy)
        _, out2, gp2, gx2 = _value_and_grads(f, params, x, dy)
    counts = {k: int(np.sum(_global(gp1[k]) != _global(gp2[k]))) for k in ("wi", "wo")}
    if jax.process_index() == 0:
        print(
            f"[{layout['name']} enable={enable}] run-to-run differing weight-grad elements:"
            f" {counts}",
            flush=True,
        )
    np.testing.assert_array_equal(_global(out1), _global(out2))
    _assert_same(gx1, gx2, "x", enable)
    for k in ("wi", "wo", "gate_kernel"):
        _assert_same(gp1[k], gp2[k], k, enable)


# ----------------------------------------------------------------------------- MXFP8


layout_mr_placeholder = None


def _mxfp8_block(enable, **kw):
    from transformer_engine.common.recipe import MXFP8BlockScaling

    return MoEBlock(
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        intermediate_size=INTER,
        dtype=DTYPE,
        input_axes=("batch", None, None),
        use_cudnn_fusion=False,
        collect_expert_counts=False,
        enable_ep_overlap=enable,
        cudnn_native_weight_layout=True,
        fc2_native_weight_layout=True,
        quantization_recipe=MXFP8BlockScaling(),
        mesh_resource=layout_mr_placeholder,
        **kw,
    )


# MXFP8 vs MXFP8 (both quantize every GEMM operand, at different points: the overlap quantizes
# SwiGLU's output before applying the routing weight in FC2's epilogue, and requantizes dY and the
# activations columnwise for the weight gradients), and each vs the BF16 overlap reference.
MXFP8_TOL = {"out": 1.5e-1, "x": 1.5e-1, "wi": 1.5e-1, "wo": 1.5e-1, "gate_kernel": 1.5e-1}


@pytest.mark.parametrize("quant_before_fsdp_ag", [False, True], ids=["gather_bf16", "gather_mxfp8"])
def test_mxfp8_overlap_matches_flag_off(layout, quant_before_fsdp_ag):
    """MXFP8 recipe: flag-on output and gradients match flag-off MXFP8 and the BF16 layer; the
    MXFP8 overlap calls ran; no CuTeDSL kernel is compiled at execution. With
    quant_before_fsdp_ag the weights are quantized on their FSDP shards and gathered as MXFP8
    (expert dimension sharded over (ep, fsdp) only)."""
    if quant_before_fsdp_ag and layout["name"] != "fsdp2_ep2_expert":
        pytest.skip("quantize-before-gather needs the expert dimension sharded over (ep, fsdp)")
    global layout_mr_placeholder  # pylint: disable=global-statement
    layout_mr_placeholder = layout["mr"]
    with _ctx(layout):
        x = _make_x(layout, 21)
        dy = jax.device_put(jax.random.normal(jax.random.PRNGKey(22), x.shape, DTYPE), x.sharding)
        off_block = _mxfp8_block(False, quant_before_fsdp_ag=quant_before_fsdp_ag)
        on_block = _mxfp8_block(True, quant_before_fsdp_ag=quant_before_fsdp_ag)
        # init traces and runs the forward: the first trace compiles this config's kernels.
        before = _COMPILE_COUNTER["calls"]
        abstract = jax.eval_shape(on_block.init, jax.random.PRNGKey(23), x)
        shardings = nn.logical_to_mesh_sharding(
            nn.get_partition_spec(abstract), layout["mesh"], layout["rules"]
        )
        variables = nn.unbox(
            jax.jit(on_block.init, out_shardings=shardings)(jax.random.PRNGKey(23), x)
        )

        def run(block):
            def loss(v, x):
                out = block.apply(v, x)[0]
                return jnp.sum(out.astype(jnp.float32) * dy.astype(jnp.float32)), out

            fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))
            (_, out), (gv, gx) = fn(variables, x)
            return fn, out, gv["params"], gx

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*EP overlap skipped.*")
            compiled = (
                jax.jit(
                    jax.value_and_grad(
                        lambda v, x: jnp.sum(on_block.apply(v, x)[0].astype(jnp.float32)),
                        argnums=(0, 1),
                    )
                )
                .lower(variables, x)
                .compile()
            )
            after_first = _COMPILE_COUNTER["calls"]
            jax.block_until_ready(compiled(variables, x))
            on_fn, out, gp, gx = run(on_block)
            jax.block_until_ready(on_fn(variables, x))
            jax.block_until_ready(on_fn(variables, x))
            after_more = _COMPILE_COUNTER["calls"]
        if jax.process_index() == 0:
            print(
                f"[mxfp8 {layout['name']} qbfag={quant_before_fsdp_ag}] CuTeDSL entries: init +"
                f" fwd/bwd trace and compile {after_first - before}, 4 executions after that"
                f" {after_more - after_first}",
                flush=True,
            )
        assert after_more == after_first, "a CuTeDSL kernel was compiled at execution"
        _, ref_out, ref_gp, ref_gx = run(off_block)
        bf16_block = on_block.clone(quantization_recipe=None, quant_before_fsdp_ag=False)
        _, b_out, b_gp, b_gx = run(bf16_block)
        hlo = on_fn.lower(variables, x).as_text()
        modes = _overlap_scaling_modes(hlo)
        for target in OVERLAP_TARGETS:
            assert modes.get(target) == {
                MXFP8_SCALING_MODE
            }, f"{target}: scaling modes {modes.get(target)}; the MXFP8 overlap path did not run"

    def errs_of(o, g, gx_, r_out, r_gp, r_gx):
        return {
            "out": _rel_err(o, r_out),
            "x": _rel_err(gx_, r_gx),
            "wi": _rel_err(g["wi"], r_gp["wi"]),
            "wo": _rel_err(g["wo"], r_gp["wo"]),
            "gate_kernel": _rel_err(g["gate_kernel"], r_gp["gate_kernel"]),
        }

    on_vs_off = errs_of(out, gp, gx, ref_out, ref_gp, ref_gx)
    on_vs_bf16 = errs_of(out, gp, gx, b_out, b_gp, b_gx)
    off_vs_bf16 = errs_of(ref_out, ref_gp, ref_gx, b_out, b_gp, b_gx)
    if jax.process_index() == 0:
        tag = f"[mxfp8 {layout['name']} qbfag={quant_before_fsdp_ag}]"
        print(f"{tag} on vs off {on_vs_off}", flush=True)
        print(f"{tag} on vs bf16 {on_vs_bf16}", flush=True)
        print(f"{tag} off vs bf16 {off_vs_bf16}", flush=True)
    for k in on_vs_off:
        # The overlap's MXFP8 error against the BF16 layer must be of the order of the flag-off
        # MXFP8 path's own error, and the two MXFP8 paths must agree within MXFP8_TOL.
        assert on_vs_bf16[k] <= 1.5 * off_vs_bf16[k] + 1e-2, (k, on_vs_bf16[k], off_vs_bf16[k])
        assert on_vs_off[k] <= MXFP8_TOL[k], (k, on_vs_off[k])

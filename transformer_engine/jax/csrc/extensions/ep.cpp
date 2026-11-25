/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifdef NVTE_WITH_NCCL_EP

#include "transformer_engine/ep.h"

#include <nccl.h>

#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "../extensions.h"
#include "common.h"
#include "ffi_collectives.h"
#include "transformer_engine/ep_gemm.h"
#include "transformer_engine/gemm.h"

namespace transformer_engine {
namespace jax {

// NCCL comm + EPBackend lifetime tracks live JAX executables via XLA stateful FFI.

// All local EP domains share one group configuration.
struct EpBootstrapParams {
  std::array<uint8_t, 128> uid_bytes{};
  int ep_size = 0;
  int rank_within_group = 0;
  int num_experts = 0;
  int max_tokens_per_rank = 0;
  int max_recv_tokens_per_rank = 0;
  int hidden_dim = 0;
  int max_num_sms = 0;
  NVTEDType max_token_dtype = kNVTEBFloat16;
  bool drop_on_overflow = false;
  // Per-token top-k (0 = unset). Lets NCCL EP pick the HT pull-push flow, which the EP/GEMM
  // overlap calls (ep_gemm.h) need for their readiness signals.
  int num_topk = 0;
  // When set, EP borrows XLA's comm (see below): no ncclCommInitRank at
  // bootstrap; nvte_ep_initialize is deferred to the first executable that
  // fetches the borrowed communicator.
  bool borrowed_comm = false;
};

static NVTEEpGroupConfig MakeEpGroupConfig(const EpBootstrapParams& p) {
  return NVTEEpGroupConfig{.struct_size = sizeof(NVTEEpGroupConfig),
                           .ep_size = p.ep_size,
                           .num_experts = p.num_experts,
                           .max_tokens_per_rank = p.max_tokens_per_rank,
                           .max_recv_tokens_per_rank = p.max_recv_tokens_per_rank,
                           .hidden_dim = p.hidden_dim,
                           .num_comm_sms = p.max_num_sms,
                           .max_token_dtype = p.max_token_dtype,
                           .zero_copy = 0,
                           .num_topk = p.num_topk,
                           .drop_on_overflow = p.drop_on_overflow};
}

class EpResources {
 public:
  explicit EpResources(const EpBootstrapParams& p) {
    NVTE_CHECK_CUDA(cudaGetDevice(&device_));
    ncclUniqueId uid;
    std::memcpy(&uid, p.uid_bytes.data(), sizeof(uid));
    NVTE_CHECK_NCCL(ncclCommInitRank(&comm_, p.ep_size, uid, p.rank_within_group));
    NVTEEpGroupConfig cfg = MakeEpGroupConfig(p);
    try {
      nvte_ep_initialize(static_cast<void*>(comm_), &cfg);
    } catch (...) {
      ncclCommDestroy(comm_);
      comm_ = nullptr;
      throw;
    }
  }

  ~EpResources() {
    if (comm_ == nullptr) return;
    // Destroy EP groups before their communicators; destructors must not throw.
    try {
      nvte_ep_shutdown();
    } catch (...) {
    }
    // Destroy the communicator on its owning device, then restore the caller's device.
    int caller_device = 0;
    if (cudaGetDevice(&caller_device) != cudaSuccess) return;
    if (caller_device != device_ && cudaSetDevice(device_) != cudaSuccess) return;
    ncclCommDestroy(comm_);
    if (caller_device != device_) cudaSetDevice(caller_device);
  }

  EpResources(const EpResources&) = delete;
  EpResources& operator=(const EpResources&) = delete;

  ncclComm_t comm() const { return comm_; }

 private:
  int device_{-1};
  ncclComm_t comm_{nullptr};
};

struct EpInstanceState {
  static ::xla::ffi::TypeId id;
  static ::xla::ffi::TypeInfo info;
  std::shared_ptr<EpResources> resources;
};

::xla::ffi::TypeId EpInstanceState::id = {};
::xla::ffi::TypeInfo EpInstanceState::info = ::xla::ffi::MakeTypeInfo<EpInstanceState>();

namespace {

std::mutex g_ep_mu;
EpBootstrapParams g_ep_params;
bool g_ep_params_set = false;
std::weak_ptr<EpResources> g_ep_resources_weak;
// Python-held anchor so trace-time handle_mem allocs find EPBackend ready.
std::shared_ptr<EpResources> g_ep_resources_anchor;
// Borrowed-comm path: EPBackend is initialized once per device from a borrowed communicator.
enum class XlaDeviceEpState { kInProgress, kDone };
std::unordered_map<int, XlaDeviceEpState> g_ep_xla_device_state;
std::condition_variable g_ep_xla_device_cv;
// Communicator mode cannot change within a process.
std::optional<bool> g_ep_last_borrowed_comm;

#ifdef XLA_FFI_COLLECTIVES_AVAILABLE
// Idempotently initialize EPBackend on a borrowed communicator. Safe to call
// from every executable that fetches the comm; only the first call per device initializes.
void EnsureEpBackendFromBorrowedComm(ncclComm_t comm) {
  int device = 0;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  EpBootstrapParams params;
  {
    std::unique_lock<std::mutex> lock(g_ep_mu);
    g_ep_xla_device_cv.wait(lock, [device] {
      auto it = g_ep_xla_device_state.find(device);
      return it == g_ep_xla_device_state.end() || it->second != XlaDeviceEpState::kInProgress;
    });
    auto it = g_ep_xla_device_state.find(device);
    if (it != g_ep_xla_device_state.end() && it->second == XlaDeviceEpState::kDone) return;
    NVTE_CHECK(g_ep_params_set, "EP bootstrap params not set before borrowing XLA comm.");
    params = g_ep_params;
    g_ep_xla_device_state[device] = XlaDeviceEpState::kInProgress;
  }
  // Release g_ep_mu so local peers can enter collective initialization.
  NVTEEpGroupConfig cfg = MakeEpGroupConfig(params);
  try {
    nvte_ep_initialize(static_cast<void*>(comm), &cfg);
  } catch (...) {
    std::lock_guard<std::mutex> lock(g_ep_mu);
    g_ep_xla_device_state.erase(device);
    g_ep_xla_device_cv.notify_all();
    throw;
  }
  std::lock_guard<std::mutex> lock(g_ep_mu);
  g_ep_xla_device_state[device] = XlaDeviceEpState::kDone;
  g_ep_xla_device_cv.notify_all();
}
#endif  // collectives header available

std::shared_ptr<EpResources> AcquireEpResources() {
  std::lock_guard<std::mutex> lock(g_ep_mu);
  NVTE_CHECK(g_ep_params_set,
             "EP bootstrap params not set; call transformer_engine_jax."
             "set_ep_bootstrap_params() (typically via ep_bootstrap) first.");
  auto sp = g_ep_resources_weak.lock();
  if (sp) return sp;
  sp = std::make_shared<EpResources>(g_ep_params);
  g_ep_resources_weak = sp;
  return sp;
}

}  // namespace

// top_k, dispatch_output_per_expert_alignment, and topk_idx_dtype are baked as static
// FFI attributes; every op passes them to the *_v2 C API as NVTEEpLayerConfig.
struct EpConfig {
  int64_t top_k;
  int64_t dispatch_output_per_expert_alignment;
  int64_t topk_idx_dtype;
};

// ── Bootstrap helpers ─────────────────────────────────────────────────────────

// Caches uid + group config and eagerly creates the NCCL comm (ranks
// synchronize via the UID broadcast).
void SetEpBootstrapParams(pybind11::bytes unique_id_bytes_obj, int ep_size, int rank_within_group,
                          int num_experts, int max_tokens_per_rank, int max_recv_tokens_per_rank,
                          int hidden_dim, int max_num_sms, int max_token_dtype,
                          bool drop_on_overflow, bool borrowed_comm, int num_topk) {
  std::string uid_str = unique_id_bytes_obj;
  NVTE_CHECK(static_cast<int>(uid_str.size()) >= 128,
             "unique_id_bytes must be at least 128 bytes (ncclUniqueId size).");
  std::shared_ptr<EpResources> anchor;
  {
    std::lock_guard<std::mutex> lock(g_ep_mu);
    NVTE_CHECK(!g_ep_resources_anchor && g_ep_xla_device_state.empty(),
               "EP bootstrap already initialized; call release_ep_resources() before re-init.");
    NVTE_CHECK(!g_ep_last_borrowed_comm.has_value() || *g_ep_last_borrowed_comm == borrowed_comm,
               "EP bootstrap: switching between self-hosted NCCL and XLA-borrowed-comm within one "
               "process is not supported (a spurious combine-backward numeric mismatch has been "
               "observed); restart the process to change comm path.");
    g_ep_last_borrowed_comm = borrowed_comm;
    std::memcpy(g_ep_params.uid_bytes.data(), uid_str.data(), 128);
    g_ep_params.ep_size = ep_size;
    g_ep_params.rank_within_group = rank_within_group;
    g_ep_params.num_experts = num_experts;
    g_ep_params.max_tokens_per_rank = max_tokens_per_rank;
    g_ep_params.max_recv_tokens_per_rank = max_recv_tokens_per_rank;
    g_ep_params.hidden_dim = hidden_dim;
    g_ep_params.max_num_sms = max_num_sms;
    g_ep_params.max_token_dtype = static_cast<NVTEDType>(max_token_dtype);
    g_ep_params.drop_on_overflow = drop_on_overflow;
    g_ep_params.num_topk = num_topk;
    g_ep_params.borrowed_comm = borrowed_comm;
    g_ep_params_set = true;
  }
  // Borrowed-comm path defers NCCL init to the first executable; nothing eager.
  if (borrowed_comm) return;
  // Acquire outside the lock: EpResources ctor runs ncclCommInitRank which is
  // a collective and may block on peer ranks.
  anchor = AcquireEpResources();
  std::lock_guard<std::mutex> lock(g_ep_mu);
  g_ep_resources_anchor = std::move(anchor);
}

// Drops the anchor; comm tears down once the last executable also releases.
// For the borrowed-comm path, tears down each EPBackend while its borrowed comm is
// still alive (call from ep_finalize, not atexit).
void ReleaseEpResources() {
  std::shared_ptr<EpResources> to_drop;
  bool had_xla_devices = false;
  {
    std::lock_guard<std::mutex> lock(g_ep_mu);
    to_drop = std::move(g_ep_resources_anchor);
    had_xla_devices = !g_ep_xla_device_state.empty();
    g_ep_xla_device_state.clear();
    g_ep_xla_device_cv.notify_all();
  }
  // Keep collective teardown outside g_ep_mu so local peers can make progress.
  if (had_xla_devices) nvte_ep_shutdown();
  // to_drop dtor runs outside the lock.
}

// Atexit-only variant: drops the self-hosted anchor (its destructor safely
// tears down its own NCCL comm while the process is still alive) but never
// touches a borrowed backend. Ep bootstrap/finalize can switch comm paths
// mid-process (e.g. tests), so by the time atexit fires the live backend may
// be borrowed-comm; XLA may already be tearing down that comm, and calling
// nvte_ep_shutdown() on it here would race that teardown. A borrowed backend
// left live at process exit is handled separately by EPBackend's own
// atexit-safe static destructor (skips NCCL calls).
void ReleaseEpResourcesAtExit() {
  std::shared_ptr<EpResources> to_drop;
  {
    std::lock_guard<std::mutex> lock(g_ep_mu);
    to_drop = std::move(g_ep_resources_anchor);
  }
  // to_drop dtor runs outside the lock.
}

size_t EpHandleMemSize(int top_k, size_t dispatch_output_per_expert_alignment) {
  NVTEEpLayerConfig layer_cfg{
      .struct_size = sizeof(NVTEEpLayerConfig),
      .top_k = top_k,
      .dispatch_output_per_expert_alignment = dispatch_output_per_expert_alignment};
  return nvte_ep_handle_mem_size(&layer_cfg);
}

pybind11::capsule GetEpInstanceStateTypeIdCapsule() {
  return pybind11::capsule(static_cast<void*>(&EpInstanceState::id), "xla.ffi.type_id");
}

pybind11::capsule GetEpInstanceStateTypeInfoCapsule() {
  return pybind11::capsule(static_cast<void*>(&EpInstanceState::info), "xla.ffi.type_info");
}

// ── Instantiate handler ─────────────────────────────────────────────────────

static ::xla::ffi::ErrorOr<std::unique_ptr<EpInstanceState>> EpInstantiateImpl() {
  auto state = std::make_unique<EpInstanceState>();
  {
    // Borrowed-comm path: XLA owns the comm, so there is nothing self-hosted to
    // acquire or pin per-executable; EPBackend is initialized by the bootstrap op.
    std::lock_guard<std::mutex> lock(g_ep_mu);
    if (g_ep_params.borrowed_comm) return state;
  }
  try {
    state->resources = AcquireEpResources();
  } catch (const std::exception& e) {
    return ::xla::ffi::Unexpected(ffi_internal_error("EP instantiate failed: ", e));
  }
  return state;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpInstantiateHandler, EpInstantiateImpl, FFI::BindInstantiate());

// ── ep_prepare ────────────────────────────────────────────────────────────────

Error_Type EpPrepareFFI(cudaStream_t stream, EpInstanceState* ep_state, Buffer_Type topk_idx,
                        Result_Type recv_tokens_per_expert, Result_Type total_recv_tokens,
                        Result_Type handle_mem, EpConfig config) {
  (void)ep_state;  // lifetime only.
  auto topk_dims = topk_idx.dimensions();
  NVTE_CHECK(topk_dims.size() >= 2,
             "topk_idx must be at least 2D [..., top_k], got ndim=", topk_dims.size());
  auto idx_etype = topk_idx.element_type();
  NVTE_CHECK(idx_etype == ::xla::ffi::DataType::S64 || idx_etype == ::xla::ffi::DataType::S32,
             "topk_idx must be int32 or int64; got element_type=", static_cast<int>(idx_etype));

  std::vector<size_t> topk_shape = {product(topk_dims, 0, topk_dims.size() - 1),
                                    static_cast<size_t>(topk_dims.back())};
  auto topk_idx_ = TensorWrapper(topk_idx.untyped_data(), topk_shape,
                                 convert_ffi_datatype_to_te_dtype(idx_etype));

  std::vector<size_t> tc_shape = {static_cast<size_t>(recv_tokens_per_expert->element_count())};
  auto recv_tokens_per_expert_ =
      TensorWrapper(recv_tokens_per_expert->untyped_data(), tc_shape, DType::kInt32);

  std::vector<size_t> trt_shape = {static_cast<size_t>(total_recv_tokens->element_count())};
  auto total_recv_tokens_ =
      TensorWrapper(total_recv_tokens->untyped_data(), trt_shape, DType::kInt32);

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem->element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem->untyped_data(), hm_shape, DType::kByte);

  NVTEEpLayerConfig layer_cfg{.struct_size = sizeof(NVTEEpLayerConfig),
                              .top_k = static_cast<int>(config.top_k),
                              .dispatch_output_per_expert_alignment =
                                  static_cast<size_t>(config.dispatch_output_per_expert_alignment),
                              .topk_idx_dtype = static_cast<NVTEDType>(config.topk_idx_dtype)};
  return ffi_guard("EP prepare failed: ", [&] {
    nvte_ep_prepare(handle_mem_.data(), topk_idx_.data(), recv_tokens_per_expert_.data(),
                    total_recv_tokens_.data(), &layer_cfg, stream);
  });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpPrepareHandler, EpPrepareFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()                     // stream
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()  // EP state
                                  .Arg<Buffer_Type>()                         // topk_idx
                                  .Ret<Buffer_Type>()  // recv_tokens_per_expert
                                  .Ret<Buffer_Type>()  // total_recv_tokens
                                  .Ret<Buffer_Type>()  // handle_mem
                                  .Attrs<EpConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_dispatch ───────────────────────────────────────────────────────────────

struct EpDispatchBuffers {
  TensorWrapper topk_idx, tokens, topk_weights, recv_tokens, recv_topk_weights;
};

// MXFP8 block size along the hidden dim.
constexpr size_t kEpMxfp8BlockSize = 32;

// Wrap the topk_idx / tokens / topk_weights inputs and the recv_tokens / recv_topk_weights
// outputs shared by dispatch and fused prepare + dispatch. E4M3 tokens are MXFP8: the compact
// (unswizzled) E8M0 scales tokens_scale_inv / recv_scale_inv ride with the data. For other
// token dtypes the scale buffers are empty placeholders and ignored.
EpDispatchBuffers WrapEpDispatchBuffers(Buffer_Type topk_idx, Buffer_Type tokens,
                                        Buffer_Type tokens_scale_inv, Buffer_Type topk_weights,
                                        Result_Type recv_tokens, Result_Type recv_scale_inv,
                                        Result_Type recv_topk_weights, const EpConfig& config) {
  auto token_dims = tokens.dimensions();
  NVTE_CHECK(token_dims.size() >= 2,
             "tokens must be at least 2D [..., H], got ndim=", token_dims.size());

  auto idx_dims = topk_idx.dimensions();
  NVTE_CHECK(idx_dims.size() >= 2,
             "topk_idx must be at least 2D [..., top_k], got ndim=", idx_dims.size());
  auto idx_etype = topk_idx.element_type();
  NVTE_CHECK(idx_etype == ::xla::ffi::DataType::S64 || idx_etype == ::xla::ffi::DataType::S32,
             "topk_idx must be int32 or int64; got element_type=", static_cast<int>(idx_etype));
  NVTE_CHECK(static_cast<int64_t>(idx_dims.back()) == config.top_k, "top_k attr (", config.top_k,
             ") must match topk_idx last dim (", idx_dims.back(), ")");
  std::vector<size_t> idx_shape = {product(idx_dims, 0, idx_dims.size() - 1),
                                   static_cast<size_t>(idx_dims.back())};
  auto topk_idx_ = TensorWrapper(topk_idx.untyped_data(), idx_shape,
                                 convert_ffi_datatype_to_te_dtype(idx_etype));

  const size_t T_flat = product(token_dims, 0, token_dims.size() - 1);
  const size_t H = static_cast<size_t>(token_dims.back());
  std::vector<size_t> tok_shape = {T_flat, H};
  auto token_dtype = convert_ffi_datatype_to_te_dtype(tokens.element_type());
  const bool is_mxfp8 = token_dtype == DType::kFloat8E4M3;
  const size_t sc_cols = H / kEpMxfp8BlockSize;
  TensorWrapper tokens_(is_mxfp8 ? NVTE_MXFP8_1D_SCALING : NVTE_DELAYED_TENSOR_SCALING);
  tokens_.set_rowwise_data(tokens.untyped_data(), token_dtype, tok_shape);
  if (is_mxfp8) {
    NVTE_CHECK(H % kEpMxfp8BlockSize == 0, "MXFP8 dispatch requires hidden dim divisible by ",
               kEpMxfp8BlockSize, "; got H=", H);
    NVTE_CHECK(tokens_scale_inv.element_type() == ::xla::ffi::DataType::F8E8M0FNU,
               "tokens_scale_inv must be float8_e8m0fnu for MXFP8 dispatch");
    NVTE_CHECK(tokens_scale_inv.element_count() == T_flat * sc_cols,
               "tokens_scale_inv must have T * H/32 elements; got ",
               tokens_scale_inv.element_count(), " for T=", T_flat, ", H=", H);
    tokens_.set_rowwise_scale_inv(tokens_scale_inv.untyped_data(), DType::kFloat8E8M0,
                                  std::vector<size_t>{T_flat, sc_cols});
  }

  auto tw_dims = topk_weights.dimensions();
  NVTE_CHECK(tw_dims.size() >= 2,
             "topk_weights must be at least 2D [..., top_k], got ndim=", tw_dims.size());
  NVTE_CHECK(topk_weights.element_type() == ::xla::ffi::DataType::F32,
             "topk_weights must be float32; got element_type=",
             static_cast<int>(topk_weights.element_type()));
  std::vector<size_t> tw_shape = {product(tw_dims, 0, tw_dims.size() - 1),
                                  static_cast<size_t>(tw_dims.back())};
  auto topk_weights_ = TensorWrapper(topk_weights.untyped_data(), tw_shape, DType::kFloat32);

  // recv_tokens: flatten any leading dims into recv_capacity_per_rank.
  auto recv_dims = recv_tokens->dimensions();
  NVTE_CHECK(recv_dims.size() >= 2,
             "recv_tokens must be at least 2D [..., recv_pr, H]; got ndim=", recv_dims.size());
  const size_t recv_capacity_per_rank = product(recv_dims, 0, recv_dims.size() - 1);
  std::vector<size_t> recv_shape = {recv_capacity_per_rank, H};
  TensorWrapper recv_tokens_(is_mxfp8 ? NVTE_MXFP8_1D_SCALING : NVTE_DELAYED_TENSOR_SCALING);
  recv_tokens_.set_rowwise_data(recv_tokens->untyped_data(), token_dtype, recv_shape);
  if (is_mxfp8) {
    NVTE_CHECK(recv_scale_inv->element_count() == recv_capacity_per_rank * sc_cols,
               "recv_scale_inv must have recv_pr * H/32 elements; got ",
               recv_scale_inv->element_count());
    recv_tokens_.set_rowwise_scale_inv(recv_scale_inv->untyped_data(), DType::kFloat8E8M0,
                                       std::vector<size_t>{recv_capacity_per_rank, sc_cols});
  }

  auto recv_w_dims = recv_topk_weights->dimensions();
  NVTE_CHECK(recv_w_dims.size() >= 1,
             "recv_topk_weights must be at least 1D; got ndim=", recv_w_dims.size());
  const size_t recv_w_total = product(recv_w_dims, 0, recv_w_dims.size());
  NVTE_CHECK(recv_w_total == recv_capacity_per_rank, "recv_topk_weights total (", recv_w_total,
             ") must match recv_tokens recv_pr (", recv_capacity_per_rank, ")");
  std::vector<size_t> recv_w_shape = {recv_capacity_per_rank};
  auto recv_topk_weights_ =
      TensorWrapper(recv_topk_weights->untyped_data(), recv_w_shape, DType::kFloat32);

  return {std::move(topk_idx_), std::move(tokens_), std::move(topk_weights_),
          std::move(recv_tokens_), std::move(recv_topk_weights_)};
}

Error_Type EpDispatchFFI(cudaStream_t stream, EpInstanceState* ep_state, Buffer_Type handle_mem,
                         Buffer_Type topk_idx, Buffer_Type tokens, Buffer_Type tokens_scale_inv,
                         Buffer_Type topk_weights, Result_Type recv_tokens,
                         Result_Type recv_scale_inv, Result_Type recv_topk_weights,
                         Result_Type out_handle_mem, EpConfig config) {
  (void)ep_state;
  (void)out_handle_mem;  // aliased to handle_mem; dispatch writes there directly
  auto bufs = WrapEpDispatchBuffers(topk_idx, tokens, tokens_scale_inv, topk_weights, recv_tokens,
                                    recv_scale_inv, recv_topk_weights, config);

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);
  NVTECommWindow no_win{nullptr, 0};
  NVTEEpLayerConfig layer_cfg{.struct_size = sizeof(NVTEEpLayerConfig),
                              .top_k = static_cast<int>(config.top_k),
                              .dispatch_output_per_expert_alignment =
                                  static_cast<size_t>(config.dispatch_output_per_expert_alignment),
                              .topk_idx_dtype = static_cast<NVTEDType>(config.topk_idx_dtype)};
  return ffi_guard("EP dispatch failed: ", [&] {
    nvte_ep_dispatch_v2(handle_mem_.data(), bufs.topk_idx.data(), bufs.tokens.data(), no_win,
                        bufs.topk_weights.data(), no_win, bufs.recv_tokens.data(), no_win,
                        bufs.recv_topk_weights.data(), no_win, &layer_cfg, stream);
  });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpDispatchHandler, EpDispatchFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()                     // stream
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()  // EP state
                                  .Arg<Buffer_Type>()                         // handle_mem
                                  .Arg<Buffer_Type>()                         // topk_idx
                                  .Arg<Buffer_Type>()                         // tokens
                                  .Arg<Buffer_Type>()                         // tokens_scale_inv
                                  .Arg<Buffer_Type>()                         // topk_weights
                                  .Ret<Buffer_Type>()                         // recv_tokens
                                  .Ret<Buffer_Type>()                         // recv_scale_inv
                                  .Ret<Buffer_Type>()                         // recv_topk_weights
                                  .Ret<Buffer_Type>()  // out_handle_mem (aliased to handle_mem)
                                  .Attrs<EpConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_prepare_and_dispatch ───────────────────────────────────────────────────

Error_Type EpPrepareAndDispatchFFI(
    cudaStream_t stream, EpInstanceState* ep_state, Buffer_Type topk_idx, Buffer_Type tokens,
    Buffer_Type tokens_scale_inv, Buffer_Type topk_weights, Result_Type recv_tokens,
    Result_Type recv_scale_inv, Result_Type recv_topk_weights, Result_Type recv_tokens_per_expert,
    Result_Type total_recv_tokens, Result_Type handle_mem, EpConfig config) {
  (void)ep_state;
  auto bufs = WrapEpDispatchBuffers(topk_idx, tokens, tokens_scale_inv, topk_weights, recv_tokens,
                                    recv_scale_inv, recv_topk_weights, config);

  std::vector<size_t> tc_shape = {static_cast<size_t>(recv_tokens_per_expert->element_count())};
  auto recv_tokens_per_expert_ =
      TensorWrapper(recv_tokens_per_expert->untyped_data(), tc_shape, DType::kInt32);

  std::vector<size_t> trt_shape = {static_cast<size_t>(total_recv_tokens->element_count())};
  auto total_recv_tokens_ =
      TensorWrapper(total_recv_tokens->untyped_data(), trt_shape, DType::kInt32);

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem->element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem->untyped_data(), hm_shape, DType::kByte);

  NVTEEpLayerConfig layer_cfg{.struct_size = sizeof(NVTEEpLayerConfig),
                              .top_k = static_cast<int>(config.top_k),
                              .dispatch_output_per_expert_alignment =
                                  static_cast<size_t>(config.dispatch_output_per_expert_alignment),
                              .topk_idx_dtype = static_cast<NVTEDType>(config.topk_idx_dtype)};
  NVTECommWindow no_win{nullptr, 0};
  return ffi_guard("EP prepare_and_dispatch failed: ", [&] {
    nvte_ep_prepare_and_dispatch(handle_mem_.data(), bufs.topk_idx.data(), bufs.tokens.data(),
                                 no_win, bufs.topk_weights.data(), no_win, bufs.recv_tokens.data(),
                                 no_win, bufs.recv_topk_weights.data(), no_win,
                                 recv_tokens_per_expert_.data(), total_recv_tokens_.data(),
                                 &layer_cfg, stream);
  });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpPrepareAndDispatchHandler, EpPrepareAndDispatchFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()                     // stream
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()  // EP state
                                  .Arg<Buffer_Type>()                         // topk_idx
                                  .Arg<Buffer_Type>()                         // tokens
                                  .Arg<Buffer_Type>()                         // tokens_scale_inv
                                  .Arg<Buffer_Type>()                         // topk_weights
                                  .Ret<Buffer_Type>()                         // recv_tokens
                                  .Ret<Buffer_Type>()                         // recv_scale_inv
                                  .Ret<Buffer_Type>()                         // recv_topk_weights
                                  .Ret<Buffer_Type>()  // recv_tokens_per_expert
                                  .Ret<Buffer_Type>()  // total_recv_tokens
                                  .Ret<Buffer_Type>()  // handle_mem
                                  .Attrs<EpConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_combine ────────────────────────────────────────────────────────────────

Error_Type EpCombineFFI(cudaStream_t stream, EpInstanceState* ep_state, Buffer_Type handle_mem,
                        Buffer_Type expert_out, Result_Type result, EpConfig config) {
  (void)ep_state;
  auto eo_dims = expert_out.dimensions();
  NVTE_CHECK(eo_dims.size() >= 2,
             "expert_out must be at least 2D [..., recv_pr, H]; got ndim=", eo_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  const size_t recv_capacity_per_rank = product(eo_dims, 0, eo_dims.size() - 1);
  const size_t H = static_cast<size_t>(eo_dims.back());
  std::vector<size_t> eo_shape = {recv_capacity_per_rank, H};
  auto eo_dtype = convert_ffi_datatype_to_te_dtype(expert_out.element_type());
  auto expert_out_ = TensorWrapper(expert_out.untyped_data(), eo_shape, eo_dtype);

  auto res_dims = result->dimensions();
  NVTE_CHECK(res_dims.size() >= 2,
             "result must be at least 2D [..., H]; got ndim=", res_dims.size());
  const size_t res_T_flat = product(res_dims, 0, res_dims.size() - 1);
  std::vector<size_t> res_shape = {res_T_flat, H};
  auto result_ = TensorWrapper(result->untyped_data(), res_shape, eo_dtype);

  NVTECommWindow no_win{nullptr, 0};
  NVTEEpLayerConfig layer_cfg{.struct_size = sizeof(NVTEEpLayerConfig),
                              .top_k = static_cast<int>(config.top_k),
                              .dispatch_output_per_expert_alignment =
                                  static_cast<size_t>(config.dispatch_output_per_expert_alignment),
                              .topk_idx_dtype = static_cast<NVTEDType>(config.topk_idx_dtype)};
  return ffi_guard("EP combine failed: ", [&] {
    nvte_ep_combine_v2(handle_mem_.data(), expert_out_.data(), no_win, result_.data(), &layer_cfg,
                       stream);
  });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpCombineHandler, EpCombineFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()                     // stream
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()  // EP state
                                  .Arg<Buffer_Type>()                         // handle_mem
                                  .Arg<Buffer_Type>()                         // expert_out
                                  .Ret<Buffer_Type>()                         // result
                                  .Attrs<EpConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_dispatch_bwd ───────────────────────────────────────────────────────────

Error_Type EpDispatchBwdFFI(cudaStream_t stream, EpInstanceState* ep_state, Buffer_Type handle_mem,
                            Buffer_Type grad, Buffer_Type g_recv_topk_weights,
                            Result_Type grad_tokens, Result_Type grad_topk_weights,
                            EpConfig config) {
  (void)ep_state;
  auto grad_dims = grad.dimensions();
  NVTE_CHECK(grad_dims.size() >= 2,
             "grad must be at least 2D [..., recv_pr, H]; got ndim=", grad_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  const size_t recv_capacity_per_rank = product(grad_dims, 0, grad_dims.size() - 1);
  const size_t H = static_cast<size_t>(grad_dims.back());
  std::vector<size_t> g_shape = {recv_capacity_per_rank, H};
  auto g_dtype = convert_ffi_datatype_to_te_dtype(grad.element_type());
  auto grad_ = TensorWrapper(grad.untyped_data(), g_shape, g_dtype);

  auto gw_dims = g_recv_topk_weights.dimensions();
  NVTE_CHECK(
      gw_dims.size() >= 1,
      "g_recv_topk_weights rank must flatten to recv_capacity_per_rank; got ndim=", gw_dims.size());
  const size_t gw_total = product(gw_dims, 0, gw_dims.size());
  NVTE_CHECK(gw_total == recv_capacity_per_rank, "g_recv_topk_weights total (", gw_total,
             ") must match grad recv_pr (", recv_capacity_per_rank, ")");
  std::vector<size_t> gw_shape = {recv_capacity_per_rank};
  auto g_recv_topk_weights_ =
      TensorWrapper(g_recv_topk_weights.untyped_data(), gw_shape, DType::kFloat32);

  auto out_dims = grad_tokens->dimensions();
  NVTE_CHECK(out_dims.size() >= 2,
             "grad_tokens must be at least 2D [..., H], got ndim=", out_dims.size());
  const size_t T_flat = product(out_dims, 0, out_dims.size() - 1);
  std::vector<size_t> out_shape = {T_flat, H};
  auto grad_tokens_ = TensorWrapper(grad_tokens->untyped_data(), out_shape, g_dtype);

  auto gtw_dims = grad_topk_weights->dimensions();
  NVTE_CHECK(gtw_dims.size() >= 2,
             "grad_topk_weights must be at least 2D [..., top_k]; got ndim=", gtw_dims.size());
  const size_t gtw_T_flat = product(gtw_dims, 0, gtw_dims.size() - 1);
  NVTE_CHECK(gtw_T_flat == T_flat, "grad_topk_weights leading-dim product (", gtw_T_flat,
             ") must equal grad_tokens leading-dim product (", T_flat, ")");
  const size_t top_k = static_cast<size_t>(gtw_dims.back());
  NVTE_CHECK(static_cast<int64_t>(top_k) == config.top_k, "top_k attr (", config.top_k,
             ") must match grad_topk_weights last dim (", top_k, ")");
  std::vector<size_t> gtw_shape = {T_flat, top_k};
  auto grad_topk_weights_ =
      TensorWrapper(grad_topk_weights->untyped_data(), gtw_shape, DType::kFloat32);

  NVTECommWindow no_win{nullptr, 0};
  NVTEEpLayerConfig layer_cfg{.struct_size = sizeof(NVTEEpLayerConfig),
                              .top_k = static_cast<int>(config.top_k),
                              .dispatch_output_per_expert_alignment =
                                  static_cast<size_t>(config.dispatch_output_per_expert_alignment),
                              .topk_idx_dtype = static_cast<NVTEDType>(config.topk_idx_dtype)};
  return ffi_guard("EP dispatch_bwd failed: ", [&] {
    nvte_ep_dispatch_bwd_v2(handle_mem_.data(), grad_.data(), no_win, g_recv_topk_weights_.data(),
                            no_win, grad_tokens_.data(), grad_topk_weights_.data(), &layer_cfg,
                            stream);
  });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpDispatchBwdHandler, EpDispatchBwdFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()                     // stream
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()  // EP state
                                  .Arg<Buffer_Type>()                         // handle_mem
                                  .Arg<Buffer_Type>()  // grad (w.r.t. recv_tokens)
                                  .Arg<Buffer_Type>()  // g_recv_topk_weights
                                  .Ret<Buffer_Type>()  // grad_tokens
                                  .Ret<Buffer_Type>()  // grad_topk_weights
                                  .Attrs<EpConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_combine_bwd ────────────────────────────────────────────────────────────

Error_Type EpCombineBwdFFI(cudaStream_t stream, EpInstanceState* ep_state, Buffer_Type handle_mem,
                           Buffer_Type grad, Result_Type grad_expert_out, EpConfig config) {
  (void)ep_state;
  auto grad_dims = grad.dimensions();
  NVTE_CHECK(grad_dims.size() >= 2,
             "grad must be at least 2D [..., H], got ndim=", grad_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  const size_t T_flat = product(grad_dims, 0, grad_dims.size() - 1);
  const size_t H = static_cast<size_t>(grad_dims.back());
  std::vector<size_t> g_shape = {T_flat, H};
  auto g_dtype = convert_ffi_datatype_to_te_dtype(grad.element_type());
  auto grad_ = TensorWrapper(grad.untyped_data(), g_shape, g_dtype);

  auto out_dims = grad_expert_out->dimensions();
  NVTE_CHECK(out_dims.size() >= 2,
             "grad_expert_out must be at least 2D [..., recv_pr, H]; got ndim=", out_dims.size());
  const size_t recv_capacity_per_rank = product(out_dims, 0, out_dims.size() - 1);
  const size_t out_H = static_cast<size_t>(out_dims.back());
  NVTE_CHECK(out_H == H, "grad_expert_out hidden dim (", out_H, ") must match grad H (", H, ")");
  std::vector<size_t> out_shape = {recv_capacity_per_rank, H};
  auto grad_expert_out_ = TensorWrapper(grad_expert_out->untyped_data(), out_shape, g_dtype);

  NVTECommWindow no_win{nullptr, 0};
  NVTEEpLayerConfig layer_cfg{.struct_size = sizeof(NVTEEpLayerConfig),
                              .top_k = static_cast<int>(config.top_k),
                              .dispatch_output_per_expert_alignment =
                                  static_cast<size_t>(config.dispatch_output_per_expert_alignment),
                              .topk_idx_dtype = static_cast<NVTEDType>(config.topk_idx_dtype)};
  return ffi_guard("EP combine_bwd failed: ", [&] {
    nvte_ep_combine_bwd_v2(handle_mem_.data(), grad_.data(), no_win, grad_expert_out_.data(),
                           no_win, &layer_cfg, stream);
  });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpCombineBwdHandler, EpCombineBwdFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()                     // stream
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()  // EP state
                                  .Arg<Buffer_Type>()                         // handle_mem
                                  .Arg<Buffer_Type>()  // grad (w.r.t. result)
                                  .Ret<Buffer_Type>()  // grad_expert_out
                                  .Attrs<EpConfig>(),
                              FFI_CudaGraph_Traits);

// ── EP/GEMM overlap (FusedMoE) ──────────────────────────────────────────────
//
// The four calls of nvte_ep_*_gemm_* (ep_gemm.h): NCCL EP dispatch/combine overlapped with the
// CuTeDSL grouped GEMMs through device readiness signals, all on XLA's stream. One handler per
// call; the `scaling_mode` attribute selects BF16 (NO_SCALING) or MXFP8 (MXFP8_1D_SCALING). Scale
// buffers are always present and are zero-size under NO_SCALING; so are the MXFP8-only
// columnwise buffers. MXFP8 operand forms follow ep_gemm.h: tokens, received rows, dY and
// grad_expert_out carry compact rowwise scales [rows, cols / 32]; weights and GEMM inputs/outputs
// carry GEMM-swizzled scales (flat, rows * cols / 32); the backward reads the weights columnwise.
// Shapes are per device (the Python side calls these inside shard_map): T local tokens, R the
// recv capacity, E local experts, H hidden, F the expert FFN size. FC1 weights are [E, 2F, H]
// with gate/up in alternating 32-row blocks; FC2 weights are [E, H, F]. The workspace is an XLA
// output buffer of EpOverlapWorkspaceSize bytes, read only during the call.

namespace {

NVTEEpLayerConfig MakeLayerCfg(const EpConfig& config) {
  return NVTEEpLayerConfig{.struct_size = sizeof(NVTEEpLayerConfig),
                           .top_k = static_cast<int>(config.top_k),
                           .dispatch_output_per_expert_alignment =
                               static_cast<size_t>(config.dispatch_output_per_expert_alignment),
                           .topk_idx_dtype = static_cast<NVTEDType>(config.topk_idx_dtype)};
}

std::vector<size_t> Dims2D(const std::vector<size_t>& d) {
  NVTE_CHECK(!d.empty(), "expected at least 1D buffer");
  if (d.size() == 1) return {d[0]};
  size_t lead = 1;
  for (size_t i = 0; i + 1 < d.size(); ++i) lead *= d[i];
  return {lead, d.back()};
}

template <typename B>
std::vector<size_t> DimsOf(const B& b) {
  std::vector<size_t> out;
  for (auto x : b.dimensions()) out.push_back(static_cast<size_t>(x));
  return out;
}

TensorWrapper WrapArg(Buffer_Type b, bool flatten = true) {
  auto d = DimsOf(b);
  return TensorWrapper(b.untyped_data(), flatten ? Dims2D(d) : d,
                       convert_ffi_datatype_to_te_dtype(b.element_type()));
}

TensorWrapper WrapRet(Result_Type b, bool flatten = true) {
  auto d = DimsOf(*b);
  return TensorWrapper(b->untyped_data(), flatten ? Dims2D(d) : d,
                       convert_ffi_datatype_to_te_dtype(b->element_type()));
}

TensorWrapper WrapBytes(void* ptr, size_t n) {
  return TensorWrapper(ptr, std::vector<size_t>{n}, DType::kByte);
}

// expert_token_offsets is allocated as [E + 4] int32 and used from element 3, so that
// expert_token_offsets[1:] starts 16-byte aligned (XLA buffers are at least 16-byte aligned).
TensorWrapper WrapOffsets(void* base, size_t padded_count) {
  NVTE_CHECK(padded_count >= 5, "expert_token_offsets must be [E + 4] int32");
  return TensorWrapper(static_cast<int32_t*>(base) + 3, std::vector<size_t>{padded_count - 3},
                       DType::kInt32);
}

}  // namespace

size_t EpOverlapWorkspaceSize(size_t hidden_dim, size_t ffn_dim, size_t num_local_experts,
                              size_t recv_capacity, bool with_weight_grad, bool mxfp8, int top_k,
                              size_t dispatch_output_per_expert_alignment, int topk_idx_dtype) {
  NVTEEpLayerConfig layer_cfg{
      .struct_size = sizeof(NVTEEpLayerConfig),
      .top_k = top_k,
      .dispatch_output_per_expert_alignment = dispatch_output_per_expert_alignment,
      .topk_idx_dtype = static_cast<NVTEDType>(topk_idx_dtype)};
  return nvte_ep_overlap_workspace_size(hidden_dim, ffn_dim, num_local_experts, recv_capacity,
                                        with_weight_grad ? 1 : 0, mxfp8 ? 1 : 0, &layer_cfg);
}

void EpOverlapCompile(size_t hidden_dim, size_t ffn_dim, size_t num_local_experts,
                      size_t recv_capacity, bool with_weight_grad, bool fp32_weight_grad,
                      bool mxfp8, int top_k, size_t dispatch_output_per_expert_alignment,
                      int topk_idx_dtype) {
  NVTEEpLayerConfig layer_cfg{
      .struct_size = sizeof(NVTEEpLayerConfig),
      .top_k = top_k,
      .dispatch_output_per_expert_alignment = dispatch_output_per_expert_alignment,
      .topk_idx_dtype = static_cast<NVTEDType>(topk_idx_dtype)};
  nvte_ep_overlap_compile(hidden_dim, ffn_dim, num_local_experts, recv_capacity,
                          with_weight_grad ? 1 : 0, fp32_weight_grad ? 1 : 0, mxfp8 ? 1 : 0,
                          &layer_cfg);
}

namespace {

// Layer config from the handlers' attributes.
NVTEEpLayerConfig MakeLayerCfg(int64_t top_k, int64_t alignment, int64_t topk_idx_dtype) {
  return MakeLayerCfg(EpConfig{top_k, alignment, topk_idx_dtype});
}

// True for MXFP8; BF16 is NO_SCALING. Other modes are not supported by the overlap calls.
bool IsMxfp8(JAXX_Scaling_Mode scaling_mode) {
  NVTE_CHECK(scaling_mode == JAXX_Scaling_Mode::NO_SCALING ||
                 scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING,
             "the EP overlap calls support NO_SCALING (BF16) or MXFP8_1D_SCALING, got ",
             static_cast<int>(scaling_mode));
  return scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING;
}

// Operand of an overlap call. BF16: the plain rowwise tensor. MXFP8: E4M3 rowwise and/or
// columnwise data (null: absent) with E8M0 scales, compact [rows, cols / 32] or GEMM-swizzled
// (flat rows * cols / 32); columnwise scales are always swizzled.
TensorWrapper MakeOperand(bool mxfp8, void* row, void* row_scale, void* col, void* col_scale,
                          const std::vector<size_t>& shape, DType plain_dtype, bool swizzled) {
  if (!mxfp8) {
    return TensorWrapper(row, shape, plain_dtype);
  }
  TensorWrapper t(NVTE_MXFP8_1D_SCALING);
  size_t rows = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i) rows *= shape[i];
  const size_t cols = shape.back();
  if (row != nullptr) {
    t.set_rowwise_data(row, DType::kFloat8E4M3, shape);
    t.set_rowwise_scale_inv(
        row_scale, DType::kFloat8E8M0,
        swizzled ? std::vector<size_t>{rows * cols / 32} : std::vector<size_t>{rows, cols / 32});
  }
  if (col != nullptr) {
    t.set_columnwise_data(col, DType::kFloat8E4M3, shape);
    t.set_columnwise_scale_inv(col_scale, DType::kFloat8E8M0,
                               std::vector<size_t>{rows * cols / 32});
  }
  t.set_with_gemm_swizzled_scales(swizzled);
  return t;
}

TensorWrapper WrapF32(void* p, size_t n) {
  return TensorWrapper(p, std::vector<size_t>{n}, DType::kFloat32);
}

TensorWrapper WrapI32(void* p, size_t n) {
  return TensorWrapper(p, std::vector<size_t>{n}, DType::kInt32);
}

DType DtypeOf(Buffer_Type b) { return convert_ffi_datatype_to_te_dtype(b.element_type()); }
DType DtypeOf(Result_Type b) { return convert_ffi_datatype_to_te_dtype(b->element_type()); }

}  // namespace

// Dispatch overlapped with FC1 + SwiGLU. BF16: act_out = silu(gate) * up * prob. MXFP8: act_out =
// silu(gate) * up (the routing weight is applied to FC2's output).
Error_Type EpDispatchGemmFwdFFI(
    cudaStream_t stream, EpInstanceState* ep_state, Buffer_Type topk_idx, Buffer_Type tokens,
    Buffer_Type tokens_scale_inv, Buffer_Type topk_weights, Buffer_Type weight,
    Buffer_Type weight_scale_inv, Result_Type recv_tokens, Result_Type recv_scale_inv,
    Result_Type recv_topk_weights, Result_Type gemm_out, Result_Type act_out,
    Result_Type act_out_scale_inv, Result_Type recv_tokens_per_expert,
    Result_Type expert_token_offsets, Result_Type handle_mem, Result_Type workspace, int64_t top_k,
    int64_t dispatch_output_per_expert_alignment, int64_t topk_idx_dtype,
    JAXX_Scaling_Mode scaling_mode) {
  (void)ep_state;
  auto layer_cfg = MakeLayerCfg(top_k, dispatch_output_per_expert_alignment, topk_idx_dtype);
  return ffi_guard("EP dispatch_gemm_fwd failed: ", [&] {
    const bool mxfp8 = IsMxfp8(scaling_mode);
    NVTE_CHECK(mxfp8 || tokens.element_type() == ::xla::ffi::DataType::BF16,
               "EP overlap: BF16 tokens must be bfloat16");
    NVTE_CHECK(topk_weights.element_type() == ::xla::ffi::DataType::F32,
               "EP overlap: topk_weights must be float32");
    auto topk_idx_ = WrapArg(topk_idx);
    auto tokens_ = MakeOperand(mxfp8, tokens.untyped_data(), tokens_scale_inv.untyped_data(),
                               nullptr, nullptr, Dims2D(DimsOf(tokens)), DtypeOf(tokens), false);
    auto topk_weights_ = WrapArg(topk_weights);
    auto weight_ = MakeOperand(mxfp8, weight.untyped_data(), weight_scale_inv.untyped_data(),
                               nullptr, nullptr, DimsOf(weight), DtypeOf(weight), true);
    auto recv_tokens_ =
        MakeOperand(mxfp8, recv_tokens->untyped_data(), recv_scale_inv->untyped_data(), nullptr,
                    nullptr, Dims2D(DimsOf(*recv_tokens)), DtypeOf(recv_tokens), false);
    auto recv_w_ = WrapF32(recv_topk_weights->untyped_data(), recv_topk_weights->element_count());
    auto gemm_out_ = WrapRet(gemm_out);
    auto act_out_ = MakeOperand(mxfp8, act_out->untyped_data(), act_out_scale_inv->untyped_data(),
                                nullptr, nullptr, Dims2D(DimsOf(*act_out)), DtypeOf(act_out), true);
    auto counts_ =
        WrapI32(recv_tokens_per_expert->untyped_data(), recv_tokens_per_expert->element_count());
    auto offsets_ =
        WrapOffsets(expert_token_offsets->untyped_data(), expert_token_offsets->element_count());
    auto handle_mem_ = WrapBytes(handle_mem->untyped_data(), handle_mem->size_bytes());
    auto ws_ = WrapBytes(workspace->untyped_data(), workspace->size_bytes());
    nvte_ep_dispatch_gemm_fwd(
        handle_mem_.data(), topk_idx_.data(), tokens_.data(), NVTECommWindow{nullptr, 0},
        topk_weights_.data(), weight_.data(), recv_tokens_.data(), recv_w_.data(), gemm_out_.data(),
        act_out_.data(), counts_.data(), offsets_.data(), nullptr, ws_.data(), &layer_cfg, stream);
  });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpDispatchGemmFwdHandler, EpDispatchGemmFwdFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()
                                  .Arg<Buffer_Type>()  // topk_idx [T, K]
                                  .Arg<Buffer_Type>()  // tokens [T, H]
                                  .Arg<Buffer_Type>()  // tokens_scale_inv [T, H/32] (MXFP8)
                                  .Arg<Buffer_Type>()  // topk_weights [T, K] f32
                                  .Arg<Buffer_Type>()  // weight [E, 2F, H]
                                  .Arg<Buffer_Type>()  // weight_scale_inv, swizzled (MXFP8)
                                  .Ret<Buffer_Type>()  // recv_tokens [R, H]
                                  .Ret<Buffer_Type>()  // recv_scale_inv [R, H/32] (MXFP8)
                                  .Ret<Buffer_Type>()  // recv_topk_weights [R] f32
                                  .Ret<Buffer_Type>()  // gemm_out [R, 2F] bf16
                                  .Ret<Buffer_Type>()  // act_out [R, F]
                                  .Ret<Buffer_Type>()  // act_out_scale_inv, swizzled (MXFP8)
                                  .Ret<Buffer_Type>()  // recv_tokens_per_expert [E] i32
                                  .Ret<Buffer_Type>()  // expert_token_offsets [E + 4] i32
                                  .Ret<Buffer_Type>()  // handle_mem
                                  .Ret<Buffer_Type>()  // workspace
                                  .Attr<int64_t>("top_k")
                                  .Attr<int64_t>("dispatch_output_per_expert_alignment")
                                  .Attr<int64_t>("topk_idx_dtype")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode"),
                              FFI_CudaGraph_Traits);

// FC2 overlapped with the combine. BF16: the routing weights are already in act_out, so
// recv_topk_weights is not read. MXFP8: FC2's output rows are scaled by recv_topk_weights.
Error_Type EpCombineGemmFwdFFI(cudaStream_t stream, EpInstanceState* ep_state,
                               Buffer_Type handle_mem, Buffer_Type act_out,
                               Buffer_Type act_out_scale_inv, Buffer_Type recv_topk_weights,
                               Buffer_Type expert_token_offsets, Buffer_Type weight,
                               Buffer_Type weight_scale_inv, Result_Type expert_out,
                               Result_Type result, Result_Type workspace, int64_t top_k,
                               int64_t dispatch_output_per_expert_alignment, int64_t topk_idx_dtype,
                               JAXX_Scaling_Mode scaling_mode) {
  (void)ep_state;
  auto layer_cfg = MakeLayerCfg(top_k, dispatch_output_per_expert_alignment, topk_idx_dtype);
  return ffi_guard("EP combine_gemm_fwd failed: ", [&] {
    const bool mxfp8 = IsMxfp8(scaling_mode);
    auto handle_mem_ = WrapBytes(handle_mem.untyped_data(), handle_mem.size_bytes());
    auto act_out_ = MakeOperand(mxfp8, act_out.untyped_data(), act_out_scale_inv.untyped_data(),
                                nullptr, nullptr, Dims2D(DimsOf(act_out)), DtypeOf(act_out), true);
    auto recv_w_ = WrapF32(recv_topk_weights.untyped_data(), recv_topk_weights.element_count());
    auto offsets_ =
        WrapOffsets(expert_token_offsets.untyped_data(), expert_token_offsets.element_count());
    auto weight_ = MakeOperand(mxfp8, weight.untyped_data(), weight_scale_inv.untyped_data(),
                               nullptr, nullptr, DimsOf(weight), DtypeOf(weight), true);
    auto expert_out_ = WrapRet(expert_out);
    auto result_ = WrapRet(result);
    auto ws_ = WrapBytes(workspace->untyped_data(), workspace->size_bytes());
    nvte_ep_combine_gemm_fwd(handle_mem_.data(), act_out_.data(), mxfp8 ? recv_w_.data() : nullptr,
                             offsets_.data(), weight_.data(), expert_out_.data(), result_.data(),
                             ws_.data(), &layer_cfg, stream);
  });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpCombineGemmFwdHandler, EpCombineGemmFwdFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()
                                  .Arg<Buffer_Type>()  // handle_mem
                                  .Arg<Buffer_Type>()  // act_out [R, F]
                                  .Arg<Buffer_Type>()  // act_out_scale_inv, swizzled (MXFP8)
                                  .Arg<Buffer_Type>()  // recv_topk_weights [R] f32 (MXFP8 reads)
                                  .Arg<Buffer_Type>()  // expert_token_offsets [E + 4] i32
                                  .Arg<Buffer_Type>()  // weight [E, H, F]
                                  .Arg<Buffer_Type>()  // weight_scale_inv, swizzled (MXFP8)
                                  .Ret<Buffer_Type>()  // expert_out [R, H] bf16
                                  .Ret<Buffer_Type>()  // result [T, H] bf16
                                  .Ret<Buffer_Type>()  // workspace
                                  .Attr<int64_t>("top_k")
                                  .Attr<int64_t>("dispatch_output_per_expert_alignment")
                                  .Attr<int64_t>("topk_idx_dtype")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode"),
                              FFI_CudaGraph_Traits);

// Combine backward overlapped with FC2 dgrad + SwiGLU backward, and the FC2 weight gradient.
// BF16: wgrad_input is the forward act_out; the MXFP8-only outputs are zero-size. MXFP8: the weight
// is the columnwise copy; the call writes grad_expert_out (compact rowwise + columnwise),
// grad_gemm_out (rowwise + columnwise) and the columnwise activation, which the weight gradient
// reads (wgrad_input is then zero-size).
Error_Type EpCombineGemmBwdFFI(
    cudaStream_t stream, EpInstanceState* ep_state, Buffer_Type handle_mem, Buffer_Type grad_result,
    Buffer_Type grad_result_scale_inv, Buffer_Type gemm_out, Buffer_Type recv_topk_weights,
    Buffer_Type recv_tokens_per_expert, Buffer_Type weight, Buffer_Type weight_scale_inv,
    Buffer_Type wgrad_input, Result_Type grad_expert_out, Result_Type grad_expert_out_scale_inv,
    Result_Type grad_expert_out_col, Result_Type grad_expert_out_col_scale_inv,
    Result_Type grad_gemm_out, Result_Type grad_gemm_out_scale_inv, Result_Type grad_gemm_out_col,
    Result_Type grad_gemm_out_col_scale_inv, Result_Type grad_recv_topk_weights,
    Result_Type act_col, Result_Type act_col_scale_inv, Result_Type weight_grad,
    Result_Type workspace, int64_t top_k, int64_t dispatch_output_per_expert_alignment,
    int64_t topk_idx_dtype, JAXX_Scaling_Mode scaling_mode) {
  (void)ep_state;
  auto layer_cfg = MakeLayerCfg(top_k, dispatch_output_per_expert_alignment, topk_idx_dtype);
  return ffi_guard("EP combine_gemm_bwd failed: ", [&] {
    const bool mxfp8 = IsMxfp8(scaling_mode);
    auto handle_mem_ = WrapBytes(handle_mem.untyped_data(), handle_mem.size_bytes());
    auto grad_result_ =
        MakeOperand(mxfp8, grad_result.untyped_data(), grad_result_scale_inv.untyped_data(),
                    nullptr, nullptr, Dims2D(DimsOf(grad_result)), DtypeOf(grad_result), false);
    auto gemm_out_ = WrapArg(gemm_out);
    auto recv_w_ = WrapF32(recv_topk_weights.untyped_data(), recv_topk_weights.element_count());
    auto counts_ =
        WrapI32(recv_tokens_per_expert.untyped_data(), recv_tokens_per_expert.element_count());
    auto weight_ =
        mxfp8 ? MakeOperand(true, nullptr, nullptr, weight.untyped_data(),
                            weight_scale_inv.untyped_data(), DimsOf(weight), DtypeOf(weight), true)
              : WrapArg(weight, false);
    auto grad_expert_out_ = MakeOperand(
        mxfp8, grad_expert_out->untyped_data(), grad_expert_out_scale_inv->untyped_data(),
        mxfp8 ? grad_expert_out_col->untyped_data() : nullptr,
        grad_expert_out_col_scale_inv->untyped_data(), Dims2D(DimsOf(*grad_expert_out)),
        DtypeOf(grad_expert_out), false);
    auto grad_gemm_out_ =
        MakeOperand(mxfp8, grad_gemm_out->untyped_data(), grad_gemm_out_scale_inv->untyped_data(),
                    mxfp8 ? grad_gemm_out_col->untyped_data() : nullptr,
                    grad_gemm_out_col_scale_inv->untyped_data(), Dims2D(DimsOf(*grad_gemm_out)),
                    DtypeOf(grad_gemm_out), true);
    auto grad_recv_w_ =
        WrapF32(grad_recv_topk_weights->untyped_data(), grad_recv_topk_weights->element_count());
    // MXFP8: the activation and the weight gradient's input are the columnwise act_col.
    TensorWrapper act_, wgrad_input_;
    if (mxfp8) {
      act_ = MakeOperand(true, nullptr, nullptr, act_col->untyped_data(),
                         act_col_scale_inv->untyped_data(), Dims2D(DimsOf(*act_col)),
                         DtypeOf(act_col), true);
      wgrad_input_ = MakeOperand(true, nullptr, nullptr, act_col->untyped_data(),
                                 act_col_scale_inv->untyped_data(), Dims2D(DimsOf(*act_col)),
                                 DtypeOf(act_col), true);
    } else {
      wgrad_input_ = WrapArg(wgrad_input);
    }
    auto weight_grad_ = WrapRet(weight_grad, false);
    auto ws_ = WrapBytes(workspace->untyped_data(), workspace->size_bytes());
    nvte_ep_combine_gemm_bwd(handle_mem_.data(), grad_result_.data(), NVTECommWindow{nullptr, 0},
                             gemm_out_.data(), recv_w_.data(), counts_.data(), weight_.data(),
                             wgrad_input_.data(), grad_expert_out_.data(), grad_gemm_out_.data(),
                             grad_recv_w_.data(), mxfp8 ? act_.data() : nullptr,
                             weight_grad_.data(), 0, ws_.data(), &layer_cfg, stream);
  });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpCombineGemmBwdHandler, EpCombineGemmBwdFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()
                                  .Arg<Buffer_Type>()  // handle_mem
                                  .Arg<Buffer_Type>()  // grad_result [T, H]
                                  .Arg<Buffer_Type>()  // grad_result_scale_inv [T, H/32] (MXFP8)
                                  .Arg<Buffer_Type>()  // gemm_out [R, 2F] bf16
                                  .Arg<Buffer_Type>()  // recv_topk_weights [R] f32
                                  .Arg<Buffer_Type>()  // recv_tokens_per_expert [E] i32
                                  .Arg<Buffer_Type>()  // weight [E, H, F] (MXFP8: columnwise)
                                  .Arg<Buffer_Type>()  // weight_scale_inv, swizzled (MXFP8)
                                  .Arg<Buffer_Type>()  // wgrad_input = act_out [R, F] (BF16)
                                  .Ret<Buffer_Type>()  // grad_expert_out [R, H]
                                  .Ret<Buffer_Type>()  // its compact scales (MXFP8)
                                  .Ret<Buffer_Type>()  // its columnwise copy (MXFP8)
                                  .Ret<Buffer_Type>()  // its swizzled scales (MXFP8)
                                  .Ret<Buffer_Type>()  // grad_gemm_out [R, 2F]
                                  .Ret<Buffer_Type>()  // its swizzled scales (MXFP8)
                                  .Ret<Buffer_Type>()  // its columnwise copy (MXFP8)
                                  .Ret<Buffer_Type>()  // its swizzled scales (MXFP8)
                                  .Ret<Buffer_Type>()  // grad_recv_topk_weights [R] f32
                                  .Ret<Buffer_Type>()  // act columnwise [R, F] (MXFP8)
                                  .Ret<Buffer_Type>()  // its swizzled scales (MXFP8)
                                  .Ret<Buffer_Type>()  // weight_grad [E, H, F]
                                  .Ret<Buffer_Type>()  // workspace
                                  .Attr<int64_t>("top_k")
                                  .Attr<int64_t>("dispatch_output_per_expert_alignment")
                                  .Attr<int64_t>("topk_idx_dtype")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode"),
                              FFI_CudaGraph_Traits);

// FC1 dgrad and the FC1 weight gradient overlapped with the dispatch backward (a combine of
// gradients). MXFP8: grad_gemm_out comes rowwise + columnwise, the weight columnwise; wgrad_input
// is the forward's rowwise recv_tokens (compact scales) and the call writes its columnwise copy
// into recv_col; the wrapper carries the swizzled flag (the weight gradient reads columnwise).
Error_Type EpDispatchGemmBwdFFI(
    cudaStream_t stream, EpInstanceState* ep_state, Buffer_Type handle_mem,
    Buffer_Type grad_gemm_out, Buffer_Type grad_gemm_out_scale_inv, Buffer_Type grad_gemm_out_col,
    Buffer_Type grad_gemm_out_col_scale_inv, Buffer_Type grad_recv_topk_weights,
    Buffer_Type recv_tokens_per_expert, Buffer_Type weight, Buffer_Type weight_scale_inv,
    Buffer_Type recv_tokens, Buffer_Type recv_scale_inv, Result_Type grad_recv_tokens,
    Result_Type grad_tokens, Result_Type grad_topk_weights, Result_Type weight_grad,
    Result_Type recv_col, Result_Type recv_col_scale_inv, Result_Type workspace, int64_t top_k,
    int64_t dispatch_output_per_expert_alignment, int64_t topk_idx_dtype,
    JAXX_Scaling_Mode scaling_mode) {
  (void)ep_state;
  auto layer_cfg = MakeLayerCfg(top_k, dispatch_output_per_expert_alignment, topk_idx_dtype);
  return ffi_guard("EP dispatch_gemm_bwd failed: ", [&] {
    const bool mxfp8 = IsMxfp8(scaling_mode);
    auto handle_mem_ = WrapBytes(handle_mem.untyped_data(), handle_mem.size_bytes());
    auto grad_gemm_out_ =
        MakeOperand(mxfp8, grad_gemm_out.untyped_data(), grad_gemm_out_scale_inv.untyped_data(),
                    mxfp8 ? grad_gemm_out_col.untyped_data() : nullptr,
                    grad_gemm_out_col_scale_inv.untyped_data(), Dims2D(DimsOf(grad_gemm_out)),
                    DtypeOf(grad_gemm_out), true);
    auto grad_recv_w_ =
        WrapF32(grad_recv_topk_weights.untyped_data(), grad_recv_topk_weights.element_count());
    auto counts_ =
        WrapI32(recv_tokens_per_expert.untyped_data(), recv_tokens_per_expert.element_count());
    auto weight_ =
        mxfp8 ? MakeOperand(true, nullptr, nullptr, weight.untyped_data(),
                            weight_scale_inv.untyped_data(), DimsOf(weight), DtypeOf(weight), true)
              : WrapArg(weight, false);
    auto wgrad_input_ =
        MakeOperand(mxfp8, recv_tokens.untyped_data(), recv_scale_inv.untyped_data(),
                    mxfp8 ? recv_col->untyped_data() : nullptr, recv_col_scale_inv->untyped_data(),
                    Dims2D(DimsOf(recv_tokens)), DtypeOf(recv_tokens), true);
    if (mxfp8) {
      // The forward's rowwise scales are compact; the wrapper's flag is for the columnwise copy.
      const size_t rows = Dims2D(DimsOf(recv_tokens))[0];
      const size_t cols = Dims2D(DimsOf(recv_tokens))[1];
      wgrad_input_.set_rowwise_scale_inv(recv_scale_inv.untyped_data(), DType::kFloat8E8M0,
                                         std::vector<size_t>{rows, cols / 32});
    }
    auto grad_recv_tokens_ = WrapRet(grad_recv_tokens);
    auto grad_tokens_ = WrapRet(grad_tokens);
    auto grad_topk_ = WrapRet(grad_topk_weights);
    auto weight_grad_ = WrapRet(weight_grad, false);
    auto ws_ = WrapBytes(workspace->untyped_data(), workspace->size_bytes());
    nvte_ep_dispatch_gemm_bwd(handle_mem_.data(), grad_gemm_out_.data(), grad_recv_w_.data(),
                              counts_.data(), weight_.data(), wgrad_input_.data(),
                              grad_recv_tokens_.data(), grad_tokens_.data(), grad_topk_.data(),
                              weight_grad_.data(), 0, ws_.data(), &layer_cfg, stream);
  });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpDispatchGemmBwdHandler, EpDispatchGemmBwdFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()
                                  .Arg<Buffer_Type>()  // handle_mem
                                  .Arg<Buffer_Type>()  // grad_gemm_out [R, 2F]
                                  .Arg<Buffer_Type>()  // its swizzled scales (MXFP8)
                                  .Arg<Buffer_Type>()  // its columnwise copy (MXFP8)
                                  .Arg<Buffer_Type>()  // its swizzled scales (MXFP8)
                                  .Arg<Buffer_Type>()  // grad_recv_topk_weights [R] f32
                                  .Arg<Buffer_Type>()  // recv_tokens_per_expert [E] i32
                                  .Arg<Buffer_Type>()  // weight [E, 2F, H] (MXFP8: columnwise)
                                  .Arg<Buffer_Type>()  // weight_scale_inv, swizzled (MXFP8)
                                  .Arg<Buffer_Type>()  // wgrad_input = recv_tokens [R, H]
                                  .Arg<Buffer_Type>()  // recv_scale_inv [R, H/32] (MXFP8)
                                  .Ret<Buffer_Type>()  // grad_recv_tokens [R, H] bf16
                                  .Ret<Buffer_Type>()  // grad_tokens [T, H] bf16
                                  .Ret<Buffer_Type>()  // grad_topk_weights [T, K] f32
                                  .Ret<Buffer_Type>()  // weight_grad [E, 2F, H]
                                  .Ret<Buffer_Type>()  // recv_tokens columnwise [R, H] (MXFP8)
                                  .Ret<Buffer_Type>()  // its swizzled scales (MXFP8)
                                  .Ret<Buffer_Type>()  // workspace
                                  .Attr<int64_t>("top_k")
                                  .Attr<int64_t>("dispatch_output_per_expert_alignment")
                                  .Attr<int64_t>("topk_idx_dtype")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode"),
                              FFI_CudaGraph_Traits);

// -- Borrowed-comm path -------------------------------------------------------
//
// Instead of a self-hosted ncclCommInitRank, EPBackend borrows the communicator
// XLA already owns for the EP replica groups. A one-shot bootstrap op fetches it
// and initializes the backend once; the per-step ops then share the same FFI
// targets as the self-hosted path (see EpInstantiateImpl). Auto-selected from
// Python (see tex.ep.use_nccl_comm_from_xla). The prepare stage is the generic
// FfiRequestCliqueHandler (see ffi_collectives.cpp); only the EP-specific
// execute stage lives here.
#ifdef XLA_FFI_COLLECTIVES_AVAILABLE

// Execute stage: fetch the borrowed comm and initialize EPBackend once. The
// token buffer is copied straight through only to give the op an input->output
// data dependency, so it stays ordered on the borrowed comm's stream.
Error_Type EpBootstrapBorrowedCommFFI(cudaStream_t stream, EpInstanceState* ep_state,
                                      FfiCollectivesCtx coll, Buffer_Type token, Result_Type out,
                                      Span_Type<int64_t> replica_groups, int64_t group_size,
                                      int64_t communication_id) {
  (void)ep_state;
  auto groups = ffi_collectives::ReplicaGroupsFromFlat(replica_groups.begin(),
                                                       replica_groups.size(), group_size);
  auto comm_or = ffi_collectives::GetComm(coll, groups, communication_id);
  if (comm_or.has_error()) return comm_or.error();
  ncclComm_t comm = comm_or.value();
  NVTE_CHECK(comm != nullptr, "XLA returned a null EP communicator.");
  try {
    EnsureEpBackendFromBorrowedComm(comm);
  } catch (const std::exception& e) {
    return ffi_internal_error("EP borrowed-comm bootstrap failed: ", e);
  }
  const size_t bytes = token.size_bytes();
  if (bytes > 0) {
    NVTE_CHECK_CUDA(cudaMemcpyAsync(out->untyped_data(), token.untyped_data(), bytes,
                                    cudaMemcpyDeviceToDevice, stream));
  }
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpBootstrapBorrowedCommHandler, EpBootstrapBorrowedCommFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()                     // stream
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()  // EP state
                                  .Ctx<::xla::ffi::Extension<FfiCollectives>>()
                                  .Arg<Buffer_Type>()  // token (identity in)
                                  .Ret<Buffer_Type>()  // token (identity out)
                                  .Attr<Span_Type<int64_t>>("replica_groups")
                                  .Attr<int64_t>("group_size")
                                  .Attr<int64_t>("communication_id"));

#endif  // collectives header available

}  // namespace jax
}  // namespace transformer_engine

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(
    transformer_engine::jax::EpConfig, ::xla::ffi::StructMember<int64_t>("top_k"),
    ::xla::ffi::StructMember<int64_t>("dispatch_output_per_expert_alignment"),
    ::xla::ffi::StructMember<int64_t>("topk_idx_dtype"));

#endif  // NVTE_WITH_NCCL_EP

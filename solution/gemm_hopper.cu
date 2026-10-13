#include "megafmm/kernels.h"

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

namespace megafmm {
namespace {

enum class Operation { CanImplement, Workspace, Launch };

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_cutlass(cutlass::Status status, const char* operation) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cutlassGetStatusString(status));
}

bool supported_shape(const Problem& p, int variant) {
  if (variant < 0 || variant > 2 || p.trans_a || p.trans_b) return false;
  if (p.dtype != DType::BF16 && p.dtype != DType::FP16) return false;
  if (p.m <= 0 || p.n <= 0 || p.k <= 0 ||
      p.m > INT_MAX || p.n > INT_MAX || p.k > INT_MAX) return false;
  if (p.lda < p.k || p.ldb < p.n || p.ldd < p.n ||
      (p.beta != 0.f && p.ldc < p.n)) return false;
  // cuTensorMapEncodeTiled bounds global strides to fewer than 2^40 bytes.
  constexpr int64_t max_input_stride = int64_t(1) << 39;
  constexpr int64_t max_output_stride = int64_t(1) << 38;
  if (p.lda >= max_input_stride || p.ldb >= max_input_stride ||
      p.ldd >= max_output_stride ||
      (p.beta != 0.f && p.ldc >= max_output_stride)) return false;
  // A/B use 16-byte TMA accesses; B's contiguous dimension is N.  The
  // FP32 epilogue has four-element alignment.  These are stride and view
  // constraints, independent of whether the CTA tile divides the shape.
  return p.k % 8 == 0 && p.n % 8 == 0 &&
         p.lda % 8 == 0 && p.ldb % 8 == 0 && p.ldd % 4 == 0 &&
         (p.beta == 0.f || p.ldc % 4 == 0);
}

bool aligned16(const void* pointer) {
  return pointer && (reinterpret_cast<std::uintptr_t>(pointer) & 15) == 0;
}

cutlass::KernelHardwareInfo hardware_info() {
  // The persistent scheduler needs the current device's actual SM count.
  // Cache the attribute per host thread/device; querying it does not allocate
  // device memory or synchronize the caller's stream.
  thread_local int cached_device = -1;
  thread_local int cached_sm_count = 0;
  int device = 0;
  check_cuda(cudaGetDevice(&device), "Hopper cudaGetDevice");
  if (cached_device != device) {
    check_cuda(cudaDeviceGetAttribute(&cached_sm_count, cudaDevAttrMultiProcessorCount, device),
               "Hopper SM count query");
    cached_device = device;
  }
  cutlass::KernelHardwareInfo result;
  result.device_id = device;
  result.sm_count = cached_sm_count;
  return result;
}

template <class Input, int BM, int BN, class MainloopSchedule, class EpilogueSchedule>
struct HopperGemm {
  using Tile = cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::_64>;
  using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using RowMajor = cutlass::layout::RowMajor;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
      float, float, float, RowMajor, 4, float, RowMajor, 4,
      EpilogueSchedule>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Input, RowMajor, 8, Input, RowMajor, 8, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, Mainloop, Epilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static typename Gemm::Arguments arguments(
      const Problem& p, const void* A, const void* B, const float* C0,
      float* D, cutlass::KernelHardwareInfo hw) {
    typename Kernel::StrideA sa = cute::make_stride(p.lda, cute::_1{}, int64_t(0));
    // CUTLASS's B tensor is indexed (N,K,L), even for the public K-by-N
    // row-major B: its strides therefore have the opposite order to A.
    typename Kernel::StrideB sb = cute::make_stride(cute::_1{}, p.ldb, int64_t(0));
    typename Kernel::StrideC sc = cute::make_stride(
        p.beta == 0.f ? p.ldd : p.ldc, cute::_1{}, int64_t(0));
    typename Kernel::StrideD sd = cute::make_stride(p.ldd, cute::_1{}, int64_t(0));
    // A beta=0 call may omit C0.  Give tensor-map construction a valid source
    // descriptor, while the epilogue's beta=0 predicate suppresses all C loads.
    const float* source = p.beta == 0.f ? D : C0;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {static_cast<int>(p.m), static_cast<int>(p.n), static_cast<int>(p.k), 1},
        {static_cast<const Input*>(A), sa, static_cast<const Input*>(B), sb},
        {{}, source, sc, D, sd}, hw};
    args.epilogue.thread.alpha = p.alpha;
    args.epilogue.thread.beta = p.beta;
    return args;
  }

  static size_t execute(const Problem& p, Operation operation,
                        const void* A, const void* B, const float* C0,
                        float* D, void* workspace, cudaStream_t stream) {
    // Structural queries never create a tensor map or dereference operands.
    cutlass::KernelHardwareInfo hw;
    hw.device_id = 0;
    hw.sm_count = 1;
    if (operation == Operation::Launch) hw = hardware_info();
    auto args = arguments(p, A, B, C0, D, hw);
    if (operation == Operation::Workspace) return Gemm::get_workspace_size(args);
    auto status = Gemm::can_implement(args);
    if (operation == Operation::CanImplement)
      return status == cutlass::Status::kSuccess ? 1 : 0;
    check_cutlass(status, "Hopper can_implement");
    if (Gemm::get_workspace_size(args) != 0 && !workspace)
      throw std::invalid_argument("Hopper GEMM requires caller-owned workspace");
    Gemm gemm;
    // initialize() builds host-side tensor-map metadata and initializes any
    // external scheduler workspace on this stream.  It does not allocate or
    // synchronize; rebuilding metadata makes changed operand addresses safe.
    check_cutlass(gemm.initialize(args, workspace, stream), "Hopper initialize");
    check_cutlass(gemm.run(stream), "Hopper run");
    return 0;
  }
};

template <class Input>
size_t dispatch_variant(const Problem& p, int variant, Operation operation,
                        const void* A, const void* B, const float* C0,
                        float* D, void* workspace, cudaStream_t stream) {
  switch (variant) {
    case 0:
      return HopperGemm<Input, 128, 128,
          cutlass::gemm::KernelTmaWarpSpecializedCooperative,
          cutlass::epilogue::TmaWarpSpecializedCooperative>::execute(
              p, operation, A, B, C0, D, workspace, stream);
    case 1:
      return HopperGemm<Input, 128, 256,
          cutlass::gemm::KernelTmaWarpSpecializedCooperative,
          cutlass::epilogue::TmaWarpSpecializedCooperative>::execute(
              p, operation, A, B, C0, D, workspace, stream);
    case 2:
      // Cooperative requires BM >= 128.  A 64-row tile uses the legal
      // pingpong mainloop and its non-cooperative warp-specialized epilogue.
      return HopperGemm<Input, 64, 128,
          cutlass::gemm::KernelTmaWarpSpecializedPingpong,
          cutlass::epilogue::TmaWarpSpecialized>::execute(
              p, operation, A, B, C0, D, workspace, stream);
    default: throw std::invalid_argument("unknown Hopper GEMM variant");
  }
}

size_t dispatch_dtype(const Problem& p, int variant, Operation operation,
                      const void* A, const void* B, const float* C0,
                      float* D, void* workspace, cudaStream_t stream) {
  if (p.dtype == DType::BF16)
    return dispatch_variant<cutlass::bfloat16_t>(p, variant, operation, A, B, C0,
                                               D, workspace, stream);
  if (p.dtype == DType::FP16)
    return dispatch_variant<cutlass::half_t>(p, variant, operation, A, B, C0,
                                           D, workspace, stream);
  throw std::invalid_argument("unknown Hopper input dtype");
}

}  // namespace

bool hopper_can_implement(const Problem& p, int variant) {
  if (!supported_shape(p, variant)) return false;
  // Only alignment is inspected by can_implement; actual pointers are checked
  // again at launch, including pointers to Winograd quadrant views.
  auto placeholder = reinterpret_cast<void*>(std::uintptr_t(0x10000));
  return dispatch_dtype(p, variant, Operation::CanImplement, placeholder,
                        placeholder, static_cast<const float*>(placeholder),
                        static_cast<float*>(placeholder), nullptr, nullptr) != 0;
}

size_t hopper_workspace_bytes(const Problem& p, int variant) {
  if (!hopper_can_implement(p, variant))
    throw std::invalid_argument("problem is not supported by Hopper GEMM");
  auto placeholder = reinterpret_cast<void*>(std::uintptr_t(0x10000));
  return dispatch_dtype(p, variant, Operation::Workspace, placeholder,
                        placeholder, static_cast<const float*>(placeholder),
                        static_cast<float*>(placeholder), nullptr, nullptr);
}

void launch_hopper_gemm(const Problem& p, const void* A, const void* B,
                       const float* C0, float* D, void* workspace,
                       cudaStream_t stream, int variant) {
  if (!supported_shape(p, variant))
    throw std::invalid_argument("problem is not supported by Hopper GEMM");
  if (!aligned16(A) || !aligned16(B) || !aligned16(D) ||
      (p.beta != 0.f && !aligned16(C0)))
    throw std::invalid_argument("Hopper TMA operand view requires 16-byte alignment");
  dispatch_dtype(p, variant, Operation::Launch, A, B, C0, D, workspace, stream);
}

}  // namespace megafmm

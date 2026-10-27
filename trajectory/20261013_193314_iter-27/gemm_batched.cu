// SM80 Tensor Core leaf backend for the uniform seven-slot Winograd routes.
// A7/B7 contain seven compact row-major operands in product order, and P
// contains seven compact FP32 products.  Packing and FP32 DAG combination
// remain caller operations; this helper performs exactly one leaf launch.
#include "megafmm/gemm.h"

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace megafmm {
namespace {

constexpr int kSmallLeafBatchCount = 7;
enum class SmallBatchOperation { CanImplement, Launch };

bool small_batch_shape_supported(const Problem& p, int tile_variant) {
  if (tile_variant < 0 || tile_variant > 1 || p.trans_a || p.trans_b ||
      p.beta != 0.f) return false;
  if (p.dtype != DType::BF16 && p.dtype != DType::FP16) return false;
  // The legacy kernel computes ceil-divisions using signed int dimensions.
  if (p.m <= 0 || p.n <= 0 || p.k <= 0 ||
      p.m > INT_MAX - 64 || p.n > INT_MAX - 64 || p.k > INT_MAX - 32)
    return false;
  if (p.lda != p.k || p.ldb != p.n || p.ldd != p.n) return false;
  if (p.k % 8 != 0 || p.n % 8 != 0) return false;
  // GemmBatchedIdentityThreadblockSwizzle maps ceil(N/64) directly to grid.y.
  // CUDA grid.y is at most 65535, even when grid.x can be much larger.
  if ((p.n + 63) / 64 > 65535) return false;
  // Bound the seven-batch byte extents and signed element offsets.  The
  // dimensions above make each multiplication safe before this comparison.
  constexpr int64_t max_elements = INT64_MAX / (kSmallLeafBatchCount * 4);
  return p.m * p.k <= max_elements && p.k * p.n <= max_elements &&
         p.m * p.n <= max_elements;
}

bool small_batch_aligned16(const void* pointer) {
  return pointer && (reinterpret_cast<std::uintptr_t>(pointer) & 15) == 0;
}

void small_batch_check(cutlass::Status status, const char* operation) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cutlassGetStatusString(status));
}

template<class Input, int BM>
struct SmallBatchedLeaf {
  using RowMajor = cutlass::layout::RowMajor;
  // OnlyAlphaScaling makes is_source_needed() compile to false, so no C
  // values are loaded.  A valid P view supplies the legacy API's C TensorRef.
  // The Winograd caller uses alpha=1 and applies alpha/beta in the FP32 DAG.
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      float, 4, float, float,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
  using Gemm = cutlass::gemm::device::GemmBatched<
      Input, RowMajor, Input, RowMajor, float, RowMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<BM, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      OutputOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      3, 8, 8>;
  // Three stages select MmaMultistage/cp.async in pinned CUTLASS 3.5.1.
  // These small tiles require no dynamic shared-memory opt-in in run().
  static_assert(sizeof(typename Gemm::GemmKernel::SharedStorage) < (48 << 10),
                "small batched leaf unexpectedly requires shared-memory opt-in");

  static typename Gemm::Arguments arguments(
      const Problem& p, const void* A7, const void* B7, float* P) {
    return typename Gemm::Arguments{
        {static_cast<int>(p.m), static_cast<int>(p.n), static_cast<int>(p.k)},
        {static_cast<const Input*>(A7), RowMajor(static_cast<int>(p.k))}, p.m * p.k,
        {static_cast<const Input*>(B7), RowMajor(static_cast<int>(p.n))}, p.k * p.n,
        {P, RowMajor(static_cast<int>(p.n))}, p.m * p.n,
        {P, RowMajor(static_cast<int>(p.n))}, p.m * p.n,
        {p.alpha, 0.f}, kSmallLeafBatchCount};
  }

  static bool execute(const Problem& p, const void* A7, const void* B7,
                      float* P, cudaStream_t stream,
                      SmallBatchOperation operation) {
    const auto args = arguments(p, A7, B7, P);
    const auto status = Gemm::can_implement(args);
    if (operation == SmallBatchOperation::CanImplement)
      return status == cutlass::Status::kSuccess;
    small_batch_check(status, "small batched can_implement");
    Gemm gemm;
    // initialize is a host Params construction; workspace size is zero.
    // No allocation, tensor-map construction or host synchronization occurs.
    small_batch_check(gemm.initialize(args, nullptr, stream),
                      "small batched initialize");
    small_batch_check(gemm.run(stream), "small batched run");
    return true;
  }
};

template<class Input>
bool small_batch_dispatch_tile(const Problem& p, const void* A7,
                               const void* B7, float* P, cudaStream_t stream,
                               int tile_variant, SmallBatchOperation operation) {
  if (tile_variant == 0)
    return SmallBatchedLeaf<Input, 64>::execute(p, A7, B7, P, stream, operation);
  return SmallBatchedLeaf<Input, 32>::execute(p, A7, B7, P, stream, operation);
}

bool small_batch_dispatch(const Problem& p, const void* A7, const void* B7,
                          float* P, cudaStream_t stream, int tile_variant,
                          SmallBatchOperation operation) {
  if (p.dtype == DType::BF16)
    return small_batch_dispatch_tile<cutlass::bfloat16_t>(
        p, A7, B7, P, stream, tile_variant, operation);
  return small_batch_dispatch_tile<cutlass::half_t>(
      p, A7, B7, P, stream, tile_variant, operation);
}

}  // namespace

// tile_variant: 0 = CTA64x64x32/128 threads, 1 = CTA32x64x32/64 threads.
bool small_batched_can_implement(const Problem& leaf, int tile_variant) {
  if (!small_batch_shape_supported(leaf, tile_variant)) return false;
  auto* placeholder = reinterpret_cast<void*>(std::uintptr_t(0x10000));
  return small_batch_dispatch(leaf, placeholder, placeholder,
      static_cast<float*>(placeholder), nullptr, tile_variant,
      SmallBatchOperation::CanImplement);
}

size_t small_batched_workspace_bytes(const Problem& leaf, int tile_variant) {
  if (!small_batched_can_implement(leaf, tile_variant))
    throw std::invalid_argument("unsupported compact seven-batch leaf problem");
  // GemmBatched::get_workspace_size in pinned CUTLASS returns zero.
  return 0;
}

void launch_small_batched_gemm(const Problem& leaf, const void* A7,
                             const void* B7, float* P, cudaStream_t stream,
                             int tile_variant) {
  if (!small_batch_shape_supported(leaf, tile_variant))
    throw std::invalid_argument("unsupported compact seven-batch leaf problem");
  if (!small_batch_aligned16(A7) || !small_batch_aligned16(B7) ||
      !small_batch_aligned16(P))
    throw std::invalid_argument("small batched leaf requires 16-byte aligned views");
  small_batch_dispatch(leaf, A7, B7, P, stream, tile_variant,
                       SmallBatchOperation::Launch);
}

}  // namespace megafmm

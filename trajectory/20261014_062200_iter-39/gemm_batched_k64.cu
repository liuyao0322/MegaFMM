// Standalone K64 seven-batch leaf prototype; not connected to the public API
// or build.  H7/H8 use the same CTA32x64x64/warp32x32x64 output ownership.
// H7 selects three-stage MmaMultistage (cp.async); H8 selects two-stage
// MmaPipelined (register global-load double buffering) in CUTLASS 3.5.1.
// A7/B7 contain seven compact row-major low-precision operands; P contains
// seven complete FP32 products.  Packing and the fixed FP32 U DAG remain
// unchanged caller operations, with exactly one leaf launch here.
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

constexpr int kK64LeafBatchCount = 7;
enum class K64BatchOperation { CanImplement, Launch };

bool k64_batch_shape_supported(const Problem& p, int variant) {
  if ((variant != 7 && variant != 8) || p.trans_a || p.trans_b ||
      p.beta != 0.f) return false;
  if (p.dtype != DType::BF16 && p.dtype != DType::FP16) return false;
  // Legacy iterator and ceil-division dimensions use signed int arithmetic.
  if (p.m <= 0 || p.n <= 0 || p.k <= 0 ||
      p.m > INT_MAX - 32 || p.n > INT_MAX - 64 || p.k > INT_MAX - 64)
    return false;
  if (p.lda != p.k || p.ldb != p.n || p.ldd != p.n) return false;
  if (p.k % 8 != 0 || p.n % 8 != 0) return false;
  // Swizzle maps ceil(N/64) directly to grid.y; CUDA grid.y <= 65535.
  if ((p.n + 63) / 64 > 65535) return false;
  // Each product above is representable given the INT_MAX dimension limits.
  // Bound seven-batch byte extents and all signed element offsets as well.
  constexpr int64_t max_elements = INT64_MAX / (kK64LeafBatchCount * 4);
  return p.m * p.k <= max_elements && p.k * p.n <= max_elements &&
         p.m * p.n <= max_elements;
}

bool k64_batch_aligned16(const void* pointer) {
  return pointer && (reinterpret_cast<std::uintptr_t>(pointer) & 15) == 0;
}

void k64_batch_check(cutlass::Status status, const char* operation) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cutlassGetStatusString(status));
}

template<class Input, int Stages>
struct K64BatchedLeaf {
  using RowMajor = cutlass::layout::RowMajor;
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  // OnlyAlphaScaling removes source C loads.  P supplies the legacy API's
  // valid C view; the Winograd caller passes alpha=1 for the complete P DAG.
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      float, 4, float, float,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
  using Gemm = cutlass::gemm::device::GemmBatched<
      Input, RowMajor, Input, RowMajor, float, RowMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape, OutputOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      Stages, 8, 8>;

  static_assert(Stages == 2 || Stages == 3, "expected two/three-stage control");
  static_assert(Gemm::GemmKernel::kThreadCount == 64,
                "K64 leaf must launch exactly two warps");
  static_assert(ThreadblockShape::kM / WarpShape::kM == 1 &&
                ThreadblockShape::kN / WarpShape::kN == 2 &&
                ThreadblockShape::kK / WarpShape::kK == 1,
                "K64 leaf must have two output warps and no split-K partition");
  // MmaPipelined::gemm_iters explicitly does not support two warp K groups.
  // Both BK and warp-K are 64 here, giving four m16n8k16 K groups, rather
  // than changing only Stages on the existing BK32/warp-K32 leaf.
  static_assert(WarpShape::kK / InstructionShape::kK == 4,
                "two-stage pipeline requires four warp K groups here");
  // Input storage is nominally 36/24 KiB for stages3/2, plus any layout
  // padding; this also checks the epilogue union.  No launch-time opt-in.
  static_assert(sizeof(typename Gemm::GemmKernel::SharedStorage) < (48 << 10),
                "K64 leaf unexpectedly requires shared-memory opt-in");

  static typename Gemm::Arguments arguments(
      const Problem& p, const void* A7, const void* B7, float* P) {
    return typename Gemm::Arguments{
        {static_cast<int>(p.m), static_cast<int>(p.n), static_cast<int>(p.k)},
        {static_cast<const Input*>(A7), RowMajor(static_cast<int>(p.k))}, p.m * p.k,
        {static_cast<const Input*>(B7), RowMajor(static_cast<int>(p.n))}, p.k * p.n,
        {P, RowMajor(static_cast<int>(p.n))}, p.m * p.n,
        {P, RowMajor(static_cast<int>(p.n))}, p.m * p.n,
        {p.alpha, 0.f}, kK64LeafBatchCount};
  }

  static bool execute(const Problem& p, const void* A7, const void* B7,
                      float* P, cudaStream_t stream,
                      K64BatchOperation operation) {
    const auto args = arguments(p, A7, B7, P);
    const auto status = Gemm::can_implement(args);
    if (operation == K64BatchOperation::CanImplement)
      return status == cutlass::Status::kSuccess;
    k64_batch_check(status, "K64 batched can_implement");
    Gemm gemm;
    // initialize constructs host Params only.  GemmBatched workspace is zero;
    // this helper performs no allocation, device query or synchronization.
    k64_batch_check(gemm.initialize(args, nullptr, stream),
                    "K64 batched initialize");
    k64_batch_check(gemm.run(stream), "K64 batched run");
    return true;
  }
};

template<class Input>
bool k64_batch_dispatch_variant(const Problem& p, const void* A7,
                                const void* B7, float* P, cudaStream_t stream,
                                int variant, K64BatchOperation operation) {
  if (variant == 7)
    return K64BatchedLeaf<Input, 3>::execute(p, A7, B7, P, stream, operation);
  return K64BatchedLeaf<Input, 2>::execute(p, A7, B7, P, stream, operation);
}

bool k64_batch_dispatch(const Problem& p, const void* A7, const void* B7,
                        float* P, cudaStream_t stream, int variant,
                        K64BatchOperation operation) {
  if (p.dtype == DType::BF16)
    return k64_batch_dispatch_variant<cutlass::bfloat16_t>(
        p, A7, B7, P, stream, variant, operation);
  return k64_batch_dispatch_variant<cutlass::half_t>(
      p, A7, B7, P, stream, variant, operation);
}

}  // namespace

// Standalone helper IDs: 7=K64/stages3, 8=K64/stages2.  Root integrates these
// separately from the existing tile IDs0/1 (public Hopper variants5/6).
bool small_batched_k64_can_implement(const Problem& leaf, int variant) {
  if (!k64_batch_shape_supported(leaf, variant)) return false;
  auto* placeholder = reinterpret_cast<void*>(std::uintptr_t(0x10000));
  return k64_batch_dispatch(leaf, placeholder, placeholder,
      static_cast<float*>(placeholder), nullptr, variant,
      K64BatchOperation::CanImplement);
}

size_t small_batched_k64_workspace_bytes(const Problem& leaf, int variant) {
  if (!small_batched_k64_can_implement(leaf, variant))
    throw std::invalid_argument("unsupported compact seven-batch K64 leaf");
  return 0;
}

void launch_small_batched_k64(const Problem& leaf, const void* A7,
                             const void* B7, float* P, cudaStream_t stream,
                             int variant) {
  if (!k64_batch_shape_supported(leaf, variant))
    throw std::invalid_argument("unsupported compact seven-batch K64 leaf");
  if (!k64_batch_aligned16(A7) || !k64_batch_aligned16(B7) ||
      !k64_batch_aligned16(P))
    throw std::invalid_argument("K64 batched leaf requires aligned compact views");
  k64_batch_dispatch(leaf, A7, B7, P, stream, variant,
                     K64BatchOperation::Launch);
}

}  // namespace megafmm

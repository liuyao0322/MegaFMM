#include "megafmm/kernels.h"

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace megafmm {
namespace {

using namespace cute;

// The first megakernel has one consumer warpgroup that also performs the
// coalesced SIMT input loads. There is deliberately no producer pipeline yet.
// Each CTA owns four output tiles, the complete K reduction, and all seven P.
// The seven operand slots use the fixed PLAN.md product order P1..P7. Raw
// A21/B12 only live in FP32 registers while constructing the shared input DAG.
template <class Element, int BN>
struct MegaTraits {
  static constexpr int BM = 64;
  static constexpr int BK = 32;
  using Tile = Shape<Int<BM>, Int<BN>, Int<BK>>;
  using LayoutA = decltype(tile_to_shape(
      GMMA::Layout_K_SW32_Atom<Element>{},
      make_shape(Int<BM>{}, Int<BK>{}, Int<7>{})));
  // B is viewed as (N,K), so Major::MN makes its contiguous N dimension
  // compatible with coalesced row-major global loads.
  using LayoutB = decltype(tile_to_shape(
      GMMA::Layout_MN_SW32_Atom<Element>{},
      make_shape(Int<BN>{}, Int<BK>{}, Int<7>{})));
  using Op = decltype(GMMA::ss_op_selector<Element, Element, float, Tile,
                                           GMMA::Major::K, GMMA::Major::MN>());
  using Mma = decltype(make_tiled_mma(Op{}));
  static constexpr int ElementsA = cosize_v<LayoutA>;
  static constexpr int ElementsB = cosize_v<LayoutB>;
  static constexpr size_t SharedBytes =
      size_t(ElementsA + ElementsB) * sizeof(Element);
};

template <class Element, int BN>
__global__ __launch_bounds__(128, 1)
void winograd_m1(Problem problem, const Element* __restrict__ A,
                 const Element* __restrict__ B, const float* C0, float* D) {
  using T = MegaTraits<Element, BN>;
  constexpr int BM = T::BM;
  constexpr int BK = T::BK;
  const int64_t mh = problem.m / 2;
  const int64_t nh = problem.n / 2;
  const int64_t kh = problem.k / 2;
  const int64_t row_base = int64_t(blockIdx.x) * BM;
  const int64_t col_base = int64_t(blockIdx.y) * BN;

  extern __shared__ __align__(128) unsigned char storage[];
  auto sA = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(storage)),
                        typename T::LayoutA{});
  auto sB = make_tensor(
      make_smem_ptr(reinterpret_cast<Element*>(storage) + T::ElementsA),
      typename T::LayoutB{});
  typename T::Mma mma;
  auto thr_mma = mma.get_thread_slice(threadIdx.x);
  auto tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA));
  auto tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB));
  auto coords = thr_mma.partition_C(
      make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{})));
  auto p1 = thr_mma.make_fragment_C(coords);
  auto p2 = thr_mma.make_fragment_C(coords);
  auto p3 = thr_mma.make_fragment_C(coords);
  auto p4 = thr_mma.make_fragment_C(coords);
  auto p5 = thr_mma.make_fragment_C(coords);
  auto p6 = thr_mma.make_fragment_C(coords);
  auto p7 = thr_mma.make_fragment_C(coords);
  clear(p1); clear(p2); clear(p3); clear(p4);
  clear(p5); clear(p6); clear(p7);

  CUTE_NO_UNROLL
  for (int64_t k_base = 0; k_base < kh; k_base += BK) {
    // All dependent expressions remain FP32. In particular S2/T2 never read
    // a rounded shared S1/T1. Every final leaf operand is cast exactly once.
    CUTE_UNROLL
    for (int index = threadIdx.x; index < BM * BK; index += 128) {
      const int row = index / BK;
      const int k = index % BK;
      const int64_t gr = row_base + row;
      const int64_t gk = k_base + k;
      float a11 = 0.0f, a12 = 0.0f, a21 = 0.0f, a22 = 0.0f;
      if (gr < mh && gk < kh) {
        a11 = float(A[gr * problem.lda + gk]);
        a12 = float(A[gr * problem.lda + gk + kh]);
        a21 = float(A[(gr + mh) * problem.lda + gk]);
        a22 = float(A[(gr + mh) * problem.lda + gk + kh]);
      }
      const float s1 = a21 + a22;
      const float s2 = s1 - a11;
      const float s3 = a11 - a21;
      const float s4 = a12 - s2;
      sA(row, k, 0) = Element(a11);
      sA(row, k, 1) = Element(a12);
      sA(row, k, 2) = Element(s4);
      sA(row, k, 3) = Element(a22);
      sA(row, k, 4) = Element(s1);
      sA(row, k, 5) = Element(s2);
      sA(row, k, 6) = Element(s3);
    }
    CUTE_UNROLL
    for (int index = threadIdx.x; index < BK * BN; index += 128) {
      const int k = index / BN;
      const int col = index % BN;
      const int64_t gk = k_base + k;
      const int64_t gc = col_base + col;
      float b11 = 0.0f, b12 = 0.0f, b21 = 0.0f, b22 = 0.0f;
      if (gk < kh && gc < nh) {
        b11 = float(B[gk * problem.ldb + gc]);
        b12 = float(B[gk * problem.ldb + gc + nh]);
        b21 = float(B[(gk + kh) * problem.ldb + gc]);
        b22 = float(B[(gk + kh) * problem.ldb + gc + nh]);
      }
      const float t1 = b12 - b11;
      const float t2 = b22 - t1;
      const float t3 = b22 - b12;
      const float t4 = t2 - b21;
      sB(col, k, 0) = Element(b11);
      sB(col, k, 1) = Element(b21);
      sB(col, k, 2) = Element(b22);
      sB(col, k, 3) = Element(t4);
      sB(col, k, 4) = Element(t1);
      sB(col, k, 5) = Element(t2);
      sB(col, k, 6) = Element(t3);
    }

    // All shared writers issue the generic -> async proxy fence before the
    // CTA barrier. WGMMA then reads only fully published operand tiles.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    warpgroup_fence_operand(p1); warpgroup_fence_operand(p2);
    warpgroup_fence_operand(p3); warpgroup_fence_operand(p4);
    warpgroup_fence_operand(p5); warpgroup_fence_operand(p6);
    warpgroup_fence_operand(p7);
    warpgroup_arrive();
    gemm(mma, tCrA(_, _, _, Int<0>{}), tCrB(_, _, _, Int<0>{}), p1);
    gemm(mma, tCrA(_, _, _, Int<1>{}), tCrB(_, _, _, Int<1>{}), p2);
    gemm(mma, tCrA(_, _, _, Int<2>{}), tCrB(_, _, _, Int<2>{}), p3);
    gemm(mma, tCrA(_, _, _, Int<3>{}), tCrB(_, _, _, Int<3>{}), p4);
    gemm(mma, tCrA(_, _, _, Int<4>{}), tCrB(_, _, _, Int<4>{}), p5);
    gemm(mma, tCrA(_, _, _, Int<5>{}), tCrB(_, _, _, Int<5>{}), p6);
    gemm(mma, tCrA(_, _, _, Int<6>{}), tCrB(_, _, _, Int<6>{}), p7);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(p1); warpgroup_fence_operand(p2);
    warpgroup_fence_operand(p3); warpgroup_fence_operand(p4);
    warpgroup_fence_operand(p5); warpgroup_fence_operand(p6);
    warpgroup_fence_operand(p7);
    // Wait plus barrier protects the single shared stage against the next
    // loop iteration's writers; __syncthreads alone would be insufficient.
    __syncthreads();
  }

  CUTE_UNROLL
  for (int index = 0; index < size(p1); ++index) {
    const auto coord = coords(index);
    const int64_t row = row_base + get<0>(coord);
    const int64_t col = col_base + get<1>(coord);
    if (row < mh && col < nh) {
      const float u1 = p1(index) + p2(index);
      const float u2 = p1(index) + p6(index);
      const float u3 = u2 + p7(index);
      const float u4 = u2 + p5(index);
      const float u5 = u4 + p3(index);
      const float u6 = u3 - p4(index);
      const float u7 = u3 + p5(index);
      const int64_t i11 = row * problem.ldd + col;
      const int64_t i12 = row * problem.ldd + col + nh;
      const int64_t i21 = (row + mh) * problem.ldd + col;
      const int64_t i22 = (row + mh) * problem.ldd + col + nh;
      float d11 = problem.alpha * u1;
      float d12 = problem.alpha * u5;
      float d21 = problem.alpha * u6;
      float d22 = problem.alpha * u7;
      if (problem.beta != 0.0f) {
        // Read all C0 values before writing any of the four owner outputs.
        // Identical C0/D pointers with identical strides are safe.
        d11 += problem.beta * C0[row * problem.ldc + col];
        d12 += problem.beta * C0[row * problem.ldc + col + nh];
        d21 += problem.beta * C0[(row + mh) * problem.ldc + col];
        d22 += problem.beta * C0[(row + mh) * problem.ldc + col + nh];
      }
      D[i11] = d11; D[i12] = d12;
      D[i21] = d21; D[i22] = d22;
    }
  }
}

template <class Element, int BN>
void prepare_m1() {
  constexpr size_t smem = MegaTraits<Element, BN>::SharedBytes;
  if constexpr (smem > 48 * 1024) {
    const auto status = cudaFuncSetAttribute(
        winograd_m1<Element, BN>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(smem));
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string("megakernel shared-memory opt-in: ") +
                               cudaGetErrorString(status));
    }
  }
}

template <class Element, int BN>
void launch_m1(const Problem& p, const void* A, const void* B,
               const float* C0, float* D, cudaStream_t stream) {
  const dim3 grid(unsigned((p.m / 2 + 63) / 64),
                  unsigned((p.n / 2 + BN - 1) / BN));
  winograd_m1<Element, BN><<<grid, 128,
      MegaTraits<Element, BN>::SharedBytes, stream>>>(
          p, static_cast<const Element*>(A), static_cast<const Element*>(B),
          C0, D);
}

} // namespace

bool megakernel_can_implement(const Problem& p, int variant) {
  if (variant < 0 || variant > 1 || p.trans_a || p.trans_b) return false;
  if (p.dtype != DType::BF16 && p.dtype != DType::FP16) return false;
  if (p.m <= 0 || p.n <= 0 || p.k <= 0 || p.m > 32768 ||
      p.n > 32768 || p.k > 32768) return false;
  if ((p.m & 1) || (p.n & 1) || (p.k & 1)) return false;
  if (p.lda < p.k || p.ldb < p.n || p.ldd < p.n ||
      (p.beta != 0.0f && p.ldc < p.n)) return false;
  const int64_t max_elements = std::numeric_limits<int64_t>::max() / 4;
  if (p.lda > max_elements / p.m || p.ldb > max_elements / p.k ||
      p.ldd > max_elements / p.m ||
      (p.beta != 0.0f && p.ldc > max_elements / p.m)) return false;
  return std::isfinite(p.alpha) && std::isfinite(p.beta);
}

// Plan creation calls preparation once; steady-state launch has no attribute
// queries, allocations, host synchronization, or global S/T/P workspace.
void prepare_megakernel(const Problem& p, int variant) {
  if (!megakernel_can_implement(p, variant))
    throw std::invalid_argument("unsupported megakernel problem or variant");
  if (p.dtype == DType::BF16) {
    if (variant == 0) prepare_m1<cutlass::bfloat16_t, 32>();
    else prepare_m1<cutlass::bfloat16_t, 64>();
  } else {
    if (variant == 0) prepare_m1<cutlass::half_t, 32>();
    else prepare_m1<cutlass::half_t, 64>();
  }
}

void launch_megakernel(const Problem& p, const void* A, const void* B,
                      const float* C0, float* D, cudaStream_t stream,
                      int variant) {
  if (!megakernel_can_implement(p, variant))
    throw std::invalid_argument("unsupported megakernel problem or variant");
  if (!A || !B || !D || (p.beta != 0.0f && !C0))
    throw std::invalid_argument("megakernel requires nonnull input/output pointers");
  if (p.dtype == DType::BF16) {
    if (variant == 0) launch_m1<cutlass::bfloat16_t, 32>(p,A,B,C0,D,stream);
    else launch_m1<cutlass::bfloat16_t, 64>(p,A,B,C0,D,stream);
  } else {
    if (variant == 0) launch_m1<cutlass::half_t, 32>(p,A,B,C0,D,stream);
    else launch_m1<cutlass::half_t, 64>(p,A,B,C0,D,stream);
  }
}

} // namespace megafmm

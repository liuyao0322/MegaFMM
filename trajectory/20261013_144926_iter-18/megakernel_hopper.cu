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

// M3 streams one complete product at a time. Three P/U fragments remain live,
// and a single shared A/B operand pair replaces M1's fourteen operand slots.
template <class Element>
struct MegaM3Traits {
  static constexpr int BM = 64;
  static constexpr int BN = 128;
  static constexpr int BK = 32;
  using Tile = Shape<Int<BM>, Int<BN>, Int<BK>>;
  using LayoutA = decltype(tile_to_shape(
      GMMA::Layout_K_SW32_Atom<Element>{}, make_shape(Int<BM>{}, Int<BK>{})));
  using LayoutB = decltype(tile_to_shape(
      GMMA::Layout_MN_SW32_Atom<Element>{}, make_shape(Int<BN>{}, Int<BK>{})));
  using Op = decltype(GMMA::ss_op_selector<Element, Element, float, Tile,
                                           GMMA::Major::K, GMMA::Major::MN>());
  using Mma = decltype(make_tiled_mma(Op{}));
  static constexpr int ElementsA = cosize_v<LayoutA>;
  static constexpr int ElementsB = cosize_v<LayoutB>;
  static constexpr size_t SharedBytes =
      size_t(ElementsA + ElementsB) * sizeof(Element);
};

// SW32 moves whole 16-byte chunks, preserving the eight-element contiguous
// groups used by the vector shared stores below. Check every logical chunk,
// including those moved by the swizzle, instead of assuming a linear layout.
template <class Layout, int ContiguousMode>
constexpr bool has_contiguous_eight() {
  constexpr int rows = size<0>(Layout{});
  constexpr int cols = size<1>(Layout{});
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int contiguous_coord = ContiguousMode == 0 ? row : col;
      if (contiguous_coord % 8 != 0) continue;
      const int base = Layout{}(make_coord(row, col));
      if (base % 8 != 0) return false;
      for (int lane = 0; lane < 8; ++lane) {
        const int r = row + (ContiguousMode == 0 ? lane : 0);
        const int c = col + (ContiguousMode == 1 ? lane : 0);
        if (Layout{}(make_coord(r, c)) != base + lane) return false;
      }
    }
  }
  return true;
}

// Use explicit packed words rather than aliasing an Element array in a union.
// Constant lane indices in the unrolled transform loops become register bit
// extracts; no temporary low-precision S/T values feed later expressions.
struct PackedEight {
  uint4 bits;

  CUTE_DEVICE uint16_t raw(int lane) const {
    const uint32_t word = lane < 2 ? bits.x : lane < 4 ? bits.y
                                           : lane < 6 ? bits.z : bits.w;
    return uint16_t(word >> (16 * (lane & 1)));
  }

  CUTE_DEVICE void insert(int lane, uint16_t value) {
    const uint32_t shifted = uint32_t(value) << (16 * (lane & 1));
    if (lane < 2) bits.x |= shifted;
    else if (lane < 4) bits.y |= shifted;
    else if (lane < 6) bits.z |= shifted;
    else bits.w |= shifted;
  }
};

template <class Element>
CUTE_DEVICE
PackedEight read_eight(const Element* ptr, bool row_valid, int64_t remaining) {
  PackedEight result{make_uint4(0, 0, 0, 0)};
  if (row_valid && remaining >= 8 &&
      (reinterpret_cast<uintptr_t>(ptr) & 15) == 0) {
    result.bits = *reinterpret_cast<const uint4*>(ptr);
  } else {
    CUTE_UNROLL
    for (int lane = 0; lane < 8; ++lane) {
      if (row_valid && int64_t(lane) < remaining)
        result.insert(lane, ptr[lane].raw());
    }
  }
  return result;
}

template <class Element>
CUTE_DEVICE float unpack(const PackedEight& value, int lane) {
  return float(Element::bitcast(value.raw(lane)));
}

// Product is zero based, with the same P1..P7 numbering as M1 and PLAN.md.
// constexpr branches remove every quadrant unused by the current product.
template <int Product, class Element>
CUTE_DEVICE
PackedEight load_m3_a(const Problem& p, const Element* A,
                     int64_t row, int64_t k) {
  const int64_t mh = p.m / 2;
  const int64_t kh = p.k / 2;
  const bool valid = row < mh;
  const int64_t r = valid ? row : 0;
  const int64_t kk = k < kh ? k : 0;
  const int64_t remaining = kh - k;
  PackedEight a11, a12, a21, a22;
  if constexpr (Product == 0 || Product == 2 || Product == 5 || Product == 6)
    a11 = read_eight(A + r * p.lda + kk, valid, remaining);
  if constexpr (Product == 1 || Product == 2)
    a12 = read_eight(A + r * p.lda + kk + kh, valid, remaining);
  if constexpr (Product == 2 || Product == 4 || Product == 5 || Product == 6)
    a21 = read_eight(A + (r + mh) * p.lda + kk, valid, remaining);
  if constexpr (Product == 2 || Product == 3 || Product == 4 || Product == 5)
    a22 = read_eight(A + (r + mh) * p.lda + kk + kh, valid, remaining);

  if constexpr (Product == 0) return a11;
  else if constexpr (Product == 1) return a12;
  else if constexpr (Product == 3) return a22;
  else {
    PackedEight result{make_uint4(0, 0, 0, 0)};
    CUTE_UNROLL
    for (int lane = 0; lane < 8; ++lane) {
      float value;
      if constexpr (Product == 2) {
        const float s1 = unpack<Element>(a21, lane) + unpack<Element>(a22, lane);
        const float s2 = s1 - unpack<Element>(a11, lane);
        value = unpack<Element>(a12, lane) - s2;
      } else if constexpr (Product == 4) {
        value = unpack<Element>(a21, lane) + unpack<Element>(a22, lane);
      } else if constexpr (Product == 5) {
        const float s1 = unpack<Element>(a21, lane) + unpack<Element>(a22, lane);
        value = s1 - unpack<Element>(a11, lane);
      } else {
        value = unpack<Element>(a11, lane) - unpack<Element>(a21, lane);
      }
      result.insert(lane, Element(value).raw());
    }
    return result;
  }
}

template <int Product, class Element>
CUTE_DEVICE
PackedEight load_m3_b(const Problem& p, const Element* B,
                     int64_t k, int64_t col) {
  const int64_t kh = p.k / 2;
  const int64_t nh = p.n / 2;
  const bool valid = k < kh;
  const int64_t kk = valid ? k : 0;
  const int64_t c = col < nh ? col : 0;
  const int64_t remaining = nh - col;
  PackedEight b11, b12, b21, b22;
  if constexpr (Product == 0 || Product == 3 || Product == 4 || Product == 5)
    b11 = read_eight(B + kk * p.ldb + c, valid, remaining);
  if constexpr (Product == 3 || Product == 4 || Product == 5 || Product == 6)
    b12 = read_eight(B + kk * p.ldb + c + nh, valid, remaining);
  if constexpr (Product == 1 || Product == 3)
    b21 = read_eight(B + (kk + kh) * p.ldb + c, valid, remaining);
  if constexpr (Product == 2 || Product == 3 || Product == 5 || Product == 6)
    b22 = read_eight(B + (kk + kh) * p.ldb + c + nh, valid, remaining);

  if constexpr (Product == 0) return b11;
  else if constexpr (Product == 1) return b21;
  else if constexpr (Product == 2) return b22;
  else {
    PackedEight result{make_uint4(0, 0, 0, 0)};
    CUTE_UNROLL
    for (int lane = 0; lane < 8; ++lane) {
      float value;
      if constexpr (Product == 3) {
        const float t1 = unpack<Element>(b12, lane) - unpack<Element>(b11, lane);
        const float t2 = unpack<Element>(b22, lane) - t1;
        value = t2 - unpack<Element>(b21, lane);
      } else if constexpr (Product == 4) {
        value = unpack<Element>(b12, lane) - unpack<Element>(b11, lane);
      } else if constexpr (Product == 5) {
        const float t1 = unpack<Element>(b12, lane) - unpack<Element>(b11, lane);
        value = unpack<Element>(b22, lane) - t1;
      } else {
        value = unpack<Element>(b22, lane) - unpack<Element>(b12, lane);
      }
      result.insert(lane, Element(value).raw());
    }
    return result;
  }
}

template <int Product, class Element, class SmemA, class SmemB,
          class DescriptorA, class DescriptorB, class Mma, class Accumulator>
CUTE_DEVICE
void full_m3_product(const Problem& p, const Element* A, const Element* B,
                     int64_t row_base, int64_t col_base,
                     SmemA sA, SmemB sB, DescriptorA tCrA, DescriptorB tCrB,
                     Mma mma, Accumulator& accumulator) {
  using T = MegaM3Traits<Element>;
  clear(accumulator);
  CUTE_NO_UNROLL
  for (int64_t k_base = 0; k_base < p.k / 2; k_base += T::BK) {
    // Keep the outer chunk loops rolled to bound temporary packed-word
    // lifetimes; only the eight-lane register transformation is unrolled.
    CUTE_NO_UNROLL
    for (int index = threadIdx.x * 8; index < T::BM * T::BK; index += 128 * 8) {
      const int row = index / T::BK;
      const int k = index % T::BK;
      const PackedEight operand = load_m3_a<Product>(p, A, row_base + row, k_base + k);
      *reinterpret_cast<uint4*>(&sA(row, k)) = operand.bits;
    }
    CUTE_NO_UNROLL
    for (int index = threadIdx.x * 8; index < T::BK * T::BN; index += 128 * 8) {
      const int k = index / T::BN;
      const int col = index % T::BN;
      const PackedEight operand = load_m3_b<Product>(p, B, k_base + k, col_base + col);
      *reinterpret_cast<uint4*>(&sB(col, k)) = operand.bits;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    warpgroup_fence_operand(accumulator);
    warpgroup_arrive();
    gemm(mma, tCrA, tCrB, accumulator);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(accumulator);
    __syncthreads();
  }
}

template <int QuadrantRow, int QuadrantCol, bool Sum, class Accumulator,
          class Coordinates>
CUTE_DEVICE
void store_m3_quadrant(const Problem& p, const float* C0, float* D,
                       int64_t row_base, int64_t col_base,
                       const Accumulator& x, const Accumulator& y,
                       const Coordinates& coords) {
  const int64_t mh = p.m / 2;
  const int64_t nh = p.n / 2;
  CUTE_UNROLL
  for (int index = 0; index < size(x); ++index) {
    const auto coord = coords(index);
    const int64_t row = row_base + get<0>(coord);
    const int64_t col = col_base + get<1>(coord);
    if (row < mh && col < nh) {
      const int64_t gr = row + QuadrantRow * mh;
      const int64_t gc = col + QuadrantCol * nh;
      const float value = Sum ? x(index) + y(index) : x(index);
      float result = p.alpha * value;
      if (p.beta != 0.0f) result += p.beta * C0[gr * p.ldc + gc];
      D[gr * p.ldd + gc] = result;
    }
  }
}

template <class Element>
__global__ __launch_bounds__(128, 1)
void winograd_m3(Problem problem, const Element* __restrict__ A,
                 const Element* __restrict__ B, const float* C0, float* D) {
  using T = MegaM3Traits<Element>;
  static_assert(sizeof(Element) == 2, "M3 packed operands must be 16 bit");
  static_assert(has_contiguous_eight<typename T::LayoutA, 1>(),
                "A vector shared stores must match the GMMA layout");
  static_assert(has_contiguous_eight<typename T::LayoutB, 0>(),
                "B vector shared stores must match the GMMA layout");
  const int64_t row_base = int64_t(blockIdx.x) * T::BM;
  const int64_t col_base = int64_t(blockIdx.y) * T::BN;
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
      make_identity_tensor(make_shape(Int<T::BM>{}, Int<T::BN>{})));
  auto x = thr_mma.make_fragment_C(coords);
  auto y = thr_mma.make_fragment_C(coords);
  auto z = thr_mma.make_fragment_C(coords);

  full_m3_product<0>(problem,A,B,row_base,col_base,sA,sB,tCrA,tCrB,mma,x); // P1
  full_m3_product<1>(problem,A,B,row_base,col_base,sA,sB,tCrA,tCrB,mma,y); // P2
  store_m3_quadrant<0,0,true>(problem,C0,D,row_base,col_base,x,y,coords); // U1

  full_m3_product<5>(problem,A,B,row_base,col_base,sA,sB,tCrA,tCrB,mma,y); // P6
  CUTE_UNROLL
  for (int index = 0; index < size(x); ++index) x(index) = x(index) + y(index); // U2
  full_m3_product<6>(problem,A,B,row_base,col_base,sA,sB,tCrA,tCrB,mma,y); // P7
  CUTE_UNROLL
  for (int index = 0; index < size(x); ++index) y(index) = x(index) + y(index); // U3
  full_m3_product<4>(problem,A,B,row_base,col_base,sA,sB,tCrA,tCrB,mma,z); // P5
  CUTE_UNROLL
  for (int index = 0; index < size(x); ++index) {
    x(index) = x(index) + z(index); // U4 must consume P5 before Z changes.
    z(index) = y(index) + z(index); // U7
  }
  store_m3_quadrant<1,1,false>(problem,C0,D,row_base,col_base,z,z,coords);

  full_m3_product<2>(problem,A,B,row_base,col_base,sA,sB,tCrA,tCrB,mma,z); // P3
  CUTE_UNROLL
  for (int index = 0; index < size(x); ++index) x(index) = x(index) + z(index); // U5
  store_m3_quadrant<0,1,false>(problem,C0,D,row_base,col_base,x,x,coords);

  full_m3_product<3>(problem,A,B,row_base,col_base,sA,sB,tCrA,tCrB,mma,x); // P4
  CUTE_UNROLL
  for (int index = 0; index < size(y); ++index) y(index) = y(index) - x(index); // U6
  store_m3_quadrant<1,0,false>(problem,C0,D,row_base,col_base,y,y,coords);
}

template <class Element>
void launch_m3(const Problem& p, const void* A, const void* B,
               const float* C0, float* D, cudaStream_t stream) {
  using T = MegaM3Traits<Element>;
  const dim3 grid(unsigned((p.m / 2 + T::BM - 1) / T::BM),
                  unsigned((p.n / 2 + T::BN - 1) / T::BN));
  winograd_m3<Element><<<grid, 128, T::SharedBytes, stream>>>(
      p, static_cast<const Element*>(A), static_cast<const Element*>(B), C0, D);
}

// Variant 3 keeps M3's full-product/U schedule, but has a full-tile beta-zero
// path and overlaps each current WGMMA batch with SIMT filling the next stage.
// The original M3 above remains the control and the general-condition fallback.
template <class Element>
struct MegaM4Traits {
  using Base = MegaM3Traits<Element>;
  static constexpr int BM = Base::BM;
  static constexpr int BN = Base::BN;
  static constexpr int BK = Base::BK;
  using LayoutA = decltype(tile_to_shape(
      GMMA::Layout_K_SW32_Atom<Element>{},
      make_shape(Int<BM>{}, Int<BK>{}, Int<2>{})));
  using LayoutB = decltype(tile_to_shape(
      GMMA::Layout_MN_SW32_Atom<Element>{},
      make_shape(Int<BN>{}, Int<BK>{}, Int<2>{})));
  // GMMA layouts carry a byte-address swizzle flag. Convert the tile layout
  // to physical element offsets for the contiguous-vector proof.
  using PhysicalTileA = decltype(as_position_independent_swizzle_layout(
      typename Base::LayoutA{}));
  using PhysicalTileB = decltype(as_position_independent_swizzle_layout(
      typename Base::LayoutB{}));
  using Mma = typename Base::Mma;
  static constexpr int ElementsA = cosize_v<LayoutA>;
  static constexpr int ElementsB = cosize_v<LayoutB>;
  static constexpr size_t SharedBytes =
      size_t(ElementsA + ElementsB) * sizeof(Element);
};

template <class Element>
CUTE_DEVICE PackedEight read_aligned_eight(const Element* ptr) {
  return PackedEight{*reinterpret_cast<const uint4*>(ptr)};
}

// Only the full-tile dispatch uses these selective loads. All quadrant bases
// and all eight lanes are known aligned and in bounds before kernel launch.
template <int Product, class Element>
CUTE_DEVICE PackedEight load_m4_a(const Problem& p, const Element* A,
                                  int64_t row, int64_t k) {
  const int64_t mh = p.m / 2;
  const int64_t kh = p.k / 2;
  PackedEight a11, a12, a21, a22;
  if constexpr (Product == 0 || Product == 2 || Product == 5 || Product == 6)
    a11 = read_aligned_eight(A + row * p.lda + k);
  if constexpr (Product == 1 || Product == 2)
    a12 = read_aligned_eight(A + row * p.lda + k + kh);
  if constexpr (Product == 2 || Product == 4 || Product == 5 || Product == 6)
    a21 = read_aligned_eight(A + (row + mh) * p.lda + k);
  if constexpr (Product == 2 || Product == 3 || Product == 4 || Product == 5)
    a22 = read_aligned_eight(A + (row + mh) * p.lda + k + kh);
  if constexpr (Product == 0) return a11;
  else if constexpr (Product == 1) return a12;
  else if constexpr (Product == 3) return a22;
  else {
    PackedEight result{make_uint4(0, 0, 0, 0)};
    CUTE_UNROLL
    for (int lane = 0; lane < 8; ++lane) {
      float value;
      if constexpr (Product == 2) {
        const float s1 = unpack<Element>(a21, lane) + unpack<Element>(a22, lane);
        const float s2 = s1 - unpack<Element>(a11, lane);
        value = unpack<Element>(a12, lane) - s2;
      } else if constexpr (Product == 4) {
        value = unpack<Element>(a21, lane) + unpack<Element>(a22, lane);
      } else if constexpr (Product == 5) {
        const float s1 = unpack<Element>(a21, lane) + unpack<Element>(a22, lane);
        value = s1 - unpack<Element>(a11, lane);
      } else {
        value = unpack<Element>(a11, lane) - unpack<Element>(a21, lane);
      }
      result.insert(lane, Element(value).raw());
    }
    return result;
  }
}

template <int Product, class Element>
CUTE_DEVICE PackedEight load_m4_b(const Problem& p, const Element* B,
                                  int64_t k, int64_t col) {
  const int64_t kh = p.k / 2;
  const int64_t nh = p.n / 2;
  PackedEight b11, b12, b21, b22;
  if constexpr (Product == 0 || Product == 3 || Product == 4 || Product == 5)
    b11 = read_aligned_eight(B + k * p.ldb + col);
  if constexpr (Product == 3 || Product == 4 || Product == 5 || Product == 6)
    b12 = read_aligned_eight(B + k * p.ldb + col + nh);
  if constexpr (Product == 1 || Product == 3)
    b21 = read_aligned_eight(B + (k + kh) * p.ldb + col);
  if constexpr (Product == 2 || Product == 3 || Product == 5 || Product == 6)
    b22 = read_aligned_eight(B + (k + kh) * p.ldb + col + nh);
  if constexpr (Product == 0) return b11;
  else if constexpr (Product == 1) return b21;
  else if constexpr (Product == 2) return b22;
  else {
    PackedEight result{make_uint4(0, 0, 0, 0)};
    CUTE_UNROLL
    for (int lane = 0; lane < 8; ++lane) {
      float value;
      if constexpr (Product == 3) {
        const float t1 = unpack<Element>(b12, lane) - unpack<Element>(b11, lane);
        const float t2 = unpack<Element>(b22, lane) - t1;
        value = t2 - unpack<Element>(b21, lane);
      } else if constexpr (Product == 4) {
        value = unpack<Element>(b12, lane) - unpack<Element>(b11, lane);
      } else if constexpr (Product == 5) {
        const float t1 = unpack<Element>(b12, lane) - unpack<Element>(b11, lane);
        value = unpack<Element>(b22, lane) - t1;
      } else {
        value = unpack<Element>(b22, lane) - unpack<Element>(b12, lane);
      }
      result.insert(lane, Element(value).raw());
    }
    return result;
  }
}

CUTE_DEVICE void store_shared_eight(void* ptr, const PackedEight& value) {
  const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
      :: "r"(address), "r"(value.bits.x), "r"(value.bits.y),
         "r"(value.bits.z), "r"(value.bits.w) : "memory");
}

template <int Product, class Element, class SmemA, class SmemB>
CUTE_DEVICE void fill_m4_stage(const Problem& p, const Element* A,
                               const Element* B, int64_t row_base,
                               int64_t col_base, int64_t k_base, int stage,
                               SmemA sA, SmemB sB) {
  using T = MegaM4Traits<Element>;
  CUTE_NO_UNROLL
  for (int index = threadIdx.x * 8; index < T::BM * T::BK; index += 128 * 8) {
    const int row = index / T::BK;
    const int k = index % T::BK;
    const PackedEight operand = load_m4_a<Product>(p, A, row_base + row, k_base + k);
    store_shared_eight(&sA(row, k, stage), operand);
  }
  CUTE_NO_UNROLL
  for (int index = threadIdx.x * 8; index < T::BK * T::BN; index += 128 * 8) {
    const int k = index / T::BN;
    const int col = index % T::BN;
    const PackedEight operand = load_m4_b<Product>(p, B, k_base + k, col_base + col);
    store_shared_eight(&sB(col, k, stage), operand);
  }
}

template <int Product, class Element, class SmemA, class SmemB,
          class DescriptorA, class DescriptorB, class Mma, class Accumulator>
CUTE_DEVICE void full_m4_product(const Problem& p, const Element* A,
                                 const Element* B, int64_t row_base,
                                 int64_t col_base, SmemA sA, SmemB sB,
                                 DescriptorA tCrA, DescriptorB tCrB,
                                 Mma mma, Accumulator& accumulator) {
  using T = MegaM4Traits<Element>;
  clear(accumulator);
  fill_m4_stage<Product>(p,A,B,row_base,col_base,0,0,sA,sB);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  int current_stage = 0;
  CUTE_NO_UNROLL
  for (int64_t k_base = 0; k_base < p.k / 2; k_base += T::BK) {
    warpgroup_fence_operand(accumulator);
    warpgroup_arrive();
    gemm(mma, tCrA(_,_,_,current_stage), tCrB(_,_,_,current_stage), accumulator);
    warpgroup_commit_batch();

    // Only ordinary transform temporaries are touched here. The targeted
    // accumulator stays live and untouched until wait<0> completes. Filling
    // the other stage cannot overwrite current WGMMA operands.
    const int next_stage = current_stage ^ 1;
    if (k_base + T::BK < p.k / 2) {
      fill_m4_stage<Product>(p,A,B,row_base,col_base,k_base + T::BK,next_stage,sA,sB);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncthreads();
    }
    warpgroup_wait<0>();
    warpgroup_fence_operand(accumulator);
    // All threads finish consumption before the older stage can be reused.
    // This also drains the last stage before U math or the next product.
    __syncthreads();
    current_stage = next_stage;
  }
}

template <int QuadrantRow, int QuadrantCol, bool Sum, class Accumulator,
          class Coordinates>
CUTE_DEVICE void store_m4_quadrant(const Problem& p, float* D,
                                   int64_t row_base, int64_t col_base,
                                   const Accumulator& x, const Accumulator& y,
                                   const Coordinates& coords) {
  CUTE_UNROLL
  for (int index = 0; index < size(x); ++index) {
    const auto coord = coords(index);
    const int64_t row = row_base + get<0>(coord) + QuadrantRow * (p.m / 2);
    const int64_t col = col_base + get<1>(coord) + QuadrantCol * (p.n / 2);
    const float value = Sum ? x(index) + y(index) : x(index);
    D[row * p.ldd + col] = p.alpha * value;
  }
}

template <class Element>
__global__ __launch_bounds__(128, 1)
void winograd_m4(Problem problem, const Element* __restrict__ A,
                 const Element* __restrict__ B, float* D) {
  using T = MegaM4Traits<Element>;
  static_assert(has_contiguous_eight<typename T::PhysicalTileA, 1>(),
                "physical A layout must preserve each eight-element vector");
  static_assert(has_contiguous_eight<typename T::PhysicalTileB, 0>(),
                "physical B layout must preserve each eight-element vector");
  static_assert(T::ElementsA == 2 * MegaM3Traits<Element>::ElementsA &&
                T::ElementsB == 2 * MegaM3Traits<Element>::ElementsB,
                "two operand stages must have independent storage");
  const int64_t row_base = int64_t(blockIdx.x) * T::BM;
  const int64_t col_base = int64_t(blockIdx.y) * T::BN;
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
      make_identity_tensor(make_shape(Int<T::BM>{}, Int<T::BN>{})));
  auto x = thr_mma.make_fragment_C(coords);
  auto y = thr_mma.make_fragment_C(coords);
  auto z = thr_mma.make_fragment_C(coords);

  full_m4_product<0>(problem,A,B,row_base,col_base,sA,sB,tCrA,tCrB,mma,x); // P1
  full_m4_product<1>(problem,A,B,row_base,col_base,sA,sB,tCrA,tCrB,mma,y); // P2
  store_m4_quadrant<0,0,true>(problem,D,row_base,col_base,x,y,coords); // U1
  full_m4_product<5>(problem,A,B,row_base,col_base,sA,sB,tCrA,tCrB,mma,y); // P6
  CUTE_UNROLL
  for (int index = 0; index < size(x); ++index) x(index) = x(index) + y(index); // U2
  full_m4_product<6>(problem,A,B,row_base,col_base,sA,sB,tCrA,tCrB,mma,y); // P7
  CUTE_UNROLL
  for (int index = 0; index < size(x); ++index) y(index) = x(index) + y(index); // U3
  full_m4_product<4>(problem,A,B,row_base,col_base,sA,sB,tCrA,tCrB,mma,z); // P5
  CUTE_UNROLL
  for (int index = 0; index < size(x); ++index) {
    x(index) = x(index) + z(index); // U4
    z(index) = y(index) + z(index); // U7
  }
  store_m4_quadrant<1,1,false>(problem,D,row_base,col_base,z,z,coords);
  full_m4_product<2>(problem,A,B,row_base,col_base,sA,sB,tCrA,tCrB,mma,z); // P3
  CUTE_UNROLL
  for (int index = 0; index < size(x); ++index) x(index) = x(index) + z(index); // U5
  store_m4_quadrant<0,1,false>(problem,D,row_base,col_base,x,x,coords);
  full_m4_product<3>(problem,A,B,row_base,col_base,sA,sB,tCrA,tCrB,mma,x); // P4
  CUTE_UNROLL
  for (int index = 0; index < size(y); ++index) y(index) = y(index) - x(index); // U6
  store_m4_quadrant<1,0,false>(problem,D,row_base,col_base,y,y,coords);
}

bool m4_fast_condition(const Problem& p, const void* A, const void* B) {
  return p.beta == 0.0f && p.m % 128 == 0 && p.n % 256 == 0 && p.k % 64 == 0 &&
         p.lda % 8 == 0 && p.ldb % 8 == 0 &&
         (reinterpret_cast<uintptr_t>(A) & 15) == 0 &&
         (reinterpret_cast<uintptr_t>(B) & 15) == 0;
}

template <class Element>
void launch_m4(const Problem& p, const void* A, const void* B,
               const float* C0, float* D, cudaStream_t stream) {
  if (!m4_fast_condition(p, A, B)) {
    launch_m3<Element>(p,A,B,C0,D,stream);
    return;
  }
  using T = MegaM4Traits<Element>;
  const dim3 grid(unsigned(p.m / (2 * T::BM)), unsigned(p.n / (2 * T::BN)));
  winograd_m4<Element><<<grid, 128, T::SharedBytes, stream>>>(
      p, static_cast<const Element*>(A), static_cast<const Element*>(B), D);
}

} // namespace

bool megakernel_can_implement(const Problem& p, int variant) {
  if (variant < 0 || variant > 3 || p.trans_a || p.trans_b) return false;
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
  if (variant == 2 || variant == 3) return; // M3/M4 need no dynamic opt-in.
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
    else if (variant == 1) launch_m1<cutlass::bfloat16_t, 64>(p,A,B,C0,D,stream);
    else if (variant == 2) launch_m3<cutlass::bfloat16_t>(p,A,B,C0,D,stream);
    else launch_m4<cutlass::bfloat16_t>(p,A,B,C0,D,stream);
  } else {
    if (variant == 0) launch_m1<cutlass::half_t, 32>(p,A,B,C0,D,stream);
    else if (variant == 1) launch_m1<cutlass::half_t, 64>(p,A,B,C0,D,stream);
    else if (variant == 2) launch_m3<cutlass::half_t>(p,A,B,C0,D,stream);
    else launch_m4<cutlass::half_t>(p,A,B,C0,D,stream);
  }
}

} // namespace megafmm

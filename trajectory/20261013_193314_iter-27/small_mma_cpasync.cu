// Prototype only: Mega17 cp.async of original quadrant bits into two raw banks.
// Complete 16x16 owner tiles use this kernel; other supported calls fall back to 7.
#include "megafmm/kernels.h"

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace megafmm {
namespace {

using namespace cute;

template <class Element>
struct SmallMmaTraits {
  static constexpr int BM = 16;
  static constexpr int BN = 16;
  static constexpr int BK = 16;
  using Op = typename std::conditional<
      std::is_same<Element, cutlass::bfloat16_t>::value,
      SM80_16x8x16_F32BF16BF16F32_TN,
      SM80_16x8x16_F32F16F16F32_TN>::type;
  using Mma = decltype(make_tiled_mma(
      Op{}, Layout<Shape<_1,_1,_1>>{}, Tile<_16,_16,_16>{}));
  // Ordinary element-unit swizzles, not GMMA descriptor/pointer layouts.
  // Product stride 256 is a multiple of the swizzle's 128-element period.
  using LayoutA = decltype(composition(Swizzle<1,3,3>{},
      Layout<Shape<_16,_16,_7>, Stride<_16,_1,_256>>{}));
  using LayoutB = decltype(composition(Swizzle<1,3,3>{},
      Layout<Shape<_16,_16,_7>, Stride<_1,_16,_256>>{}));
  static constexpr int ElementsA = cosize_v<LayoutA>;
  static constexpr int ElementsB = cosize_v<LayoutB>;
  static constexpr int WarpElements = ElementsA + ElementsB;
  static constexpr int WarpSharedBytes = WarpElements * sizeof(Element);
  // Original quadrant slots are untransformed and have plain contiguous rows.
  using RawLayoutA = Layout<Shape<_16,_16,_4>, Stride<_16,_1,_256>>;
  using RawLayoutB = Layout<Shape<_16,_16,_4>, Stride<_1,_16,_256>>;
  static constexpr int RawElementsA = cosize_v<RawLayoutA>;
  static constexpr int RawElementsB = cosize_v<RawLayoutB>;
  static constexpr int RawBankElements = RawElementsA + RawElementsB;
  static constexpr int RawBankBytes = RawBankElements * sizeof(Element);
  static constexpr int TotalElements = 2 * RawBankElements + WarpElements;
  static_assert(RawBankBytes == 4096, "one original-input bank occupies 4 KiB");
  static_assert(TotalElements * sizeof(Element) == 15360,
                "two raw banks plus seven transformed pairs occupy 15 KiB");
  static_assert(ElementsA == 7 * BM * BK && ElementsB == 7 * BN * BK,
                "private operand slots must occupy exactly 7 KiB per warp");
  static_assert(size(Mma{}) == 32, "one owner must be one independent warp");
};

struct SmallPackedEight {
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
CUTE_DEVICE void small_async_eight(const Element* global, Element* shared) {
  // Pinned CuTe emits cp.async.ca.shared.global.L2::128B for this 16-byte copy.
  SM80_CP_ASYNC_CACHEALWAYS<uint4>::copy(
      *reinterpret_cast<const uint4*>(global), *reinterpret_cast<uint4*>(shared));
}

CUTE_DEVICE void small_async_commit() {
  asm volatile("" ::: "memory");
  cp_async_fence();
  asm volatile("" ::: "memory");
}

CUTE_DEVICE void small_async_wait_publish() {
  cp_async_wait<0>();
  // The pinned cp.async helpers have no compiler memory clobber. Publish all
  // lane copies before any ordinary shared read or raw-bank reuse.
  asm volatile("" ::: "memory");
  __syncwarp(0xffffffffu);
}

template <class Element>
CUTE_DEVICE SmallPackedEight small_read_raw_eight(const Element* shared) {
  SmallPackedEight result;
  const uint32_t address = uint32_t(__cvta_generic_to_shared(shared));
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(result.bits.x), "=r"(result.bits.y),
        "=r"(result.bits.z), "=r"(result.bits.w)
      : "r"(address) : "memory");
  return result;
}

template <class Element>
CUTE_DEVICE float small_unpack(const SmallPackedEight& value, int lane) {
  return float(Element::bitcast(value.raw(lane)));
}

CUTE_DEVICE void small_store_eight(void* ptr, const SmallPackedEight& value) {
  const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
      :: "r"(address), "r"(value.bits.x), "r"(value.bits.y),
         "r"(value.bits.z), "r"(value.bits.w) : "memory");
}

// The pinned CUTLASS RN converter returns two destination elements in one word;
// no intermediate low-precision value enters a subsequent S/T dependency.
template <class Element>
CUTE_DEVICE uint32_t small_round_pair(float even, float odd) {
  cutlass::Array<float,2> source;
  source[0] = even;
  source[1] = odd;
  using Converter = cutlass::NumericArrayConverter<
      Element,float,2,cutlass::FloatRoundStyle::round_to_nearest>;
  const auto rounded = Converter{}(source);
  static_assert(sizeof(rounded) == sizeof(uint32_t), "one pair is one packed word");
  // Same representation cast used by the pinned BF16 converter itself.
  return reinterpret_cast<const uint32_t&>(rounded);
}

CUTE_DEVICE void small_put_word(SmallPackedEight& output, int pair, uint32_t word) {
  if (pair == 0) output.bits.x = word;
  else if (pair == 1) output.bits.y = word;
  else if (pair == 2) output.bits.z = word;
  else output.bits.w = word;
}

template <class Element, class RawA, class RawB>
CUTE_DEVICE void small_issue_raw(const Problem& p, const Element* A,
                                 const Element* B, int64_t row_base,
                                 int64_t col_base, int64_t k_base, int lane,
                                 RawA rA, RawB rB) {
  const int64_t mh = p.m / 2, nh = p.n / 2, kh = p.k / 2;
  const int row = lane / 2;
  const int ka = (lane & 1) * 8;
  const int64_t gr = row_base + row, gka = k_base + ka;
  small_async_eight(A + gr*p.lda + gka, &rA(row,ka,0));
  small_async_eight(A + gr*p.lda + gka + kh, &rA(row,ka,1));
  small_async_eight(A + (gr+mh)*p.lda + gka, &rA(row,ka,2));
  small_async_eight(A + (gr+mh)*p.lda + gka + kh, &rA(row,ka,3));
  const int kb = lane / 2;
  const int col = (lane & 1) * 8;
  const int64_t gkb = k_base + kb, gc = col_base + col;
  small_async_eight(B + gkb*p.ldb + gc, &rB(col,kb,0));
  small_async_eight(B + gkb*p.ldb + gc + nh, &rB(col,kb,1));
  small_async_eight(B + (gkb+kh)*p.ldb + gc, &rB(col,kb,2));
  small_async_eight(B + (gkb+kh)*p.ldb + gc + nh, &rB(col,kb,3));
}

template <class Element, class RawA, class RawB, class SmemA, class SmemB>
CUTE_DEVICE void small_fill_from_raw(int lane, RawA rA, RawB rB,
                                    SmemA sA, SmemB sB) {
  const int row = lane / 2;
  const int k_a = (lane & 1) * 8;
  const SmallPackedEight a11 = small_read_raw_eight(&rA(row,k_a,0));
  const SmallPackedEight a12 = small_read_raw_eight(&rA(row,k_a,1));
  const SmallPackedEight a21 = small_read_raw_eight(&rA(row,k_a,2));
  const SmallPackedEight a22 = small_read_raw_eight(&rA(row,k_a,3));
  small_store_eight(&sA(row,k_a,0), a11);
  small_store_eight(&sA(row,k_a,1), a12);
  small_store_eight(&sA(row,k_a,3), a22);
  SmallPackedEight s1{make_uint4(0,0,0,0)}, s2{make_uint4(0,0,0,0)};
  SmallPackedEight s3{make_uint4(0,0,0,0)}, s4{make_uint4(0,0,0,0)};
  CUTE_UNROLL
  for (int pair = 0; pair < 4; ++pair) {
    float f1[2], f2[2], f3[2], f4[2];
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int i = 2 * pair + j;
      const float x11 = small_unpack<Element>(a11,i);
      const float x12 = small_unpack<Element>(a12,i);
      const float x21 = small_unpack<Element>(a21,i);
      const float x22 = small_unpack<Element>(a22,i);
      f1[j] = x21 + x22;
      f2[j] = f1[j] - x11;
      f3[j] = x11 - x21;
      f4[j] = x12 - f2[j];
    }
    small_put_word(s1,pair,small_round_pair<Element>(f1[0],f1[1]));
    small_put_word(s2,pair,small_round_pair<Element>(f2[0],f2[1]));
    small_put_word(s3,pair,small_round_pair<Element>(f3[0],f3[1]));
    small_put_word(s4,pair,small_round_pair<Element>(f4[0],f4[1]));
  }
  small_store_eight(&sA(row,k_a,2), s4);
  small_store_eight(&sA(row,k_a,4), s1);
  small_store_eight(&sA(row,k_a,5), s2);
  small_store_eight(&sA(row,k_a,6), s3);

  // B is logical (N,K), with N contiguous in shared and original global B.
  const int k_b = lane / 2;
  const int col = (lane & 1) * 8;
  const SmallPackedEight b11 = small_read_raw_eight(&rB(col,k_b,0));
  const SmallPackedEight b12 = small_read_raw_eight(&rB(col,k_b,1));
  const SmallPackedEight b21 = small_read_raw_eight(&rB(col,k_b,2));
  const SmallPackedEight b22 = small_read_raw_eight(&rB(col,k_b,3));
  small_store_eight(&sB(col,k_b,0), b11);
  small_store_eight(&sB(col,k_b,1), b21);
  small_store_eight(&sB(col,k_b,2), b22);
  SmallPackedEight t1{make_uint4(0,0,0,0)}, t2{make_uint4(0,0,0,0)};
  SmallPackedEight t3{make_uint4(0,0,0,0)}, t4{make_uint4(0,0,0,0)};
  CUTE_UNROLL
  for (int pair = 0; pair < 4; ++pair) {
    float f1[2], f2[2], f3[2], f4[2];
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int i = 2 * pair + j;
      const float x11 = small_unpack<Element>(b11,i);
      const float x12 = small_unpack<Element>(b12,i);
      const float x21 = small_unpack<Element>(b21,i);
      const float x22 = small_unpack<Element>(b22,i);
      f1[j] = x12 - x11;
      f2[j] = x22 - f1[j];
      f3[j] = x22 - x12;
      f4[j] = f2[j] - x21;
    }
    small_put_word(t1,pair,small_round_pair<Element>(f1[0],f1[1]));
    small_put_word(t2,pair,small_round_pair<Element>(f2[0],f2[1]));
    small_put_word(t3,pair,small_round_pair<Element>(f3[0],f3[1]));
    small_put_word(t4,pair,small_round_pair<Element>(f4[0],f4[1]));
  }
  small_store_eight(&sB(col,k_b,3), t4);
  small_store_eight(&sB(col,k_b,4), t1);
  small_store_eight(&sB(col,k_b,5), t2);
  small_store_eight(&sB(col,k_b,6), t3);
}

template <int Product, class SmemA, class SmemB, class Mma,
          class ThrMma, class CopyA, class CopyB, class Accumulator>
CUTE_DEVICE void small_leaf(SmemA sA, SmemB sB, Mma mma, ThrMma thr_mma,
                            CopyA copy_a, CopyB copy_b, int lane,
                            Accumulator& accumulator) {
  auto a = sA(_,_,Int<Product>{});
  auto b = sB(_,_,Int<Product>{});
  auto rA = thr_mma.partition_fragment_A(a);
  auto rB = thr_mma.partition_fragment_B(b);
  auto thr_copy_a = copy_a.get_thread_slice(lane);
  auto thr_copy_b = copy_b.get_thread_slice(lane);
  auto srcA = thr_copy_a.partition_S(a);
  auto srcB = thr_copy_b.partition_S(b);
  auto dstA = thr_copy_a.retile_D(rA);
  auto dstB = thr_copy_b.retile_D(rB);
  copy(copy_a, srcA, dstA);
  copy(copy_b, srcB, dstB);
  gemm(mma, rA, rB, accumulator);
}

template <class Element>
__global__ __launch_bounds__(32)
void small_cpasync_kernel(Problem p, const Element* __restrict__ A,
                      const Element* __restrict__ B, const float* C0, float* D) {
  using T = SmallMmaTraits<Element>;
  const int lane = threadIdx.x & 31;
  const int64_t row_base = int64_t(blockIdx.y) * T::BM;
  const int64_t col_base = int64_t(blockIdx.x) * T::BN;
  __shared__ __align__(128) Element storage[T::TotalElements];
  Element* operand_storage = storage + 2 * T::RawBankElements;
  auto sA = make_tensor(make_smem_ptr(operand_storage), typename T::LayoutA{});
  auto sB = make_tensor(make_smem_ptr(operand_storage + T::ElementsA), typename T::LayoutB{});
  typename T::Mma mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto coords = thr_mma.partition_C(make_identity_tensor(Shape<_16,_16>{}));
  auto p1 = thr_mma.make_fragment_C(coords);
  auto p2 = thr_mma.make_fragment_C(coords);
  auto p3 = thr_mma.make_fragment_C(coords);
  auto p4 = thr_mma.make_fragment_C(coords);
  auto p5 = thr_mma.make_fragment_C(coords);
  auto p6 = thr_mma.make_fragment_C(coords);
  auto p7 = thr_mma.make_fragment_C(coords);
  clear(p1); clear(p2); clear(p3); clear(p4);
  clear(p5); clear(p6); clear(p7);
  static_assert(size(p1) == 8, "16x16 owner must have eight FP32 values per lane");
  auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{}, mma);
  auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T,Element>{}, mma);
  // Raw bank 0 is ready before its first generic shared read.
  int current = 0;
  auto raw0A = make_tensor(make_smem_ptr(storage), typename T::RawLayoutA{});
  auto raw0B = make_tensor(make_smem_ptr(storage + T::RawElementsA), typename T::RawLayoutB{});
  small_issue_raw(p,A,B,row_base,col_base,0,lane,raw0A,raw0B);
  small_async_commit();
  small_async_wait_publish();
  const int64_t kh = p.k / 2;
  CUTE_NO_UNROLL
  for (int64_t k_base = 0; k_base < kh; k_base += T::BK) {
    const int next = current ^ 1;
    const int64_t next_k = k_base + T::BK;
    // At the previous end barrier all users of this reused raw bank finished.
    // Only original quadrant bits are copied; S/T transforms stay FP32 below.
    if (next_k < kh) {
      Element* next_storage = storage + next * T::RawBankElements;
      auto nextA = make_tensor(make_smem_ptr(next_storage), typename T::RawLayoutA{});
      auto nextB = make_tensor(make_smem_ptr(next_storage + T::RawElementsA), typename T::RawLayoutB{});
      small_issue_raw(p,A,B,row_base,col_base,next_k,lane,nextA,nextB);
      small_async_commit();
    }
    Element* current_storage = storage + current * T::RawBankElements;
    auto rawA = make_tensor(make_smem_ptr(current_storage), typename T::RawLayoutA{});
    auto rawB = make_tensor(make_smem_ptr(current_storage + T::RawElementsA), typename T::RawLayoutB{});
    small_fill_from_raw<Element>(lane,rawA,rawB,sA,sB);
    __syncwarp(0xffffffffu);
    small_leaf<0>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,p1);
    small_leaf<1>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,p2);
    small_leaf<2>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,p3);
    small_leaf<3>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,p4);
    small_leaf<4>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,p5);
    small_leaf<5>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,p6);
    small_leaf<6>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,p7);
    // Per-lane wait completes next raw writes; the full-warp barrier publishes
    // them and releases the current raw/transformed banks before flip or reuse.
    small_async_wait_publish();
    current = next;
  }
  CUTE_UNROLL
  for (int i = 0; i < size(p1); ++i) {
    const auto coord = coords(i);
    const int64_t row = row_base + get<0>(coord);
    const int64_t col = col_base + get<1>(coord);
    const int64_t mh = p.m / 2;
    const int64_t nh = p.n / 2;
    { // Full owner tiles put every output coordinate inside its quadrant.
      const float u1 = p1(i) + p2(i);
      const float u2 = p1(i) + p6(i);
      const float u3 = u2 + p7(i);
      const float u4 = u2 + p5(i);
      const float u5 = u4 + p3(i);
      const float u6 = u3 - p4(i);
      const float u7 = u3 + p5(i);
      float d11 = __fmul_rn(p.alpha,u1);
      float d12 = __fmul_rn(p.alpha,u5);
      float d21 = __fmul_rn(p.alpha,u6);
      float d22 = __fmul_rn(p.alpha,u7);
      if (p.beta != 0.0f) {
        d11 = fmaf(p.beta,C0[row*p.ldc + col],d11);
        d12 = fmaf(p.beta,C0[row*p.ldc + col + nh],d12);
        d21 = fmaf(p.beta,C0[(row+mh)*p.ldc + col],d21);
        d22 = fmaf(p.beta,C0[(row+mh)*p.ldc + col + nh],d22);
      }
      D[row*p.ldd + col] = d11;
      D[row*p.ldd + col + nh] = d12;
      D[(row+mh)*p.ldd + col] = d21;
      D[(row+mh)*p.ldd + col + nh] = d22;
    }
  }
}

template <class Element>
void launch_small_cpasync_typed(const Problem& p, const void* A, const void* B,
                                const float* C0, float* D, cudaStream_t stream) {
  const dim3 grid(unsigned(p.n / 32),unsigned(p.m / 32));
  small_cpasync_kernel<Element><<<grid,32,0,stream>>>(
      p,static_cast<const Element*>(A),static_cast<const Element*>(B),C0,D);
}

} // namespace

bool small_cpasync_can_implement(const Problem& p, int variant) {
  // Preserve generic candidate 7 support for tails and unusual valid strides.
  return variant == 17 && packed_mma_can_implement(p,7);
}

bool small_cpasync_full_tiles(const Problem& p, const void* A, const void* B) {
  return small_cpasync_can_implement(p,17) && A && B &&
      p.m % 32 == 0 && p.n % 32 == 0 && p.k % 32 == 0 &&
      p.lda % 8 == 0 && p.ldb % 8 == 0 &&
      (reinterpret_cast<uintptr_t>(A) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(B) & 15) == 0;
}

void prepare_small_cpasync(const Problem& p, int variant) {
  if (!small_cpasync_can_implement(p,variant))
    throw std::invalid_argument("unsupported raw cp.async problem or variant");
  // Prepare the supported fallback. The full kernel's 15 KiB static shared
  // needs no opt-in, runtime setup, allocations, or host synchronization.
  prepare_packed_mma(p,7);
}

void launch_small_cpasync(const Problem& p, const void* A, const void* B,
                          const float* C0, float* D, cudaStream_t stream,
                          int variant) {
  if (!small_cpasync_can_implement(p,variant))
    throw std::invalid_argument("unsupported raw cp.async problem or variant");
  if (!D || (p.beta != 0.0f && !C0) ||
      (reinterpret_cast<uintptr_t>(D) & 3) != 0 ||
      (p.beta != 0.0f && (reinterpret_cast<uintptr_t>(C0) & 3) != 0))
    throw std::invalid_argument("raw cp.async requires aligned FP32 output/source");
  if (!small_cpasync_full_tiles(p,A,B)) {
    launch_packed_mma(p,A,B,C0,D,stream,7);
    return;
  }
  if (p.dtype == DType::BF16)
    launch_small_cpasync_typed<cutlass::bfloat16_t>(p,A,B,C0,D,stream);
  else launch_small_cpasync_typed<cutlass::half_t>(p,A,B,C0,D,stream);
}

} // namespace megafmm

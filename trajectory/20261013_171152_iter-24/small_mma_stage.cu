// Prototype only: Mega18, PLAN section 6.5 M2 stage contribution accumulation.
// Same complete 16x16/BK16 owner and operand DAG as variant8; no async pipeline.
// M2 changes FP32 accumulation order and is not bitwise-equivalent to M1.
#include "megafmm/kernels.h"

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>
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

template <class Element, bool FullTiles>
CUTE_DEVICE SmallPackedEight small_read_eight(
    const Element* ptr, bool row_valid, int64_t remaining) {
  SmallPackedEight result{make_uint4(0,0,0,0)};
  if constexpr (FullTiles) {
    result.bits = *reinterpret_cast<const uint4*>(ptr);
  } else if (row_valid && remaining >= 8 &&
      (reinterpret_cast<uintptr_t>(ptr) & 15) == 0) {
    result.bits = *reinterpret_cast<const uint4*>(ptr);
  } else {
    CUTE_UNROLL
    for (int lane = 0; lane < 8; ++lane)
      if (row_valid && int64_t(lane) < remaining)
        result.insert(lane, ptr[lane].raw());
  }
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

template <class Element, bool FullTiles, class SmemA, class SmemB>
CUTE_DEVICE void small_fill_dag(const Problem& p, const Element* A,
                                const Element* B, int64_t row_base,
                                int64_t col_base, int64_t k_base, int lane,
                                SmemA sA, SmemB sB) {
  const int64_t mh = p.m / 2;
  const int64_t nh = p.n / 2;
  const int64_t kh = p.k / 2;

  // Each lane handles eight consecutive A elements: two lanes per row.
  const int row = lane / 2;
  const int k_a = (lane & 1) * 8;
  const int64_t gr = row_base + row;
  const int64_t gk_a = k_base + k_a;
  const bool valid_a = FullTiles || gr < mh;
  const int64_t r = FullTiles ? gr : (valid_a ? gr : 0);
  const int64_t ka = FullTiles ? gk_a : (gk_a < kh ? gk_a : 0);
  const int64_t remaining_a = kh - gk_a;
  const SmallPackedEight a11 = small_read_eight<Element,FullTiles>(A + r*p.lda + ka, valid_a, remaining_a);
  const SmallPackedEight a12 = small_read_eight<Element,FullTiles>(A + r*p.lda + ka + kh, valid_a, remaining_a);
  const SmallPackedEight a21 = small_read_eight<Element,FullTiles>(A + (r+mh)*p.lda + ka, valid_a, remaining_a);
  const SmallPackedEight a22 = small_read_eight<Element,FullTiles>(A + (r+mh)*p.lda + ka + kh, valid_a, remaining_a);
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
  const int64_t gk_b = k_base + k_b;
  const int64_t gc = col_base + col;
  const bool valid_b = FullTiles || gk_b < kh;
  const int64_t kb = FullTiles ? gk_b : (valid_b ? gk_b : 0);
  const int64_t c = FullTiles ? gc : (gc < nh ? gc : 0);
  const int64_t remaining_b = nh - gc;
  const SmallPackedEight b11 = small_read_eight<Element,FullTiles>(B + kb*p.ldb + c, valid_b, remaining_b);
  const SmallPackedEight b12 = small_read_eight<Element,FullTiles>(B + kb*p.ldb + c + nh, valid_b, remaining_b);
  const SmallPackedEight b21 = small_read_eight<Element,FullTiles>(B + (kb+kh)*p.ldb + c, valid_b, remaining_b);
  const SmallPackedEight b22 = small_read_eight<Element,FullTiles>(B + (kb+kh)*p.ldb + c + nh, valid_b, remaining_b);
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

// One product scratch is cleared and completed for this K16 stage before any
// contribution is applied. Product order is P1,P2,P3,P4,P5,P6,P7 for every stage.
// Coefficients (P1..P7):
// C11: +1,+1, 0, 0, 0, 0, 0
// C12: +1, 0,+1, 0,+1,+1, 0
// C21: +1, 0, 0,-1, 0,+1,+1
// C22: +1, 0, 0, 0,+1,+1,+1
// These are 14 separately rounded FP32 updates per output coordinate/stage;
// they deliberately replace M1's complete-P accumulation followed by the U DAG.
template <int Product, class SmemA, class SmemB, class Mma,
          class ThrMma, class CopyA, class CopyB, class Accumulator>
CUTE_DEVICE void small_stage_product(SmemA sA, SmemB sB, Mma mma,
                                     ThrMma thr_mma, CopyA copy_a, CopyB copy_b,
                                     int lane, Accumulator& scratch,
                                     Accumulator& c11, Accumulator& c12,
                                     Accumulator& c21, Accumulator& c22) {
  static_assert(Product >= 0 && Product < 7, "seven stage products");
  clear(scratch);
  small_leaf<Product>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,scratch);
  CUTE_UNROLL
  for (int i = 0; i < size(scratch); ++i) {
    const float value = scratch(i);
    if constexpr (Product == 0) {
      c11(i) = __fadd_rn(c11(i),value);
      c12(i) = __fadd_rn(c12(i),value);
      c21(i) = __fadd_rn(c21(i),value);
      c22(i) = __fadd_rn(c22(i),value);
    } else if constexpr (Product == 1) {
      c11(i) = __fadd_rn(c11(i),value);
    } else if constexpr (Product == 2) {
      c12(i) = __fadd_rn(c12(i),value);
    } else if constexpr (Product == 3) {
      c21(i) = __fsub_rn(c21(i),value);
    } else if constexpr (Product == 4) {
      c12(i) = __fadd_rn(c12(i),value);
      c22(i) = __fadd_rn(c22(i),value);
    } else if constexpr (Product == 5) {
      c12(i) = __fadd_rn(c12(i),value);
      c21(i) = __fadd_rn(c21(i),value);
      c22(i) = __fadd_rn(c22(i),value);
    } else {
      c21(i) = __fadd_rn(c21(i),value);
      c22(i) = __fadd_rn(c22(i),value);
    }
  }
}

template <class Element>
__global__ __launch_bounds__(32)
void small_stage_kernel(Problem p, const Element* __restrict__ A,
                        const Element* __restrict__ B, const float* C0, float* D) {
  using T = SmallMmaTraits<Element>;
  const int lane = threadIdx.x & 31;
  const int64_t row_base = int64_t(blockIdx.y) * T::BM;
  const int64_t col_base = int64_t(blockIdx.x) * T::BN;
  __shared__ __align__(128) Element storage[T::WarpElements];
  auto sA = make_tensor(make_smem_ptr(storage), typename T::LayoutA{});
  auto sB = make_tensor(make_smem_ptr(storage + T::ElementsA), typename T::LayoutB{});
  typename T::Mma mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto coords = thr_mma.partition_C(make_identity_tensor(Shape<_16,_16>{}));
  auto c11 = thr_mma.make_fragment_C(coords);
  auto c12 = thr_mma.make_fragment_C(coords);
  auto c21 = thr_mma.make_fragment_C(coords);
  auto c22 = thr_mma.make_fragment_C(coords);
  auto scratch = thr_mma.make_fragment_C(coords);
  clear(c11); clear(c12); clear(c21); clear(c22);
  static_assert(size(c11) == 8 && size(scratch) == 8,
                "M2 keeps 32 C plus eight product-scratch FP32 values per lane");
  auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{}, mma);
  auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T,Element>{}, mma);
  CUTE_NO_UNROLL
  for (int64_t k_base = 0; k_base < p.k / 2; k_base += T::BK) {
    small_fill_dag<Element,true>(p,A,B,row_base,col_base,k_base,lane,sA,sB);
    // All seven shared operands are published to the whole owner warp.
    __syncwarp(0xffffffffu);
    small_stage_product<0>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,scratch,c11,c12,c21,c22);
    small_stage_product<1>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,scratch,c11,c12,c21,c22);
    small_stage_product<2>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,scratch,c11,c12,c21,c22);
    small_stage_product<3>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,scratch,c11,c12,c21,c22);
    small_stage_product<4>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,scratch,c11,c12,c21,c22);
    small_stage_product<5>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,scratch,c11,c12,c21,c22);
    small_stage_product<6>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,scratch,c11,c12,c21,c22);
    // Synchronous register MMA has consumed the current operands before reuse.
    __syncwarp(0xffffffffu);
  }
  CUTE_UNROLL
  for (int i = 0; i < size(c11); ++i) {
    const auto coord = coords(i);
    const int64_t row = row_base + get<0>(coord);
    const int64_t col = col_base + get<1>(coord);
    const int64_t mh = p.m / 2;
    const int64_t nh = p.n / 2;
    // Apply alpha/beta once after all stage contributions, including all four
    // C0 reads before any output store; beta zero never dereferences C0.
    float d11 = __fmul_rn(p.alpha,c11(i));
    float d12 = __fmul_rn(p.alpha,c12(i));
    float d21 = __fmul_rn(p.alpha,c21(i));
    float d22 = __fmul_rn(p.alpha,c22(i));
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

template <class Element>
void launch_small_stage_typed(const Problem& p, const void* A, const void* B,
                              const float* C0, float* D, cudaStream_t stream) {
  const dim3 grid(unsigned(p.n / 32),unsigned(p.m / 32));
  small_stage_kernel<Element><<<grid,32,0,stream>>>(
      p,static_cast<const Element*>(A),static_cast<const Element*>(B),C0,D);
}

} // namespace

bool small_stage_can_implement(const Problem& p, int variant) {
  return variant == 18 && packed_mma_can_implement(p,7);
}

bool small_stage_full_tiles(const Problem& p, const void* A, const void* B) {
  return small_stage_can_implement(p,18) && A && B &&
      p.m % 32 == 0 && p.n % 32 == 0 && p.k % 32 == 0 &&
      p.lda % 8 == 0 && p.ldb % 8 == 0 &&
      (reinterpret_cast<uintptr_t>(A) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(B) & 15) == 0;
}

void prepare_small_stage(const Problem& p, int variant) {
  if (!small_stage_can_implement(p,variant))
    throw std::invalid_argument("unsupported stage-contribution problem or variant");
  // Both this 7 KiB static-shared kernel and generic7 need no runtime setup.
  prepare_packed_mma(p,7);
}

void launch_small_stage(const Problem& p, const void* A, const void* B,
                        const float* C0, float* D, cudaStream_t stream,
                        int variant) {
  if (!small_stage_can_implement(p,variant))
    throw std::invalid_argument("unsupported stage-contribution problem or variant");
  if (!D || (p.beta != 0.0f && !C0) ||
      (reinterpret_cast<uintptr_t>(D) & 3) != 0 ||
      (p.beta != 0.0f && (reinterpret_cast<uintptr_t>(C0) & 3) != 0))
    throw std::invalid_argument("stage-contribution requires aligned FP32 output/source");
  if (!small_stage_full_tiles(p,A,B)) {
    launch_packed_mma(p,A,B,C0,D,stream,7);
    return;
  }
  if (p.dtype == DType::BF16)
    launch_small_stage_typed<cutlass::bfloat16_t>(p,A,B,C0,D,stream);
  else launch_small_stage_typed<cutlass::half_t>(p,A,B,C0,D,stream);
}

} // namespace megafmm

// Prototype only: Mega variant 15, m16n8k16 owner with identity alpha/beta float2 output.
// Identity coefficients and complete tiles only; host fallback chain is 15 -> 14 -> 12 -> 7.
#include "megafmm/gemm.h"

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
  static constexpr int BN = 8;
  static constexpr int BK = 16;
  using Op = typename std::conditional<
      std::is_same<Element, cutlass::bfloat16_t>::value,
      SM80_16x8x16_F32BF16BF16F32_TN,
      SM80_16x8x16_F32F16F16F32_TN>::type;
  using Mma = decltype(make_tiled_mma(
      Op{}, Layout<Shape<_1,_1,_1>>{}, Tile<_16,_8,_16>{}));
  // Ordinary element-unit swizzles, not GMMA descriptor/pointer layouts.
  // Product stride 256 is a multiple of the swizzle's 128-element period.
  using LayoutA = decltype(composition(Swizzle<1,3,3>{},
      Layout<Shape<_16,_16,_7>, Stride<_16,_1,_256>>{}));
  // B has eight contiguous N elements per K row. Each 8x8 LDSM matrix
  // occupies 128 bytes without the stride-16 layout bank conflict.
  using LayoutB = Layout<Shape<_8,_16,_7>, Stride<_1,_8,_128>>;
  static constexpr int ElementsA = cosize_v<LayoutA>;
  static constexpr int ElementsB = cosize_v<LayoutB>;
  static constexpr int WarpElements = ElementsA + ElementsB;
  static constexpr int WarpSharedBytes = WarpElements * sizeof(Element);
  static_assert(ElementsA == 7 * BM * BK && ElementsB == 7 * BN * BK,
                "private operand slots must occupy exactly 5.25 KiB per warp");
  static_assert(WarpSharedBytes == 5376, "seven operand pairs use 5376 bytes");
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

// All 32 lanes own four adjacent B elements, two lanes per logical K row.
struct SmallPackedFour {
  uint2 bits;
  CUTE_DEVICE uint16_t raw(int lane) const {
    const uint32_t word = lane < 2 ? bits.x : bits.y;
    return uint16_t(word >> (16 * (lane & 1)));
  }
};

template <class Element>
CUTE_DEVICE SmallPackedFour small_read_four(const Element* ptr) {
  return SmallPackedFour{*reinterpret_cast<const uint2*>(ptr)};
}

template <class Element>
CUTE_DEVICE float small_unpack_four(const SmallPackedFour& value, int lane) {
  return float(Element::bitcast(value.raw(lane)));
}

CUTE_DEVICE void small_store_four(void* ptr, const SmallPackedFour& value) {
  const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.v2.b32 [%0], {%1, %2};"
      :: "r"(address), "r"(value.bits.x), "r"(value.bits.y) : "memory");
}

CUTE_DEVICE void small_put_four_word(SmallPackedFour& output, int pair,
                                    uint32_t word) {
  if (pair == 0) output.bits.x = word;
  else output.bits.y = word;
}

template <class Element>
CUTE_DEVICE SmallPackedEight small_read_eight(const Element* ptr) {
  // The public complete-tile guard proves bounds and actual 16-byte alignment.
  return SmallPackedEight{*reinterpret_cast<const uint4*>(ptr)};
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

template <class Element, class SmemA, class SmemB>
CUTE_DEVICE void small_fill_dag(const Problem& p, const Element* A,
                                const Element* B, int row_base,
                                int col_base, int k_base, int lane,
                                SmemA sA, SmemB sB) {
  const int mh = int(p.m) / 2;
  const int nh = int(p.n) / 2;
  const int kh = int(p.k) / 2;

  // Each lane handles eight consecutive A elements: two lanes per row.
  const int row = lane / 2;
  const int k_a = (lane & 1) * 8;
  const int gr = row_base + row;
  const int gk_a = k_base + k_a;
  const SmallPackedEight a11 = small_read_eight(A + gr*p.lda + gk_a);
  const SmallPackedEight a12 = small_read_eight(A + gr*p.lda + gk_a + kh);
  const SmallPackedEight a21 = small_read_eight(A + (gr+mh)*p.lda + gk_a);
  const SmallPackedEight a22 = small_read_eight(A + (gr+mh)*p.lda + gk_a + kh);
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

  // Every lane handles four N elements: two adjacent half-rows per K row.
  // No divergent B fill block; the shared B layout and MMA copy are unchanged.
  const int k_b = lane / 2;
  const int col = (lane & 1) * 4;
  const int gk_b = k_base + k_b;
  const int gc = col_base + col;
  const SmallPackedFour b11 = small_read_four(B + gk_b*p.ldb + gc);
  const SmallPackedFour b12 = small_read_four(B + gk_b*p.ldb + gc + nh);
  const SmallPackedFour b21 = small_read_four(B + (gk_b+kh)*p.ldb + gc);
  const SmallPackedFour b22 = small_read_four(B + (gk_b+kh)*p.ldb + gc + nh);
  small_store_four(&sB(col,k_b,0), b11);
  small_store_four(&sB(col,k_b,1), b21);
  small_store_four(&sB(col,k_b,2), b22);
  SmallPackedFour t1{make_uint2(0,0)}, t2{make_uint2(0,0)};
  SmallPackedFour t3{make_uint2(0,0)}, t4{make_uint2(0,0)};
  CUTE_UNROLL
  for (int pair = 0; pair < 2; ++pair) {
    float f1[2], f2[2], f3[2], f4[2];
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int i = 2 * pair + j;
      const float x11 = small_unpack_four<Element>(b11,i);
      const float x12 = small_unpack_four<Element>(b12,i);
      const float x21 = small_unpack_four<Element>(b21,i);
      const float x22 = small_unpack_four<Element>(b22,i);
      f1[j] = x12 - x11;
      f2[j] = x22 - f1[j];
      f3[j] = x22 - x12;
      f4[j] = f2[j] - x21;
    }
    small_put_four_word(t1,pair,small_round_pair<Element>(f1[0],f1[1]));
    small_put_four_word(t2,pair,small_round_pair<Element>(f2[0],f2[1]));
    small_put_four_word(t3,pair,small_round_pair<Element>(f3[0],f3[1]));
    small_put_four_word(t4,pair,small_round_pair<Element>(f4[0],f4[1]));
  }
  small_store_four(&sB(col,k_b,3), t4);
  small_store_four(&sB(col,k_b,4), t1);
  small_store_four(&sB(col,k_b,5), t2);
  small_store_four(&sB(col,k_b,6), t3);

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
void mma16x8_identity_kernel(Problem p, const Element* __restrict__ A,
                      const Element* __restrict__ B, float* D) {
  using T = SmallMmaTraits<Element>;
  const int lane = threadIdx.x & 31;
  const int row_base = int(blockIdx.y) * T::BM;
  const int col_base = int(blockIdx.x) * T::BN;
  const int mh = int(p.m) / 2;
  const int nh = int(p.n) / 2;
  const int kh = int(p.k) / 2;
  __shared__ __align__(128) Element storage[T::WarpElements];
  Element* warp_storage = storage;
  auto sA = make_tensor(make_smem_ptr(warp_storage), typename T::LayoutA{});
  auto sB = make_tensor(make_smem_ptr(warp_storage + T::ElementsA), typename T::LayoutB{});
  typename T::Mma mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto coords = thr_mma.partition_C(make_identity_tensor(Shape<_16,_8>{}));
  auto p1 = thr_mma.make_fragment_C(coords);
  auto p2 = thr_mma.make_fragment_C(coords);
  auto p3 = thr_mma.make_fragment_C(coords);
  auto p4 = thr_mma.make_fragment_C(coords);
  auto p5 = thr_mma.make_fragment_C(coords);
  auto p6 = thr_mma.make_fragment_C(coords);
  auto p7 = thr_mma.make_fragment_C(coords);
  clear(p1); clear(p2); clear(p3); clear(p4);
  clear(p5); clear(p6); clear(p7);
  static_assert(size(p1) == 4, "16x8 owner must have four FP32 values per lane");
  auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{}, mma);
  auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T,Element>{}, mma);
  CUTE_NO_UNROLL
  for (int k_base = 0; k_base < kh; k_base += T::BK) {
    small_fill_dag<Element>(p,A,B,row_base,col_base,k_base,lane,sA,sB);
    // Generic shared stores -> synchronous ldmatrix reads, ordered within the
    // owner warp. These register-based MMAs need no WGMMA async proxy protocol.
    __syncwarp(0xffffffffu);
    small_leaf<0>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,p1);
    small_leaf<1>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,p2);
    small_leaf<2>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,p3);
    small_leaf<3>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,p4);
    small_leaf<4>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,p5);
    small_leaf<5>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,p6);
    small_leaf<6>(sA,sB,mma,thr_mma,copy_a,copy_b,lane,p7);
    __syncwarp(0xffffffffu);
  }
  // In the native m16n8 atom C values (0,1) and (2,3) are adjacent
  // columns in the same row. Full tiles keep both values within a quadrant.
  CUTE_UNROLL
  for (int pair = 0; pair < 2; ++pair) {
    const int i = 2 * pair;
    const auto coord = coords(i);
    const int row = row_base + int(get<0>(coord));
    const int col = col_base + int(get<1>(coord));
    float2 d11, d12, d21, d22;
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int v = i + j;
      const float u1 = p1(v) + p2(v);
      const float u2 = p1(v) + p6(v);
      const float u3 = u2 + p7(v);
      const float u4 = u2 + p5(v);
      const float u5 = u4 + p3(v);
      const float u6 = u3 - p4(v);
      const float u7 = u3 + p5(v);
      if (j == 0) {
        d11.x = u1;
        d12.x = u5;
        d21.x = u6;
        d22.x = u7;
      } else {
        d11.y = u1;
        d12.y = u5;
        d21.y = u6;
        d22.y = u7;
      }
    }
    // Row/column counters fit int, while multiplying by a stride keeps every
    // matrix address in int64_t, as required by the public overflow contract.
    const int64_t offset11 = row*p.ldd + col;
    const int64_t offset12 = row*p.ldd + col + nh;
    const int64_t offset21 = (row+mh)*p.ldd + col;
    const int64_t offset22 = (row+mh)*p.ldd + col + nh;
    // Immutable alpha==1 and beta==0: directly preserve each U bit pattern,
    // including signed zero. The kernel has no C0 argument or C0 address path.
    *reinterpret_cast<float2*>(D + offset11) = d11;
    *reinterpret_cast<float2*>(D + offset12) = d12;
    *reinterpret_cast<float2*>(D + offset21) = d21;
    *reinterpret_cast<float2*>(D + offset22) = d22;
  }
}

template <class Element>
void launch_mma16x8_identity_typed(const Problem& p, const void* A, const void* B,
                         const float* C0, float* D, cudaStream_t stream) {
  const dim3 grid(unsigned(p.n / 16), unsigned(p.m / 32));
  mma16x8_identity_kernel<Element><<<grid,32,0,stream>>>(
      p,static_cast<const Element*>(A),static_cast<const Element*>(B),D);
}

} // namespace

bool mma16x8_identity_can_implement(const Problem& p, int variant) {
  if (variant != 15 || p.trans_a || p.trans_b ||
      p.alpha != 1.0f || p.beta != 0.0f) return false;
  if (p.dtype != DType::BF16 && p.dtype != DType::FP16) return false;
  if (p.m <= 0 || p.n <= 0 || p.k <= 0 || p.m > 32768 ||
      p.n > 32768 || p.k > 32768 || p.m % 32 || p.n % 16 || p.k % 32)
    return false;
  if (p.lda < p.k || p.ldb < p.n || p.ldd < p.n ||
      p.lda % 8 || p.ldb % 8 || p.ldd % 2) return false;
  const int64_t max_elements = std::numeric_limits<int64_t>::max() / 4;
  if (p.lda > max_elements / p.m || p.ldb > max_elements / p.k ||
      p.ldd > max_elements / p.m) return false;
  return std::isfinite(p.alpha) && std::isfinite(p.beta);
}

bool mma16x8_identity_full_tiles(const Problem& p, const void* A, const void* B,
                                 const float* C0, const float* D) {
  return mma16x8_identity_can_implement(p,15) && A && B && D &&
      (reinterpret_cast<uintptr_t>(A) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(B) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(D) & 7) == 0;
}

void prepare_mma16x8_identity(const Problem& p, int variant) {
  if (!mma16x8_identity_can_implement(p,variant))
    throw std::invalid_argument("unsupported 16x8 mma.sync problem or variant");
  // One warp and 5376 bytes static shared: no opt-in, allocations, or runtime setup.
}

void launch_mma16x8_identity(const Problem& p, const void* A, const void* B,
                    const float* C0, float* D, cudaStream_t stream, int variant) {
  if (!mma16x8_identity_can_implement(p,variant) ||
      !mma16x8_identity_full_tiles(p,A,B,C0,D))
    throw std::invalid_argument("16x8 identity output requires alpha1 beta0 aligned complete tiles");
  if (p.dtype == DType::BF16)
    launch_mma16x8_identity_typed<cutlass::bfloat16_t>(p,A,B,C0,D,stream);
  else launch_mma16x8_identity_typed<cutlass::half_t>(p,A,B,C0,D,stream);
}

} // namespace megafmm

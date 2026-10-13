// Paired final Winograd operand conversions, with a complete-tile specialization.
// Candidate 7 stays general; candidate 8 uses a two-dimensional complete-tile grid.
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

template <class Element, bool FullTiles>
__global__ __launch_bounds__(32)
void small_mma_kernel(Problem p, const Element* __restrict__ A,
                      const Element* __restrict__ B, const float* C0, float* D) {
  using T = SmallMmaTraits<Element>;
  const int lane = threadIdx.x & 31;
  int64_t row_base, col_base;
  if constexpr (FullTiles) {
    row_base = int64_t(blockIdx.y) * T::BM;
    col_base = int64_t(blockIdx.x) * T::BN;
  } else {
    const int64_t tiles_n = (p.n / 2 + T::BN - 1) / T::BN;
    const int64_t tiles_m = (p.m / 2 + T::BM - 1) / T::BM;
    const int64_t owner = int64_t(blockIdx.x);
    // Every generic CTA is one complete owner warp.
    if (owner >= tiles_m * tiles_n) return;
    row_base = (owner / tiles_n) * T::BM;
    col_base = (owner % tiles_n) * T::BN;
  }
  __shared__ __align__(128) Element storage[T::WarpElements];
  Element* warp_storage = storage;
  auto sA = make_tensor(make_smem_ptr(warp_storage), typename T::LayoutA{});
  auto sB = make_tensor(make_smem_ptr(warp_storage + T::ElementsA), typename T::LayoutB{});
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
  CUTE_NO_UNROLL
  for (int64_t k_base = 0; k_base < p.k / 2; k_base += T::BK) {
    small_fill_dag<Element,FullTiles>(p,A,B,row_base,col_base,k_base,lane,sA,sB);
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
  CUTE_UNROLL
  for (int i = 0; i < size(p1); ++i) {
    const auto coord = coords(i);
    const int64_t row = row_base + get<0>(coord);
    const int64_t col = col_base + get<1>(coord);
    const int64_t mh = p.m / 2;
    const int64_t nh = p.n / 2;
    if (FullTiles || (row < mh && col < nh)) {
      const float u1 = p1(i) + p2(i);
      const float u2 = p1(i) + p6(i);
      const float u3 = u2 + p7(i);
      const float u4 = u2 + p5(i);
      const float u5 = u4 + p3(i);
      const float u6 = u3 - p4(i);
      const float u7 = u3 + p5(i);
      float d11 = p.alpha * u1;
      float d12 = p.alpha * u5;
      float d21 = p.alpha * u6;
      float d22 = p.alpha * u7;
      if (p.beta != 0.0f) {
        d11 += p.beta * C0[row*p.ldc + col];
        d12 += p.beta * C0[row*p.ldc + col + nh];
        d21 += p.beta * C0[(row+mh)*p.ldc + col];
        d22 += p.beta * C0[(row+mh)*p.ldc + col + nh];
      }
      D[row*p.ldd + col] = d11;
      D[row*p.ldd + col + nh] = d12;
      D[(row+mh)*p.ldd + col] = d21;
      D[(row+mh)*p.ldd + col + nh] = d22;
    }
  }
}

template <class Element, bool FullTiles>
void launch_packed_mma_typed(const Problem& p, const void* A, const void* B,
                             const float* C0, float* D, cudaStream_t stream) {
  const unsigned tiles_m = unsigned((p.m / 2 + 15) / 16);
  const unsigned tiles_n = unsigned((p.n / 2 + 15) / 16);
  const dim3 grid = FullTiles ? dim3(tiles_n,tiles_m) : dim3(tiles_m*tiles_n);
  small_mma_kernel<Element,FullTiles><<<grid,32,0,stream>>>(
      p,static_cast<const Element*>(A),static_cast<const Element*>(B),C0,D);
}

bool full_tiles(const Problem& p, const void* A, const void* B) {
  return p.m % 32 == 0 && p.n % 32 == 0 && p.k % 32 == 0 &&
      p.lda % 8 == 0 && p.ldb % 8 == 0 &&
      (reinterpret_cast<uintptr_t>(A) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(B) & 15) == 0;
}

template <class Element>
void launch_packed_mma_dtype(const Problem& p, const void* A, const void* B,
                             const float* C0, float* D, cudaStream_t stream,
                             int variant) {
  if (variant == 8 && full_tiles(p,A,B))
    launch_packed_mma_typed<Element,true>(p,A,B,C0,D,stream);
  else launch_packed_mma_typed<Element,false>(p,A,B,C0,D,stream);
}

} // namespace

bool packed_mma_can_implement(const Problem& p, int variant) {
  if (variant < 7 || variant > 8 || p.trans_a || p.trans_b) return false;
  if (p.dtype != DType::BF16 && p.dtype != DType::FP16) return false;
  if (p.m <= 0 || p.n <= 0 || p.k <= 0 || p.m > 32768 ||
      p.n > 32768 || p.k > 32768 || (p.m & 1) || (p.n & 1) || (p.k & 1)) return false;
  if (p.lda < p.k || p.ldb < p.n || p.ldd < p.n ||
      (p.beta != 0.0f && p.ldc < p.n)) return false;
  const int64_t max_elements = std::numeric_limits<int64_t>::max() / 4;
  if (p.lda > max_elements / p.m || p.ldb > max_elements / p.k ||
      p.ldd > max_elements / p.m ||
      (p.beta != 0.0f && p.ldc > max_elements / p.m)) return false;
  return std::isfinite(p.alpha) && std::isfinite(p.beta);
}

void prepare_packed_mma(const Problem& p, int variant) {
  if (!packed_mma_can_implement(p,variant))
    throw std::invalid_argument("unsupported small mma.sync problem or variant");
  // Both variants use one warp and 7 KiB static shared, with no runtime setup.
}

void launch_packed_mma(const Problem& p, const void* A, const void* B,
                      const float* C0, float* D, cudaStream_t stream,
                      int variant) {
  if (!packed_mma_can_implement(p,variant))
    throw std::invalid_argument("unsupported small mma.sync problem or variant");
  if (!A || !B || !D || (p.beta != 0.0f && !C0))
    throw std::invalid_argument("small mma.sync requires nonnull input/output pointers");
  if (p.dtype == DType::BF16)
    launch_packed_mma_dtype<cutlass::bfloat16_t>(p,A,B,C0,D,stream,variant);
  else launch_packed_mma_dtype<cutlass::half_t>(p,A,B,C0,D,stream,variant);
}

} // namespace megafmm

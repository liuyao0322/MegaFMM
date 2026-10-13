// Experimental cooperative transformed-tile M1. Each CTA owns the matching
// four C quadrant tiles, and output warps share the transformed A/B arrays.
// Policy26:32x16x16,two warps; policy27:32x32x16,four warps.
// Seven complete FP32 P products survive full K; fixed U follows only afterward.
// Formal policy26/27. No global operand/product scratch or workspace.
#include "megafmm/gemm.h"
#include "megafmm/kernels.h"

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace megafmm {
namespace {
using namespace cute;

template <class Element, int Cols>
struct CoopMmaTraits {
  static_assert(Cols == 16 || Cols == 32, "only the two reviewed tiles are supported");
  static constexpr int BM = 32;
  static constexpr int BN = Cols;
  static constexpr int BK = 16;
  static constexpr int Threads = 2 * (BN / 16) * 32;
  using Op = typename std::conditional<
      std::is_same<Element, cutlass::bfloat16_t>::value,
      SM80_16x8x16_F32BF16BF16F32_TN,
      SM80_16x8x16_F32F16F16F32_TN>::type;
  using Mma = decltype(make_tiled_mma(
      Op{}, Layout<Shape<_2,Int<BN / 16>,_1>>{},
      Tile<_32,Int<BN>,_16>{}));
  // TiledMMA broadcasts A to N warps and B to M warps. The LDSM copies are
  // read-only; each transformed element is produced only once by this CTA.
  using LayoutA = decltype(composition(Swizzle<1,3,3>{},
      Layout<Shape<_32,_16,_7>, Stride<_16,_1,_512>>{}));
  // For BN32, the B LDSM source row stride32 needs two swizzle bits:
  // the eight16B rows of each matrix group then address distinct word banks.
  // This is a layout derivation, not a measured bank-conflict/performance claim.
  static constexpr int BSwizzleBits = BN == 32 ? 2 : 1;
  using LayoutB = decltype(composition(Swizzle<BSwizzleBits,3,3>{},
      Layout<Shape<Int<BN>,_16,_7>, Stride<_1,Int<BN>,Int<BN * 16>>>{}));
  static constexpr int ElementsA = cosize_v<LayoutA>;
  static constexpr int ElementsB = cosize_v<LayoutB>;
  static constexpr int SharedElements = ElementsA + ElementsB;
  static constexpr int SharedBytes = SharedElements * sizeof(Element);
  static_assert(ElementsA == 7 * BM * BK && ElementsB == 7 * BN * BK,
                "seven product operand slots must not overlap or contain padding");
  static_assert(size(Mma{}) == Threads, "M/N atom layout must own every CTA thread");
  static_assert(SharedBytes == (BN == 16 ? 10752 : 14336),
                "one final operand bank must remain below static shared limit");
};

struct CoopPackedEight {
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
CUTE_DEVICE CoopPackedEight coop_read_eight(const Element* ptr) {
  CoopPackedEight result;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(result.bits.x), "=r"(result.bits.y),
        "=r"(result.bits.z), "=r"(result.bits.w)
      : "l"(ptr) : "memory");
  return result;
}

template <class Element>
CUTE_DEVICE float coop_unpack(const CoopPackedEight& value, int lane) {
  return float(Element::bitcast(value.raw(lane)));
}

CUTE_DEVICE void coop_store_eight(void* ptr, const CoopPackedEight& value) {
  const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
      :: "r"(address), "r"(value.bits.x), "r"(value.bits.y),
         "r"(value.bits.z), "r"(value.bits.w) : "memory");
}

// The pinned CUTLASS RN converter returns two destination elements in one word;
// no intermediate low-precision value enters a subsequent S/T dependency.
template <class Element>
CUTE_DEVICE uint32_t coop_round_pair(float even, float odd) {
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

CUTE_DEVICE void coop_put_word(CoopPackedEight& output, int pair, uint32_t word) {
  if (pair == 0) output.bits.x = word;
  else if (pair == 1) output.bits.y = word;
  else if (pair == 2) output.bits.z = word;
  else output.bits.w = word;
}


// Config0 uses four consecutive B elements per thread to balance its64 threads.
// All vector addresses have8B alignment under the stronger AB16B binding guard.
struct CoopPackedFour {
  uint2 bits;
  CUTE_DEVICE uint16_t raw(int lane) const {
    const uint32_t word = lane < 2 ? bits.x : bits.y;
    return uint16_t(word >> (16 * (lane & 1)));
  }
};

template <class Element>
CUTE_DEVICE CoopPackedFour coop_read_four(const Element* ptr) {
  CoopPackedFour result;
  asm volatile("ld.global.v2.u32 {%0, %1}, [%2];"
      : "=r"(result.bits.x), "=r"(result.bits.y) : "l"(ptr) : "memory");
  return result;
}

template <class Element>
CUTE_DEVICE float coop_unpack_four(const CoopPackedFour& value, int lane) {
  return float(Element::bitcast(value.raw(lane)));
}

CUTE_DEVICE void coop_store_four(void* ptr, const CoopPackedFour& value) {
  const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.v2.b32 [%0], {%1, %2};"
      :: "r"(address), "r"(value.bits.x), "r"(value.bits.y) : "memory");
}

CUTE_DEVICE void coop_put_four_word(CoopPackedFour& output, int pair,
                                     uint32_t word) {
  if (pair == 0) output.bits.x = word;
  else output.bits.y = word;
}

// Each of the64 A tasks handles8K elements of one of32 rows. It emits final
// operands immediately, so A raw/conversion temporaries die before B filling.
template <class Element, class SmemA>
CUTE_DEVICE void coop_fill_a(const Problem& p, const Element* A,
                             int row_base, int k_base, int task, SmemA sA) {
  const int64_t mh = p.m / 2;
  const int64_t kh = p.k / 2;
  const int row = task / 2;
  const int k = (task & 1) * 8;
  const int64_t gr = int64_t(row_base) + row;
  const int64_t gk = int64_t(k_base) + k;
  const CoopPackedEight a11 = coop_read_eight(A + gr*p.lda + gk);
  const CoopPackedEight a12 = coop_read_eight(A + gr*p.lda + gk + kh);
  const CoopPackedEight a21 = coop_read_eight(A + (gr+mh)*p.lda + gk);
  const CoopPackedEight a22 = coop_read_eight(A + (gr+mh)*p.lda + gk + kh);
  coop_store_eight(&sA(row,k,0),a11);
  coop_store_eight(&sA(row,k,1),a12);
  coop_store_eight(&sA(row,k,3),a22);
  CoopPackedEight s1{make_uint4(0,0,0,0)}, s2{make_uint4(0,0,0,0)};
  CoopPackedEight s3{make_uint4(0,0,0,0)}, s4{make_uint4(0,0,0,0)};
  CUTE_UNROLL
  for (int pair = 0; pair < 4; ++pair) {
    float f1[2], f2[2], f3[2], f4[2];
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int i = 2 * pair + j;
      const float x11 = coop_unpack<Element>(a11,i);
      const float x12 = coop_unpack<Element>(a12,i);
      const float x21 = coop_unpack<Element>(a21,i);
      const float x22 = coop_unpack<Element>(a22,i);
      f1[j] = x21 + x22;
      f2[j] = f1[j] - x11;
      f3[j] = x11 - x21;
      f4[j] = x12 - f2[j];
    }
    coop_put_word(s1,pair,coop_round_pair<Element>(f1[0],f1[1]));
    coop_put_word(s2,pair,coop_round_pair<Element>(f2[0],f2[1]));
    coop_put_word(s3,pair,coop_round_pair<Element>(f3[0],f3[1]));
    coop_put_word(s4,pair,coop_round_pair<Element>(f4[0],f4[1]));
  }
  coop_store_eight(&sA(row,k,2),s4);
  coop_store_eight(&sA(row,k,4),s1);
  coop_store_eight(&sA(row,k,5),s2);
  coop_store_eight(&sA(row,k,6),s3);
}

// The64 B tasks cover the common BNx16 tile once. Config0 vector width4
// makes every output thread a balanced producer; config1 uses width8 while
// the other64 threads produce A. All dependencies use FP32 before final RN.
template <class Element, int Cols, class SmemB>
CUTE_DEVICE void coop_fill_b(const Problem& p, const Element* B,
                             int col_base, int k_base, int task, SmemB sB) {
  const int64_t nh = p.n / 2;
  const int64_t kh = p.k / 2;
  constexpr int Width = Cols == 16 ? 4 : 8;
  const int k = task / 4;
  const int col = (task & 3) * Width;
  const int64_t gk = int64_t(k_base) + k;
  const int64_t gc = int64_t(col_base) + col;
  if constexpr (Width == 4) {
    const CoopPackedFour b11 = coop_read_four(B + gk*p.ldb + gc);
    const CoopPackedFour b12 = coop_read_four(B + gk*p.ldb + gc + nh);
    const CoopPackedFour b21 = coop_read_four(B + (gk+kh)*p.ldb + gc);
    const CoopPackedFour b22 = coop_read_four(B + (gk+kh)*p.ldb + gc + nh);
    coop_store_four(&sB(col,k,0),b11);
    coop_store_four(&sB(col,k,1),b21);
    coop_store_four(&sB(col,k,2),b22);
    CoopPackedFour t1{make_uint2(0,0)}, t2{make_uint2(0,0)};
    CoopPackedFour t3{make_uint2(0,0)}, t4{make_uint2(0,0)};
    CUTE_UNROLL
    for (int pair = 0; pair < 2; ++pair) {
      float f1[2], f2[2], f3[2], f4[2];
      CUTE_UNROLL
      for (int j = 0; j < 2; ++j) {
        const int i = 2 * pair + j;
        const float x11 = coop_unpack_four<Element>(b11,i);
        const float x12 = coop_unpack_four<Element>(b12,i);
        const float x21 = coop_unpack_four<Element>(b21,i);
        const float x22 = coop_unpack_four<Element>(b22,i);
        f1[j] = x12 - x11;
        f2[j] = x22 - f1[j];
        f3[j] = x22 - x12;
        f4[j] = f2[j] - x21;
      }
      coop_put_four_word(t1,pair,coop_round_pair<Element>(f1[0],f1[1]));
      coop_put_four_word(t2,pair,coop_round_pair<Element>(f2[0],f2[1]));
      coop_put_four_word(t3,pair,coop_round_pair<Element>(f3[0],f3[1]));
      coop_put_four_word(t4,pair,coop_round_pair<Element>(f4[0],f4[1]));
    }
    coop_store_four(&sB(col,k,3),t4);
    coop_store_four(&sB(col,k,4),t1);
    coop_store_four(&sB(col,k,5),t2);
    coop_store_four(&sB(col,k,6),t3);
  } else {
    const CoopPackedEight b11 = coop_read_eight(B + gk*p.ldb + gc);
    const CoopPackedEight b12 = coop_read_eight(B + gk*p.ldb + gc + nh);
    const CoopPackedEight b21 = coop_read_eight(B + (gk+kh)*p.ldb + gc);
    const CoopPackedEight b22 = coop_read_eight(B + (gk+kh)*p.ldb + gc + nh);
    coop_store_eight(&sB(col,k,0),b11);
    coop_store_eight(&sB(col,k,1),b21);
    coop_store_eight(&sB(col,k,2),b22);
    CoopPackedEight t1{make_uint4(0,0,0,0)}, t2{make_uint4(0,0,0,0)};
    CoopPackedEight t3{make_uint4(0,0,0,0)}, t4{make_uint4(0,0,0,0)};
    CUTE_UNROLL
    for (int pair = 0; pair < 4; ++pair) {
      float f1[2], f2[2], f3[2], f4[2];
      CUTE_UNROLL
      for (int j = 0; j < 2; ++j) {
        const int i = 2 * pair + j;
        const float x11 = coop_unpack<Element>(b11,i);
        const float x12 = coop_unpack<Element>(b12,i);
        const float x21 = coop_unpack<Element>(b21,i);
        const float x22 = coop_unpack<Element>(b22,i);
        f1[j] = x12 - x11;
        f2[j] = x22 - f1[j];
        f3[j] = x22 - x12;
        f4[j] = f2[j] - x21;
      }
      coop_put_word(t1,pair,coop_round_pair<Element>(f1[0],f1[1]));
      coop_put_word(t2,pair,coop_round_pair<Element>(f2[0],f2[1]));
      coop_put_word(t3,pair,coop_round_pair<Element>(f3[0],f3[1]));
      coop_put_word(t4,pair,coop_round_pair<Element>(f4[0],f4[1]));
    }
    coop_store_eight(&sB(col,k,3),t4);
    coop_store_eight(&sB(col,k,4),t1);
    coop_store_eight(&sB(col,k,5),t2);
    coop_store_eight(&sB(col,k,6),t3);
  }
}

template <int Product, class SmemA, class SmemB, class Mma,
          class ThrMma, class CopyA, class CopyB, class Accumulator>
CUTE_DEVICE void coop_leaf(SmemA sA, SmemB sB, Mma mma, ThrMma thr_mma,
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


template <class Element, int Cols>
__global__ __launch_bounds__((CoopMmaTraits<Element,Cols>::Threads))
void cooperative_mma_kernel(Problem p, const Element* __restrict__ A,
                             const Element* __restrict__ B,
                             const float* C0, float* D) {
  using T = CoopMmaTraits<Element,Cols>;
  const int tid = int(threadIdx.x);
  const int row_base = int(blockIdx.y) * T::BM;
  const int col_base = int(blockIdx.x) * T::BN;
  // The host full-tile guard launches only valid whole-CTA owners. No lane,
  // warp, or producer role can return before either block barrier below.
  __shared__ __align__(128) Element storage[T::SharedElements];
  auto sA = make_tensor(make_smem_ptr(storage),typename T::LayoutA{});
  auto sB = make_tensor(make_smem_ptr(storage + T::ElementsA),typename T::LayoutB{});
  typename T::Mma mma;
  auto thr_mma = mma.get_thread_slice(tid);
  // BN32's N-warps own interleaved nativeN8 blocks (warpN*8,+repeat16).
  // Use the CuTe identity partition for every P and final store coordinate.
  auto coords = thr_mma.partition_C(make_identity_tensor(Shape<_32,Int<Cols>>{}));
  auto p1 = thr_mma.make_fragment_C(coords);
  auto p2 = thr_mma.make_fragment_C(coords);
  auto p3 = thr_mma.make_fragment_C(coords);
  auto p4 = thr_mma.make_fragment_C(coords);
  auto p5 = thr_mma.make_fragment_C(coords);
  auto p6 = thr_mma.make_fragment_C(coords);
  auto p7 = thr_mma.make_fragment_C(coords);
  clear(p1); clear(p2); clear(p3); clear(p4);
  clear(p5); clear(p6); clear(p7);
  static_assert(size(p1) == 8, "each output thread owns8 CuTe C coordinates per product");
  auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{},mma);
  auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T,Element>{},mma);
  CUTE_NO_UNROLL
  for (int k_base = 0; k_base < int(p.k / 2); k_base += T::BK) {
    if constexpr (Cols == 16) {
      coop_fill_a(p,A,row_base,k_base,tid,sA);
      coop_fill_b<Element,Cols>(p,B,col_base,k_base,tid,sB);
    } else {
      if (tid < 64) coop_fill_a(p,A,row_base,k_base,tid,sA);
      else coop_fill_b<Element,Cols>(p,B,col_base,k_base,tid-64,sB);
    }
    // Ordinary shared writes -> synchronous LDSM and mma.sync. Every output
    // warp reads the common arrays after all producers have published them.
    __syncthreads();
    coop_leaf<0>(sA,sB,mma,thr_mma,copy_a,copy_b,tid,p1);
    coop_leaf<1>(sA,sB,mma,thr_mma,copy_a,copy_b,tid,p2);
    coop_leaf<2>(sA,sB,mma,thr_mma,copy_a,copy_b,tid,p3);
    coop_leaf<3>(sA,sB,mma,thr_mma,copy_a,copy_b,tid,p4);
    coop_leaf<4>(sA,sB,mma,thr_mma,copy_a,copy_b,tid,p5);
    coop_leaf<5>(sA,sB,mma,thr_mma,copy_a,copy_b,tid,p6);
    coop_leaf<6>(sA,sB,mma,thr_mma,copy_a,copy_b,tid,p7);
    // All synchronous readers have finished before any producer overwrites
    // this single operand bank in the next ascending K16 stage.
    __syncthreads();
  }
  CUTE_UNROLL
  for (int i = 0; i < size(p1); ++i) {
    const auto coord = coords(i);
    const int64_t row = int64_t(row_base) + get<0>(coord);
    const int64_t col = int64_t(col_base) + get<1>(coord);
    const int64_t mh = p.m / 2;
    const int64_t nh = p.n / 2;
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
      // All four active C0 reads occur before the four D stores. C0 may alias D.
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

template <class Element, int Cols>
void launch_cooperative_mma_typed(const Problem& p, const void* A, const void* B,
                                  const float* C0, float* D, cudaStream_t stream) {
  using T = CoopMmaTraits<Element,Cols>;
  const dim3 grid(unsigned(p.n / (2 * T::BN)),unsigned(p.m / (2 * T::BM)));
  cooperative_mma_kernel<Element,Cols><<<grid,T::Threads,0,stream>>>(
      p,static_cast<const Element*>(A),static_cast<const Element*>(B),C0,D);
}

template <int Cols>
void launch_cooperative_mma_dtype(const Problem& p, const void* A, const void* B,
                                  const float* C0, float* D, cudaStream_t stream) {
  if (p.dtype == DType::BF16)
    launch_cooperative_mma_typed<cutlass::bfloat16_t,Cols>(p,A,B,C0,D,stream);
  else launch_cooperative_mma_typed<cutlass::half_t,Cols>(p,A,B,C0,D,stream);
}
} // namespace

// Native metadata is strict; Plan retains packed7 for unsupported tile shapes.
bool cooperative_mma_can_implement(const Problem& p, int variant) {
  if ((variant != 26 && variant != 27) || !packed_mma_can_implement(p,7)) return false;
  const int cols = variant == 26 ? 16 : 32;
  return p.m % 64 == 0 && p.n % (2 * cols) == 0 && p.k % 32 == 0 &&
      p.lda % 8 == 0 && p.ldb % 8 == 0;
}

bool cooperative_mma_full_tiles(const Problem& p, const void* A, const void* B,
                                 int variant) {
  return cooperative_mma_can_implement(p,variant) && A && B &&
      (reinterpret_cast<uintptr_t>(A) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(B) & 15) == 0;
}

void prepare_cooperative_mma(const Problem& p, int variant) {
  if ((variant != 26 && variant != 27) || !packed_mma_can_implement(p,7))
    throw std::invalid_argument("unsupported cooperative mma.sync problem/variant");
  prepare_packed_mma(p,7);
  //10.5/14KiB static shared; no device setup, opt-in, allocation, or workspace.
}

void launch_cooperative_mma(const Problem& p, const void* A, const void* B,
                             const float* C0, float* D, cudaStream_t stream,
                             int variant) {
  if ((variant != 26 && variant != 27) || !packed_mma_can_implement(p,7))
    throw std::invalid_argument("unsupported cooperative mma.sync problem/variant");
  if (!A || !B || !D || (p.beta != 0.0f && !C0))
    throw std::invalid_argument("cooperative mma.sync requires nonnull active pointers");
  // Current AB bases and original matrix metadata are checked on every call.
  // Plan applies its common Dense guard before this helper; direct helper users
  // retain the existing generic packed7 fallback for nonnative shapes/bindings.
  if (variant == 26 && cooperative_mma_full_tiles(p,A,B,variant))
    launch_cooperative_mma_dtype<16>(p,A,B,C0,D,stream);
  else if (variant == 27 && cooperative_mma_full_tiles(p,A,B,variant))
    launch_cooperative_mma_dtype<32>(p,A,B,C0,D,stream);
  else launch_packed_mma(p,A,B,C0,D,stream,7);
}
} // namespace megafmm

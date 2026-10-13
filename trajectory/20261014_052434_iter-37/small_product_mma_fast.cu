// Experimental product-parallel M1, variant 10: unchecked complete-tile loads.
// Standalone prototype: link with variant 9 for the generic fallback.
// Every launch rechecks its actual A/B addresses before choosing this fast path.
// One 224-thread CTA owns matching 16x16 tiles in all four output quadrants.
// Every warp keeps one complete FP32 P tile through the entire half-K loop;
// only those seven final fragments cross a CTA barrier for the fixed U DAG.
#include "megafmm/gemm.h"

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace megafmm {
// Variant 9 owns the generic tail/unaligned path and common problem contract.
// These forward declarations keep this scratch prototype independent of API edits.
bool small_product_can_implement(const Problem& p, int variant);
void prepare_small_product(const Problem& p, int variant);
void launch_small_product(const Problem& p, const void* A, const void* B,
                          const float* C0, float* D, cudaStream_t stream, int variant);

namespace {

using namespace cute;

template <class Element>
struct FastProductMmaTraits {
  static constexpr int BM = 16;
  static constexpr int BN = 16;
  static constexpr int BK = 16;
  static constexpr int Products = 7;
  static constexpr int Threads = 32 * Products;
  using Op = typename std::conditional<
      std::is_same<Element, cutlass::bfloat16_t>::value,
      SM80_16x8x16_F32BF16BF16F32_TN,
      SM80_16x8x16_F32F16F16F32_TN>::type;
  using Mma = decltype(make_tiled_mma(
      Op{}, Layout<Shape<_1,_1,_1>>{}, Tile<_16,_16,_16>{}));
  // Each warp has only its own A/B operand tile. The 256-element offsets
  // preserve the swizzle's 128-element period and 16-byte packed store alignment.
  using LayoutA = decltype(composition(Swizzle<1,3,3>{},
      Layout<Shape<_16,_16>, Stride<_16,_1>>{}));
  using LayoutB = decltype(composition(Swizzle<1,3,3>{},
      Layout<Shape<_16,_16>, Stride<_1,_16>>{}));
  static constexpr int ElementsA = cosize_v<LayoutA>;
  static constexpr int ElementsB = cosize_v<LayoutB>;
  static constexpr int WarpElements = ElementsA + ElementsB;
  static constexpr int SharedBytes =
      Products * WarpElements * sizeof(Element) + Products * BM * BN * sizeof(float);
  static_assert(ElementsA == BM * BK && ElementsB == BN * BK,
                "a product warp needs exactly one 1 KiB operand pair");
  static_assert(size(Mma{}) == 32, "one product must use one independent warp");
  static_assert(SharedBytes == 14 * 1024, "operand and product storage is 14 KiB");
};

struct FastProductPackedEight {
  uint4 bits;
  CUTE_DEVICE uint16_t raw(int index) const {
    const uint32_t word = index < 2 ? bits.x : index < 4 ? bits.y
                                              : index < 6 ? bits.z : bits.w;
    return uint16_t(word >> (16 * (index & 1)));
  }
  CUTE_DEVICE void insert(int index, uint16_t value) {
    const uint32_t shifted = uint32_t(value) << (16 * (index & 1));
    if (index < 2) bits.x |= shifted;
    else if (index < 4) bits.y |= shifted;
    else if (index < 6) bits.z |= shifted;
    else bits.w |= shifted;
  }
};

template <class Element>
CUTE_DEVICE FastProductPackedEight fast_product_read_eight(const Element* ptr) {
  // Host eligibility proves every vector has eight valid, aligned elements.
  return FastProductPackedEight{*reinterpret_cast<const uint4*>(ptr)};
}

template <class Element>
CUTE_DEVICE float fast_product_unpack(const FastProductPackedEight& packed, int index) {
  return float(Element::bitcast(packed.raw(index)));
}

CUTE_DEVICE void fast_product_store_eight(void* ptr, const FastProductPackedEight& packed) {
  const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
      :: "r"(address), "r"(packed.bits.x), "r"(packed.bits.y),
         "r"(packed.bits.z), "r"(packed.bits.w) : "memory");
}

// Quadrant indices are 0=11, 1=12, 2=21, 3=22. Under the host gate,
// half-M/N/K are divisible by 16, lda/ldb by 8, and A/B by 16 bytes.
// All lane vectors therefore fit entirely within one original quadrant.
template <int Quadrant, class Element>
CUTE_DEVICE FastProductPackedEight fast_product_read_a_quadrant(
    const Problem& p, const Element* A, int64_t row_base, int64_t k_base, int lane) {
  const int64_t mh = p.m / 2;
  const int64_t kh = p.k / 2;
  const int64_t row = row_base + lane / 2 + (Quadrant / 2) * mh;
  const int64_t k = k_base + (lane & 1) * 8 + (Quadrant & 1) * kh;
  return fast_product_read_eight(A + row * p.lda + k);
}

template <int Quadrant, class Element>
CUTE_DEVICE FastProductPackedEight fast_product_read_b_quadrant(
    const Problem& p, const Element* B, int64_t col_base, int64_t k_base, int lane) {
  const int64_t nh = p.n / 2;
  const int64_t kh = p.k / 2;
  const int64_t k = k_base + lane / 2 + (Quadrant / 2) * kh;
  const int64_t col = col_base + (lane & 1) * 8 + (Quadrant & 1) * nh;
  return fast_product_read_eight(B + k * p.ldb + col);
}

template <int Product, class Element>
CUTE_DEVICE FastProductPackedEight fast_product_make_a(
    const Problem& p, const Element* A, int64_t row_base, int64_t k_base, int lane) {
  static_assert(Product >= 0 && Product < 7, "invalid Winograd product");
  if constexpr (Product == 0) {
    return fast_product_read_a_quadrant<0>(p,A,row_base,k_base,lane); // A11
  } else if constexpr (Product == 1) {
    return fast_product_read_a_quadrant<1>(p,A,row_base,k_base,lane); // A12
  } else if constexpr (Product == 3) {
    return fast_product_read_a_quadrant<3>(p,A,row_base,k_base,lane); // A22
  } else {
    // Load only the dependencies of this product, and quantize only its final
    // operand. S2/S4 are formed from FP32 S1; they never read rounded S1/S2.
    FastProductPackedEight a11{make_uint4(0,0,0,0)}, a12{make_uint4(0,0,0,0)};
    FastProductPackedEight a21{make_uint4(0,0,0,0)}, a22{make_uint4(0,0,0,0)};
    if constexpr (Product == 2 || Product == 5 || Product == 6)
      a11 = fast_product_read_a_quadrant<0>(p,A,row_base,k_base,lane);
    if constexpr (Product == 2)
      a12 = fast_product_read_a_quadrant<1>(p,A,row_base,k_base,lane);
    a21 = fast_product_read_a_quadrant<2>(p,A,row_base,k_base,lane);
    if constexpr (Product != 6)
      a22 = fast_product_read_a_quadrant<3>(p,A,row_base,k_base,lane);
    FastProductPackedEight result{make_uint4(0,0,0,0)};
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
      float value;
      if constexpr (Product == 6) { // S3 = A11 - A21
        value = __fsub_rn(fast_product_unpack<Element>(a11,i), fast_product_unpack<Element>(a21,i));
      } else {
        const float s1 = __fadd_rn(fast_product_unpack<Element>(a21,i), fast_product_unpack<Element>(a22,i));
        if constexpr (Product == 4) { // S1
          value = s1;
        } else {
          const float s2 = __fsub_rn(s1, fast_product_unpack<Element>(a11,i));
          if constexpr (Product == 5) value = s2; // S2
          else value = __fsub_rn(fast_product_unpack<Element>(a12,i), s2); // S4
        }
      }
      result.insert(i, Element(value).raw());
    }
    return result;
  }
}

template <int Product, class Element>
CUTE_DEVICE FastProductPackedEight fast_product_make_b(
    const Problem& p, const Element* B, int64_t col_base, int64_t k_base, int lane) {
  static_assert(Product >= 0 && Product < 7, "invalid Winograd product");
  if constexpr (Product == 0) {
    return fast_product_read_b_quadrant<0>(p,B,col_base,k_base,lane); // B11
  } else if constexpr (Product == 1) {
    return fast_product_read_b_quadrant<2>(p,B,col_base,k_base,lane); // B21
  } else if constexpr (Product == 2) {
    return fast_product_read_b_quadrant<3>(p,B,col_base,k_base,lane); // B22
  } else {
    FastProductPackedEight b11{make_uint4(0,0,0,0)}, b12{make_uint4(0,0,0,0)};
    FastProductPackedEight b21{make_uint4(0,0,0,0)}, b22{make_uint4(0,0,0,0)};
    if constexpr (Product != 6)
      b11 = fast_product_read_b_quadrant<0>(p,B,col_base,k_base,lane);
    b12 = fast_product_read_b_quadrant<1>(p,B,col_base,k_base,lane);
    if constexpr (Product == 3)
      b21 = fast_product_read_b_quadrant<2>(p,B,col_base,k_base,lane);
    if constexpr (Product != 4)
      b22 = fast_product_read_b_quadrant<3>(p,B,col_base,k_base,lane);
    FastProductPackedEight result{make_uint4(0,0,0,0)};
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
      float value;
      if constexpr (Product == 6) { // T3 = B22 - B12
        value = __fsub_rn(fast_product_unpack<Element>(b22,i), fast_product_unpack<Element>(b12,i));
      } else {
        const float t1 = __fsub_rn(fast_product_unpack<Element>(b12,i), fast_product_unpack<Element>(b11,i));
        if constexpr (Product == 4) { // T1
          value = t1;
        } else {
          const float t2 = __fsub_rn(fast_product_unpack<Element>(b22,i), t1);
          if constexpr (Product == 5) value = t2; // T2
          else value = __fsub_rn(t2, fast_product_unpack<Element>(b21,i)); // T4
        }
      }
      result.insert(i, Element(value).raw());
    }
    return result;
  }
}

template <int Product, class Element, class SmemA, class SmemB, class Mma,
          class ThrMma, class CopyA, class CopyB, class Accumulator>
CUTE_DEVICE void fast_product_compute(
    const Problem& p, const Element* A, const Element* B, int64_t row_base,
    int64_t col_base, int lane, SmemA sA, SmemB sB, Mma mma, ThrMma thr_mma,
    CopyA copy_a, CopyB copy_b, Accumulator& accumulator) {
  auto rA = thr_mma.partition_fragment_A(sA);
  auto rB = thr_mma.partition_fragment_B(sB);
  auto thr_copy_a = copy_a.get_thread_slice(lane);
  auto thr_copy_b = copy_b.get_thread_slice(lane);
  auto srcA = thr_copy_a.partition_S(sA);
  auto srcB = thr_copy_b.partition_S(sB);
  auto dstA = thr_copy_a.retile_D(rA);
  auto dstB = thr_copy_b.retile_D(rB);
  CUTE_NO_UNROLL
  for (int k_base = 0; k_base < int(p.k / 2); k_base += 16) {
    const auto a = fast_product_make_a<Product>(p,A,row_base,k_base,lane);
    const auto b = fast_product_make_b<Product>(p,B,col_base,k_base,lane);
    fast_product_store_eight(&sA(lane/2,(lane&1)*8), a);
    fast_product_store_eight(&sB((lane&1)*8,lane/2), b);
    // Only this warp reads or overwrites its 1 KiB operand pair. Synchronous
    // ldmatrix and register MMA use generic shared memory, with no async proxy.
    __syncwarp(0xffffffffu);
    copy(copy_a, srcA, dstA);
    copy(copy_b, srcB, dstB);
    gemm(mma, rA, rB, accumulator);
    __syncwarp(0xffffffffu);
  }
}

template <class Element>
__global__ __launch_bounds__(224)
void small_product_fast_kernel(Problem p, const Element* __restrict__ A,
                          const Element* __restrict__ B, const float* C0, float* D) {
  using T = FastProductMmaTraits<Element>;
  const int lane = threadIdx.x & 31;
  const int product = threadIdx.x / 32;
  const int64_t row_base = int64_t(blockIdx.y) * T::BM;
  const int64_t col_base = int64_t(blockIdx.x) * T::BN;
  // The exact host grid contains only complete owners. There are no early
  // returns: all 224 threads reach the single product-to-U CTA barrier.
  __shared__ __align__(128) Element operands[T::Products * T::WarpElements];
  __shared__ __align__(128) float products[T::Products][T::BM * T::BN];
  Element* private_operands = operands + product * T::WarpElements;
  auto sA = make_tensor(make_smem_ptr(private_operands), typename T::LayoutA{});
  auto sB = make_tensor(make_smem_ptr(private_operands + T::ElementsA), typename T::LayoutB{});
  typename T::Mma mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto coords = thr_mma.partition_C(make_identity_tensor(Shape<_16,_16>{}));
  auto accumulator = thr_mma.make_fragment_C(coords);
  clear(accumulator);
  static_assert(size(accumulator) == 8, "one product has eight FP32 values per lane");
  auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{}, mma);
  auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T,Element>{}, mma);
  // Product is uniform within each complete warp, so its full K loop and
  // syncwarp masks remain converged. Products 0..6 are P1..P7 in PLAN.md.
  switch (product) {
    case 0: fast_product_compute<0>(p,A,B,row_base,col_base,lane,sA,sB,mma,thr_mma,copy_a,copy_b,accumulator); break;
    case 1: fast_product_compute<1>(p,A,B,row_base,col_base,lane,sA,sB,mma,thr_mma,copy_a,copy_b,accumulator); break;
    case 2: fast_product_compute<2>(p,A,B,row_base,col_base,lane,sA,sB,mma,thr_mma,copy_a,copy_b,accumulator); break;
    case 3: fast_product_compute<3>(p,A,B,row_base,col_base,lane,sA,sB,mma,thr_mma,copy_a,copy_b,accumulator); break;
    case 4: fast_product_compute<4>(p,A,B,row_base,col_base,lane,sA,sB,mma,thr_mma,copy_a,copy_b,accumulator); break;
    case 5: fast_product_compute<5>(p,A,B,row_base,col_base,lane,sA,sB,mma,thr_mma,copy_a,copy_b,accumulator); break;
    case 6: fast_product_compute<6>(p,A,B,row_base,col_base,lane,sA,sB,mma,thr_mma,copy_a,copy_b,accumulator); break;
  }
  CUTE_UNROLL
  for (int j = 0; j < size(accumulator); ++j)
    products[product][j * 32 + lane] = accumulator(j);
  __syncthreads();
  if (product == 0) {
    // All product warps use the same lane-local CuTe C fragment mapping. The
    // shared slot j*32+lane therefore matches coords(j) across all seven P's.
    CUTE_NO_UNROLL
    for (int j = 0; j < size(accumulator); ++j) {
      const auto coord = coords(j);
      const int64_t row = row_base + get<0>(coord);
      const int64_t col = col_base + get<1>(coord);
      const int64_t mh = p.m / 2;
      const int64_t nh = p.n / 2;
      { // Full output tiles are proved by the host gate; no per-value mask.
        const int slot = j * 32 + lane;
        const float p1 = products[0][slot];
        const float p2 = products[1][slot];
        const float p3 = products[2][slot];
        const float p4 = products[3][slot];
        const float p5 = products[4][slot];
        const float p6 = products[5][slot];
        const float p7 = products[6][slot];
        const float u1 = __fadd_rn(p1,p2);
        const float u2 = __fadd_rn(p1,p6);
        const float u3 = __fadd_rn(u2,p7);
        const float u4 = __fadd_rn(u2,p5);
        const float u5 = __fadd_rn(u4,p3);
        const float u6 = __fsub_rn(u3,p4);
        const float u7 = __fadd_rn(u3,p5);
        float d11 = __fmul_rn(p.alpha,u1);
        float d12 = __fmul_rn(p.alpha,u5);
        float d21 = __fmul_rn(p.alpha,u6);
        float d22 = __fmul_rn(p.alpha,u7);
        if (p.beta != 0.0f) {
          // Read all four C0 values before any corresponding output store;
          // C0==D with identical strides is safe: each owner writes unique coords.
          const float c11 = C0[row*p.ldc + col];
          const float c12 = C0[row*p.ldc + col + nh];
          const float c21 = C0[(row+mh)*p.ldc + col];
          const float c22 = C0[(row+mh)*p.ldc + col + nh];
          // Match combine_fp32: alpha*U is rounded first, then beta*C0 uses FMA.
          d11 = fmaf(p.beta,c11,d11);
          d12 = fmaf(p.beta,c12,d12);
          d21 = fmaf(p.beta,c21,d21);
          d22 = fmaf(p.beta,c22,d22);
        }
        D[row*p.ldd + col] = d11;
        D[row*p.ldd + col + nh] = d12;
        D[(row+mh)*p.ldd + col] = d21;
        D[(row+mh)*p.ldd + col + nh] = d22;
      }
    }
  }
}

template <class Element>
void launch_small_product_fast_typed(const Problem& p, const void* A, const void* B,
                                const float* C0, float* D, cudaStream_t stream) {
  const int64_t tiles_m = p.m / 32;
  const int64_t tiles_n = p.n / 32;
  const dim3 grid{unsigned(tiles_n),unsigned(tiles_m)};
  small_product_fast_kernel<Element><<<grid,224,0,stream>>>(
      p,static_cast<const Element*>(A),static_cast<const Element*>(B),C0,D);
}

bool fast_product_problem_eligible(const Problem& p) {
  return (p.m % 32) == 0 && (p.n % 32) == 0 && (p.k % 32) == 0 &&
         (p.lda % 8) == 0 && (p.ldb % 8) == 0 &&
         (p.ldc % 8) == 0 && (p.ldd % 8) == 0;
}

bool fast_product_addresses_eligible(const void* A, const void* B) {
  return (reinterpret_cast<uintptr_t>(A) & 15) == 0 &&
         (reinterpret_cast<uintptr_t>(B) & 15) == 0;
}

} // namespace

bool small_product_fast_can_implement(const Problem& p, int variant) {
  // Public Plan handles degenerate cases and validates overlap/in-place strides.
  // Eligibility here includes the generic fallback, so partial tiles remain legal.
  return variant == 10 && small_product_can_implement(p,9);
}

void prepare_small_product_fast(const Problem& p, int variant) {
  if (!small_product_fast_can_implement(p,variant))
    throw std::invalid_argument("unsupported full-tile product mma.sync problem or variant");
  prepare_small_product(p,9);
  // Both routes use zero workspace and 14 KiB static shared; no runtime opt-in.
}

void launch_small_product_fast(const Problem& p, const void* A, const void* B,
                               const float* C0, float* D, cudaStream_t stream,
                               int variant) {
  if (!small_product_fast_can_implement(p,variant))
    throw std::invalid_argument("unsupported full-tile product mma.sync problem or variant");
  if (!A || !B || !D || (p.beta != 0.0f && !C0))
    throw std::invalid_argument("full-tile product mma.sync requires nonnull input/output pointers");
  if ((reinterpret_cast<uintptr_t>(A) & 1) ||
      (reinterpret_cast<uintptr_t>(B) & 1) ||
      (reinterpret_cast<uintptr_t>(D) & 3) ||
      (p.beta != 0.0f && (reinterpret_cast<uintptr_t>(C0) & 3)))
    throw std::invalid_argument("full-tile product mma.sync pointers require element alignment");
  // Binding-specific eligibility is evaluated every call. A Plan rebound to a
  // two-byte-aligned input must select the generic path even after a fast launch.
  if (!fast_product_problem_eligible(p) || !fast_product_addresses_eligible(A,B)) {
    launch_small_product(p,A,B,C0,D,stream,9);
    return;
  }
  if (p.dtype == DType::BF16)
    launch_small_product_fast_typed<cutlass::bfloat16_t>(p,A,B,C0,D,stream);
  else launch_small_product_fast_typed<cutlass::half_t>(p,A,B,C0,D,stream);
}

} // namespace megafmm

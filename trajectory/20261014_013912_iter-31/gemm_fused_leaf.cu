// Fused original-input seven-product leaf launch; public Hopper variants 9/10.
// The existing combine_winograd is still REQUIRED as a separate invocation.
// Full timing includes this leaf and the required existing combine invocation.
#include "megafmm/gemm.h"
#include "megafmm/kernels.h"
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace megafmm {
namespace {
using namespace cute;

template <class Element, int BN>
struct FusedLeafTraits {
  static constexpr int BM=32, BK=32, Threads=32*2*(BN/16);
  using Op = typename std::conditional<
      std::is_same<Element,cutlass::bfloat16_t>::value,
      SM80_16x8x16_F32BF16BF16F32_TN,SM80_16x8x16_F32F16F16F32_TN>::type;
  using Mma = decltype(make_tiled_mma(Op{},
      Layout<Shape<_2,Int<BN/16>,_1>>{},Tile<_32,Int<BN>,_32>{}));
  using LayoutA = decltype(composition(Swizzle<2,3,3>{},
      Layout<Shape<_32,_32>,Stride<_32,_1>>{}));
  using LayoutB = decltype(composition(Swizzle<(BN==64?3:2),3,3>{},
      Layout<Shape<Int<BN>,_32>,Stride<_1,Int<BN>>>{}));
  static constexpr int ElementsA=cosize_v<LayoutA>, ElementsB=cosize_v<LayoutB>;
  static_assert(BN==32 || BN==64,"N32/N64 fused-input configurations");
  static_assert(size(Mma{})==Threads,"one output warp per16x16");
  static_assert(ElementsA==BM*BK && ElementsB==BN*BK,"compact final operands");
  static_assert((ElementsA+ElementsB)*sizeof(Element)<=6144,"one operand bank");
};

struct FusedLeafPackedEight {
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
CUTE_DEVICE FusedLeafPackedEight fused_leaf_read_eight(const Element* ptr) {
  // Host eligibility proves every vector has eight valid, aligned elements.
  return FusedLeafPackedEight{*reinterpret_cast<const uint4*>(ptr)};
}

template <class Element>
CUTE_DEVICE float fused_leaf_unpack(const FusedLeafPackedEight& packed, int index) {
  return float(Element::bitcast(packed.raw(index)));
}

CUTE_DEVICE void fused_leaf_store_eight(void* ptr, const FusedLeafPackedEight& packed) {
  const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
      :: "r"(address), "r"(packed.bits.x), "r"(packed.bits.y),
         "r"(packed.bits.z), "r"(packed.bits.w) : "memory");
}

// Quadrant indices are 0=11, 1=12, 2=21, 3=22. Under the host gate,
// half-M/N/K are divisible by 16, lda/ldb by 8, and A/B by 16 bytes.
// Generalized row/col and K vectors fit within a complete original quadrant.
template <int Quadrant, class Element>
CUTE_DEVICE FusedLeafPackedEight fused_leaf_read_a_quadrant(
    const Problem& p, const Element* A, int64_t row, int64_t k) {
  const int64_t mh = p.m / 2;
  const int64_t kh = p.k / 2;
  return fused_leaf_read_eight(A + (row + (Quadrant / 2) * mh) * p.lda + k + (Quadrant & 1) * kh);
}

template <int Quadrant, class Element>
CUTE_DEVICE FusedLeafPackedEight fused_leaf_read_b_quadrant(
    const Problem& p, const Element* B, int64_t col, int64_t k) {
  const int64_t nh = p.n / 2;
  const int64_t kh = p.k / 2;
  return fused_leaf_read_eight(B + (k + (Quadrant / 2) * kh) * p.ldb + col + (Quadrant & 1) * nh);
}

template <int Product, class Element>
CUTE_DEVICE FusedLeafPackedEight fused_leaf_make_a(
    const Problem& p, const Element* A, int64_t row, int64_t k) {
  static_assert(Product >= 0 && Product < 7, "invalid Winograd product");
  if constexpr (Product == 0) {
    return fused_leaf_read_a_quadrant<0>(p,A,row,k); // A11
  } else if constexpr (Product == 1) {
    return fused_leaf_read_a_quadrant<1>(p,A,row,k); // A12
  } else if constexpr (Product == 3) {
    return fused_leaf_read_a_quadrant<3>(p,A,row,k); // A22
  } else {
    // Load only the dependencies of this product, and quantize only its final
    // operand. S2/S4 are formed from FP32 S1; they never read rounded S1/S2.
    FusedLeafPackedEight a11{make_uint4(0,0,0,0)}, a12{make_uint4(0,0,0,0)};
    FusedLeafPackedEight a21{make_uint4(0,0,0,0)}, a22{make_uint4(0,0,0,0)};
    if constexpr (Product == 2 || Product == 5 || Product == 6)
      a11 = fused_leaf_read_a_quadrant<0>(p,A,row,k);
    if constexpr (Product == 2)
      a12 = fused_leaf_read_a_quadrant<1>(p,A,row,k);
    a21 = fused_leaf_read_a_quadrant<2>(p,A,row,k);
    if constexpr (Product != 6)
      a22 = fused_leaf_read_a_quadrant<3>(p,A,row,k);
    FusedLeafPackedEight result{make_uint4(0,0,0,0)};
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
      float value;
      if constexpr (Product == 6) { // S3 = A11 - A21
        value = __fsub_rn(fused_leaf_unpack<Element>(a11,i), fused_leaf_unpack<Element>(a21,i));
      } else {
        const float s1 = __fadd_rn(fused_leaf_unpack<Element>(a21,i), fused_leaf_unpack<Element>(a22,i));
        if constexpr (Product == 4) { // S1
          value = s1;
        } else {
          const float s2 = __fsub_rn(s1, fused_leaf_unpack<Element>(a11,i));
          if constexpr (Product == 5) value = s2; // S2
          else value = __fsub_rn(fused_leaf_unpack<Element>(a12,i), s2); // S4
        }
      }
      result.insert(i, Element(value).raw());
    }
    return result;
  }
}

template <int Product, class Element>
CUTE_DEVICE FusedLeafPackedEight fused_leaf_make_b(
    const Problem& p, const Element* B, int64_t col, int64_t k) {
  static_assert(Product >= 0 && Product < 7, "invalid Winograd product");
  if constexpr (Product == 0) {
    return fused_leaf_read_b_quadrant<0>(p,B,col,k); // B11
  } else if constexpr (Product == 1) {
    return fused_leaf_read_b_quadrant<2>(p,B,col,k); // B21
  } else if constexpr (Product == 2) {
    return fused_leaf_read_b_quadrant<3>(p,B,col,k); // B22
  } else {
    FusedLeafPackedEight b11{make_uint4(0,0,0,0)}, b12{make_uint4(0,0,0,0)};
    FusedLeafPackedEight b21{make_uint4(0,0,0,0)}, b22{make_uint4(0,0,0,0)};
    if constexpr (Product != 6)
      b11 = fused_leaf_read_b_quadrant<0>(p,B,col,k);
    b12 = fused_leaf_read_b_quadrant<1>(p,B,col,k);
    if constexpr (Product == 3)
      b21 = fused_leaf_read_b_quadrant<2>(p,B,col,k);
    if constexpr (Product != 4)
      b22 = fused_leaf_read_b_quadrant<3>(p,B,col,k);
    FusedLeafPackedEight result{make_uint4(0,0,0,0)};
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
      float value;
      if constexpr (Product == 6) { // T3 = B22 - B12
        value = __fsub_rn(fused_leaf_unpack<Element>(b22,i), fused_leaf_unpack<Element>(b12,i));
      } else {
        const float t1 = __fsub_rn(fused_leaf_unpack<Element>(b12,i), fused_leaf_unpack<Element>(b11,i));
        if constexpr (Product == 4) { // T1
          value = t1;
        } else {
          const float t2 = __fsub_rn(fused_leaf_unpack<Element>(b22,i), t1);
          if constexpr (Product == 5) value = t2; // T2
          else value = __fsub_rn(t2, fused_leaf_unpack<Element>(b21,i)); // T4
        }
      }
      result.insert(i, Element(value).raw());
    }
    return result;
  }
}


// The selected Product is CTA-uniform and all output warps share its operands.
template<int Product,class Element,int BN>
CUTE_DEVICE void fused_leaf_product_body(Problem const& p,Element const* A,
    Element const* B,float* P,Element* operands) {
  using T=FusedLeafTraits<Element,BN>;
  const int tid=int(threadIdx.x);
  const int row_base=int(blockIdx.y)*T::BM, col_base=int(blockIdx.x)*BN;
  const int mh=int(p.m/2),nh=int(p.n/2),kh=int(p.k/2);
  auto sA=make_tensor(make_smem_ptr(operands),typename T::LayoutA{});
  auto sB=make_tensor(make_smem_ptr(operands+T::ElementsA),typename T::LayoutB{});
  typename T::Mma mma;
  auto thr_mma=mma.get_thread_slice(tid);
  auto coords=thr_mma.partition_C(make_identity_tensor(Shape<_32,Int<BN>>{}));
  auto accumulator=thr_mma.make_fragment_C(coords);
  clear(accumulator);
  static_assert(size(accumulator)==8,"one complete product per output warp");
  auto rA=thr_mma.partition_fragment_A(sA);
  auto rB=thr_mma.partition_fragment_B(sB);
  auto copy_a=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{},mma);
  auto copy_b=make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T,Element>{},mma);
  auto thr_copy_a=copy_a.get_thread_slice(tid);
  auto thr_copy_b=copy_b.get_thread_slice(tid);
  auto srcA=thr_copy_a.partition_S(sA);
  auto srcB=thr_copy_b.partition_S(sB);
  auto dstA=thr_copy_a.retile_D(rA);
  auto dstB=thr_copy_b.retile_D(rB);
  CUTE_NO_UNROLL
  for(int k_base=0;k_base<kh;k_base+=T::BK) {
    // One writer/vector. No conditional branch below contains a CTA barrier.
    if(tid<T::BM*T::BK/8) {
      const int row=tid/(T::BK/8), k=(tid%(T::BK/8))*8;
      const auto a=fused_leaf_make_a<Product>(p,A,row_base+row,k_base+k);
      fused_leaf_store_eight(&sA(row,k),a);
    }
    const int k=tid/(BN/8),col=(tid%(BN/8))*8;
    const auto b=fused_leaf_make_b<Product>(p,B,col_base+col,k_base+k);
    fused_leaf_store_eight(&sB(col,k),b);
    __syncthreads();
    copy(copy_a,srcA,dstA);
    copy(copy_b,srcB,dstB);
    // Both chronological K16 groups update the same full-P accumulator.
    gemm(mma,rA,rB,accumulator);
    __syncthreads();
  }
  CUTE_UNROLL
  for(int j=0;j<size(accumulator);++j) {
    const auto coord=coords(j);
    const int64_t row=row_base+get<0>(coord),col=col_base+get<1>(coord);
    P[int64_t(Product)*mh*nh+row*nh+col]=accumulator(j);
  }
  // No alpha/beta/U here: the existing fixed FP32 U combine consumes all P.
}

template<class Element,int BN>
__global__ __launch_bounds__(32*2*(BN/16))
void fused_leaf_kernel(Problem p,Element const* A,Element const* B,float* P) {
  using T=FusedLeafTraits<Element,BN>;
  __shared__ __align__(128) Element operands[T::ElementsA+T::ElementsB];
  // Exact grid.z=7. All threads select one branch, with no early thread exit.
  switch(int(blockIdx.z)) {
    case 0:fused_leaf_product_body<0,Element,BN>(p,A,B,P,operands);break;
    case 1:fused_leaf_product_body<1,Element,BN>(p,A,B,P,operands);break;
    case 2:fused_leaf_product_body<2,Element,BN>(p,A,B,P,operands);break;
    case 3:fused_leaf_product_body<3,Element,BN>(p,A,B,P,operands);break;
    case 4:fused_leaf_product_body<4,Element,BN>(p,A,B,P,operands);break;
    case 5:fused_leaf_product_body<5,Element,BN>(p,A,B,P,operands);break;
    case 6:fused_leaf_product_body<6,Element,BN>(p,A,B,P,operands);break;
  }
}

template<class Element,int BN>
void launch_fused_leaf_typed(Problem const& p,void const* A,void const* B,
                            float* P,cudaStream_t stream) {
  const dim3 grid{unsigned(p.n/(2*BN)),unsigned(p.m/64),7};
  fused_leaf_kernel<Element,BN><<<grid,FusedLeafTraits<Element,BN>::Threads,0,stream>>>(
      p,static_cast<Element const*>(A),static_cast<Element const*>(B),P);
}
} // namespace

// Internal configs0=N32/128threads,1=N64/256threads map to Hopper9/10.
bool fused_leaf_can_implement(Problem const& p,int config) {
  const int bn=config==0?32:64;
  return (config==0 || config==1) && packed_mma_can_implement(p,7) &&
      p.m<=32768 && p.n<=32768 && p.k<=32768 &&
      p.m%64==0 && p.n%(2*bn)==0 && p.k%64==0 &&
      p.lda%8==0 && p.ldb%8==0;
}
bool fused_leaf_full_tiles(Problem const& p,void const* A,void const* B,
                           float const* P,int config) {
  return fused_leaf_can_implement(p,config) && A && B && P &&
      (reinterpret_cast<uintptr_t>(A)&15)==0 &&
      (reinterpret_cast<uintptr_t>(B)&15)==0 &&
      (reinterpret_cast<uintptr_t>(P)&15)==0;
}
size_t fused_leaf_product_workspace_bytes(Problem const& p,int config) {
  if(!fused_leaf_can_implement(p,config))
    throw std::invalid_argument("unsupported fused leaf metadata");
  return size_t(7)*size_t(p.m/2)*size_t(p.n/2)*sizeof(float);
}
void prepare_fused_leaf(Problem const& p,int config) {
  if(!fused_leaf_can_implement(p,config))
    throw std::invalid_argument("unsupported fused leaf metadata");
  // <=6KiB static shared. No allocation, opt-in, synchronization or caches.
}
void launch_fused_leaf(Problem const& p,void const* A,void const* B,float* P,
                       cudaStream_t stream,int config) {
  if(!fused_leaf_full_tiles(p,A,B,P,config))
    throw std::invalid_argument("unsupported fused leaf binding");
  if(p.dtype==DType::BF16) {
    if(config==0)launch_fused_leaf_typed<cutlass::bfloat16_t,32>(p,A,B,P,stream);
    else launch_fused_leaf_typed<cutlass::bfloat16_t,64>(p,A,B,P,stream);
  } else {
    if(config==0)launch_fused_leaf_typed<cutlass::half_t,32>(p,A,B,P,stream);
    else launch_fused_leaf_typed<cutlass::half_t,64>(p,A,B,P,stream);
  }
}
} // namespace megafmm

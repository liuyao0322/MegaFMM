// Matched packed-input pair of gemm_fused_leaf.cu.
// REQUIRED complete run: pack_winograd_batched_inputs, this leaf, combine_winograd.
#include "megafmm/gemm.h"
#include "megafmm/kernels.h"
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace megafmm {
namespace {
using namespace cute;

template <class Element, int BN>
struct PackedLeafTraits {
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
  static_assert(BN==32 || BN==64,"N32/N64 configs");
  static_assert(size(Mma{})==Threads,"one output warp per16x16");
  static_assert(ElementsA==BM*BK && ElementsB==BN*BK,"compact final operands");
  static_assert((ElementsA+ElementsB)*sizeof(Element)<=6144,"one operand bank");
};

struct PackedLeafPackedEight {
  uint4 bits;
};

template <class Element>
CUTE_DEVICE PackedLeafPackedEight packed_leaf_read_eight(const Element* ptr) {
  // Host metadata/binding gates prove an aligned, complete compact vector.
  return PackedLeafPackedEight{*reinterpret_cast<const uint4*>(ptr)};
}

CUTE_DEVICE void packed_leaf_store_eight(void* ptr, const PackedLeafPackedEight& packed) {
  const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
      :: "r"(address), "r"(packed.bits.x), "r"(packed.bits.y),
         "r"(packed.bits.z), "r"(packed.bits.w) : "memory");
}

// Inputs are NOT original A/B. They are the fresh compact uniform slots from
// detail::pack_winograd_batched_inputs with the same parent Problem metadata.
// A7 = A11,A12,S4,A22,S1,S2,S3; B7 = B11,B21,B22,T4,T1,T2,T3.
// No conversion or S/T arithmetic here: copy the already quantized bit pattern.
template <int Product, class Element>
CUTE_DEVICE PackedLeafPackedEight packed_leaf_read_a(
    const Problem& p, const Element* A7, int64_t row, int64_t k) {
  static_assert(Product >= 0 && Product < 7, "invalid Winograd product");
  const int64_t mh = p.m / 2;
  const int64_t kh = p.k / 2;
  return packed_leaf_read_eight(A7 + int64_t(Product)*mh*kh + row*kh + k);
}

template <int Product, class Element>
CUTE_DEVICE PackedLeafPackedEight packed_leaf_read_b(
    const Problem& p, const Element* B7, int64_t col, int64_t k) {
  static_assert(Product >= 0 && Product < 7, "invalid Winograd product");
  const int64_t nh = p.n / 2;
  const int64_t kh = p.k / 2;
  return packed_leaf_read_eight(B7 + int64_t(Product)*kh*nh + k*nh + col);
}

// The selected Product is CTA-uniform and all output warps share its operands.
template<int Product,class Element,int BN>
CUTE_DEVICE void packed_leaf_product_body(Problem const& p,Element const* A,
    Element const* B,float* P,Element* operands) {
  using T=PackedLeafTraits<Element,BN>;
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
      const auto a=packed_leaf_read_a<Product>(p,A,row_base+row,k_base+k);
      packed_leaf_store_eight(&sA(row,k),a);
    }
    const int k=tid/(BN/8),col=(tid%(BN/8))*8;
    const auto b=packed_leaf_read_b<Product>(p,B,col_base+col,k_base+k);
    packed_leaf_store_eight(&sB(col,k),b);
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
void packed_leaf_kernel(Problem p,Element const* A,Element const* B,float* P) {
  using T=PackedLeafTraits<Element,BN>;
  __shared__ __align__(128) Element operands[T::ElementsA+T::ElementsB];
  // Exact grid.z=7. All threads select one branch, with no early thread exit.
  switch(int(blockIdx.z)) {
    case 0:packed_leaf_product_body<0,Element,BN>(p,A,B,P,operands);break;
    case 1:packed_leaf_product_body<1,Element,BN>(p,A,B,P,operands);break;
    case 2:packed_leaf_product_body<2,Element,BN>(p,A,B,P,operands);break;
    case 3:packed_leaf_product_body<3,Element,BN>(p,A,B,P,operands);break;
    case 4:packed_leaf_product_body<4,Element,BN>(p,A,B,P,operands);break;
    case 5:packed_leaf_product_body<5,Element,BN>(p,A,B,P,operands);break;
    case 6:packed_leaf_product_body<6,Element,BN>(p,A,B,P,operands);break;
  }
}

template<class Element,int BN>
void launch_packed_leaf_typed(Problem const& p,void const* A,void const* B,
                            float* P,cudaStream_t stream) {
  const dim3 grid{unsigned(p.n/(2*BN)),unsigned(p.m/64),7};
  packed_leaf_kernel<Element,BN><<<grid,PackedLeafTraits<Element,BN>::Threads,0,stream>>>(
      p,static_cast<Element const*>(A),static_cast<Element const*>(B),P);
}
} // namespace

// Internal configs0/1 map to Hopper11/12, N32/128threads and N64/256threads.
// This retains the fused pair's parent metadata gate deliberately. Original
// lda/ldb describe the preceding pack, NOT the compact leaf's row strides.
bool packed_leaf_can_implement(Problem const& p,int config) {
  const int bn=config==0?32:64;
  return (config==0 || config==1) && packed_mma_can_implement(p,7) &&
      p.m<=32768 && p.n<=32768 && p.k<=32768 &&
      p.m%64==0 && p.n%(2*bn)==0 && p.k%64==0 &&
      p.lda%8==0 && p.ldb%8==0;
}

// All strides are in elements; byte extents include all seven compact slots.
// Publish equivalent declarations in a formal header only if this is integrated.


namespace {
size_t packed_leaf_checked_mul(size_t a,size_t b) {
  if(b && a>std::numeric_limits<size_t>::max()/b)
    throw std::invalid_argument("packed leaf extent overflows size_t");
  return a*b;
}
size_t packed_leaf_checked_add(size_t a,size_t b) {
  if(a>std::numeric_limits<size_t>::max()-b)
    throw std::invalid_argument("packed leaf workspace overflows size_t");
  return a+b;
}
bool packed_leaf_range_fits(void const* ptr,size_t bytes) {
  return reinterpret_cast<uintptr_t>(ptr)<=std::numeric_limits<uintptr_t>::max()-bytes;
}
} // namespace

PackedLeafMetadata packed_leaf_metadata(Problem const& p,int config) {
  if(!packed_leaf_can_implement(p,config))
    throw std::invalid_argument("unsupported packed leaf metadata");
  const int64_t mh=p.m/2,nh=p.n/2,kh=p.k/2;
  // Guarded dimensions <=32768 make these int64_t products safe.
  const int64_t sa=mh*kh,sb=kh*nh,sp=mh*nh;
  const size_t a_bytes=packed_leaf_checked_mul(packed_leaf_checked_mul(size_t(7),size_t(sa)),size_t(2));
  const size_t b_bytes=packed_leaf_checked_mul(packed_leaf_checked_mul(size_t(7),size_t(sb)),size_t(2));
  const size_t p_bytes=packed_leaf_checked_mul(packed_leaf_checked_mul(size_t(7),size_t(sp)),sizeof(float));
  const size_t total=packed_leaf_checked_add(packed_leaf_checked_add(a_bytes,b_bytes),p_bytes);
  return {mh,nh,kh,kh,nh,nh,sa,sb,sp,a_bytes,b_bytes,p_bytes,total};
}

// Original binding is validated before the current inputs are packed. Public
// Plan validation also checks original strides, byte spans and disjointness.
bool packed_leaf_original_input_guard(Problem const& p,void const* A,void const* B,int config) {
  return packed_leaf_can_implement(p,config) && A && B &&
      (reinterpret_cast<uintptr_t>(A)&15)==0 && (reinterpret_cast<uintptr_t>(B)&15)==0;
}

bool packed_leaf_full_tiles(Problem const& p,void const* A7,void const* B7,
                            float const* P,int config) {
  if(!packed_leaf_can_implement(p,config) || !A7 || !B7 || !P ||
      (reinterpret_cast<uintptr_t>(A7)&15)!=0 ||
      (reinterpret_cast<uintptr_t>(B7)&15)!=0 ||
      (reinterpret_cast<uintptr_t>(P)&15)!=0) return false;
  const auto meta=packed_leaf_metadata(p,config);
  return packed_leaf_range_fits(A7,meta.a_bytes) &&
      packed_leaf_range_fits(B7,meta.b_bytes) && packed_leaf_range_fits(P,meta.p_bytes);
  // This cannot discover an allocation's capacity or legal disjointness. Plan
  // must reserve/validate all extents, cap and overlaps before launching pack.
}
size_t packed_leaf_workspace_bytes(Problem const& p,int config) {
  return packed_leaf_metadata(p,config).compact_workspace_bytes;
}
void prepare_packed_leaf(Problem const& p,int config) {
  (void)packed_leaf_metadata(p,config);
  // <=6KiB static shared. No allocation, opt-in, synchronization or caches.
}
void launch_packed_leaf(Problem const& p,void const* A7,void const* B7,float* P,
                        cudaStream_t stream,int config) {
  if(!packed_leaf_full_tiles(p,A7,B7,P,config))
    throw std::invalid_argument("unsupported packed leaf binding");
  // Leaf ONLY. Caller must repack current original inputs on this same stream
  // before every complete run, then launch the existing combine after this.
  if(p.dtype==DType::BF16) {
    if(config==0)launch_packed_leaf_typed<cutlass::bfloat16_t,32>(p,A7,B7,P,stream);
    else launch_packed_leaf_typed<cutlass::bfloat16_t,64>(p,A7,B7,P,stream);
  } else {
    if(config==0)launch_packed_leaf_typed<cutlass::half_t,32>(p,A7,B7,P,stream);
    else launch_packed_leaf_typed<cutlass::half_t,64>(p,A7,B7,P,stream);
  }
}
} // namespace megafmm

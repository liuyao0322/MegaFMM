// Coalesced-register controls: variant22 owns 16 rows, variant23 owns 8 rows.
// M16 A64/B64 uses two K3 phases; M8 A64/B64 uses one upper-row producer.
// Exact full-warp butterfly/PRMT routing follows the frozen CPU proof.
// No shared memory, LDSM, warp barrier, or next-stage raw prefetch.
// Preserve direct24 FP32 S/T -> final RN, seven complete P, fixed U, identity.
// Explicit load counts alone do not establish total issue, DRAM, or speed gains.
// Caller retains public Plan certification/range/overlap/binding validation.
#include "megafmm/gemm.h"

#include <cute/arch/mma_sm80.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace megafmm {
namespace {

template <class Element>
using NativeMma = typename std::conditional<
    std::is_same<Element,cutlass::bfloat16_t>::value,
    cute::SM80_16x8x16_F32BF16BF16F32_TN,
    cute::SM80_16x8x16_F32F16F16F32_TN>::type;

struct SevenWords { uint32_t word[7]; };
struct NativeOperands { uint32_t a[7][4]; uint32_t b[7][2]; };
struct SevenProducts { float value[7][4]; };

struct RawPair { uint32_t x,y; };

template <class Element>
CUTE_DEVICE RawPair read_four(const Element* ptr) {
  // Four consecutive original row-major half elements, guarded 8B alignment.
  RawPair raw;
  asm volatile("ld.global.v2.u32 {%0, %1}, [%2];"
      : "=r"(raw.x), "=r"(raw.y) : "l"(ptr) : "memory");
  return raw;
}

CUTE_DEVICE void pair_exchange(RawPair& raw,bool p,int xor_mask) {
  // send is evaluated in each SOURCE lane. Snapshot both original registers;
  // receiver-local indexing cannot select an arbitrary source-lane raw word.
  const uint32_t old_x=raw.x, old_y=raw.y;
  const uint32_t send=p ? old_x:old_y;
  const uint32_t got=__shfl_xor_sync(0xffffffffu,send,xor_mask,32);
  raw.x=p ? got:old_x;
  raw.y=p ? old_y:got;
}

template <class Element,int OwnerRows,int Phase>
CUTE_DEVICE RawPair load_a_phase(const Element* quadrant,int64_t lda,
                                 int row_base,int k_base,int lane) {
  static_assert(OwnerRows==8 || OwnerRows==16, "one native owner geometry");
  static_assert(Phase==0 || (Phase==1 && OwnerRows==16), "two M16 K3 phases");
  const int r=lane>>2;
  const int row=row_base+r+(OwnerRows==16 ? 8*(lane&1):0);
  const int k=k_base+4*((lane>>1)&1)+
      (OwnerRows==16 ? 8*Phase:8*(lane&1));
  RawPair raw=read_four(quadrant+row*lda+k);
  // Phase supplies M16 K3: one lane0/register-bit0 exchange is sufficient.
  // M8 sends only actual upper rows; every lane still executes this shuffle.
  pair_exchange(raw,(lane&1)!=0,1);
  return raw;
}

template <class Element>
CUTE_DEVICE RawPair load_b_words(const Element* quadrant,int64_t ldb,
                                 int col_base,int k_base,int lane) {
  const int k=k_base+2*(lane&1)+4*((lane>>1)&1)+
      ((lane>>2)&1)+8*((lane>>3)&1);
  const int col=col_base+4*((lane>>4)&1);
  RawPair raw=read_four(quadrant+k*ldb+col);
  // First exchange lane3 with the packed-word index, then lane2 with half.
  pair_exchange(raw,(lane&8)!=0,8);
  const uint32_t own0=raw.x, own1=raw.y;
  const uint32_t peer0=__shfl_xor_sync(0xffffffffu,own0,4,32);
  const uint32_t peer1=__shfl_xor_sync(0xffffffffu,own1,4,32);
  const uint32_t selector=(lane&4) ? 0x3276u:0x5410u;
  // Fixed own/peer argument order; the alternate 0x7632 selector is WRONG here.
  raw.x=__byte_perm(own0,peer0,selector);
  raw.y=__byte_perm(own1,peer1,selector);
  return raw;
}

template <class Element>
CUTE_DEVICE float unpack(uint32_t word, int component) {
  return float(Element::bitcast(uint16_t(word >> (16*component))));
}

template <class Element>
CUTE_DEVICE uint32_t round_pair(float even, float odd) {
  cutlass::Array<float,2> source;
  source[0] = even;
  source[1] = odd;
  using Converter = cutlass::NumericArrayConverter<
      Element,float,2,cutlass::FloatRoundStyle::round_to_nearest>;
  const auto converted = Converter{}(source);
  static_assert(sizeof(converted)==sizeof(uint32_t), "native pair is one word");
  return reinterpret_cast<const uint32_t&>(converted);
}

template <class Element>
CUTE_DEVICE SevenWords a_dag(uint32_t x11, uint32_t x12,
                             uint32_t x21, uint32_t x22) {
  float s1[2],s2[2],s3[2],s4[2];
  CUTE_UNROLL
  for (int i=0;i<2;++i) {
    const float a11=unpack<Element>(x11,i), a12=unpack<Element>(x12,i);
    const float a21=unpack<Element>(x21,i), a22=unpack<Element>(x22,i);
    // Every dependency remains FP32. Only the final four leaf operands cast.
    s1[i]=__fadd_rn(a21,a22);
    s2[i]=__fsub_rn(s1[i],a11);
    s3[i]=__fsub_rn(a11,a21);
    s4[i]=__fsub_rn(a12,s2[i]);
  }
  SevenWords output;
  output.word[0]=x11;
  output.word[1]=x12;
  output.word[2]=round_pair<Element>(s4[0],s4[1]);
  output.word[3]=x22;
  output.word[4]=round_pair<Element>(s1[0],s1[1]);
  output.word[5]=round_pair<Element>(s2[0],s2[1]);
  output.word[6]=round_pair<Element>(s3[0],s3[1]);
  return output;
}

template <class Element>
CUTE_DEVICE SevenWords b_dag(uint32_t x11, uint32_t x12,
                             uint32_t x21, uint32_t x22) {
  float t1[2],t2[2],t3[2],t4[2];
  CUTE_UNROLL
  for (int i=0;i<2;++i) {
    const float b11=unpack<Element>(x11,i), b12=unpack<Element>(x12,i);
    const float b21=unpack<Element>(x21,i), b22=unpack<Element>(x22,i);
    t1[i]=__fsub_rn(b12,b11);
    t2[i]=__fsub_rn(b22,t1[i]);
    t3[i]=__fsub_rn(b22,b12);
    t4[i]=__fsub_rn(t2[i],b21);
  }
  SevenWords output;
  output.word[0]=x11;
  output.word[1]=x21;
  output.word[2]=x22;
  output.word[3]=round_pair<Element>(t4[0],t4[1]);
  output.word[4]=round_pair<Element>(t1[0],t1[1]);
  output.word[5]=round_pair<Element>(t2[0],t2[1]);
  output.word[6]=round_pair<Element>(t3[0],t3[1]);
  return output;
}

template <class Element,int OwnerRows,int Phase>
CUTE_DEVICE void fill_a_phase(const Problem& p,const Element* A,
    int row_base,int k_base,int lane,NativeOperands& output) {
  const int mh=int(p.m)/2, kh=int(p.k)/2;
  // All four quadrants are independently routed BEFORE applying any FP32 DAG.
  const RawPair a11=load_a_phase<Element,OwnerRows,Phase>(
      A,p.lda,row_base,k_base,lane);
  const RawPair a12=load_a_phase<Element,OwnerRows,Phase>(
      A+kh,p.lda,row_base,k_base,lane);
  const RawPair a21=load_a_phase<Element,OwnerRows,Phase>(
      A+mh*p.lda,p.lda,row_base,k_base,lane);
  const RawPair a22=load_a_phase<Element,OwnerRows,Phase>(
      A+mh*p.lda+kh,p.lda,row_base,k_base,lane);
  CUTE_UNROLL
  for (int word=0;word<2;++word) {
    const SevenWords packed=a_dag<Element>(
        word==0 ? a11.x:a11.y,word==0 ? a12.x:a12.y,
        word==0 ? a21.x:a21.y,word==0 ? a22.x:a22.y);
    const int native_word=OwnerRows==16 ? 2*Phase+word:2*word;
    CUTE_UNROLL
    for (int product=0;product<7;++product)
      output.a[product][native_word]=packed.word[product];
  }
}

template <class Element,int OwnerRows>
CUTE_DEVICE NativeOperands load_operands(const Problem& p,
    const Element* A,const Element* B,int row_base,int col_base,int k_base,
    int lane) {
  static_assert(OwnerRows==8 || OwnerRows==16, "one native owner geometry");
  NativeOperands output;
  if constexpr (OwnerRows==8) {
    CUTE_UNROLL
    for (int product=0;product<7;++product) {
      output.a[product][1]=0;
      output.a[product][3]=0;
    }
  }
  fill_a_phase<Element,OwnerRows,0>(p,A,row_base,k_base,lane,output);
  if constexpr (OwnerRows==16)
    fill_a_phase<Element,OwnerRows,1>(p,A,row_base,k_base,lane,output);
  const int nh=int(p.n)/2, kh=int(p.k)/2;
  const RawPair b11=load_b_words(B,p.ldb,col_base,k_base,lane);
  const RawPair b12=load_b_words(B+nh,p.ldb,col_base,k_base,lane);
  const RawPair b21=load_b_words(B+kh*p.ldb,p.ldb,col_base,k_base,lane);
  const RawPair b22=load_b_words(B+kh*p.ldb+nh,p.ldb,col_base,k_base,lane);
  CUTE_UNROLL
  for (int word=0;word<2;++word) {
    const SevenWords packed=b_dag<Element>(
        word==0 ? b11.x:b11.y,word==0 ? b12.x:b12.y,
        word==0 ? b21.x:b21.y,word==0 ? b22.x:b22.y);
    CUTE_UNROLL
    for (int product=0;product<7;++product)
      output.b[product][word]=packed.word[product];
  }
  return output;
}

template <class Element,int OwnerRows>
__global__ __launch_bounds__(32)
void coalesced_mma_kernel(Problem p,const Element* __restrict__ A,
                   const Element* __restrict__ B,float* D) {
  const int lane=int(threadIdx.x)&31;
  const int row_base=int(blockIdx.y)*OwnerRows;
  const int col_base=int(blockIdx.x)*8;
  const int mh=int(p.m)/2, nh=int(p.n)/2, kh=int(p.k)/2;
  SevenProducts products{};
  // No shared communication is required: each lane directly owns the native
  // fragment entries proved by pinned CuTe traits. All 32 lanes execute each
  // full-warp MMA, and every product accumulates in the same increasing-K order.
  CUTE_NO_UNROLL
  for (int k_base=0;k_base<kh;k_base+=16) {
    const NativeOperands input=load_operands<Element,OwnerRows>(
        p,A,B,row_base,col_base,k_base,lane);
    CUTE_UNROLL
    for (int product=0;product<7;++product) {
      float* v=products.value[product];
      const uint32_t* a=input.a[product];
      const uint32_t* b=input.b[product];
      NativeMma<Element>::fma(v[0],v[1],v[2],v[3],
          a[0],a[1],a[2],a[3],b[0],b[1],v[0],v[1],v[2],v[3]);
    }
  }
  CUTE_UNROLL
  for (int pair=0;pair<OwnerRows/8;++pair) {
    const int row=row_base+lane/4+8*pair;
    const int col=col_base+2*(lane%4);
    float2 d11,d12,d21,d22;
    CUTE_UNROLL
    for (int j=0;j<2;++j) {
      const int v=2*pair+j;
      const float u1=__fadd_rn(products.value[0][v],products.value[1][v]);
      const float u2=__fadd_rn(products.value[0][v],products.value[5][v]);
      const float u3=__fadd_rn(u2,products.value[6][v]);
      const float u4=__fadd_rn(u2,products.value[4][v]);
      const float u5=__fadd_rn(u4,products.value[2][v]);
      const float u6=__fsub_rn(u3,products.value[3][v]);
      const float u7=__fadd_rn(u3,products.value[4][v]);
      if (j==0) { d11.x=u1; d12.x=u5; d21.x=u6; d22.x=u7; }
      else { d11.y=u1; d12.y=u5; d21.y=u6; d22.y=u7; }
    }
    *reinterpret_cast<float2*>(D+row*p.ldd+col)=d11;
    *reinterpret_cast<float2*>(D+row*p.ldd+col+nh)=d12;
    *reinterpret_cast<float2*>(D+(row+mh)*p.ldd+col)=d21;
    *reinterpret_cast<float2*>(D+(row+mh)*p.ldd+col+nh)=d22;
  }
}

template <class Element,int OwnerRows>
void launch_typed(const Problem& p,const void* A,const void* B,float* D,
                  cudaStream_t stream) {
  const dim3 grid(unsigned(p.n/16),unsigned(p.m/(2*OwnerRows)));
  coalesced_mma_kernel<Element,OwnerRows><<<grid,32,0,stream>>>(
      p,static_cast<const Element*>(A),static_cast<const Element*>(B),D);
}

} // namespace

bool coalesced_mma_can_implement(const Problem& p,int variant) {
  if ((variant!=22 && variant!=23) || p.trans_a || p.trans_b ||
      p.alpha!=1.0f || p.beta!=0.0f) return false;
  if (p.dtype!=DType::BF16 && p.dtype!=DType::FP16) return false;
  if (p.m<=0 || p.n<=0 || p.k<=0 || p.m>32768 || p.n>32768 || p.k>32768 ||
      p.m%32 || p.n%32 || p.k%32) return false;
  if (p.lda<p.k || p.ldb<p.n || p.ldd<p.n ||
      p.lda%8 || p.ldb%8 || p.ldd%2) return false;
  const int64_t limit=std::numeric_limits<int64_t>::max()/4;
  if (p.lda>limit/p.m || p.ldb>limit/p.k || p.ldd>limit/p.m) return false;
  return std::isfinite(p.alpha) && std::isfinite(p.beta);
}

bool coalesced_mma_full_tiles(const Problem& p,const void* A,const void* B,
                           const float* C0,const float* D,int variant) {
  (void)C0; // Identity beta +/-0 never consumes C0 or its unused stride.
  return coalesced_mma_can_implement(p,variant) && A && B && D &&
      (reinterpret_cast<uintptr_t>(A)&15)==0 &&
      (reinterpret_cast<uintptr_t>(B)&15)==0 &&
      (reinterpret_cast<uintptr_t>(D)&7)==0;
}

void prepare_coalesced_mma(const Problem& p,int variant) {
  if (!coalesced_mma_can_implement(p,variant))
    throw std::invalid_argument("unsupported coalesced-register mma.sync problem or variant");
  // One warp, zero shared/workspace; no opt-in or runtime initialization.
}

void launch_coalesced_mma(const Problem& p,const void* A,const void* B,
                       const float* C0,float* D,cudaStream_t stream,int variant) {
  if (!coalesced_mma_full_tiles(p,A,B,C0,D,variant))
    throw std::invalid_argument("coalesced-register requires aligned identity complete tiles");
  const int owner_rows=variant==22 ? 16:8;
  if (p.dtype==DType::BF16) {
    if (owner_rows==8) launch_typed<cutlass::bfloat16_t,8>(p,A,B,D,stream);
    else launch_typed<cutlass::bfloat16_t,16>(p,A,B,D,stream);
  } else {
    if (owner_rows==8) launch_typed<cutlass::half_t,8>(p,A,B,D,stream);
    else launch_typed<cutlass::half_t,16>(p,A,B,D,stream);
  }
}

} // namespace megafmm

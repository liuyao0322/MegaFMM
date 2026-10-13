// Mega20/21: direct native SM80 m16n8k16 operands with 16/8 logical rows.
// No shared memory, LDSM, warp shuffle, warp barrier, or register prefetch.
// Keep seven complete M1 P fragments, final-only RN S/T casts, and fixed U.
// Identity coefficients only; Plan retains certification/range/overlap checks.
// Direct LDGs change cache policy and load granularity relative to identity15;
// comparisons with prefetch16/19 additionally change the prefetch schedule.
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

template <class Element>
CUTE_DEVICE uint32_t read_a_pair(const Element* ptr) {
  // Adjacent K elements; full-tile/ld/input guards prove four-byte alignment.
  uint32_t word;
  asm volatile("ld.global.u32 %0, [%1];"
      : "=r"(word) : "l"(ptr) : "memory");
  return word;
}

template <class Element>
CUTE_DEVICE uint32_t read_b_pair(const Element* ptr, int64_t ldb) {
  // Native B pair crosses two original row-major K rows. Neither a contiguous
  // u32 nor a receiver-indexed shuffle is equivalent to these two coordinates.
  uint16_t even, odd;
  asm volatile("ld.global.u16 %0, [%1];"
      : "=h"(even) : "l"(ptr) : "memory");
  asm volatile("ld.global.u16 %0, [%1];"
      : "=h"(odd) : "l"(ptr+ldb) : "memory");
  return uint32_t(even) | (uint32_t(odd) << 16);
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

template <class Element,int OwnerRows>
CUTE_DEVICE NativeOperands load_operands(const Problem& p,
    const Element* A,const Element* B,int row_base,int col_base,int k_base,
    int lane) {
  static_assert(OwnerRows==8 || OwnerRows==16, "one native owner geometry");
  const int r=lane/4, g=lane%4;
  const int mh=int(p.m)/2, nh=int(p.n)/2, kh=int(p.k)/2;
  NativeOperands output;
  CUTE_UNROLL
  for (int word=0;word<4;++word) {
    if constexpr (OwnerRows==8) {
      // Native a1/a3 cover the padded lower eight rows. No input/DAG work.
      if (word&1) {
        CUTE_UNROLL
        for (int product=0;product<7;++product) output.a[product][word]=0;
        continue;
      }
    }
    const int row=row_base+r+8*(word&1);
    const int k=k_base+2*g+8*(word/2);
    const SevenWords packed=a_dag<Element>(
        read_a_pair(A+row*p.lda+k),
        read_a_pair(A+row*p.lda+k+kh),
        read_a_pair(A+(row+mh)*p.lda+k),
        read_a_pair(A+(row+mh)*p.lda+k+kh));
    CUTE_UNROLL
    for (int product=0;product<7;++product)
      output.a[product][word]=packed.word[product];
  }
  CUTE_UNROLL
  for (int word=0;word<2;++word) {
    const int k=k_base+2*g+8*word;
    const int col=col_base+r;
    const SevenWords packed=b_dag<Element>(
        read_b_pair(B+k*p.ldb+col,p.ldb),
        read_b_pair(B+k*p.ldb+col+nh,p.ldb),
        read_b_pair(B+(k+kh)*p.ldb+col,p.ldb),
        read_b_pair(B+(k+kh)*p.ldb+col+nh,p.ldb));
    CUTE_UNROLL
    for (int product=0;product<7;++product)
      output.b[product][word]=packed.word[product];
  }
  return output;
}

template <class Element,int OwnerRows>
__global__ __launch_bounds__(32)
void direct_kernel(Problem p,const Element* __restrict__ A,
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
  direct_kernel<Element,OwnerRows><<<grid,32,0,stream>>>(
      p,static_cast<const Element*>(A),static_cast<const Element*>(B),D);
}

} // namespace

bool direct_mma_can_implement(const Problem& p,int variant) {
  if ((variant!=20 && variant!=21) || p.trans_a || p.trans_b ||
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

bool direct_mma_full_tiles(const Problem& p,const void* A,const void* B,
                           const float* C0,const float* D,int variant) {
  (void)C0; // Identity beta +/-0 never consumes C0 or its unused stride.
  return direct_mma_can_implement(p,variant) && A && B && D &&
      (reinterpret_cast<uintptr_t>(A)&15)==0 &&
      (reinterpret_cast<uintptr_t>(B)&15)==0 &&
      (reinterpret_cast<uintptr_t>(D)&7)==0;
}

void prepare_direct_mma(const Problem& p,int variant) {
  if (!direct_mma_can_implement(p,variant))
    throw std::invalid_argument("unsupported direct-register mma.sync problem or variant");
  // One warp, zero shared/workspace; no opt-in or runtime initialization.
}

void launch_direct_mma(const Problem& p,const void* A,const void* B,
                       const float* C0,float* D,cudaStream_t stream,int variant) {
  if (!direct_mma_full_tiles(p,A,B,C0,D,variant))
    throw std::invalid_argument("direct-register requires aligned identity complete tiles");
  const int owner_rows=variant==20 ? 16:8;
  if (p.dtype==DType::BF16) {
    if (owner_rows==8) launch_typed<cutlass::bfloat16_t,8>(p,A,B,D,stream);
    else launch_typed<cutlass::bfloat16_t,16>(p,A,B,D,stream);
  } else {
    if (owner_rows==8) launch_typed<cutlass::half_t,8>(p,A,B,D,stream);
    else launch_typed<cutlass::half_t,16>(p,A,B,D,stream);
  }
}

} // namespace megafmm

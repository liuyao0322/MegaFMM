#include "internal.h"
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace sw_small_native {
using namespace megafmm;
using megafmm::detail::SWResources;
namespace {
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
template<class T,int N,int K,int S> struct alignas(128) Shared {
  T operand[S][(64+N)*K];
  uint64_t ready[S],free[S];
};
#include "sw_small_16_128_4_k2.cuh"
#include "sw_small_16_128_4_rt.cuh"
#include "sw_small_16_128_4_split.cuh"
#include "sw_small_16_128_8_k2.cuh"
#include "sw_small_16_128_8_rt.cuh"
#include "sw_small_16_64_4_rt.cuh"
#include "sw_small_32_128_4_split.cuh"
#include "sw_small_32_128_8_k2.cuh"
__device__ __forceinline__ void wait_free(uint64_t* bar,int phase) {
  Barrier::wait(bar,phase);asm volatile("" ::: "memory");
}
__device__ constexpr int product_for_step(int step) {
  return step<2?step:step==2?5:step==3?6:step==4?4:step==5?2:3;
}
template<class T> __device__ uint32_t pair(float lo,float hi) {
  uint32_t bits;
  if constexpr(std::is_same_v<T,cute::bfloat16_t>)
    asm("cvt.rn.bf16x2.f32 %0,%2,%1;":"=r"(bits):"f"(lo),"f"(hi));
  else asm("cvt.rn.f16x2.f32 %0,%2,%1;":"=r"(bits):"f"(lo),"f"(hi));
  return bits;
}
// A/B planes have independent sizes and pitches, including rectangular cases.
// No input-dependent values persist across calls. Aligned raw P1/P2 overlap packing;
// padded raw operands wait for the pack kernel before their first TMA load.
template<class T> __global__ __launch_bounds__(256)
void transform(const T* A,const T* B,T* packed,int mh,int nh,int kh) {
  cudaTriggerProgrammaticLaunchCompletion();
  const int64_t sa=int64_t(mh)*kh,sb=int64_t(kh)*nh;
  for(int side=0;side<2;++side) {
    const int cols=side?nh:kh,ld=2*cols;
    const int64_t plane=side?sb:sa;
    const T* input=side?B:A;
    T* output=packed+(side?4*sa:0);
    for(int64_t i=2*(int64_t(blockIdx.x)*256+threadIdx.x);i<plane;i+=2*256*gridDim.x) {
      int row=int(i/cols),col=int(i%cols);
      int64_t at=int64_t(row)*ld+col;
      const uint32_t* words=reinterpret_cast<const uint32_t*>(input);
      const int halfrows=side?kh:mh;
      uint32_t bits[4]={words[at/2],words[(at+cols)/2],words[(at+int64_t(halfrows)*ld)/2],words[(at+int64_t(halfrows)*ld+cols)/2]};
      float values[4][2];
      #pragma unroll
      for(int lane=0;lane<2;++lane) {
        float x[4];
        #pragma unroll
        for(int j=0;j<4;++j)x[j]=float(T::bitcast(uint16_t(bits[j]>>(16*lane))));
        if(side==0) {
          float s1=__fadd_rn(x[2],x[3]),s2=__fsub_rn(s1,x[0]);
          values[0][lane]=s1;values[1][lane]=s2;
          values[2][lane]=__fsub_rn(x[0],x[2]);values[3][lane]=__fsub_rn(x[1],s2);
        } else {
          float t1=__fsub_rn(x[1],x[0]),t2=__fsub_rn(x[3],t1);
          values[0][lane]=t1;values[1][lane]=t2;
          values[2][lane]=__fsub_rn(x[3],x[1]);values[3][lane]=__fsub_rn(t2,x[2]);
        }
      }
      #pragma unroll
      for(int j=0;j<4;++j)reinterpret_cast<uint32_t*>(output)[(j*plane+i)/2]=pair<T>(values[j][0],values[j][1]);
    }
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
}


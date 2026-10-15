#include "internal.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdexcept>

namespace megafmm::detail {
namespace {
template<class T> __device__ __forceinline__ T zero();
template<> __device__ __forceinline__ __half zero(){return __float2half_rn(0.0f);}
template<> __device__ __forceinline__ __nv_bfloat16 zero(){return __float2bfloat16_rn(0.0f);}
template<class T> __device__ __forceinline__ T at(const T* p,int64_t r,int64_t c,
    int64_t rows,int64_t cols,int64_t ld,bool transpose) {
  return r<rows && c<cols ? p[transpose?c*ld+r:r*ld+c]:zero<T>();
}
__device__ __forceinline__ void copy16(void* dst,const void* src,bool valid) {
  const unsigned smem=unsigned(__cvta_generic_to_shared(dst));
  const int bytes=valid?16:0;
  asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(smem),"l"(src),"r"(bytes) : "memory");
}
template<class T,int BK>
__device__ __forceinline__ void load_tile(T* sa,T* sb,Problem p,const T* A,const T* B,
                                         int64_t r0,int64_t c0,int64_t q,bool asynchronous) {
  constexpr int AS=BK+8,BS=72;
  if(asynchronous) {
    #pragma unroll
    for(int chunk=threadIdx.x;chunk<32*BK/8;chunk+=128) {
      int r=chunk/(BK/8),c=(chunk%(BK/8))*8;
      bool valid=r0+r<p.m && q+c<p.k;
      copy16(sa+r*AS+c,valid?A+(r0+r)*p.lda+q+c:A,valid);
    }
    #pragma unroll
    for(int chunk=threadIdx.x;chunk<BK*64/8;chunk+=128) {
      int r=chunk/8,c=(chunk%8)*8;
      bool valid=q+r<p.k && c0+c<p.n;
      copy16(sb+r*BS+c,valid?B+(q+r)*p.ldb+c0+c:B,valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  } else {
    #pragma unroll
    for(int i=threadIdx.x;i<32*BK;i+=128) {
      int r=i/BK,c=i%BK;
      sa[r*AS+c]=at(A,r0+r,q+c,p.m,p.k,p.lda,p.trans_a);
    }
    #pragma unroll
    for(int i=threadIdx.x;i<BK*64;i+=128) {
      int r=i/64,c=i%64;
      sb[r*BS+c]=at(B,q+r,c0+c,p.k,p.n,p.ldb,p.trans_b);
    }
  }
}
__device__ __forceinline__ void ready(bool asynchronous) {
  if(asynchronous) asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
}
// A small-GEMM kernel avoids TMA descriptor construction. Four warps cover
// 32x64 outputs with two accumulator fragments each; padded shared pitches
// reduce bank aliasing. A/B ping-pong buffers overlap cp.async with warp MMA.
template<class T,int BK>
__global__ __launch_bounds__(128) void direct_wmma(Problem p,const T* A,const T* B,
                                                   const float* C,float* D) {
  using namespace nvcuda;
  constexpr int AS=BK+8,BS=72,CS=72;
  __shared__ __align__(32) T sa[2][32*AS];
  __shared__ __align__(32) T sb[2][BK*BS];
  __shared__ __align__(32) float sc[32*CS];
  const int warp=threadIdx.x/32,wm=warp/2,wn=warp%2;
  const int64_t r0=int64_t(blockIdx.y)*32,c0=int64_t(blockIdx.x)*64;
  const bool asynchronous=!p.trans_a && !p.trans_b && p.k%8==0 && p.n%8==0 &&
      p.lda%8==0 && p.ldb%8==0 && reinterpret_cast<uintptr_t>(A)%16==0 && reinterpret_cast<uintptr_t>(B)%16==0;
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[2];
  wmma::fill_fragment(acc[0],0.0f);wmma::fill_fragment(acc[1],0.0f);
  load_tile<T,BK>(sa[0],sb[0],p,A,B,r0,c0,0,asynchronous);ready(asynchronous);
  int slot=0;
  for(int64_t q=0;q<p.k;q+=BK,slot^=1) {
    const bool next=q+BK<p.k;
    if(next) load_tile<T,BK>(sa[slot^1],sb[slot^1],p,A,B,r0,c0,q+BK,asynchronous);
    #pragma unroll
    for(int kk=0;kk<BK;kk+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> af;
      wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> bf;
      wmma::load_matrix_sync(af,sa[slot]+wm*16*AS+kk,AS);
      #pragma unroll
      for(int j=0;j<2;++j) {
        wmma::load_matrix_sync(bf,sb[slot]+kk*BS+wn*32+j*16,BS);
        wmma::mma_sync(acc[j],af,bf,acc[j]);
      }
    }
    if(next) ready(asynchronous);
  }
  #pragma unroll
  for(int j=0;j<2;++j)
    wmma::store_matrix_sync(sc+wm*16*CS+wn*32+j*16,acc[j],CS,wmma::mem_row_major);
  __syncthreads();
  for(int i=threadIdx.x;i<32*64;i+=128) {
    const int64_t r=r0+i/64,c=c0+i%64;
    if(r<p.m && c<p.n) {
      float value=p.alpha*sc[(i/64)*CS+i%64];
      if(p.beta!=0) value=fmaf(p.beta,C[r*p.ldc+c],value);
      D[r*p.ldd+c]=value;
    }
  }
}
}
bool wmma_dense(int bk,const Problem& p,const void* A,const void* B,const float* C,float* D,
                 cudaStream_t stream,int operation) {
  if(operation==2) return true;
  if(p.m<=0 || p.n<=0 || p.k<=0 || p.m>2048 || p.n>2048 || p.k>2048) return false;
  if(operation==0) return true;
  dim3 grid(unsigned((p.n+63)/64),unsigned((p.m+31)/32));
  if(p.dtype==DType::BF16) {
    if(bk==32) direct_wmma<__nv_bfloat16,32><<<grid,128,0,stream>>>(p,static_cast<const __nv_bfloat16*>(A),static_cast<const __nv_bfloat16*>(B),C,D);
    else direct_wmma<__nv_bfloat16,64><<<grid,128,0,stream>>>(p,static_cast<const __nv_bfloat16*>(A),static_cast<const __nv_bfloat16*>(B),C,D);
  } else {
    if(bk==32) direct_wmma<__half,32><<<grid,128,0,stream>>>(p,static_cast<const __half*>(A),static_cast<const __half*>(B),C,D);
    else direct_wmma<__half,64><<<grid,128,0,stream>>>(p,static_cast<const __half*>(A),static_cast<const __half*>(B),C,D);
  }
  auto e=cudaPeekAtLastError();if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
  return true;
}
}

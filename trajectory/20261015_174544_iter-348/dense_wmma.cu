#include "internal.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cute/arch/mma_sm80.hpp>
#include <type_traits>
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
__device__ __forceinline__ void copy4(void* dst,const void* src,bool valid) {
  const unsigned smem=unsigned(__cvta_generic_to_shared(dst));
  const int bytes=valid?4:0;
  asm volatile("cp.async.ca.shared.global [%0],[%1],4,%2;" :: "r"(smem),"l"(src),"r"(bytes) : "memory");
}
template<class T,int BK,int BN=64>
__device__ __forceinline__ void load_tile(T* sa,T* sb,Problem p,const T* A,const T* B,
                                         int64_t r0,int64_t c0,int64_t q,int copy_bytes) {
  constexpr int AS=BK+8,BS=BN+8;
  if(copy_bytes==16) {
    #pragma unroll
    for(int chunk=threadIdx.x;chunk<32*BK/8;chunk+=128) {
      int r=chunk/(BK/8),c=(chunk%(BK/8))*8;
      bool valid=r0+r<p.m && q+c<p.k;
      copy16(sa+r*AS+c,valid?A+(r0+r)*p.lda+q+c:A,valid);
    }
    #pragma unroll
    for(int chunk=threadIdx.x;chunk<BK*BN/8;chunk+=128) {
      int r=chunk/(BN/8),c=(chunk%(BN/8))*8;
      bool valid=q+r<p.k && c0+c<p.n;
      copy16(sb+r*BS+c,valid?B+(q+r)*p.ldb+c0+c:B,valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  } else if(copy_bytes==4) {
    #pragma unroll
    for(int chunk=threadIdx.x;chunk<32*BK/2;chunk+=128) {
      int r=chunk/(BK/2),c=(chunk%(BK/2))*2;
      bool valid=r0+r<p.m && q+c<p.k;
      copy4(sa+r*AS+c,valid?A+(r0+r)*p.lda+q+c:A,valid);
    }
    #pragma unroll
    for(int chunk=threadIdx.x;chunk<BK*BN/2;chunk+=128) {
      int r=chunk/(BN/2),c=(chunk%(BN/2))*2;
      bool valid=q+r<p.k && c0+c<p.n;
      copy4(sb+r*BS+c,valid?B+(q+r)*p.ldb+c0+c:B,valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  } else {
    #pragma unroll
    for(int i=threadIdx.x;i<32*BK;i+=128) {
      int r=i/BK,c=i%BK;
      sa[r*AS+c]=at(A,r0+r,q+c,p.m,p.k,p.lda,p.trans_a);
    }
    #pragma unroll
    for(int i=threadIdx.x;i<BK*BN;i+=128) {
      int r=i/BN,c=i%BN;
      sb[r*BS+c]=at(B,q+r,c0+c,p.k,p.n,p.ldb,p.trans_b);
    }
  }
}
__device__ __forceinline__ void ready(int copy_bytes) {
  if(copy_bytes) asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
}
// A small-GEMM kernel avoids TMA descriptor construction. Four warps cover
// 32x64 outputs with two accumulator fragments each; padded shared pitches
// reduce bank aliasing. A/B ping-pong buffers overlap cp.async with warp MMA.
template<class T,int BK,int BN=64>
__global__ __launch_bounds__(128) void direct_wmma(Problem p,const T* A,const T* B,
                                                   const float* C,float* D) {
  using namespace nvcuda;
  constexpr int AS=BK+8,BS=BN+8,CS=BN+8,ACC=BN/32;
  __shared__ __align__(32) T sa[2][32*AS];
  __shared__ __align__(32) T sb[2][BK*BS];
  __shared__ __align__(32) float sc[32*CS];
  const int warp=threadIdx.x/32,wm=warp/2,wn=warp%2;
  const int64_t r0=int64_t(blockIdx.y)*32,c0=int64_t(blockIdx.x)*BN;
  const bool wide=!p.trans_a && !p.trans_b && p.k%8==0 && p.n%8==0 &&
      p.lda%8==0 && p.ldb%8==0 && reinterpret_cast<uintptr_t>(A)%16==0 && reinterpret_cast<uintptr_t>(B)%16==0;
  const bool pair=!p.trans_a && !p.trans_b && p.k%2==0 && p.n%2==0 &&
      p.lda%2==0 && p.ldb%2==0 && reinterpret_cast<uintptr_t>(A)%4==0 && reinterpret_cast<uintptr_t>(B)%4==0;
  const int copy_bytes=wide?16:pair?4:0;
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[ACC];
  #pragma unroll
  for(int j=0;j<ACC;++j)wmma::fill_fragment(acc[j],0.0f);
  load_tile<T,BK,BN>(sa[0],sb[0],p,A,B,r0,c0,0,copy_bytes);ready(copy_bytes);
  int slot=0;
  for(int64_t q=0;q<p.k;q+=BK,slot^=1) {
    const bool next=q+BK<p.k;
    if(next) load_tile<T,BK,BN>(sa[slot^1],sb[slot^1],p,A,B,r0,c0,q+BK,copy_bytes);
    #pragma unroll
    for(int kk=0;kk<BK;kk+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> af;
      wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> bf;
      wmma::load_matrix_sync(af,sa[slot]+wm*16*AS+kk,AS);
      #pragma unroll
      for(int j=0;j<ACC;++j) {
        wmma::load_matrix_sync(bf,sb[slot]+kk*BS+wn*(BN/2)+j*16,BS);
        wmma::mma_sync(acc[j],af,bf,acc[j]);
      }
    }
    if(next) ready(copy_bytes);
  }
  #pragma unroll
  for(int j=0;j<ACC;++j)
    wmma::store_matrix_sync(sc+wm*16*CS+wn*(BN/2)+j*16,acc[j],CS,wmma::mem_row_major);
  __syncthreads();
  for(int i=threadIdx.x;i<32*BN;i+=128) {
    const int64_t r=r0+i/BN,c=c0+i%BN;
    if(r<p.m && c<p.n) {
      float value=p.alpha*sc[(i/BN)*CS+i%BN];
      if(p.beta!=0) value=fmaf(p.beta,C[r*p.ldc+c],value);
      D[r*p.ldd+c]=value;
    }
  }
}
// Explicit fragment addressing avoids a shared-memory accumulator round trip.
// The register layout is the SM80 m16n8k16 accumulator mapping used by CuTe.
template<class T,int BK>
__global__ __launch_bounds__(128) void direct_mma(Problem p,const T* A,const T* B,const float* C,float* D) {
  constexpr int AS=BK+8,BS=40;
  __shared__ __align__(32) T sa[2][32*AS];
  __shared__ __align__(32) T sb[2][BK*BS];
  const int lane=threadIdx.x%32,warp=threadIdx.x/32,wm=warp/2,wn=warp%2;
  const int64_t r0=int64_t(blockIdx.y)*32,c0=int64_t(blockIdx.x)*32;
  const bool wide=!p.trans_a && !p.trans_b && p.k%8==0 && p.n%8==0 &&
      p.lda%8==0 && p.ldb%8==0 && reinterpret_cast<uintptr_t>(A)%16==0 && reinterpret_cast<uintptr_t>(B)%16==0;
  const bool pair=!p.trans_a && !p.trans_b && p.k%2==0 && p.n%2==0 &&
      p.lda%2==0 && p.ldb%2==0 && reinterpret_cast<uintptr_t>(A)%4==0 && reinterpret_cast<uintptr_t>(B)%4==0;
  const int copy_bytes=wide?16:pair?4:0;
  float acc[2][4]={{0,0,0,0},{0,0,0,0}};
  load_tile<T,BK,32>(sa[0],sb[0],p,A,B,r0,c0,0,copy_bytes);ready(copy_bytes);
  int slot=0;
  for(int64_t q=0;q<p.k;q+=BK,slot^=1) {
    const bool next=q+BK<p.k;
    if(next)load_tile<T,BK,32>(sa[slot^1],sb[slot^1],p,A,B,r0,c0,q+BK,copy_bytes);
    #pragma unroll
    for(int kk=0;kk<BK;kk+=16) {
      uint32_t a[4];
      unsigned ap=unsigned(__cvta_generic_to_shared(sa[slot]+(wm*16+lane%16)*AS+kk+(lane/16)*8));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];" : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]) : "r"(ap));
      #pragma unroll
      for(int j=0;j<2;++j) {
        uint32_t b[2];
        unsigned bp=unsigned(__cvta_generic_to_shared(sb[slot]+(kk+lane%16)*BS+wn*16+j*8));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];" : "=r"(b[0]),"=r"(b[1]) : "r"(bp));
        using Mma=std::conditional_t<std::is_same_v<T,__half>,cute::SM80_16x8x16_F32F16F16F32_TN,cute::SM80_16x8x16_F32BF16BF16F32_TN>;
        Mma::fma(acc[j][0],acc[j][1],acc[j][2],acc[j][3],a[0],a[1],a[2],a[3],b[0],b[1],acc[j][0],acc[j][1],acc[j][2],acc[j][3]);
      }
    }
    if(next)ready(copy_bytes);
  }
  #pragma unroll
  for(int j=0;j<2;++j) {
    #pragma unroll
    for(int i=0;i<4;++i) {
      int64_t r=r0+wm*16+lane/4+(i/2)*8,c=c0+wn*16+j*8+(lane%4)*2+i%2;
      if(r<p.m && c<p.n) {
        float value=p.alpha*acc[j][i];
        if(p.beta!=0)value=fmaf(p.beta,C[r*p.ldc+c],value);
        D[r*p.ldd+c]=value;
      }
    }
  }
}
struct SmallProblem {int m,n,k,lda,ldb,ldc,ldd;float alpha,beta;};
template<int Stride,bool Swizzle>
__device__ __forceinline__ int shared_offset(int r,int c) {
  int i=r*Stride+c;
  if constexpr(Swizzle)i^=(r&7)<<3;
  return i;
}
template<class T,int BK,int Copy,int LoadK=BK,int BM=32,int BN=32,bool Swizzle=false>
__device__ __forceinline__ void load_even(T* sa,T* sb,SmallProblem p,const T* A,const T* B,
                                         int r0,int c0,int q) {
  constexpr int AS=Swizzle?BK:BK+8,BS=Swizzle?BN:BN+8,W=Copy/2;
  #pragma unroll
  for(int i=threadIdx.x;i<BM*LoadK/W;i+=128) {
    const int r=i/(LoadK/W),c=i%(LoadK/W)*W;
    const bool valid=r0+r<p.m && q+c<p.k;
    const int offset=(r0+r)*p.lda+q+c;
    if constexpr(Copy==16) copy16(sa+shared_offset<AS,Swizzle>(r,c),valid?A+offset:A,valid);
    else copy4(sa+shared_offset<AS,Swizzle>(r,c),valid?A+offset:A,valid);
  }
  #pragma unroll
  for(int i=threadIdx.x;i<LoadK*BN/W;i+=128) {
    const int r=i/(BN/W),c=i%(BN/W)*W;
    const bool valid=q+r<p.k && c0+c<p.n;
    const int offset=(q+r)*p.ldb+c0+c;
    if constexpr(Copy==16) copy16(sb+shared_offset<BS,Swizzle>(r,c),valid?B+offset:B,valid);
    else copy4(sb+shared_offset<BS,Swizzle>(r,c),valid?B+offset:B,valid);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template<class T,int BK,int Copy,int BM=32,int BN=32,bool Swizzle=false>
__device__ __forceinline__ void load_even_tile(T* sa,T* sb,SmallProblem p,const T* A,const T* B,int r0,int c0,int q) {
  if(p.k-q<=16)load_even<T,BK,Copy,16,BM,BN,Swizzle>(sa,sb,p,A,B,r0,c0,q);
  else load_even<T,BK,Copy,BK,BM,BN,Swizzle>(sa,sb,p,A,B,r0,c0,q);
}
template<class T,int BK,int Copy,int BM=32,int BN=32,bool Swizzle=false>
__global__ __launch_bounds__(128) void even_mma(SmallProblem p,const T* A,const T* B,const float* C,float* D) {
  constexpr int AS=Swizzle?BK:BK+8,BS=Swizzle?BN:BN+8,Cols=4/(BM/16),WN=BN/Cols,ACC=WN/8;
  __shared__ __align__(32) T sa[2][BM*AS];
  __shared__ __align__(32) T sb[2][BK*BS];
  const int lane=threadIdx.x%32,warp=threadIdx.x/32,wm=warp/Cols,wn=warp%Cols;
  const int r0=int(blockIdx.y)*BM,c0=int(blockIdx.x)*BN;
  float acc[ACC][4]={};
  load_even_tile<T,BK,Copy,BM,BN,Swizzle>(sa[0],sb[0],p,A,B,r0,c0,0);ready(Copy);
  int slot=0;
  for(int q=0;q<p.k;q+=BK,slot^=1) {
    const bool next=q+BK<p.k;
    if(next)load_even_tile<T,BK,Copy,BM,BN,Swizzle>(sa[slot^1],sb[slot^1],p,A,B,r0,c0,q+BK);
    #pragma unroll
    for(int kk=0;kk<BK;kk+=16) {
      if(kk && q+kk>=p.k)continue;
      uint32_t a[4];
      unsigned ap=unsigned(__cvta_generic_to_shared(sa[slot]+shared_offset<AS,Swizzle>(wm*16+lane%16,kk+(lane/16)*8)));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];" : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]) : "r"(ap));
      #pragma unroll
      for(int j=0;j<ACC;++j) {
        uint32_t b[2];
        unsigned bp=unsigned(__cvta_generic_to_shared(sb[slot]+shared_offset<BS,Swizzle>(kk+lane%16,wn*WN+j*8)));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];" : "=r"(b[0]),"=r"(b[1]) : "r"(bp));
        using Mma=std::conditional_t<std::is_same_v<T,__half>,cute::SM80_16x8x16_F32F16F16F32_TN,cute::SM80_16x8x16_F32BF16BF16F32_TN>;
        Mma::fma(acc[j][0],acc[j][1],acc[j][2],acc[j][3],a[0],a[1],a[2],a[3],b[0],b[1],acc[j][0],acc[j][1],acc[j][2],acc[j][3]);
      }
    }
    if(next)ready(Copy);
  }
  #pragma unroll
  for(int j=0;j<ACC;++j) {
    #pragma unroll
    for(int pair=0;pair<2;++pair) {
      const int r=r0+wm*16+lane/4+pair*8,c=c0+wn*WN+j*8+(lane%4)*2;
      if(r<p.m && c<p.n) {
        float2 value=make_float2(p.alpha*acc[j][pair*2],p.alpha*acc[j][pair*2+1]);
        if(p.beta!=0) {
          float2 prior=*reinterpret_cast<const float2*>(C+r*p.ldc+c);
          value.x=fmaf(p.beta,prior.x,value.x);value.y=fmaf(p.beta,prior.y,value.y);
        }
        *reinterpret_cast<float2*>(D+r*p.ldd+c)=value;
      }
    }
  }
}

}
bool wmma_dense(int bk,const Problem& p,const void* A,const void* B,const float* C,float* D,
                 cudaStream_t stream,int operation) {
  if(operation==2) return true;
  if(p.m<=0 || p.n<=0 || p.k<=0 || p.m>2048 || p.n>2048 || p.k>2048) return false;
  if(operation==0) return true;
  const int bn=bk>64?32:64;
  dim3 grid(unsigned((p.n+bn-1)/bn),unsigned((p.m+31)/32));
  if(p.dtype==DType::BF16) {
    if(bk==256) direct_mma<__nv_bfloat16,32><<<grid,128,0,stream>>>(p,static_cast<const __nv_bfloat16*>(A),static_cast<const __nv_bfloat16*>(B),C,D);
    else if(bk==257) direct_mma<__nv_bfloat16,64><<<grid,128,0,stream>>>(p,static_cast<const __nv_bfloat16*>(A),static_cast<const __nv_bfloat16*>(B),C,D);
    else if(bk==96) direct_wmma<__nv_bfloat16,32,32><<<grid,128,0,stream>>>(p,static_cast<const __nv_bfloat16*>(A),static_cast<const __nv_bfloat16*>(B),C,D);
    else if(bk==128) direct_wmma<__nv_bfloat16,64,32><<<grid,128,0,stream>>>(p,static_cast<const __nv_bfloat16*>(A),static_cast<const __nv_bfloat16*>(B),C,D);
    else if(bk==32) direct_wmma<__nv_bfloat16,32><<<grid,128,0,stream>>>(p,static_cast<const __nv_bfloat16*>(A),static_cast<const __nv_bfloat16*>(B),C,D);
    else direct_wmma<__nv_bfloat16,64><<<grid,128,0,stream>>>(p,static_cast<const __nv_bfloat16*>(A),static_cast<const __nv_bfloat16*>(B),C,D);
  } else {
    if(bk==256) direct_mma<__half,32><<<grid,128,0,stream>>>(p,static_cast<const __half*>(A),static_cast<const __half*>(B),C,D);
    else if(bk==257) direct_mma<__half,64><<<grid,128,0,stream>>>(p,static_cast<const __half*>(A),static_cast<const __half*>(B),C,D);
    else if(bk==96) direct_wmma<__half,32,32><<<grid,128,0,stream>>>(p,static_cast<const __half*>(A),static_cast<const __half*>(B),C,D);
    else if(bk==128) direct_wmma<__half,64,32><<<grid,128,0,stream>>>(p,static_cast<const __half*>(A),static_cast<const __half*>(B),C,D);
    else if(bk==32) direct_wmma<__half,32><<<grid,128,0,stream>>>(p,static_cast<const __half*>(A),static_cast<const __half*>(B),C,D);
    else direct_wmma<__half,64><<<grid,128,0,stream>>>(p,static_cast<const __half*>(A),static_cast<const __half*>(B),C,D);
  }
  auto e=cudaPeekAtLastError();if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
  return true;
}

bool mma_even_dense(int bk,const Problem& p,const void* A,const void* B,const float* C,float* D,
                    cudaStream_t stream,int operation) {
  if(operation==2)return true;
  if(p.trans_a || p.trans_b || p.m<=0 || p.n<=0 || p.k<=0 || p.m>2048 || p.n>2048 || p.k>2048 ||
     p.n%2 || p.k%2 || p.lda%2 || p.ldb%2 || p.lda>(1<<20) || p.ldb>(1<<20) ||
     p.ldc>(1<<20) || p.ldd>(1<<20) || p.ldd%2 || reinterpret_cast<uintptr_t>(D)%8 ||
     (p.beta!=0 && (p.ldc%2 || reinterpret_cast<uintptr_t>(C)%8)) || reinterpret_cast<uintptr_t>(A)%4 || reinterpret_cast<uintptr_t>(B)%4)return false;
  if(operation==0)return true;
  SmallProblem q{int(p.m),int(p.n),int(p.k),int(p.lda),int(p.ldb),int(p.ldc),int(p.ldd),p.alpha,p.beta};
  const bool wide=p.k%8==0 && p.n%8==0 && p.lda%8==0 && p.ldb%8==0 &&
       reinterpret_cast<uintptr_t>(A)%16==0 && reinterpret_cast<uintptr_t>(B)%16==0;
  const int bm=bk==128?16:32,bn=bk==129?16:32;
  dim3 grid(unsigned((p.n+bn-1)/bn),unsigned((p.m+bm-1)/bm));
  #define EVEN_LAUNCH(T,K,M,N,S) do { \
    if(wide)even_mma<T,K,16,M,N,S><<<grid,128,0,stream>>>(q,static_cast<const T*>(A),static_cast<const T*>(B),C,D); \
    else even_mma<T,K,4,M,N,S><<<grid,128,0,stream>>>(q,static_cast<const T*>(A),static_cast<const T*>(B),C,D); \
  } while(0)
  #define EVEN_DISPATCH(T) do { \
    if(bk==128)EVEN_LAUNCH(T,64,16,32,true); \
    else if(bk==129)EVEN_LAUNCH(T,64,32,16,true); \
    else if(bk==256)EVEN_LAUNCH(T,64,32,32,true); \
    else if(bk==32)EVEN_LAUNCH(T,32,32,32,false); \
    else EVEN_LAUNCH(T,64,32,32,false); \
  } while(0)
  if(p.dtype==DType::BF16)EVEN_DISPATCH(__nv_bfloat16);
  else EVEN_DISPATCH(__half);
  #undef EVEN_DISPATCH
  #undef EVEN_LAUNCH
  auto e=cudaPeekAtLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
  return true;
}
}

#include "sw_fp32.h"
#include <cuda.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <algorithm>
#include <stdexcept>

namespace megafmm::detail {
namespace {
void check(cudaError_t e) {
  if(e!=cudaSuccess)throw std::runtime_error(std::string("SW FP32: ")+cudaGetErrorString(e));
}
__device__ float load(const float* x,int64_t r,int64_t c,int64_t rows,
                      int64_t cols,int64_t ld,bool transpose) {
  return r<rows && c<cols ? x[transpose?c*ld+r:r*ld+c] : 0.0f;
}

// Each CTA stages seven transformed 16xBK / BKx16 operand tiles. Each thread
// retains all seven FP32 leaf sums for one position in the four quadrants.
// Every stage is consumed before reuse; no cross-CTA state or global scratch.
template<int BK,bool Guarded=false>
__global__ void sw_fp32(Problem p,const float* A,const float* B,const float* C,
                        float* D,int64_t tiles_n,int64_t tiles) {
  __shared__ float lhs[7][16][BK],rhs[7][BK][16];
  const int tid=threadIdx.x,r=tid/16,c=tid%16;
  const int64_t mh=(p.m+1)/2,nh=(p.n+1)/2,kh=(p.k+1)/2;
  for(int64_t tile=blockIdx.x;tile<tiles;tile+=gridDim.x) {
    const int64_t r0=(tile/tiles_n)*16,c0=(tile%tiles_n)*16;
    float v[7]={};
    for(int64_t q=0;q<kh;q+=BK) {
      for(int i=tid;i<16*BK;i+=256) {
        const int ar=i/BK,ak=i%BK,bk=i/16,bc=i%16;
        float a[4]={},b[4]={};
        if(r0+ar<mh && q+ak<kh) {
          #pragma unroll
          for(int j=0;j<4;++j)
            a[j]=load(A,r0+ar+(j/2)*mh,q+ak+(j%2)*kh,p.m,p.k,p.lda,p.trans_a);
        }
        if(q+bk<kh && c0+bc<nh) {
          #pragma unroll
          for(int j=0;j<4;++j)
            b[j]=load(B,q+bk+(j/2)*kh,c0+bc+(j%2)*nh,p.k,p.n,p.ldb,p.trans_b);
        }
        const float s1=__fadd_rn(a[2],a[3]),s2=__fsub_rn(s1,a[0]);
        const float t1=__fsub_rn(b[1],b[0]),t2=__fsub_rn(b[3],t1);
        const float av[7]={a[0],a[1],__fsub_rn(a[1],s2),a[3],s1,s2,__fsub_rn(a[0],a[2])};
        const float bv[7]={b[0],b[2],b[3],__fsub_rn(t2,b[2]),t1,t2,__fsub_rn(b[3],b[1])};
        #pragma unroll
        for(int j=0;j<7;++j) {
          if constexpr(Guarded) {
            if(r0+ar<mh && q+ak<kh)lhs[j][ar][ak]=av[j];
            if(q+bk<kh && c0+bc<nh)rhs[j][bk][bc]=bv[j];
          } else {lhs[j][ar][ak]=av[j];rhs[j][bk][bc]=bv[j];}
        }
      }
      __syncthreads();
      if constexpr(Guarded) {
        if(r0+r<mh && c0+c<nh) {
          const int valid_k=int(kh-q<BK?kh-q:BK);
          if(valid_k==BK) {
            #pragma unroll
            for(int k=0;k<BK;++k) {
              #pragma unroll
              for(int j=0;j<7;++j)v[j]=__fmaf_rn(lhs[j][r][k],rhs[j][k][c],v[j]);
            }
          } else for(int k=0;k<valid_k;++k) {
            #pragma unroll
            for(int j=0;j<7;++j)v[j]=__fmaf_rn(lhs[j][r][k],rhs[j][k][c],v[j]);
          }
          // Padded zero FMAs turn an underflowed -0 into +0. One such FMA
          // preserves the exact original finite seven-chain rounding semantics.
          if(valid_k<BK) {
            #pragma unroll
            for(int j=0;j<7;++j)v[j]=__fmaf_rn(0.0f,0.0f,v[j]);
          }
        }
      } else {
        #pragma unroll
        for(int k=0;k<BK;++k) {
          #pragma unroll
          for(int j=0;j<7;++j) v[j]=__fmaf_rn(lhs[j][r][k],rhs[j][k][c],v[j]);
        }
      }
      __syncthreads();
    }
    const float u2=__fadd_rn(v[0],v[5]),u3=__fadd_rn(u2,v[6]),u4=__fadd_rn(u2,v[4]);
    const float out[4]={__fadd_rn(v[0],v[1]),__fadd_rn(u4,v[2]),
                        __fsub_rn(u3,v[3]),__fadd_rn(u3,v[4])};
    if(r0+r<mh && c0+c<nh) {
      #pragma unroll
      for(int j=0;j<4;++j) {
        const int64_t row=r0+r+(j/2)*mh,col=c0+c+(j%2)*nh;
        if(row<p.m && col<p.n) {
          float value=__fmul_rn(p.alpha,out[j]);
          if(p.beta!=0) value=__fadd_rn(value,__fmul_rn(p.beta,C[row*p.ldc+col]));
          D[row*p.ldd+col]=value;
        }
      }
    }
  }
}

// TMA writes only the raw slots. All 256 threads retain the original transform,
// seven FP32 accumulation chains and reconstruction. One elected thread issues
// the next slot before the current slot's transform/compute. A CTA barrier after
// compute proves that the previous raw slot is free before it is reused.
template<int BK> struct alignas(128) TMAShared {
  alignas(128) float a[2][4][16][BK];
  alignas(128) float b[2][4][BK][16];
  float lhs[7][16][BK],rhs[7][BK][16];
  uint64_t ready[2];
};
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;

template<int BK> __device__ __forceinline__
void prefetch(TMAShared<BK>& s,const CUtensorMap& amap,const CUtensorMap& bmap,
              int slot,int q,int r0,int c0,int mh,int nh,int kh) {
  TxBarrier::arrive_and_expect_tx(&s.ready[slot],8*16*BK*sizeof(float));
  #pragma unroll
  for(int j=0;j<4;++j) {
    cute::SM90_TMA_LOAD_2D::copy(&amap,&s.ready[slot],0,s.a[slot][j],
                                q+(j%2)*kh,r0+(j/2)*mh);
    cute::SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],0,s.b[slot][j],
                                c0+(j%2)*nh,q+(j/2)*kh);
  }
}

template<int BK> __global__
void sw_fp32_tma(Problem p,const __grid_constant__ CUtensorMap amap,
                  const __grid_constant__ CUtensorMap bmap,const float* C,
                  float* D,int64_t tiles_n,int64_t tiles) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s=*reinterpret_cast<TMAShared<BK>*>(storage);
  const int tid=threadIdx.x,r=tid/16,c=tid%16;
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2),kh=int((p.k+1)/2);
  if(tid==0) {
    Barrier::init(&s.ready[0],1);Barrier::init(&s.ready[1],1);
    cutlass::arch::fence_barrier_init();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  int phase0=0,phase1=0;
  for(int64_t tile=blockIdx.x;tile<tiles;tile+=gridDim.x) {
    const int r0=int(tile/tiles_n)*16,c0=int(tile%tiles_n)*16;
    if(tid==0)prefetch(s,amap,bmap,0,0,r0,c0,mh,nh,kh);
    float v[7]={};
    for(int q=0,seq=0;q<kh;q+=BK,++seq) {
      const int slot=seq%2;
      Barrier::wait(&s.ready[slot],slot?phase1:phase0);
      if(slot)phase1^=1;else phase0^=1;
      asm volatile("" ::: "memory");
      if(tid==0 && q+BK<kh)
        prefetch(s,amap,bmap,slot^1,q+BK,r0,c0,mh,nh,kh);
      for(int i=tid;i<16*BK;i+=256) {
        const int ar=i/BK,ak=i%BK,bk=i/16,bc=i%16;
        float a[4]={},b[4]={};
        // TMA handles original-matrix OOB. Mask logical half boundaries too:
        // an upper quadrant's padded tile can otherwise read the lower half.
        if(r0+ar<mh && q+ak<kh) {
          #pragma unroll
          for(int j=0;j<4;++j)a[j]=s.a[slot][j][ar][ak];
        }
        if(q+bk<kh && c0+bc<nh) {
          #pragma unroll
          for(int j=0;j<4;++j)b[j]=s.b[slot][j][bk][bc];
        }
        const float s1=__fadd_rn(a[2],a[3]),s2=__fsub_rn(s1,a[0]);
        const float t1=__fsub_rn(b[1],b[0]),t2=__fsub_rn(b[3],t1);
        const float av[7]={a[0],a[1],__fsub_rn(a[1],s2),a[3],s1,s2,__fsub_rn(a[0],a[2])};
        const float bv[7]={b[0],b[2],b[3],__fsub_rn(t2,b[2]),t1,t2,__fsub_rn(b[3],b[1])};
        #pragma unroll
        for(int j=0;j<7;++j) {s.lhs[j][ar][ak]=av[j];s.rhs[j][bk][bc]=bv[j];}
      }
      __syncthreads();
      #pragma unroll
      for(int k=0;k<BK;++k) {
        #pragma unroll
        for(int j=0;j<7;++j)v[j]=__fmaf_rn(s.lhs[j][r][k],s.rhs[j][k][c],v[j]);
      }
      __syncthreads();
    }
    const float u2=__fadd_rn(v[0],v[5]),u3=__fadd_rn(u2,v[6]),u4=__fadd_rn(u2,v[4]);
    const float out[4]={__fadd_rn(v[0],v[1]),__fadd_rn(u4,v[2]),
                        __fsub_rn(u3,v[3]),__fadd_rn(u3,v[4])};
    if(r0+r<mh && c0+c<nh) {
      #pragma unroll
      for(int j=0;j<4;++j) {
        const int64_t row=r0+r+(j/2)*mh,col=c0+c+(j%2)*nh;
        if(row<p.m && col<p.n) {
          float value=__fmul_rn(p.alpha,out[j]);
          if(p.beta!=0)value=__fadd_rn(value,__fmul_rn(p.beta,C[row*p.ldc+col]));
          D[row*p.ldd+col]=value;
        }
      }
    }
    __syncthreads(); // Keep slot phases across grid-stride output tiles.
  }
}

template<int BK> void encode(CUtensorMap& map,const float* ptr,const Problem& p,bool a) {
  const uint64_t dims[2]={uint64_t(a?p.k:p.n),uint64_t(a?p.m:p.k)};
  const uint64_t strides[1]={uint64_t(a?p.lda:p.ldb)*sizeof(float)};
  const uint32_t box[2]={uint32_t(a?BK:16),uint32_t(a?16:BK)},es[2]={1,1};
  const auto result=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,
    const_cast<float*>(ptr),dims,strides,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(result!=CUDA_SUCCESS)throw std::runtime_error("SW FP32 tensor map: "+std::to_string(int(result)));
}

template<int BK> SWResources tma_resources() {
  check(cudaFuncSetAttribute(sw_fp32_tma<BK>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(TMAShared<BK>)));
  cudaFuncAttributes attr{};int resident=0;
  check(cudaFuncGetAttributes(&attr,sw_fp32_tma<BK>));
  check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,sw_fp32_tma<BK>,256,sizeof(TMAShared<BK>)));
  return {sizeof(TMAShared<BK>)+attr.sharedSizeBytes,attr.numRegs,resident};
}

template<int BK> void launch_tma(const Problem& p,const float* A,const float* B,
                                const float* C,float* D,cudaStream_t stream,
                                int blocks,int64_t cols,int64_t tiles) {
  // Per-binding stack descriptors become immutable grid-constant parameters.
  // No allocation, input cache, host synchronization or shared Plan mutation.
  CUtensorMap amap{},bmap{};encode<BK>(amap,A,p,true);encode<BK>(bmap,B,p,false);
  sw_fp32_tma<BK><<<blocks,256,sizeof(TMAShared<BK>),stream>>>(p,amap,bmap,C,D,cols,tiles);
}
template<int BK> SWResources guarded_resources() {
  cudaFuncAttributes attr{};int resident=0;
  check(cudaFuncGetAttributes(&attr,sw_fp32<BK,true>));
  check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,sw_fp32<BK,true>,256,0));
  return {attr.sharedSizeBytes,attr.numRegs,resident};
}
template<int BK> SWResources resources() {
  cudaFuncAttributes attr{};int resident=0;
  check(cudaFuncGetAttributes(&attr,sw_fp32<BK>));
  check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,sw_fp32<BK>,256,0));
  return {attr.sharedSizeBytes,attr.numRegs,resident};
}
}
bool sw_fp32_tma_problem(const Problem& p) {
  // A/B row pitches and both half-column offsets must be 16-byte aligned.
  return p.dtype==DType::FP32 && !p.trans_a && !p.trans_b &&
    p.m>0 && p.n>0 && p.k>0 && std::max({p.m,p.n,p.k})<=32768 &&
    p.n%8==0 && p.k%8==0 && p.lda%4==0 && p.ldb%4==0 &&
    uint64_t(p.lda)<(uint64_t(1)<<38) && uint64_t(p.ldb)<(uint64_t(1)<<38);
}
bool sw_fp32_tma_binding(const Problem& p,const void* A,const void* B) {
  return sw_fp32_tma_problem(p) && A && B &&
    reinterpret_cast<uintptr_t>(A)%16==0 && reinterpret_cast<uintptr_t>(B)%16==0;
}
SWResources setup_sw_fp32_tma(int bk) {
  if(bk==16)return tma_resources<16>();
  if(bk==32)return tma_resources<32>();
  throw std::invalid_argument("SW FP32 TMA BK must be 16 or 32");
}
SWResources setup_sw_fp32_guarded(int bk) {
  if(bk==16)return guarded_resources<16>();
  if(bk==32)return guarded_resources<32>();
  throw std::invalid_argument("SW FP32 guarded BK must be 16 or 32");
}
SWResources setup_sw_fp32(int bk) {
  if(bk==16)return resources<16>();
  if(bk==32)return resources<32>();
  throw std::invalid_argument("SW FP32 BK must be 16 or 32");
}
void run_sw_fp32_guarded_config(const Problem& p,const float* A,const float* B,const float* C,
                       float* D,cudaStream_t stream,int bk,bool tma,int max_blocks,bool guarded) {
  if(max_blocks<1)throw std::invalid_argument("SW FP32 grid cap must be positive");
  const int64_t rows=((p.m+1)/2+15)/16,cols=((p.n+1)/2+15)/16;
  const int blocks=int(std::min<int64_t>(rows*cols,max_blocks));
  if(tma && sw_fp32_tma_binding(p,A,B)) {
    if(bk==16)launch_tma<16>(p,A,B,C,D,stream,blocks,cols,rows*cols);
    else if(bk==32)launch_tma<32>(p,A,B,C,D,stream,blocks,cols,rows*cols);
    else throw std::invalid_argument("SW FP32 TMA BK must be 16 or 32");
  }
  else if(guarded && bk==16)sw_fp32<16,true><<<blocks,256,0,stream>>>(p,A,B,C,D,cols,rows*cols);
  else if(guarded && bk==32)sw_fp32<32,true><<<blocks,256,0,stream>>>(p,A,B,C,D,cols,rows*cols);
  else if(bk==16)sw_fp32<16><<<blocks,256,0,stream>>>(p,A,B,C,D,cols,rows*cols);
  else if(bk==32)sw_fp32<32><<<blocks,256,0,stream>>>(p,A,B,C,D,cols,rows*cols);
  else throw std::invalid_argument("SW FP32 BK must be 16 or 32");
  check(cudaPeekAtLastError());
}
bool sw_fp32_guarded_tail(const Problem& p,int bk) {
  if(bk!=16 && bk!=32)throw std::invalid_argument("SW FP32 guarded BK must be 16 or 32");
  return ((p.m+1)/2)%16 || ((p.n+1)/2)%16 || ((p.k+1)/2)%bk;
}
SWResources setup_sw_fp32(const Problem& p,int bk) {
  return sw_fp32_guarded_tail(p,bk)?setup_sw_fp32_guarded(bk):setup_sw_fp32(bk);
}
void run_sw_fp32_config(const Problem& p,const float* A,const float* B,const float* C,
                       float* D,cudaStream_t stream,int bk,bool tma,int max_blocks) {
  run_sw_fp32_guarded_config(p,A,B,C,D,stream,bk,tma,max_blocks,sw_fp32_guarded_tail(p,bk));
}
void run_sw_fp32(const Problem& p,const float* A,const float* B,const float* C,
                 float* D,cudaStream_t stream,int bk) {
  run_sw_fp32_config(p,A,B,C,D,stream,bk,true);
}
}

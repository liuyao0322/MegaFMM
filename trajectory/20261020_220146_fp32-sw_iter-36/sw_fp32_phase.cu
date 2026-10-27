#include "sw_fp32_phase.h"
#include <cuda.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <algorithm>
#include <stdexcept>

namespace megafmm::detail::sw_fp32_phase {
namespace {
constexpr int BK=16;
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
struct alignas(64) Maps {CUtensorMap a[4],b[4];};
template<int Rows,int Cols> struct alignas(128) Slab {float v[Rows][Cols];};
template<int BM,int BN,int APad> struct alignas(128) Shared {
  struct Stage {
    float lhs[7][BK][BM+APad],rhs[7][BK][BN];
    Slab<BM/4,BK+4> a[2][4][4];
    Slab<BK/4,BN+4> b[2][4][4];
  };
  union {Stage stage;float products[7][BM][BN];};
  // These barriers have a longer lifetime than either union member.
  uint64_t ready[2];
};
void check(cudaError_t e) {
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}

template<int BM,int BN,int APad> __device__ __forceinline__
void prefetch(Shared<BM,BN,APad>& s,const Maps& maps,int slot,int q,
              int r0,int c0,int mh,int nh,int kh,int lda,int ldb) {
  constexpr int Bytes=4*BM*(BK+4)*sizeof(float)+4*BK*(BN+4)*sizeof(float);
  TxBarrier::arrive_and_expect_tx(&s.ready[slot],Bytes);
  #pragma unroll
  for(int j=0;j<4;++j) {
    #pragma unroll
    for(int t=0;t<4;++t) {
      const int ar=r0+(j/2)*mh+t,ap=ar%4;
      const int ax=(q+(j%2)*kh+(ap*lda)%4)&~3;
      cute::SM90_TMA_LOAD_2D::copy(&maps.a[ap],&s.ready[slot],0,
        &s.stage.a[slot][j][t].v[0][0],ax,ar/4);
      const int br=q+(j/2)*kh+t,bp=br%4;
      const int bx=(c0+(j%2)*nh+(bp*ldb)%4)&~3;
      cute::SM90_TMA_LOAD_2D::copy(&maps.b[bp],&s.ready[slot],0,
        &s.stage.b[slot][j][t].v[0][0],bx,br/4);
    }
  }
}

template<int BM,int BN,int RM,int RN>
__global__ __launch_bounds__(256,2)
void kernel(Problem p,const __grid_constant__ Maps maps,const float* C,float* D,
            int tiles_n,int tiles) {
  constexpr int APad=RM==2?2:4,Cols=BN/RN;
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s=*reinterpret_cast<Shared<BM,BN,APad>*>(storage);
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
  const int row=(lane/Cols)*RM,col=(lane%Cols)*4;
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2),kh=int((p.k+1)/2);
  const int lda=int(p.lda),ldb=int(p.ldb);
  if(tid==224) {
    Barrier::init(&s.ready[0],1);Barrier::init(&s.ready[1],1);
    cutlass::arch::fence_barrier_init();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  int phase0=0,phase1=0;
  for(int tile=blockIdx.x;tile<tiles;tile+=gridDim.x) {
    const int r0=tile/tiles_n*BM,c0=tile%tiles_n*BN;
    float v[RM][RN]={};
    if(tid==224)prefetch(s,maps,0,0,r0,c0,mh,nh,kh,lda,ldb);
    for(int q=0,seq=0;q<kh;q+=BK,++seq) {
      const int slot=seq%2;
      Barrier::wait(&s.ready[slot],slot?phase1:phase0);
      if(slot)phase1^=1;else phase0^=1;
      asm volatile("" ::: "memory");
      if(tid==224 && q+BK<kh)
        prefetch(s,maps,slot^1,q+BK,r0,c0,mh,nh,kh,lda,ldb);
      for(int i=tid;i<BM*BK;i+=256) {
        const int ar=i/BK,ak=i%BK;
        float a[4]={};
        if(r0+ar<mh && q+ak<kh) {
          #pragma unroll
          for(int j=0;j<4;++j) {
            const int rr=r0+ar+(j/2)*mh,cc=q+ak+(j%2)*kh;
            const int phase=rr%4,delta=(q+(j%2)*kh+(phase*lda)%4)%4;
            if(rr<p.m && cc<p.k)a[j]=s.stage.a[slot][j][ar%4].v[ar/4][ak+delta];
          }
        }
        const float s1=__fadd_rn(a[2],a[3]),s2=__fsub_rn(s1,a[0]);
        const float av[7]={a[0],a[1],__fsub_rn(a[1],s2),a[3],s1,s2,__fsub_rn(a[0],a[2])};
        #pragma unroll
        for(int j=0;j<7;++j)s.stage.lhs[j][ak][ar]=av[j];
      }
      for(int i=tid;i<BK*BN;i+=256) {
        const int bk=i/BN,bc=i%BN;
        float b[4]={};
        if(q+bk<kh && c0+bc<nh) {
          #pragma unroll
          for(int j=0;j<4;++j) {
            const int rr=q+bk+(j/2)*kh,cc=c0+bc+(j%2)*nh;
            const int phase=rr%4,delta=(c0+(j%2)*nh+(phase*ldb)%4)%4;
            if(rr<p.k && cc<p.n)b[j]=s.stage.b[slot][j][bk%4].v[bk/4][bc+delta];
          }
        }
        const float t1=__fsub_rn(b[1],b[0]),t2=__fsub_rn(b[3],t1);
        const float bv[7]={b[0],b[2],b[3],__fsub_rn(t2,b[2]),t1,t2,__fsub_rn(b[3],b[1])};
        #pragma unroll
        for(int j=0;j<7;++j)s.stage.rhs[j][bk][bc]=bv[j];
      }
      __syncthreads();
      if(warp<7) {
        #pragma unroll
        for(int k=0;k<BK;++k) {
          float a[RM],b[RN];
          if constexpr(RM==2) {
            const float2 x=*reinterpret_cast<const float2*>(&s.stage.lhs[warp][k][row]);
            a[0]=x.x;a[1]=x.y;
          } else {
            #pragma unroll
            for(int r=0;r<RM;r+=4) {
              const float4 x=*reinterpret_cast<const float4*>(&s.stage.lhs[warp][k][row+r]);
              a[r]=x.x;a[r+1]=x.y;a[r+2]=x.z;a[r+3]=x.w;
            }
          }
          #pragma unroll
          for(int c=0;c<RN;c+=4) {
            const float4 x=*reinterpret_cast<const float4*>(&s.stage.rhs[warp][k][col+(c/4)*(BN/(RN/4))]);
            b[c]=x.x;b[c+1]=x.y;b[c+2]=x.z;b[c+3]=x.w;
          }
          #pragma unroll
          for(int r=0;r<RM;++r) {
            #pragma unroll
            for(int c=0;c<RN;++c)v[r][c]=__fmaf_rn(a[r],b[c],v[r][c]);
          }
        }
      }
      __syncthreads();
    }
    if(warp<7) {
      #pragma unroll
      for(int r=0;r<RM;++r) {
        #pragma unroll
        for(int c=0;c<RN;++c)
          s.products[warp][row+r][col+(c/4)*(BN/(RN/4))+c%4]=v[r][c];
      }
    }
    __syncthreads();
    for(int i=tid;i<BM*BN;i+=256) {
      const int r=i/BN,c=i%BN;
      float v[7];
      #pragma unroll
      for(int j=0;j<7;++j)v[j]=s.products[j][r][c];
      const float u2=__fadd_rn(v[0],v[5]),u3=__fadd_rn(u2,v[6]),u4=__fadd_rn(u2,v[4]);
      const float out[4]={__fadd_rn(v[0],v[1]),__fadd_rn(u4,v[2]),__fsub_rn(u3,v[3]),__fadd_rn(u3,v[4])};
      if(r0+r<mh && c0+c<nh) {
        #pragma unroll
        for(int j=0;j<4;++j) {
          const int rr=r0+r+(j/2)*mh,cc=c0+c+(j%2)*nh;
          if(rr<p.m && cc<p.n) {
            float value=__fmul_rn(p.alpha,out[j]);
            if(p.beta!=0)value=__fadd_rn(value,__fmul_rn(p.beta,C[int64_t(rr)*p.ldc+cc]));
            D[int64_t(rr)*p.ldd+cc]=value;
          }
        }
      }
    }
    __syncthreads();
  }
}

template<int BM,int BN,int RM,int RN>
SWResources dispatch(const Problem* p,const float* A,const float* B,const float* C,
                     float* D,cudaStream_t stream,int max_blocks) {
  constexpr size_t Bytes=sizeof(Shared<BM,BN,RM==2?2:4>);
  if(!p) {
    check(cudaFuncSetAttribute(kernel<BM,BN,RM,RN>,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;
    check(cudaFuncGetAttributes(&attr,kernel<BM,BN,RM,RN>));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel<BM,BN,RM,RN>,256,Bytes));
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  Maps maps{};
  for(int side=0;side<2;++side)for(int f=0;f<4;++f) {
    const int64_t rows=side?p->k:p->m,cols=side?p->n:p->k,ld=side?p->ldb:p->lda;
    const int64_t phase=f<rows?f:0,delta=(phase*ld)%4;
    const float* base=(side?B:A)+(phase*ld/4)*4;
    const uint64_t dims[2]={uint64_t(cols+delta),uint64_t((rows-phase+3)/4)};
    const uint64_t strides[1]={uint64_t(ld)*16};
    const uint32_t box[2]={uint32_t(side?BN+4:BK+4),uint32_t(side?BK/4:BM/4)},es[2]={1,1};
    const auto result=cuTensorMapEncodeTiled(side?&maps.b[f]:&maps.a[f],CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      2,const_cast<float*>(base),dims,strides,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(result!=CUDA_SUCCESS)throw std::runtime_error("FP32 row-phase tensor map: "+std::to_string(int(result)));
  }
  const int rows=int(((p->m+1)/2+BM-1)/BM),cols=int(((p->n+1)/2+BN-1)/BN);
  kernel<BM,BN,RM,RN><<<std::min(rows*cols,max_blocks),256,Bytes,stream>>>(*p,maps,C,D,cols,rows*cols);
  check(cudaPeekAtLastError());return {};
}
SWResources select(int id,const Problem* p,const float* A,const float* B,const float* C,
                   float* D,cudaStream_t stream,int max_blocks) {
  switch(id) {
#define CONFIG(ID,BM,BN,RM,RN) case ID:return dispatch<BM,BN,RM,RN>(p,A,B,C,D,stream,max_blocks)
    CONFIG(0,32,64,4,16);CONFIG(1,64,32,8,8);CONFIG(2,32,32,4,8);
    CONFIG(3,32,16,4,4);CONFIG(4,16,32,2,8);
#undef CONFIG
    default:throw std::invalid_argument("invalid FP32 row-phase configuration");
  }
}
}
bool problem(const Problem& p) {
  return p.dtype==DType::FP32 && !p.trans_a && !p.trans_b && p.m>0 && p.n>0 && p.k>0 &&
    std::max({p.m,p.n,p.k,p.lda,p.ldb})<=32768;
}
bool binding(const Problem& p,const void* A,const void* B) {
  return problem(p) && A && B && reinterpret_cast<uintptr_t>(A)%16==0 && reinterpret_cast<uintptr_t>(B)%16==0;
}
SWResources setup(int id) {return select(id,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,65535);}
std::string name(int id) {
  const auto& c=configurations.at(id);
  return "sw_fp32_phase;tile="+std::to_string(c.bm)+"x"+std::to_string(c.bn)+
    ";thread_tile="+std::to_string(c.rm)+"x"+std::to_string(c.rn)+
    ";bk=16;threads=256;leaf_terms=7;transforms=fp32;leaf_math=fp32_simt;accum=fp32;workspace=0;"
    "load=tma;row_phases=4;copies_per_stage=32;raw_stages=2;issuer=thread224;product_warps=7;residual_correction=off";
}
void run(const Problem& p,int id,const float* A,const float* B,const float* C,
         float* D,cudaStream_t stream,int max_blocks) {
  if(!binding(p,A,B) || max_blocks<=0)throw std::invalid_argument("FP32 row-phase TMA binding is incompatible");
  select(id,&p,A,B,C,D,stream,max_blocks);
}
}

#include "sw_fp32_tiled.h"
#include "sw_fp32.h"
#include <cuda.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace megafmm::detail::sw_fp32_tiled {
namespace {
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Empty {};
template<int BM,int BN,int BK> struct alignas(128) Raw {
  alignas(128) float a[2][4][BM][BK];
  alignas(128) float b[2][4][BK][BN];
  uint64_t ready[2];
};
template<int BM,int BN> struct ProductTiles {float v[7][BM][BN];};
template<int BM,int BN,int BK,bool TMA,bool Vectorized,bool Async,bool WarpProduct,bool Overlay=false,int APad=0> struct alignas(128) Shared {
  // The extra A column avoids conflicts between warp row groups reading K.
  float lhs[7][Vectorized?BK:BM][Vectorized?BM+APad:BK+1],rhs[7][BK][BN];
  std::conditional_t<TMA || Async,Raw<BM,BN,BK>,Empty> raw;
  std::conditional_t<WarpProduct,ProductTiles<BM,BN>,Empty> partial;
};
template<int BM,int BN,int BK,bool TMA,bool Vectorized,bool Async,bool WarpProduct,int APad>
struct alignas(128) Shared<BM,BN,BK,TMA,Vectorized,Async,WarpProduct,true,APad> {
  struct alignas(128) Stage {
    float lhs[7][Vectorized?BK:BM][Vectorized?BM+APad:BK+1],rhs[7][BK][BN];
    std::conditional_t<TMA || Async,Raw<BM,BN,BK>,Empty> raw;
  };
  static_assert(WarpProduct,"shared overlay requires product warps");
  // ready[] occupies the final128-byte segment. Output stores must not touch
  // these live barriers, whose phases continue into the next output tile.
  static_assert(!TMA || sizeof(ProductTiles<BM,BN>)<=sizeof(Stage)-128,"overlay would overwrite TMA barriers");
  union {Stage stage;ProductTiles<BM,BN> partial;};
};
__device__ __forceinline__ float load(const float* x,int64_t r,int64_t c,
                                     int64_t rows,int64_t cols,int64_t ld,bool transpose) {
  return r<rows && c<cols?x[transpose?c*ld+r:r*ld+c]:0.0f;
}
template<int BM,int BN,int BK> __device__ __forceinline__
void prefetch(Raw<BM,BN,BK>& raw,const CUtensorMap& amap,const CUtensorMap& bmap,
              int slot,int q,int r0,int c0,int mh,int nh,int kh) {
  TxBarrier::arrive_and_expect_tx(&raw.ready[slot],4*(BM+BN)*BK*sizeof(float));
  #pragma unroll
  for(int j=0;j<4;++j) {
    cute::SM90_TMA_LOAD_2D::copy(&amap,&raw.ready[slot],0,raw.a[slot][j],q+(j%2)*kh,r0+(j/2)*mh);
    cute::SM90_TMA_LOAD_2D::copy(&bmap,&raw.ready[slot],0,raw.b[slot][j],c0+(j%2)*nh,q+(j/2)*kh);
  }
}

template<int BM,int BN,int BK,int Threads> __device__ __forceinline__
void prefetch_async(Raw<BM,BN,BK>& raw,const Problem& p,const float* A,const float* B,
                    int slot,int q,int r0,int c0,int mh,int nh,int kh) {
  for(int side=0;side<2;++side) {
    const int count=side?BK*BN:BM*BK;
    for(int i=threadIdx.x;i<count;i+=Threads) {
      const int ir=side?i/BN:i/BK,ic=side?i%BN:i%BK;
      #pragma unroll
      for(int j=0;j<4;++j) {
        const int64_t r=(side?q:r0)+ir+(j/2)*(side?kh:mh);
        const int64_t c=(side?c0:q)+ic+(j%2)*(side?nh:kh);
        const bool valid=r<(side?p.k:p.m) && c<(side?p.n:p.k);
        const bool trans=side?p.trans_b:p.trans_a;
        const int64_t offset=trans?c*(side?p.ldb:p.lda)+r:r*(side?p.ldb:p.lda)+c;
        const float* source=(side?B:A)+(valid?offset:0);
        float* target=side?&raw.b[slot][j][ir][ic]:&raw.a[slot][j][ir][ic];
        const uint32_t shared=uint32_t(__cvta_generic_to_shared(target));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::
          "r"(shared),"l"(source),"r"(valid?4:0):"memory");
      }
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int BM,int BN,int RM,int RN,int BK,bool TMA,bool ProductMajor,bool Vectorized=false,bool Async=false,bool WarpProduct=false,int Unroll=2,bool Fast=false,bool Overlay=false,int MinBlocks=1,bool Shuffle=false,int APad=0>
__global__ __launch_bounds__((WarpProduct?256:BM*BN/(RM*RN)),MinBlocks)
void kernel(Problem p,const float* A,const float* B,const float* C,float* D,
            const __grid_constant__ CUtensorMap amap,const __grid_constant__ CUtensorMap bmap,
            int64_t tiles_n,int64_t tiles) {
  constexpr int Threads=WarpProduct?256:BM*BN/(RM*RN),WarpCols=BN/(8*RN)>0?BN/(8*RN):1;
  constexpr int VCols=RN>=4?4:RN,LaneCols=BN/RN,Issuer=WarpProduct?224:0;
  extern __shared__ __align__(128) unsigned char storage[];
  auto& root=*reinterpret_cast<Shared<BM,BN,BK,TMA,Vectorized,Async,WarpProduct,Overlay,APad>*>(storage);
  auto& s=[&]() -> auto& {if constexpr(Overlay)return root.stage;else return root;}();
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
  const int row=WarpProduct?(lane/LaneCols)*RM:(warp/WarpCols)*(4*RM)+(lane/8)*RM;
  const int col=WarpProduct?(lane%LaneCols)*VCols:(warp%WarpCols)*(8*RN)+(lane%8)*RN;
  const int64_t mh=(p.m+1)/2,nh=(p.n+1)/2,kh=(p.k+1)/2;
  if constexpr(TMA) {
    if(tid==Issuer) {
      Barrier::init(&s.raw.ready[0],1);Barrier::init(&s.raw.ready[1],1);
      cutlass::arch::fence_barrier_init();
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
  }
  int phase0=0,phase1=0;
  for(int64_t tile=blockIdx.x;tile<tiles;tile+=gridDim.x) {
    const int64_t r0=tile/tiles_n*BM,c0=tile%tiles_n*BN;
    float v[WarpProduct?1:7][RM][RN]={};
    if constexpr(TMA)if(tid==Issuer)prefetch(s.raw,amap,bmap,0,0,int(r0),int(c0),int(mh),int(nh),int(kh));
    if constexpr(Async)prefetch_async<BM,BN,BK,Threads>(s.raw,p,A,B,0,0,int(r0),int(c0),int(mh),int(nh),int(kh));
    for(int64_t q=0,seq=0;q<kh;q+=BK,++seq) {
      const int slot=int(seq%2);
      if constexpr(TMA) {
        Barrier::wait(&s.raw.ready[slot],slot?phase1:phase0);
        if(slot)phase1^=1;else phase0^=1;
        asm volatile("" ::: "memory");
        if(tid==Issuer && q+BK<kh)prefetch(s.raw,amap,bmap,slot^1,int(q+BK),int(r0),int(c0),int(mh),int(nh),int(kh));
      }
      if constexpr(Async) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if(q+BK<kh)prefetch_async<BM,BN,BK,Threads>(s.raw,p,A,B,slot^1,int(q+BK),int(r0),int(c0),int(mh),int(nh),int(kh));
      }
      for(int i=tid;i<BM*BK;i+=Threads) {
        const int ar=i/BK,ak=i%BK;
        float a[4]={};
        if constexpr(Fast && !TMA) {
          if(!p.trans_a && p.lda<=32768 && p.m<=32768 && r0+BM<=p.m-mh && q+BK<=p.k-kh) {
            const int offset=int(r0+ar)*int(p.lda)+int(q)+ak,lower=int(mh)*int(p.lda);
            a[0]=A[offset];a[1]=A[offset+int(kh)];a[2]=A[offset+lower];a[3]=A[offset+lower+int(kh)];
          } else if(r0+ar<mh && q+ak<kh) {
            #pragma unroll
            for(int j=0;j<4;++j)a[j]=load(A,r0+ar+(j/2)*mh,q+ak+(j%2)*kh,p.m,p.k,p.lda,p.trans_a);
          }
        } else if((Fast && TMA) || (r0+ar<mh && q+ak<kh)) {
          #pragma unroll
          for(int j=0;j<4;++j) {
            if constexpr(TMA || Async)a[j]=s.raw.a[slot][j][ar][ak];
            else a[j]=load(A,r0+ar+(j/2)*mh,q+ak+(j%2)*kh,p.m,p.k,p.lda,p.trans_a);
          }
        }
        const float s1=__fadd_rn(a[2],a[3]),s2=__fsub_rn(s1,a[0]);
        const float av[7]={a[0],a[1],__fsub_rn(a[1],s2),a[3],s1,s2,__fsub_rn(a[0],a[2])};
        #pragma unroll
        for(int j=0;j<7;++j) {
          if constexpr(Vectorized)s.lhs[j][ak][ar ^ (APad?0:((ak%(BM>=32?8:BM/4))*4))]=av[j];
          else s.lhs[j][ar][ak]=av[j];
        }
      }
      for(int i=tid;i<BK*BN;i+=Threads) {
        const int bk=i/BN,bc=i%BN;
        float b[4]={};
        if constexpr(Fast && !TMA) {
          if(!p.trans_b && p.ldb<=32768 && p.k<=32768 && q+BK<=p.k-kh && c0+BN<=p.n-nh) {
            const int offset=int(q+bk)*int(p.ldb)+int(c0)+bc,lower=int(kh)*int(p.ldb);
            b[0]=B[offset];b[1]=B[offset+int(nh)];b[2]=B[offset+lower];b[3]=B[offset+lower+int(nh)];
          } else if(q+bk<kh && c0+bc<nh) {
            #pragma unroll
            for(int j=0;j<4;++j)b[j]=load(B,q+bk+(j/2)*kh,c0+bc+(j%2)*nh,p.k,p.n,p.ldb,p.trans_b);
          }
        } else if((Fast && TMA) || (q+bk<kh && c0+bc<nh)) {
          #pragma unroll
          for(int j=0;j<4;++j) {
            if constexpr(TMA || Async)b[j]=s.raw.b[slot][j][bk][bc];
            else b[j]=load(B,q+bk+(j/2)*kh,c0+bc+(j%2)*nh,p.k,p.n,p.ldb,p.trans_b);
          }
        }
        const float t1=__fsub_rn(b[1],b[0]),t2=__fsub_rn(b[3],t1);
        const float bv[7]={b[0],b[2],b[3],__fsub_rn(t2,b[2]),t1,t2,__fsub_rn(b[3],b[1])};
        #pragma unroll
        for(int j=0;j<7;++j)s.rhs[j][bk][bc]=bv[j];
      }
      __syncthreads();
      // Multiple outputs reuse each A/B register. Keep ascending K and each
      // product's independent FMA chain; partial unroll bounds instruction size.
      const int active_k=BM<=8?int(kh-q<BK?kh-q:BK):BK;
      if constexpr(WarpProduct) {
        if(warp<7) {
          #pragma unroll Unroll
          for(int k=0;k<active_k;++k) {
            float a[RM],b[RN];
            if constexpr(Shuffle) {
              float aa[(BM+31)/32],bb[(BN+31)/32];
              #pragma unroll
              for(int i=0;i<(BM+31)/32;++i) {
                const int rr=i*32+lane%(BM<32?BM:32);
                aa[i]=s.lhs[warp][k][rr ^ (APad?0:((k%(BM>=32?8:BM/4))*4))];
              }
              #pragma unroll
              for(int i=0;i<(BN+31)/32;++i)bb[i]=s.rhs[warp][k][i*32+lane%(BN<32?BN:32)];
              #pragma unroll
              for(int r=0;r<RM;++r) {
                const int rr=row+r;
                const float x=__shfl_sync(0xffffffff,aa[0],rr%32);
                if constexpr(BM>32) {
                  const float y=__shfl_sync(0xffffffff,aa[1],rr%32);
                  a[r]=rr<32?x:y;
                } else a[r]=x;
              }
              #pragma unroll
              for(int c=0;c<RN;++c) {
                const int cc=col+(c/VCols)*(BN/(RN/VCols))+c%VCols;
                const float x=__shfl_sync(0xffffffff,bb[0],cc%32);
                if constexpr(BN>32) {
                  const float y=__shfl_sync(0xffffffff,bb[1],cc%32);
                  b[c]=cc<32?x:y;
                } else b[c]=x;
              }
            } else if constexpr(Vectorized) {
              if constexpr(RM>=4) {
                #pragma unroll
                for(int r=0;r<RM;r+=4) {
                  const float4 x=*reinterpret_cast<const float4*>(&s.lhs[warp][k][(row+r) ^ (APad?0:((k%(BM>=32?8:BM/4))*4))]);
                  a[r]=x.x;a[r+1]=x.y;a[r+2]=x.z;a[r+3]=x.w;
                }
              } else {
                const float2 x=*reinterpret_cast<const float2*>(&s.lhs[warp][k][row ^ (APad?0:((k%(BM>=32?8:BM/4))*4))]);
                a[0]=x.x;a[1]=x.y;
              }
              #pragma unroll
              for(int c=0;c<RN;c+=4) {
                const float4 x=*reinterpret_cast<const float4*>(&s.rhs[warp][k][col+(c/4)*(BN/(RN/4))]);
                b[c]=x.x;b[c+1]=x.y;b[c+2]=x.z;b[c+3]=x.w;
              }
            } else {
              #pragma unroll
              for(int r=0;r<RM;++r)a[r]=s.lhs[warp][row+r][k];
              #pragma unroll
              for(int c=0;c<RN;++c)b[c]=s.rhs[warp][k][col+c];
            }
            #pragma unroll
            for(int r=0;r<RM;++r) {
              #pragma unroll
              for(int c=0;c<RN;++c)v[0][r][c]=__fmaf_rn(a[r],b[c],v[0][r][c]);
            }
          }
        }
      } else if constexpr(ProductMajor) {
        // Finish a product's current K block before loading the next product.
        // The other products' accumulators stay live, but their operands do not.
        #pragma unroll
        for(int j=0;j<7;++j) {
          #pragma unroll Unroll
          for(int k=0;k<active_k;++k) {
            float a[RM],b[RN];
            if constexpr(Vectorized) {
              if constexpr(RM==4) {
                const float4 x=*reinterpret_cast<const float4*>(&s.lhs[j][k][row ^ (APad?0:((k%(BM>=32?8:BM/4))*4))]);
                a[0]=x.x;a[1]=x.y;a[2]=x.z;a[3]=x.w;
              } else {
                const float2 x=*reinterpret_cast<const float2*>(&s.lhs[j][k][row ^ (APad?0:((k%(BM>=32?8:BM/4))*4))]);
                a[0]=x.x;a[1]=x.y;
              }
              if constexpr(RN==4) {
                const float4 x=*reinterpret_cast<const float4*>(&s.rhs[j][k][col]);
                b[0]=x.x;b[1]=x.y;b[2]=x.z;b[3]=x.w;
              } else {
                const float2 x=*reinterpret_cast<const float2*>(&s.rhs[j][k][col]);
                b[0]=x.x;b[1]=x.y;
              }
            } else {
              #pragma unroll
              for(int r=0;r<RM;++r)a[r]=s.lhs[j][row+r][k];
              #pragma unroll
              for(int c=0;c<RN;++c)b[c]=s.rhs[j][k][col+c];
            }
            #pragma unroll
            for(int r=0;r<RM;++r) {
              #pragma unroll
              for(int c=0;c<RN;++c)v[j][r][c]=__fmaf_rn(a[r],b[c],v[j][r][c]);
            }
          }
        }
      } else {
        #pragma unroll Unroll
        for(int k=0;k<active_k;++k) {
          #pragma unroll
          for(int j=0;j<7;++j) {
            float a[RM],b[RN];
            if constexpr(Vectorized) {
              if constexpr(RM==4) {
                const float4 x=*reinterpret_cast<const float4*>(&s.lhs[j][k][row ^ (APad?0:((k%(BM>=32?8:BM/4))*4))]);
                a[0]=x.x;a[1]=x.y;a[2]=x.z;a[3]=x.w;
              } else {
                const float2 x=*reinterpret_cast<const float2*>(&s.lhs[j][k][row ^ (APad?0:((k%(BM>=32?8:BM/4))*4))]);
                a[0]=x.x;a[1]=x.y;
              }
              if constexpr(RN==4) {
                const float4 x=*reinterpret_cast<const float4*>(&s.rhs[j][k][col]);
                b[0]=x.x;b[1]=x.y;b[2]=x.z;b[3]=x.w;
              } else {
                const float2 x=*reinterpret_cast<const float2*>(&s.rhs[j][k][col]);
                b[0]=x.x;b[1]=x.y;
              }
            } else {
              #pragma unroll
              for(int r=0;r<RM;++r)a[r]=s.lhs[j][row+r][k];
              #pragma unroll
              for(int c=0;c<RN;++c)b[c]=s.rhs[j][k][col+c];
            }
            #pragma unroll
            for(int r=0;r<RM;++r) {
              #pragma unroll
              for(int c=0;c<RN;++c)v[j][r][c]=__fmaf_rn(a[r],b[c],v[j][r][c]);
            }
          }
        }
      }
      __syncthreads();
    }
    if constexpr(!WarpProduct) {
    #pragma unroll
    for(int r=0;r<RM;++r) {
      #pragma unroll
      for(int c=0;c<RN;++c) {
        const float u2=__fadd_rn(v[0][r][c],v[5][r][c]);
        const float u3=__fadd_rn(u2,v[6][r][c]),u4=__fadd_rn(u2,v[4][r][c]);
        const float out[4]={__fadd_rn(v[0][r][c],v[1][r][c]),__fadd_rn(u4,v[2][r][c]),
                            __fsub_rn(u3,v[3][r][c]),__fadd_rn(u3,v[4][r][c])};
        if((Fast && TMA) || (r0+row+r<mh && c0+col+c<nh)) {
          #pragma unroll
          for(int j=0;j<4;++j) {
            const int64_t rr=r0+row+r+(j/2)*mh,cc=c0+col+c+(j%2)*nh;
            if((Fast && TMA) || (rr<p.m && cc<p.n)) {
              float value=__fmul_rn(p.alpha,out[j]);
              if(p.beta!=0)value=__fadd_rn(value,__fmul_rn(p.beta,C[rr*p.ldc+cc]));
              D[rr*p.ldd+cc]=value;
            }
          }
        }
      }
    }
    } else {
      if(warp<7) {
        #pragma unroll
        for(int r=0;r<RM;++r) {
          #pragma unroll
          for(int c=0;c<RN;++c) {
            const int cc=Vectorized?col+(c/VCols)*(BN/(RN/VCols))+c%VCols:col+c;
            root.partial.v[warp][row+r][cc]=v[0][r][c];
          }
        }
      }
      __syncthreads();
      for(int i=tid;i<BM*BN;i+=Threads) {
        const int r=i/BN,c=i%BN;
        float vv[7];
        #pragma unroll
        for(int j=0;j<7;++j)vv[j]=root.partial.v[j][r][c];
        const float u2=__fadd_rn(vv[0],vv[5]),u3=__fadd_rn(u2,vv[6]),u4=__fadd_rn(u2,vv[4]);
        const float out[4]={__fadd_rn(vv[0],vv[1]),__fadd_rn(u4,vv[2]),__fsub_rn(u3,vv[3]),__fadd_rn(u3,vv[4])};
        if((Fast && TMA) || (r0+r<mh && c0+c<nh)) {
          #pragma unroll
          for(int j=0;j<4;++j) {
            const int64_t rr=r0+r+(j/2)*mh,cc=c0+c+(j%2)*nh;
            if((Fast && TMA) || (rr<p.m && cc<p.n)) {
              float value=__fmul_rn(p.alpha,out[j]);
              if(p.beta!=0)value=__fadd_rn(value,__fmul_rn(p.beta,C[rr*p.ldc+cc]));
              D[rr*p.ldd+cc]=value;
            }
          }
        }
      }
    }
    __syncthreads();
  }
}

template<int BM,int BN,int RM,int RN,int BK,bool TMA,bool ProductMajor,bool Vectorized=false,bool Async=false,bool WarpProduct=false,int Unroll=2,bool Fast=false,bool Overlay=false,int MinBlocks=1,bool Shuffle=false,int APad=0>
SWResources dispatch(const Problem* p,const float* A,const float* B,const float* C,
                     float* D,cudaStream_t stream,int max_blocks) {
  constexpr int Threads=WarpProduct?256:BM*BN/(RM*RN);
  constexpr size_t Bytes=sizeof(Shared<BM,BN,BK,TMA,Vectorized,Async,WarpProduct,Overlay,APad>);
  if(!p) {
    check(cudaFuncSetAttribute(kernel<BM,BN,RM,RN,BK,TMA,ProductMajor,Vectorized,Async,WarpProduct,Unroll,Fast,Overlay,MinBlocks,Shuffle,APad>,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;
    check(cudaFuncGetAttributes(&attr,kernel<BM,BN,RM,RN,BK,TMA,ProductMajor,Vectorized,Async,WarpProduct,Unroll,Fast,Overlay,MinBlocks,Shuffle,APad>));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel<BM,BN,RM,RN,BK,TMA,ProductMajor,Vectorized,Async,WarpProduct,Unroll,Fast,Overlay,MinBlocks,Shuffle,APad>,Threads,Bytes));
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  CUtensorMap amap{},bmap{};
  if constexpr(TMA) {
    for(int side=0;side<2;++side) {
      uint64_t dims[2]={uint64_t(side?p->n:p->k),uint64_t(side?p->k:p->m)};
      uint64_t strides[1]={uint64_t(side?p->ldb:p->lda)*sizeof(float)};
      uint32_t box[2]={uint32_t(side?BN:BK),uint32_t(side?BK:BM)},es[2]={1,1};
      const auto result=cuTensorMapEncodeTiled(side?&bmap:&amap,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,
        const_cast<float*>(side?B:A),dims,strides,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if(result!=CUDA_SUCCESS)throw std::runtime_error("FP32 tiled tensor map: "+std::to_string(int(result)));
    }
  }
  const int64_t rows=((p->m+1)/2+BM-1)/BM,cols=((p->n+1)/2+BN-1)/BN;
  const int blocks=int(std::min<int64_t>(rows*cols,max_blocks));
  kernel<BM,BN,RM,RN,BK,TMA,ProductMajor,Vectorized,Async,WarpProduct,Unroll,Fast,Overlay,MinBlocks,Shuffle,APad><<<blocks,Threads,Bytes,stream>>>(*p,A,B,C,D,amap,bmap,cols,rows*cols);
  check(cudaPeekAtLastError());return {};
}

SWResources select(int id,const Problem* p,const float* A,const float* B,const float* C,
                   float* D,cudaStream_t stream,int max_blocks) {
  switch(id) {
#define CONFIG(ID,BM,BN,RM,RN,TMA) case ID:return dispatch<BM,BN,RM,RN,16,TMA,false>(p,A,B,C,D,stream,max_blocks)
    CONFIG(0,32,16,2,2,false);CONFIG(1,32,16,2,2,true);
    CONFIG(2,32,32,4,2,false);CONFIG(3,32,32,4,2,true);
    CONFIG(4,64,32,4,2,false);CONFIG(5,64,32,4,2,true);
    CONFIG(6,32,64,2,4,false);CONFIG(7,32,64,2,4,true);
    CONFIG(8,64,64,4,4,false);CONFIG(9,64,64,4,4,true);
#undef CONFIG
#define CONFIG(ID,BM,BN,RM,RN,TMA) case ID:return dispatch<BM,BN,RM,RN,16,TMA,true>(p,A,B,C,D,stream,max_blocks)
    CONFIG(10,32,16,2,2,false);CONFIG(11,32,16,2,2,true);
    CONFIG(12,32,32,4,2,false);CONFIG(13,32,32,4,2,true);
    CONFIG(14,64,32,4,2,false);CONFIG(15,64,32,4,2,true);
    CONFIG(16,32,64,2,4,false);CONFIG(17,32,64,2,4,true);
    CONFIG(18,64,64,4,4,false);CONFIG(19,64,64,4,4,true);
    CONFIG(20,32,32,4,4,false);CONFIG(21,32,32,4,4,true);
    CONFIG(22,64,32,4,4,false);CONFIG(23,64,32,4,4,true);
    CONFIG(24,32,64,4,4,false);CONFIG(25,32,64,4,4,true);
    CONFIG(26,16,16,4,2,false);CONFIG(27,16,16,4,2,true);
#undef CONFIG
#define CONFIG(ID,BM,BN,RM,RN,TMA) case ID:return dispatch<BM,BN,RM,RN,16,TMA,false,true>(p,A,B,C,D,stream,max_blocks)
    CONFIG(28,32,16,2,2,false);CONFIG(29,32,16,2,2,true);
    CONFIG(30,32,32,4,2,false);CONFIG(31,32,32,4,2,true);
    CONFIG(32,64,32,4,2,false);CONFIG(33,64,32,4,2,true);
    CONFIG(34,32,64,2,4,false);CONFIG(35,32,64,2,4,true);
    CONFIG(36,64,64,4,4,false);CONFIG(37,64,64,4,4,true);
    CONFIG(38,32,32,4,4,false);CONFIG(39,32,32,4,4,true);
    CONFIG(40,64,32,4,4,false);CONFIG(41,64,32,4,4,true);
    CONFIG(42,32,64,4,4,false);CONFIG(43,32,64,4,4,true);
#undef CONFIG
#define SMALL(ID,BM,BN,BK) case ID:return dispatch<BM,BN,1,1,BK,false,false>(p,A,B,C,D,stream,max_blocks)
    SMALL(44,4,8,128);SMALL(45,8,8,128);SMALL(46,8,16,128);SMALL(47,4,16,128);
    SMALL(48,4,8,32);SMALL(49,8,8,32);SMALL(50,8,16,32);SMALL(51,4,16,32);
    SMALL(52,4,8,64);SMALL(53,8,8,64);SMALL(54,8,16,64);SMALL(55,4,16,64);
#undef SMALL
#define ASYNC(ID,BM,BN,RM,RN) case ID:return dispatch<BM,BN,RM,RN,16,false,false,true,true>(p,A,B,C,D,stream,max_blocks)
    ASYNC(56,32,16,2,2);ASYNC(57,32,32,4,2);ASYNC(58,64,32,4,2);ASYNC(59,32,64,2,4);
    ASYNC(60,32,32,4,4);ASYNC(61,64,32,4,4);ASYNC(62,32,64,4,4);
#undef ASYNC
#define WP(ID,BM,BN,RM,RN,TMA,ASYNC) case ID:return dispatch<BM,BN,RM,RN,16,TMA,false,true,ASYNC,true>(p,A,B,C,D,stream,max_blocks)
    WP(63,32,32,4,8,true,false);WP(64,32,32,4,8,false,false);WP(65,32,32,4,8,false,true);
    WP(66,32,16,4,4,true,false);WP(67,32,16,4,4,false,false);WP(68,32,16,4,4,false,true);
    WP(69,16,32,2,8,true,false);WP(70,16,32,2,8,false,false);WP(71,16,32,2,8,false,true);
    WP(72,16,16,2,4,true,false);WP(73,16,16,2,4,false,false);WP(74,16,16,2,4,false,true);
#undef WP
#define WPS(ID,BM,BN,RM,RN,BK) case ID:return dispatch<BM,BN,RM,RN,BK,false,false,false,false,true>(p,A,B,C,D,stream,max_blocks)
    WPS(75,4,8,1,1,128);WPS(76,8,8,1,2,128);WPS(77,4,16,1,2,128);
    WPS(78,4,8,1,1,32);WPS(79,4,8,1,1,64);WPS(80,8,8,1,2,64);
#undef WPS
#define WPU(ID,BM,BN,RM,RN,BK,TMA,U) case ID:return dispatch<BM,BN,RM,RN,BK,TMA,false,true,false,true,U>(p,A,B,C,D,stream,max_blocks)
    WPU(81,32,32,4,8,16,true,4);
    WPU(82,32,16,4,4,16,true,4);
    WPU(83,16,32,2,8,16,true,4);
    WPU(84,16,16,2,4,16,true,4);
    WPU(85,32,32,4,8,16,true,8);
    WPU(86,32,16,4,4,16,true,8);
    WPU(87,16,32,2,8,16,true,8);
    WPU(88,16,16,2,4,16,true,8);
    WPU(89,32,32,4,8,16,true,16);
    WPU(90,32,16,4,4,16,true,16);
    WPU(91,16,32,2,8,16,true,16);
    WPU(92,16,16,2,4,16,true,16);
    WPU(93,64,32,8,8,16,false,2);
    WPU(94,64,32,8,8,16,false,8);
    WPU(95,64,32,8,8,16,false,16);
    WPU(96,64,32,8,8,16,true,2);
    WPU(97,64,32,8,8,16,true,8);
    WPU(98,64,32,8,8,16,true,16);
    WPU(99,32,64,4,16,16,false,2);
    WPU(100,32,64,4,16,16,false,8);
    WPU(101,32,64,4,16,16,false,16);
    WPU(102,32,64,4,16,16,true,2);
    WPU(103,32,64,4,16,16,true,8);
    WPU(104,32,64,4,16,16,true,16);
    WPU(105,32,16,4,4,32,false,2);
    WPU(106,32,16,4,4,32,false,8);
    WPU(107,32,16,4,4,32,true,2);
    WPU(108,32,16,4,4,32,true,8);
    WPU(109,32,32,4,8,32,false,2);
    WPU(110,32,32,4,8,32,false,8);
    WPU(111,32,32,4,8,32,true,2);
    WPU(112,32,32,4,8,32,true,8);
#undef WPU
#define FAST(ID,BM,BN,RM,RN,BK,TMA,WP,U) case ID: \
    if(TMA && !p)dispatch<BM,BN,RM,RN,BK,TMA,false,true,false,WP,U,false>(p,A,B,C,D,stream,max_blocks); \
    if(TMA && p && (p->m%(2*BM) || p->n%(2*BN) || p->k%(2*BK))) \
      return dispatch<BM,BN,RM,RN,BK,TMA,false,true,false,WP,U,false>(p,A,B,C,D,stream,max_blocks); \
    return dispatch<BM,BN,RM,RN,BK,TMA,false,true,false,WP,U,true>(p,A,B,C,D,stream,max_blocks)
    FAST(113,32,32,4,2,16,false,false,2);
    FAST(114,64,32,4,2,16,false,false,2);
    FAST(115,32,64,2,4,16,false,false,2);
    FAST(116,64,64,4,4,16,false,false,2);
    FAST(117,32,32,4,8,16,false,true,2);
    FAST(118,32,32,4,8,16,false,true,16);
    FAST(119,64,32,8,8,16,false,true,16);
    FAST(120,32,64,4,16,16,false,true,16);
    FAST(121,32,16,4,4,16,false,true,16);
    FAST(122,16,32,2,8,16,false,true,16);
    FAST(123,32,32,4,8,16,true,true,16);
    FAST(124,64,32,8,8,16,true,true,16);
    FAST(125,32,64,4,16,16,true,true,16);
    FAST(126,32,16,4,4,16,true,true,16);
    FAST(127,32,16,4,4,32,true,true,2);
    FAST(128,16,32,2,8,16,true,true,16);
#undef FAST
#define OVERLAY(ID,BM,BN,RM,RN,BK,TMA,U,MB) case ID: \
    if(TMA && !p)dispatch<BM,BN,RM,RN,BK,TMA,false,true,false,true,U,false,true,MB>(p,A,B,C,D,stream,max_blocks); \
    if(TMA && p && (p->m%(2*BM) || p->n%(2*BN) || p->k%(2*BK))) \
      return dispatch<BM,BN,RM,RN,BK,TMA,false,true,false,true,U,false,true,MB>(p,A,B,C,D,stream,max_blocks); \
    return dispatch<BM,BN,RM,RN,BK,TMA,false,true,false,true,U,true,true,MB>(p,A,B,C,D,stream,max_blocks)
    OVERLAY(129,32,32,4,8,16,false,16,1);
    OVERLAY(130,32,32,4,8,16,false,16,2);
    OVERLAY(131,32,32,4,8,16,true,16,1);
    OVERLAY(132,32,32,4,8,16,true,16,2);
    OVERLAY(133,64,32,8,8,16,false,16,1);
    OVERLAY(134,64,32,8,8,16,false,16,2);
    OVERLAY(135,64,32,8,8,16,true,16,1);
    OVERLAY(136,64,32,8,8,16,true,16,2);
    OVERLAY(137,32,64,4,16,16,false,16,1);
    OVERLAY(138,32,64,4,16,16,false,16,2);
    OVERLAY(139,32,64,4,16,16,true,16,1);
    OVERLAY(140,32,64,4,16,16,true,16,2);
    OVERLAY(141,32,16,4,4,16,false,16,1);
    OVERLAY(142,32,16,4,4,16,false,16,2);
    OVERLAY(143,32,16,4,4,16,true,16,1);
    OVERLAY(144,32,16,4,4,16,true,16,2);
    OVERLAY(145,16,64,4,8,16,false,16,2);
    OVERLAY(146,16,64,4,8,16,true,16,2);
    OVERLAY(147,64,16,8,4,16,false,16,2);
    OVERLAY(148,64,16,8,4,16,true,16,2);
    OVERLAY(149,32,16,4,4,32,true,2,1);
    OVERLAY(150,32,16,4,4,32,true,2,2);
#undef OVERLAY
#define OPERANDS(ID,BM,BN,RM,RN,TMA,SHFL,PAD) case ID: \
    if(TMA && !p)dispatch<BM,BN,RM,RN,16,TMA,false,true,false,true,16,false,true,2,SHFL,PAD>(p,A,B,C,D,stream,max_blocks); \
    if(TMA && p && (p->m%(2*BM) || p->n%(2*BN) || p->k%32)) \
      return dispatch<BM,BN,RM,RN,16,TMA,false,true,false,true,16,false,true,2,SHFL,PAD>(p,A,B,C,D,stream,max_blocks); \
    return dispatch<BM,BN,RM,RN,16,TMA,false,true,false,true,16,true,true,2,SHFL,PAD>(p,A,B,C,D,stream,max_blocks)
    OPERANDS(151,32,16,4,4,false,false,4);
    OPERANDS(152,32,16,4,4,true,false,4);
    OPERANDS(153,32,32,4,8,false,false,4);
    OPERANDS(154,32,32,4,8,true,false,4);
    OPERANDS(155,64,32,8,8,false,false,4);
    OPERANDS(156,64,32,8,8,true,false,4);
    OPERANDS(157,32,64,4,16,false,false,4);
    OPERANDS(158,32,64,4,16,true,false,4);
    OPERANDS(159,16,32,2,8,false,false,2);
    OPERANDS(160,16,32,2,8,true,false,2);
    OPERANDS(161,16,16,2,4,false,false,2);
    OPERANDS(162,16,16,2,4,true,false,2);
    OPERANDS(163,32,16,4,4,false,true,0);
    OPERANDS(164,32,16,4,4,true,true,0);
    OPERANDS(165,32,32,4,8,false,true,0);
    OPERANDS(166,32,32,4,8,true,true,0);
    OPERANDS(167,64,32,8,8,false,true,0);
    OPERANDS(168,64,32,8,8,true,true,0);
    OPERANDS(169,32,64,4,16,false,true,0);
    OPERANDS(170,32,64,4,16,true,true,0);
    OPERANDS(171,16,32,2,8,false,true,0);
    OPERANDS(172,16,32,2,8,true,true,0);
    OPERANDS(173,16,16,2,4,false,true,0);
    OPERANDS(174,16,16,2,4,true,true,0);
    OPERANDS(175,8,32,2,4,false,false,2);
    OPERANDS(176,8,32,2,4,true,false,2);
#undef OPERANDS
    default:throw std::invalid_argument("invalid FP32 tiled configuration");
  }
}
}
namespace {
template<bool Grouped> __global__ void small_kernel(Problem p,const float* A,const float* B,
                                                   const float* C,float* D) {
  constexpr int Width=Grouped?8:1;
  const int product=threadIdx.x%Width;
  const int64_t mh=(p.m+1)/2,nh=(p.n+1)/2,kh=(p.k+1)/2;
  const int64_t pos=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/Width;
  const int64_t r=pos/nh,c=pos%nh;
  if(r>=mh)return; // Entire eight-lane subgroup takes the same branch.
  float v[Grouped?1:7]={};
  #pragma unroll 1
  for(int64_t q=0;q<kh;++q) {
    float a[4],b[4];
    #pragma unroll
    for(int j=0;j<4;++j) {
      a[j]=load(A,r+(j/2)*mh,q+(j%2)*kh,p.m,p.k,p.lda,p.trans_a);
      b[j]=load(B,q+(j/2)*kh,c+(j%2)*nh,p.k,p.n,p.ldb,p.trans_b);
    }
    const float s1=__fadd_rn(a[2],a[3]),s2=__fsub_rn(s1,a[0]);
    const float t1=__fsub_rn(b[1],b[0]),t2=__fsub_rn(b[3],t1);
    if constexpr(Grouped) {
      const float aa=product==0?a[0]:product==1?a[1]:product==2?__fsub_rn(a[1],s2):
        product==3?a[3]:product==4?s1:product==5?s2:__fsub_rn(a[0],a[2]);
      const float bb=product==0?b[0]:product==1?b[2]:product==2?b[3]:
        product==3?__fsub_rn(t2,b[2]):product==4?t1:product==5?t2:__fsub_rn(b[3],b[1]);
      if(product<7)v[0]=__fmaf_rn(aa,bb,v[0]);
    } else {
      const float av[7]={a[0],a[1],__fsub_rn(a[1],s2),a[3],s1,s2,__fsub_rn(a[0],a[2])};
      const float bv[7]={b[0],b[2],b[3],__fsub_rn(t2,b[2]),t1,t2,__fsub_rn(b[3],b[1])};
      #pragma unroll
      for(int j=0;j<7;++j)v[j]=__fmaf_rn(av[j],bv[j],v[j]);
    }
  }
  float values[7];
  if constexpr(Grouped) {
    const unsigned mask=__activemask();
    #pragma unroll
    for(int j=0;j<7;++j)values[j]=__shfl_sync(mask,v[0],j,8);
  } else {
    #pragma unroll
    for(int j=0;j<7;++j)values[j]=v[j];
  }
  if(product==0) {
    const float u2=__fadd_rn(values[0],values[5]);
    const float u3=__fadd_rn(u2,values[6]),u4=__fadd_rn(u2,values[4]);
    const float out[4]={__fadd_rn(values[0],values[1]),__fadd_rn(u4,values[2]),
                        __fsub_rn(u3,values[3]),__fadd_rn(u3,values[4])};
    #pragma unroll
    for(int j=0;j<4;++j) {
      const int64_t rr=r+(j/2)*mh,cc=c+(j%2)*nh;
      if(rr<p.m && cc<p.n) {
        float value=__fmul_rn(p.alpha,out[j]);
        if(p.beta!=0)value=__fadd_rn(value,__fmul_rn(p.beta,C[rr*p.ldc+cc]));
        D[rr*p.ldd+cc]=value;
      }
    }
  }
}
template<bool Grouped> SWResources small_dispatch(const Problem* p,const float* A,const float* B,
                                                 const float* C,float* D,cudaStream_t stream) {
  if(!p) {
    cudaFuncAttributes attr{};int resident=0;
    check(cudaFuncGetAttributes(&attr,small_kernel<Grouped>));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,small_kernel<Grouped>,128,0));
    return {attr.sharedSizeBytes,attr.numRegs,resident};
  }
  const int64_t count=((p->m+1)/2)*((p->n+1)/2)*(Grouped?8:1);
  small_kernel<Grouped><<<int((count+127)/128),128,0,stream>>>(*p,A,B,C,D);
  check(cudaPeekAtLastError());return {};
}
}
bool small_problem(const Problem& p) {return p.dtype==DType::FP32 && p.m>0 && p.n>0 && p.k>0 && p.m<=256 && p.n<=256 && p.k<=1024;}
SWResources small_setup(int id) {
  if(id==0)return small_dispatch<false>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr);
  if(id==1)return small_dispatch<true>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr);
  throw std::invalid_argument("invalid FP32 small configuration");
}
std::string small_name(int id) {
  return "sw_fp32_small;leaf_terms=7;transforms=fp32;leaf_math=fp32_simt;accum=fp32;workspace=0;load=synchronous;subgroup="+
    std::to_string(id?8:1)+";residual_correction=off";
}
void small_run(const Problem& p,int id,const float* A,const float* B,const float* C,float* D,cudaStream_t stream) {
  if(!small_problem(p))throw std::invalid_argument("FP32 small dimensions unsupported");
  if(id==0)small_dispatch<false>(&p,A,B,C,D,stream);
  else if(id==1)small_dispatch<true>(&p,A,B,C,D,stream);
  else throw std::invalid_argument("invalid FP32 small configuration");
}
bool eligible(const Problem& p,const Config& c) {
  return (!c.tma || sw_fp32_tma_problem(p)) &&
    (!c.fast || !c.tma || (p.m%(2*c.bm)==0 && p.n%(2*c.bn)==0 && p.k%(2*c.bk)==0)) && (!((c.id>=44 && c.id<=55) || (c.id>=75 && c.id<=80)) || small_problem(p));
}
SWResources setup(int id) {return select(id,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,65535);}
std::string name(int id) {
  const auto& c=configurations.at(id);
  return "sw_fp32_tiled;tile="+std::to_string(c.bm)+"x"+std::to_string(c.bn)+
    ";thread_tile="+std::to_string(c.rm)+"x"+std::to_string(c.rn)+";bk="+std::to_string(c.bk)+";threads="+std::to_string(c.threads)+
    ";product_major="+std::to_string(c.product_major)+";vectorized_shared="+std::to_string(c.vectorized)+
    ";operand_shuffle="+std::to_string(c.shuffle)+";a_pad="+std::to_string(c.a_pad)+";shared_overlay="+std::to_string(c.overlay)+";min_blocks="+std::to_string(c.min_blocks)+";fast_interior="+std::to_string(c.fast)+";k_unroll="+std::to_string(c.unroll)+";product_warps="+std::to_string(c.warp_product?7:0)+";leaf_terms=7;transforms=fp32;leaf_math=fp32_simt;accum=fp32;workspace=0;load="+
    (c.tma?(c.warp_product?"tma;raw_stages=2;issuer=thread224":"tma;raw_stages=2;issuer=thread0"):c.async?"cp_async;raw_stages=2;issuer=cta_threads":"synchronous")+";residual_correction=off";
}
void run(const Problem& p,int id,const float* A,const float* B,const float* C,
         float* D,cudaStream_t stream,int max_blocks) {
  const auto& c=configurations.at(id);
  if(max_blocks<=0)throw std::invalid_argument("FP32 tiled grid cap must be positive");
  if(c.tma && !sw_fp32_tma_binding(p,A,B))
    throw std::invalid_argument("FP32 tiled TMA binding is incompatible");
  select(id,&p,A,B,C,D,stream,max_blocks);
}
}

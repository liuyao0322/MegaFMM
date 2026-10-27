#include "internal.h"
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace megafmm::detail {
namespace {
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
template<class T,int N,int K,int S> struct alignas(128) Shared {
  T operand[S][(64+N)*K];
  uint64_t ready[S],free[S];
};
#include "sw_padded_64.cuh"
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
template<class T,int N,int K,int Stages,bool Pad=false> __global__ __launch_bounds__(256,1)
void compute(const __grid_constant__ CUtensorMap amap,const __grid_constant__ CUtensorMap bmap,
             const __grid_constant__ CUtensorMap smap,const __grid_constant__ CUtensorMap tmap,
             int mh,int nh,int kh,float* D,int M,int NN,int packed,int raster=-1) {
  using namespace cute;
  extern __shared__ __align__(128) unsigned char bytes[];
  auto& s=*reinterpret_cast<Shared<T,N,K,Stages>*>(bytes);
  int tid=threadIdx.x,m0=blockIdx.y*64,n0=blockIdx.x*N,kt=kh/K;
  {
    if(raster>0 || (!Pad && raster<0 && kh>=1024)) {
      int group_size=raster>0?raster:(kh>=8192?2:4),gx=gridDim.x,gy=gridDim.y;
      int pid=blockIdx.x+blockIdx.y*gx,group=pid/(group_size*gy),first=group*group_size;
      int count=min(group_size,gx-first),within=pid%(group_size*gy);
      m0=(within/count)*64;n0=(first+within%count)*N;
    }
  }
  if(tid==32) asm volatile("prefetch.tensormap [%0]; prefetch.tensormap [%1]; prefetch.tensormap [%2]; prefetch.tensormap [%3];"::"l"(&amap),"l"(&bmap),"l"(&smap),"l"(&tmap):"memory");
  if(tid==0) {
    for(int i=0;i<Stages;++i){Barrier::init(&s.ready[i],1);Barrier::init(&s.free[i],4);}
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  if(tid==0) {
    // Long-K row groups reuse A planes while B streams through L2.
    const uint64_t a_hint=kh>=8192?uint64_t(TMA::CacheHintSm90::EVICT_LAST):0;
    const uint64_t b_hint=kh>=8192?uint64_t(TMA::CacheHintSm90::EVICT_FIRST):0;
    for(int step=0;step<7;++step)for(int q=0;q<kt;++q) {
      int p=product_for_step(step),seq=step*kt+q,slot=seq%Stages,k0=q*K;
      // All transformed global operands are complete before their first TMA.
      if(seq==(Pad?0:2*kt))cudaGridDependencySynchronize();
      if(seq>=Stages)wait_free(&s.free[slot],((seq-Stages)/Stages)%2);
      TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*(64+N)*K);
      T* dst=s.operand[slot];
      if(p==3 || p>=4) {
        SM90_TMA_LOAD_3D::copy(&tmap,&s.ready[slot],b_hint,dst,m0,k0,p==3?3:p-4);
      } else {
        int col=p==2?nh:0,row=p==1 || p==2?kh:0;
        SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],b_hint,dst,m0+col,k0+row);
      }
      if(p==2 || p>=4) {
        SM90_TMA_LOAD_4D::copy(&smap,&s.ready[slot],a_hint,dst+64*K,0,n0,k0/64,p==2?3:p-4);
      } else {
        int col=p==1 || p==3?kh:0,row=p==3?mh:0;
        SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],a_hint,dst+64*K,0,n0+row,(k0+col)/64);
      }
    }
  } else if(tid>=128) {
    consume_pad_64(s,tid,m0,n0,(M+1)/2,(NN+1)/2,kt,D,M,NN);
  }
}
#include "sw_padded.cuh"
void check(cudaError_t e,const char* where) {
  if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
void encode(CUtensorMap& map,const void* ptr,DType type,bool is_a,bool transformed,int mh,int nh,int kh,int N,int K) {
  auto dtype=type==DType::BF16?CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  uint64_t dims[4],strides[3];uint32_t box[4],es[4]={1,1,1,1};
  int rank;
  if(is_a) {
    dims[0]=64;dims[1]=transformed?mh:2*mh;dims[2]=(transformed?kh:2*kh)/64;dims[3]=4;
    strides[0]=uint64_t(transformed?kh:2*kh)*2;strides[1]=128;strides[2]=uint64_t(mh)*kh*2;
    box[0]=64;box[1]=N;box[2]=K/64;box[3]=1;rank=transformed?4:3;
  } else {
    dims[0]=transformed?nh:2*nh;dims[1]=transformed?kh:2*kh;dims[2]=4;
    strides[0]=dims[0]*2;strides[1]=dims[0]*dims[1]*2;
    box[0]=64;box[1]=K;box[2]=1;rank=transformed?3:2;
  }
  auto r=cuTensorMapEncodeTiled(&map,dtype,rank,const_cast<void*>(ptr),dims,strides,box,es,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(r!=CUDA_SUCCESS)throw std::runtime_error("multisize SW tensor map: "+std::to_string(int(r)));
}
template<class T,int N,int K,int Stages,bool Pad=false> void setup() {
  check(cudaFuncSetAttribute(compute<T,N,K,Stages,Pad>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"SW carveout");
  check(cudaFuncSetAttribute(compute<T,N,K,Stages,Pad>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<T,N,K,Stages>)),"SW shared opt-in");
}
template<class T,int N,int K,int Stages,bool Pad=false> void launch(const Problem& p,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream,int pack_cap=0,int raster=-1) {
  constexpr bool padded=true;auto shape=pad_shape(p);
  int mh=padded?shape.mh:int(p.m/2),nh=padded?shape.nh:int(p.n/2),kh=padded?shape.kh:int(p.k/2);
  const void* rawA=padded?static_cast<char*>(workspace)+8*(shape.sa+shape.sb):A;
  const void* rawB=padded?static_cast<const char*>(rawA)+8*shape.sa:B;
  CUtensorMap amap{},bmap{},smap{},tmap{};
  void* tbase=static_cast<char*>(workspace)+size_t(4)*mh*kh*2;
  encode(amap,rawA,p.dtype,true,false,mh,nh,kh,N,K);encode(bmap,rawB,p.dtype,false,false,mh,nh,kh,N,K);
  encode(smap,workspace,p.dtype,true,true,mh,nh,kh,N,K);encode(tmap,tbase,p.dtype,false,true,mh,nh,kh,N,K);
  const int blocks=int(std::min<int64_t>(128,(std::max(int64_t(mh)*kh,int64_t(kh)*nh)+511)/512));
  pack_padded<T><<<blocks,256,0,stream>>>(static_cast<const T*>(A),static_cast<const T*>(B),static_cast<T*>(workspace),int(p.m),int(p.n),int(p.k),mh,nh,kh);

  check(cudaPeekAtLastError(),"SW transform launch");
  cudaLaunchAttribute attribute{};attribute.id=cudaLaunchAttributeProgrammaticStreamSerialization;attribute.val.programmaticStreamSerializationAllowed=1;
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(mh/N,nh/64);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=sizeof(Shared<T,N,K,Stages>);cfg.stream=stream;cfg.attrs=&attribute;cfg.numAttrs=1;
  check(cudaLaunchKernelEx(&cfg,compute<T,N,K,Stages,Pad>,amap,bmap,smap,tmap,mh,nh,kh,D,int(p.m),int(p.n),int(padded),raster),"SW compute launch");
}
} // namespace
size_t sw_hopper_workspace(const Problem& p) {
  if(p.dtype==DType::FP32 || p.m<=0 || p.n<=0 || p.k<=0 || p.m>32768 || p.n>32768 || p.k>32768 ||
     p.trans_a || p.trans_b || p.lda!=p.k || p.ldb!=p.n || p.ldd!=p.n || p.alpha!=1 || p.beta!=0)return 0;
  return pad_shape(p).bytes;
}
void setup_sw_hopper(const Problem& p) {
  if(p.dtype==DType::BF16)setup<cute::bfloat16_t,64,128,4,true>();
  else setup<cute::half_t,64,128,4,true>();
}
void run_sw_hopper(const Problem& p,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream) {
  if(p.dtype==DType::BF16)launch<cute::bfloat16_t,64,128,4,true>(p,A,B,D,workspace,stream);
  else launch<cute::half_t,64,128,4,true>(p,A,B,D,workspace,stream);
}
const char* sw_hopper_config(const Problem&) {
  return "sw_hopper;tile=64x64;bk=128;stages=4;leaf_terms=7;padding=logical_quadrants;pack=paired;pack_cta_cap=128;residual_correction=off";
}
std::vector<SWConfig> sw_configs(const Problem& p) {
  if(!sw_hopper_workspace(p))return {};
  return {{0,64,128,4,128,0,true}};
}
size_t sw_config_workspace(const Problem& p,const SWConfig&) {return sw_hopper_workspace(p);}
SWResources setup_sw_config(const Problem& p,const SWConfig&) {
  setup_sw_hopper(p);
  cudaFuncAttributes attr{};int resident=0;
  if(p.dtype==DType::BF16) {
    check(cudaFuncGetAttributes(&attr,compute<cute::bfloat16_t,64,128,4,true>),"SW attributes");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,compute<cute::bfloat16_t,64,128,4,true>,256,sizeof(Shared<cute::bfloat16_t,64,128,4>)),"SW occupancy");
  } else {
    check(cudaFuncGetAttributes(&attr,compute<cute::half_t,64,128,4,true>),"SW attributes");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,compute<cute::half_t,64,128,4,true>,256,sizeof(Shared<cute::half_t,64,128,4>)),"SW occupancy");
  }
  return {sizeof(Shared<cute::half_t,64,128,4>)+attr.sharedSizeBytes,attr.numRegs,resident};
}
void run_sw_config(const Problem& p,const SWConfig&,const void* A,const void* B,float* D,void* ws,cudaStream_t stream) {
  run_sw_hopper(p,A,B,D,ws,stream);
}
std::string sw_config_name(const SWConfig&) {return sw_hopper_config({});}
} // namespace megafmm::detail

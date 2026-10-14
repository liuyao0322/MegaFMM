#include "internal.h"
#include <cuda.h>
#include <cooperative_groups.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <stdexcept>
#include <type_traits>

namespace megafmm::detail {
namespace {
constexpr int BM=64,BN=32,BK=128,Stages=2,ASize=BM*BK,BSize=BK*BN,Elements=BM*BN;
constexpr int Plane=512*512,Threads=320;
constexpr size_t WorkspaceBytes=size_t(16)*Plane*2;
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
template<class T> struct alignas(128) Shared {
  T raw[Stages][4*(ASize+BSize)];
  T extra_a[ASize],extra_b[BSize];
  uint64_t ready[Stages],free[Stages],transform_request,transform_ready;
};
static_assert(sizeof(Shared<cute::half_t>)==221312,"shared budget");
__device__ void wait(const uint64_t* b,int phase){Barrier::wait(b,phase);asm volatile("" ::: "memory");}
__device__ void arrive(const uint64_t* b){asm volatile("" ::: "memory");Barrier::arrive(b);}

template<class T,class TA,class TB>
__device__ void mma_term(TA a,TB b,float (&d)[16]) {
  using namespace cute;
  using Op=std::conditional_t<std::is_same_v<T,bfloat16_t>,
      SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>,
      SM90_64x32x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::MN>>;
  // Build the canonical descriptor once; CuTe preserves the swizzled K offsets
  // in descriptor units instead of rebuilding address bitfields for each MMA.
  auto da_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(
      tiled_divide(a,Shape<_64,_16>{}));
  auto db_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(
      tiled_divide(b,Shape<_32,_16>{}));
  auto da_base=da_tiles(0,0,0);
  auto db_base=db_tiles(0,0,0);
  #pragma unroll
  for(int kk=0;kk<BK/16;++kk){
    // All operands fit below 256 KiB of shared memory. K offsets cannot carry
    // out of the 14-bit start-address field (16-byte units), so keep the high
    // descriptor word invariant rather than generating a 64-bit add/carry.
    auto da=da_base,db=db_base;
    da.reg32_[0]+=uint32_t(da_tiles.layout()(0,0,kk));
    db.reg32_[0]+=uint32_t(db_tiles.layout()(0,0,kk));
    Op::fma(da,db,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15]);
  }
}

template<class T,bool IsA,class Layout>
__device__ __forceinline__ void transform_shared(T* raw,T* extra,Layout layout,int worker){
  constexpr int Size=IsA ? ASize:BSize;
  for(int w=worker;w<Size/2;w+=160){
    int row=IsA ? (2*w)/BK:(2*w)%BN;
    int col=IsA ? (2*w)%BK:(2*w)/BN;
    int idx=cute::as_position_independent_swizzle_layout(layout)(row,col);
    uint32_t bits[4];
    #pragma unroll
    for(int j=0;j<4;++j)bits[j]=*reinterpret_cast<uint32_t*>(raw+j*Size+idx);
    *reinterpret_cast<uint32_t*>(extra+idx)=bits[3];
    uint32_t words[4]={};
    #pragma unroll
    for(int lane=0;lane<2;++lane){
      float v0=float(T::bitcast(uint16_t(bits[0]>>(16*lane))));
      float v1=float(T::bitcast(uint16_t(bits[1]>>(16*lane))));
      float v2=float(T::bitcast(uint16_t(bits[2]>>(16*lane))));
      float v3=float(T::bitcast(uint16_t(bits[3]>>(16*lane))));
      float values[4];
      if constexpr(IsA){
        float s1=__fadd_rn(v2,v3),s2=__fsub_rn(s1,v0);
        values[0]=s1;values[1]=s2;values[2]=__fsub_rn(v0,v2);values[3]=__fsub_rn(v1,s2);
      }else{
        float t1=__fsub_rn(v1,v0),t2=__fsub_rn(v3,t1);
        values[0]=t1;values[1]=t2;values[2]=__fsub_rn(v3,v1);values[3]=__fsub_rn(t2,v2);
      }
      #pragma unroll
      for(int j=0;j<4;++j)words[j]|=uint32_t(T(values[j]).raw())<<(16*lane);
    }
    #pragma unroll
    for(int j=0;j<4;++j)*reinterpret_cast<uint32_t*>(raw+j*Size+idx)=words[j];
  }
}
template<class T,class LA,class LB>
__device__ __forceinline__ void issue_leaf(T* a,T* b,LA la,LB lb,float (&acc)[16]){
  using namespace cute;
  #pragma unroll
  for(int j=0;j<16;++j)warpgroup_fence_operand(acc[j]);
  warpgroup_arrive();
  mma_term<T>(make_tensor(make_smem_ptr(a),la),make_tensor(make_smem_ptr(b),lb),acc);
  warpgroup_commit_batch();
}
template<class T> __global__ __launch_bounds__(Threads,1)
void sw_cooperative_hilo(const __grid_constant__ CUtensorMap amap,
                         const __grid_constant__ CUtensorMap bmap,float* D){
  using namespace cute;
  extern __shared__ __align__(128) unsigned char buffer[];
  auto& s=*reinterpret_cast<Shared<T>*>(buffer);
  int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;
  if(tid==0){
    for(int j=0;j<Stages;++j){Barrier::init(&s.ready[j],1);Barrier::init(&s.free[j],4);}
    Barrier::init(&s.transform_request,4);Barrier::init(&s.transform_ready,160);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
  constexpr auto lb=tile_to_shape(GMMA::Layout_MN_SW64_Atom<T>{},Shape<_32,Int<BK>>{});
  if(tid==0){
    for(int q=0;q<512/BK;++q){
      int slot=q%Stages,k0=q*BK;
      if(q>=Stages)wait(&s.free[slot],((q-Stages)/Stages)%2);
      TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*4*(ASize+BSize));
      #pragma unroll
      for(int j=0;j<4;++j){
        SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],0,s.raw[slot]+j*ASize,0,m0+(j/2)*512,(k0+(j%2)*512)/64);
        SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],0,s.raw[slot]+4*ASize+j*BSize,n0+(j%2)*512,k0+(j/2)*512);
      }
    }
  }else if((tid>=32&&tid<128)||tid>=256){
    int worker=tid>=256 ? tid-160:tid-32;
    for(int q=0;q<512/BK;++q){
      wait(&s.transform_request,q%2);
      transform_shared<T,true>(s.raw[q%Stages],s.extra_a,la,worker);
      transform_shared<T,false>(s.raw[q%Stages]+4*ASize,s.extra_b,lb,worker);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      arrive(&s.transform_ready);
    }
  }else if(tid>=128&&tid<256){
    float acc[7][16]={};
    #pragma unroll 1
    for(int q=0;q<512/BK;++q){
      int slot=q%Stages;
      wait(&s.ready[slot],(q/Stages)%2);
      T* a=s.raw[slot];T* b=a+4*ASize;
      issue_leaf<T>(a,b,la,lb,acc[0]);
      issue_leaf<T>(a+ASize,b+2*BSize,la,lb,acc[1]);
      warpgroup_wait<0>();
      #pragma unroll
      for(int p=0;p<7;++p)for(int j=0;j<16;++j)warpgroup_fence_operand(acc[p][j]);
      __syncwarp();if((tid&31)==0)arrive(&s.transform_request);
      wait(&s.transform_ready,q%2);
      issue_leaf<T>(s.extra_a,b+3*BSize,la,lb,acc[3]);
      issue_leaf<T>(a+3*ASize,s.extra_b,la,lb,acc[2]);
      issue_leaf<T>(a,b,la,lb,acc[4]);
      issue_leaf<T>(a+ASize,b+BSize,la,lb,acc[5]);
      issue_leaf<T>(a+2*ASize,b+2*BSize,la,lb,acc[6]);
      warpgroup_wait<0>();
      #pragma unroll
      for(int p=0;p<7;++p)for(int j=0;j<16;++j)warpgroup_fence_operand(acc[p][j]);
      __syncwarp();if((tid&31)==0)arrive(&s.free[slot]);
    }
    #pragma unroll
    for(int j=0;j<8;++j){
      int mn=GMMA::CLayout_64x32{}(tid-128,2*j);
      int at=(m0+mn%64)*1024+n0+mn/64;
      float2 out[4];
      #pragma unroll
      for(int lane=0;lane<2;++lane){
        int i=2*j+lane;
        float u1=__fadd_rn(acc[0][i],acc[1][i]);
        float u2=__fadd_rn(acc[0][i],acc[5][i]);
        float u3=__fadd_rn(u2,acc[6][i]);
        float u4=__fadd_rn(u2,acc[4][i]);
        float v[4]={u1,__fadd_rn(u4,acc[2][i]),__fsub_rn(u3,acc[3][i]),__fadd_rn(u3,acc[4][i])};
        #pragma unroll
        for(int p=0;p<4;++p){if(lane==0)out[p].x=v[p];else out[p].y=v[p];}
      }
      *reinterpret_cast<float2*>(D+at)=out[0];
      *reinterpret_cast<float2*>(D+at+512)=out[1];
      *reinterpret_cast<float2*>(D+at+512*1024)=out[2];
      *reinterpret_cast<float2*>(D+at+512*1024+512)=out[3];
    }
  }
  __syncthreads();
}
void check(cudaError_t e,const char* where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
void encode(CUtensorMap& map,const void* ptr,DType dtype,bool is_a,bool transformed){
  if(reinterpret_cast<uintptr_t>(ptr)%16)throw std::invalid_argument("SW TMA input must be16-byte aligned");
  if(is_a){
    // Expose K/64 as an outer tensor dimension. The shared box still packs
    // (64 K elements, BM rows, BK/64 chunks), matching the existing GMMA layout.
    uint64_t dims[4]={64,transformed ? 512ULL:1024ULL,transformed ? 8ULL:16ULL,8};
    uint64_t strides[3]={transformed ? 1024ULL:2048ULL,128,512ULL*512*2};
    uint32_t box[4]={64,BM,BK/64,1},element_strides[4]={1,1,1,1};
    auto result=cuTensorMapEncodeTiled(&map,dtype==DType::BF16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        transformed ? 4:3,const_cast<void*>(ptr),dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(result!=CUDA_SUCCESS)throw std::runtime_error("SW A tensor map status "+std::to_string(int(result)));
    return;
  }
  uint64_t dims[3]={transformed ? 512ULL:1024ULL,transformed ? 512ULL:1024ULL,8};
  uint64_t strides[2]={dims[0]*2,dims[0]*dims[1]*2};
  uint32_t box[3]={uint32_t(is_a ? 64:BN),uint32_t(is_a ? BM:BK),1},element_strides[3]={1,1,1};
  auto result=cuTensorMapEncodeTiled(&map,dtype==DType::BF16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
      transformed ? 3:2,const_cast<void*>(ptr),dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,
      is_a ? CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_64B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(result!=CUDA_SUCCESS)throw std::runtime_error("SW tensor map status "+std::to_string(int(result)));
}
} // namespace
struct SWState{DType dtype;};
std::shared_ptr<SWState> make_sw_pipeline(const Problem& p){
  if(p.m!=1024 || p.n!=1024 || p.k!=1024 || p.trans_a || p.trans_b || p.lda!=1024 || p.ldb!=1024 || p.ldd!=1024 || p.alpha!=1 || p.beta!=0)
    throw std::invalid_argument("SW requires contiguous non-transposed1024^3 alpha1 beta0");
  int device;cudaDeviceProp prop{};check(cudaGetDevice(&device),"SW device");check(cudaGetDeviceProperties(&prop,device),"SW properties");
  if(prop.major!=9 || prop.minor!=0)throw std::invalid_argument("SW needs cooperative Hopper launch");
  int blocks=0;
  if(p.dtype==DType::BF16){
    check(cudaFuncSetAttribute(sw_cooperative_hilo<cute::bfloat16_t>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<cute::bfloat16_t>)),"SW shared opt-in");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,sw_cooperative_hilo<cute::bfloat16_t>,Threads,sizeof(Shared<cute::bfloat16_t>)),"SW occupancy capacity");
  }else{
    check(cudaFuncSetAttribute(sw_cooperative_hilo<cute::half_t>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<cute::half_t>)),"SW shared opt-in");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,sw_cooperative_hilo<cute::half_t>,Threads,sizeof(Shared<cute::half_t>)),"SW occupancy capacity");
  }
  if(blocks*prop.multiProcessorCount<128)throw std::invalid_argument("SW cooperative grid requires128 resident CTA capacity");
  return std::make_shared<SWState>(SWState{p.dtype});
}
void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,void*,cudaStream_t stream){
  CUtensorMap a{},b{};encode(a,A,state.dtype,true,false);encode(b,B,state.dtype,false,false);
  dim3 grid(16,8);
  if(state.dtype==DType::BF16)sw_cooperative_hilo<cute::bfloat16_t><<<grid,Threads,sizeof(Shared<cute::bfloat16_t>),stream>>>(a,b,D);
  else sw_cooperative_hilo<cute::half_t><<<grid,Threads,sizeof(Shared<cute::half_t>),stream>>>(a,b,D);
  check(cudaGetLastError(),"SW local transform launch");
}
} // namespace megafmm::detail

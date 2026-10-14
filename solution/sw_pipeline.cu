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
constexpr int BM=64,BN=256,BK=64,Stages=3,ASize=BM*BK,BSize=BK*BN,Elements=BM*BN;
constexpr int Plane=512*512,Threads=256;
constexpr size_t WorkspaceBytes=size_t(16)*Plane*2;
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
template<class T> struct alignas(128) Shared {
  T operand[Stages][ASize+BSize];
  uint64_t ready[Stages],free[Stages];
};
static_assert(sizeof(Shared<cute::half_t>)==123008,"shared budget");
__device__ void wait(const uint64_t* b,int phase){Barrier::wait(b,phase);asm volatile("" ::: "memory");}
__device__ void arrive(const uint64_t* b){asm volatile("" ::: "memory");Barrier::arrive(b);}

__device__ __forceinline__ int product_index(int row,int col){
  // Fragment lanes vary column bits 1,2 and row bits 0,1,2.
  // XOR the latter into column bits 0,3,4, the three remaining bank bits.
  return row*BN+(col^((row&1)|((row&6)<<2)));
}

// Zero-based P IDs. Producer and consumers advance barriers by step, not P ID.
__device__ constexpr int product_for_step(int step){
  return step<2 ? step:step==2 ? 5:step==3 ? 6:step==4 ? 4:step==5 ? 2:3;
}

template<class T,class TA,class TB>
__device__ void mma_term(TA a,TB b,float (&d)[128]) {
  using namespace cute;
  using Op=std::conditional_t<std::is_same_v<T,bfloat16_t>,
      SM90_64x256x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>,
      SM90_64x256x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::MN>>;
  // Build the canonical descriptor once; CuTe preserves the swizzled K offsets
  // in descriptor units instead of rebuilding address bitfields for each MMA.
  auto da_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(
      tiled_divide(a,Shape<_64,_16>{}));
  auto db_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(
      tiled_divide(b,Shape<_256,_16>{}));
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
    Op::fma(da,db,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],d[16],d[17],d[18],d[19],d[20],d[21],d[22],d[23],d[24],d[25],d[26],d[27],d[28],d[29],d[30],d[31],d[32],d[33],d[34],d[35],d[36],d[37],d[38],d[39],d[40],d[41],d[42],d[43],d[44],d[45],d[46],d[47],d[48],d[49],d[50],d[51],d[52],d[53],d[54],d[55],d[56],d[57],d[58],d[59],d[60],d[61],d[62],d[63],d[64],d[65],d[66],d[67],d[68],d[69],d[70],d[71],d[72],d[73],d[74],d[75],d[76],d[77],d[78],d[79],d[80],d[81],d[82],d[83],d[84],d[85],d[86],d[87],d[88],d[89],d[90],d[91],d[92],d[93],d[94],d[95],d[96],d[97],d[98],d[99],d[100],d[101],d[102],d[103],d[104],d[105],d[106],d[107],d[108],d[109],d[110],d[111],d[112],d[113],d[114],d[115],d[116],d[117],d[118],d[119],d[120],d[121],d[122],d[123],d[124],d[125],d[126],d[127]);
  }
}

template<class T> __global__ __launch_bounds__(Threads,1)
void sw_cooperative_hilo(const __grid_constant__ CUtensorMap amap,
                         const __grid_constant__ CUtensorMap bmap,
                         const __grid_constant__ CUtensorMap smap,
                         const __grid_constant__ CUtensorMap tmap,
                         const T* A,const T* B,T* packed,float* D){
  using namespace cute;
  extern __shared__ __align__(128) unsigned char buffer[];
  auto& s=*reinterpret_cast<Shared<T>*>(buffer);
  int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;
  int cta=(blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x;
  int blocks=gridDim.x*gridDim.y*gridDim.z;
  int p=blockIdx.z;
  if(tid==0){
    for(int j=0;j<Stages;++j){Barrier::init(&s.ready[j],1);Barrier::init(&s.free[j],4);}
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  // All CTAs cooperate in one input transformation per invocation, never across calls.
  // Transform two adjacent elements per thread with aligned 32-bit IO. Experimental seven-product path:
  // retain FP32 transforms and high rounding, omit residual generation.
  for(int i=2*(cta*Threads+tid);i<Plane;i+=2*blocks*Threads){
    int row=i/512,col=i%512,at=row*1024+col;
    auto av=reinterpret_cast<const uint32_t*>(A);
    auto bv=reinterpret_cast<const uint32_t*>(B);
    uint32_t abits[4]={av[at/2],av[(at+512)/2],av[(at+512*1024)/2],av[(at+512*1024+512)/2]};
    uint32_t bbits[4]={bv[at/2],bv[(at+512)/2],bv[(at+512*1024)/2],bv[(at+512*1024+512)/2]};
    uint32_t words[16]={};
    #pragma unroll
    for(int lane=0;lane<2;++lane){
      float a11=float(T::bitcast(uint16_t(abits[0]>>(16*lane)))),a12=float(T::bitcast(uint16_t(abits[1]>>(16*lane))));
      float a21=float(T::bitcast(uint16_t(abits[2]>>(16*lane)))),a22=float(T::bitcast(uint16_t(abits[3]>>(16*lane))));
      float b11=float(T::bitcast(uint16_t(bbits[0]>>(16*lane)))),b12=float(T::bitcast(uint16_t(bbits[1]>>(16*lane))));
      float b21=float(T::bitcast(uint16_t(bbits[2]>>(16*lane)))),b22=float(T::bitcast(uint16_t(bbits[3]>>(16*lane))));
      float s1=__fadd_rn(a21,a22),s2=__fsub_rn(s1,a11);
      float t1=__fsub_rn(b12,b11),t2=__fsub_rn(b22,t1);
      float sv[4]={s1,s2,__fsub_rn(a11,a21),__fsub_rn(a12,s2)};
      float tv[4]={t1,t2,__fsub_rn(b22,b12),__fsub_rn(t2,b21)};
      #pragma unroll
      for(int j=0;j<4;++j){
        T sh=T(sv[j]),th=T(tv[j]);
        words[2*j]|=uint32_t(sh.raw())<<(16*lane);
        words[8+2*j]|=uint32_t(th.raw())<<(16*lane);
      }
    }
    #pragma unroll
    for(int j=0;j<8;++j)reinterpret_cast<uint32_t*>(packed)[(2*j*Plane+i)/2]=words[2*j];
  }
  // Generic global stores become visible to TMA's async proxy before publication.
  asm volatile("fence.proxy.async.global;" ::: "memory");
  cooperative_groups::this_grid().sync();
  constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
  constexpr auto lb=tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_256,Int<BK>>{},Step<_2,_1>{});
  if(tid==0){
    for(int q=0;q<512/BK;++q){
      int sequence=q,slot=sequence%Stages,k0=q*BK;
      if(sequence>=Stages)wait(&s.free[slot],((sequence-Stages)/Stages)%2);
      bool alo=p==2 || p>=4,blo=p==3 || p>=4;
      // Seven-product experiment: only publish high operands; no residual loads.
      TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*(ASize+BSize));
      T* op=s.operand[slot];
      if(alo){
        int plane=p==2 ? 6:2*(p-4);
        SM90_TMA_LOAD_4D::copy(&smap,&s.ready[slot],0,op,0,m0,k0/64,plane);
      }else{
        int col=(p==1 || p==3) ? 512:0,row=p==3 ? 512:0;
        SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],0,op,0,m0+row,(k0+col)/64);
      }
      if(blo){
        int plane=p==3 ? 6:2*(p-4);
        SM90_TMA_LOAD_4D::copy(&tmap,&s.ready[slot],0,op+ASize,0,k0,n0/64,plane);
      }else{
        int col=p==2 ? 512:0,row=p==1 || p==2 ? 512:0;
        SM90_TMA_LOAD_3D::copy(&bmap,&s.ready[slot],0,op+ASize,0,k0+row,(n0+col)/64);
      }

    }
  }else if(tid>=128 && tid<256){
    float acc[128]={};
    #pragma unroll 1
    for(int q=0;q<512/BK;++q){
      int slot=q%Stages;
      wait(&s.ready[slot],(q/Stages)%2);
      T* op=s.operand[slot];
      auto ah=make_tensor(make_smem_ptr(op),la);
      auto bh=make_tensor(make_smem_ptr(op+ASize),lb);
      #pragma unroll
      for(int j=0;j<128;++j)warpgroup_fence_operand(acc[j]);
      warpgroup_arrive();mma_term<T>(ah,bh,acc);warpgroup_commit_batch();warpgroup_wait<0>();
      #pragma unroll
      for(int j=0;j<128;++j)warpgroup_fence_operand(acc[j]);
      __syncwarp();if((tid&31)==0)arrive(&s.free[slot]);
    }
    float* products=reinterpret_cast<float*>(packed+16*Plane);
    #pragma unroll
    for(int j=0;j<64;++j){
      int mn=GMMA::CLayout_64x256{}(tid-128,2*j);
      int at=p*Plane+(m0+mn%64)*512+n0+mn/64;
      *reinterpret_cast<float2*>(products+at)=make_float2(acc[2*j],acc[2*j+1]);
    }
  }
  cooperative_groups::this_grid().sync();
  const float* products=reinterpret_cast<const float*>(packed+16*Plane);
  for(int i=2*(cta*Threads+tid);i<Plane;i+=2*blocks*Threads){
    float2 values[7];
    #pragma unroll
    for(int j=0;j<7;++j)values[j]=*reinterpret_cast<const float2*>(products+j*Plane+i);
    float2 out[4];
    #pragma unroll
    for(int lane=0;lane<2;++lane){
      float v[7];
      #pragma unroll
      for(int j=0;j<7;++j)v[j]=lane==0 ? values[j].x:values[j].y;
      float u1=__fadd_rn(v[0],v[1]);
      float u2=__fadd_rn(v[0],v[5]);
      float u3=__fadd_rn(u2,v[6]);
      float u4=__fadd_rn(u2,v[4]);
      float c[4]={u1,__fadd_rn(u4,v[2]),__fsub_rn(u3,v[3]),__fadd_rn(u3,v[4])};
      #pragma unroll
      for(int j=0;j<4;++j){if(lane==0)out[j].x=c[j];else out[j].y=c[j];}
    }
    int at=(i/512)*1024+i%512;
    *reinterpret_cast<float2*>(D+at)=out[0];
    *reinterpret_cast<float2*>(D+at+512)=out[1];
    *reinterpret_cast<float2*>(D+at+512*1024)=out[2];
    *reinterpret_cast<float2*>(D+at+512*1024+512)=out[3];
  }
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
  uint64_t dims[4]={64,transformed ? 512ULL:1024ULL,transformed ? 8ULL:16ULL,8};
  uint64_t strides[3]={transformed ? 1024ULL:2048ULL,128,512ULL*512*2};
  uint32_t box[4]={64,BK,BN/64,1},element_strides[4]={1,1,1,1};
  auto result=cuTensorMapEncodeTiled(&map,dtype==DType::BF16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
      transformed ? 4:3,const_cast<void*>(ptr),dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(result!=CUDA_SUCCESS)throw std::runtime_error("SW B tensor map status "+std::to_string(int(result)));
}
} // namespace
struct SWState{DType dtype;};
std::shared_ptr<SWState> make_sw_pipeline(const Problem& p){
  if(p.m!=1024 || p.n!=1024 || p.k!=1024 || p.trans_a || p.trans_b || p.lda!=1024 || p.ldb!=1024 || p.ldd!=1024 || p.alpha!=1 || p.beta!=0)
    throw std::invalid_argument("SW requires contiguous non-transposed1024^3 alpha1 beta0");
  int device;cudaDeviceProp prop{};check(cudaGetDevice(&device),"SW device");check(cudaGetDeviceProperties(&prop,device),"SW properties");
  if(prop.major!=9 || prop.minor!=0 || !prop.cooperativeLaunch)throw std::invalid_argument("SW needs cooperative Hopper launch");
  int blocks=0;
  if(p.dtype==DType::BF16){
    check(cudaFuncSetAttribute(sw_cooperative_hilo<cute::bfloat16_t>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<cute::bfloat16_t>)),"SW shared opt-in");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,sw_cooperative_hilo<cute::bfloat16_t>,Threads,sizeof(Shared<cute::bfloat16_t>)),"SW occupancy capacity");
  }else{
    check(cudaFuncSetAttribute(sw_cooperative_hilo<cute::half_t>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<cute::half_t>)),"SW shared opt-in");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,sw_cooperative_hilo<cute::half_t>,Threads,sizeof(Shared<cute::half_t>)),"SW occupancy capacity");
  }
  if(blocks*prop.multiProcessorCount<112)throw std::invalid_argument("SW cooperative grid requires112 resident CTA capacity");
  return std::make_shared<SWState>(SWState{p.dtype});
}
void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream){
  CUtensorMap a{},b{},s{},t{};encode(a,A,state.dtype,true,false);encode(b,B,state.dtype,false,false);
  encode(s,workspace,state.dtype,true,true);encode(t,static_cast<char*>(workspace)+WorkspaceBytes/2,state.dtype,false,true);
  void* args[]={&a,&b,&s,&t,&A,&B,&workspace,&D};dim3 grid(2,8,7);
  if(state.dtype==DType::BF16)
    check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(sw_cooperative_hilo<cute::bfloat16_t>),grid,Threads,args,sizeof(Shared<cute::bfloat16_t>),stream),"SW cooperative launch");
  else
    check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(sw_cooperative_hilo<cute::half_t>),grid,Threads,args,sizeof(Shared<cute::half_t>),stream),"SW cooperative launch");
}
} // namespace megafmm::detail

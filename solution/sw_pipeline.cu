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
constexpr int BM=64,BN=32,BK=256,ASize=BM*BK,BSize=BK*BN,Elements=BM*BN;
constexpr int Plane=512*512,Threads=512;
constexpr size_t WorkspaceBytes=size_t(16)*Plane*2;
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
template<class T> struct alignas(128) Shared {
  T operand[2][2*ASize+2*BSize];
  // Row-major P with an XOR permutation inside each 32-float row.
  // Both WGMMA fragment stores and row-major consumer loads use every bank once.
  float product[2][Elements];
  uint64_t ready[2],low_ready[2],free[2],product_ready[2],product_free[2];
};
static_assert(sizeof(Shared<cute::half_t>)==213120,"shared budget");
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

template<class T> __global__ __launch_bounds__(Threads,1)
void sw_cooperative_hilo(const __grid_constant__ CUtensorMap amap,
                         const __grid_constant__ CUtensorMap bmap,
                         const __grid_constant__ CUtensorMap smap,
                         const __grid_constant__ CUtensorMap tmap,
                         const T* __restrict__ A,const T* __restrict__ B,
                         T* __restrict__ packed,float* __restrict__ D){
  using namespace cute;
  extern __shared__ __align__(128) unsigned char buffer[];
  auto& s=*reinterpret_cast<Shared<T>*>(buffer);
  int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;
  if(tid==0){
    for(int j=0;j<2;++j){Barrier::init(&s.ready[j],1);Barrier::init(&s.low_ready[j],1);Barrier::init(&s.free[j],4);}
    for(int j=0;j<2;++j){Barrier::init(&s.product_ready[j],128);Barrier::init(&s.product_free[j],352);}
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  // All CTAs cooperate in one input transformation per invocation, never across calls.
  // Transform two adjacent elements per thread with aligned 32-bit IO. Keep
  // every scalar FP32 operation and each high/low rounding identical.
  for(int i=2*((blockIdx.y*gridDim.x+blockIdx.x)*Threads+tid);i<Plane;i+=2*gridDim.x*gridDim.y*Threads){
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
        words[2*j+1]|=uint32_t(T(__fsub_rn(sv[j],float(sh))).raw())<<(16*lane);
        words[8+2*j]|=uint32_t(th.raw())<<(16*lane);
        words[8+2*j+1]|=uint32_t(T(__fsub_rn(tv[j],float(th))).raw())<<(16*lane);
      }
    }
    #pragma unroll
    for(int j=0;j<16;++j)reinterpret_cast<uint32_t*>(packed)[(j*Plane+i)/2]=words[j];
  }
  // Generic global stores become visible to TMA's async proxy before publication.
  asm volatile("fence.proxy.async.global;" ::: "memory");
  cooperative_groups::this_grid().sync();
  constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
  constexpr auto lb=tile_to_shape(GMMA::Layout_MN_SW64_Atom<T>{},Shape<_32,Int<BK>>{});
  if(tid==0){
    for(int step=0;step<7;++step)for(int q=0;q<512/BK;++q){
      int p=product_for_step(step);
      int sequence=step*(512/BK)+q,slot=sequence%2,k0=q*BK;
      if(sequence>=2)wait(&s.free[slot],((sequence-2)/2)%2);
      bool alo=p==2 || p>=4,blo=p==3 || p>=4;
      // Publish high operands first so their MMA can overlap residual loads.
      // Advance both barrier phases on every stage, including zero-byte lows.
      TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*(ASize+BSize));
      TxBarrier::arrive_and_expect_tx(&s.low_ready[slot],(alo ? 2*ASize:0)+(blo ? 2*BSize:0));
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
        SM90_TMA_LOAD_3D::copy(&tmap,&s.ready[slot],0,op+2*ASize,n0,k0,plane);
      }else{
        int col=p==2 ? 512:0,row=p==1 || p==2 ? 512:0;
        SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],0,op+2*ASize,n0+col,k0+row);
      }
      if(alo){
        int plane=p==2 ? 6:2*(p-4);
        SM90_TMA_LOAD_4D::copy(&smap,&s.low_ready[slot],0,op+ASize,0,m0,k0/64,plane+1);
      }
      if(blo){
        int plane=p==3 ? 6:2*(p-4);
        SM90_TMA_LOAD_3D::copy(&tmap,&s.low_ready[slot],0,op+2*ASize+BSize,n0,k0,plane+1);
      }
    }
  }else if((tid>=32 && tid<128) || tid>=256){
    int worker=tid>=256 ? tid-160:tid-32;
    constexpr int Chunks=(Elements+351)/352;
    float first[Chunks],second[Chunks],current[Chunks];
    #pragma unroll
    for(int step=0;step<7;++step){
      wait(&s.product_ready[step%2],(step/2)%2);
      #pragma unroll
      for(int j=0;j<Chunks;++j){
        int i=worker+j*352;
        if(i<Elements)current[j]=s.product[step%2][product_index(i/BN,i%BN)];
      }
      // Release shared P immediately after reading; arithmetic/global stores can
      // overlap the Tensor warpgroup's reuse of this slot two steps later.
      arrive(&s.product_free[step%2]);
      #pragma unroll
      for(int j=0;j<Chunks;++j){
        int i=worker+j*352;
        if(i<Elements){
          int at=(m0+i/BN)*1024+n0+i%BN;
          if(step==0)first[j]=current[j]; // P1
          else if(step==1)D[at]=__fadd_rn(first[j],current[j]); // U1
          else if(step==2)first[j]=__fadd_rn(first[j],current[j]); // U2
          else if(step==3)second[j]=__fadd_rn(first[j],current[j]); // U3
          else if(step==4){
            D[at+512*1024+512]=__fadd_rn(second[j],current[j]); // D22
            first[j]=__fadd_rn(first[j],current[j]); // U4
          }else if(step==5)D[at+512]=__fadd_rn(first[j],current[j]); // D12
          else D[at+512*1024]=__fsub_rn(second[j],current[j]); // D21
        }
      }
    }
  }else if(tid>=128 && tid<256){
    for(int step=0;step<7;++step){
      int p=product_for_step(step);
      float acc[16]={};
      // Two fixed slots: expose their shared addresses to descriptor lowering.
      #pragma unroll
      for(int q=0;q<512/BK;++q){
        int sequence=step*(512/BK)+q,slot=sequence%2;
        wait(&s.ready[slot],(sequence/2)%2);
        T* op=s.operand[slot];
        auto ah=make_tensor(make_smem_ptr(op),la),al=make_tensor(make_smem_ptr(op+ASize),la);
        auto bh=make_tensor(make_smem_ptr(op+2*ASize),lb),bl=make_tensor(make_smem_ptr(op+2*ASize+BSize),lb);
        warpgroup_arrive();
        mma_term<T>(ah,bh,acc);
        if(p>=2)wait(&s.low_ready[slot],(sequence/2)%2);
        if(p==2 || p>=4)mma_term<T>(al,bh,acc);
        if(p==3 || p>=4)mma_term<T>(ah,bl,acc);
        warpgroup_commit_batch();warpgroup_wait<0>();
        __syncwarp();
        if((tid&31)==0)arrive(&s.free[slot]);
      }
      if(step>=2)wait(&s.product_free[step%2],((step-2)/2)%2);
      #pragma unroll
      for(int j=0;j<16;++j){
        int mn=GMMA::CLayout_64x32{}(tid-128,j);
        s.product[step%2][product_index(mn%64,mn/64)]=acc[j];
      }
      arrive(&s.product_ready[step%2]);
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
  if(prop.major!=9 || prop.minor!=0 || !prop.cooperativeLaunch)throw std::invalid_argument("SW needs cooperative Hopper launch");
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
void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream){
  CUtensorMap a{},b{},s{},t{};encode(a,A,state.dtype,true,false);encode(b,B,state.dtype,false,false);
  encode(s,workspace,state.dtype,true,true);encode(t,static_cast<char*>(workspace)+WorkspaceBytes/2,state.dtype,false,true);
  void* args[]={&a,&b,&s,&t,&A,&B,&workspace,&D};dim3 grid(16,8);
  if(state.dtype==DType::BF16)
    check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(sw_cooperative_hilo<cute::bfloat16_t>),grid,Threads,args,sizeof(Shared<cute::bfloat16_t>),stream),"SW cooperative launch");
  else
    check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(sw_cooperative_hilo<cute::half_t>),grid,Threads,args,sizeof(Shared<cute::half_t>),stream),"SW cooperative launch");
}
} // namespace megafmm::detail

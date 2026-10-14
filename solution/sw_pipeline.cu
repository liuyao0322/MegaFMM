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
constexpr int BM=64,BN=64,BK=256,Stages=2,ASize=BM*BK,BSize=BK*BN,Elements=BM*BN;
constexpr int Plane=512*512,Threads=768;
constexpr size_t WorkspaceBytes=size_t(16)*Plane*2;
constexpr size_t PackedBytes=size_t(8)*Plane*2;
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
template<class T> struct alignas(128) Shared {
  T operand[Stages][ASize+BSize];
  // Row-major P with an XOR permutation inside each 32-float row.
  // Both WGMMA fragment stores and row-major consumer loads use every bank once.
  float product[2][Elements];
  uint64_t ready[Stages],free[Stages],product_ready[2],product_free[2];
  float state[3][Elements];
  uint64_t pack_ready;
};
static_assert(sizeof(Shared<cute::half_t>)==213120,"shared budget");
__device__ void wait(const uint64_t* b,int phase){Barrier::wait(b,phase);asm volatile("" ::: "memory");}
__device__ void arrive(const uint64_t* b){asm volatile("" ::: "memory");Barrier::arrive(b);}

__device__ __forceinline__ int product_index(int row,int col){
  // Fragment lanes vary column bits 1,2 and row bits 0,1,2.
  // XOR the latter into column bits 0,3,4, the three remaining bank bits.
  return row*BN+(col^((row&1)|((row&6)<<2)));
}

// Each two-CTA cluster owns one64x64 quadrant tile.
__device__ int product_for_step(int step,int rank){
  return rank==0 ? (step<2 ? step:step==2 ? 5:6) : (step==0 ? 4:step==1 ? 2:3);
}

template<class T,class TA,class TB>
__device__ void mma_term(TA a,TB b,float (&d)[32]) {
  using namespace cute;
  using Op=std::conditional_t<std::is_same_v<T,bfloat16_t>,
      SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>,
      SM90_64x64x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::MN>>;
  // Build the canonical descriptor once; CuTe preserves the swizzled K offsets
  // in descriptor units instead of rebuilding address bitfields for each MMA.
  auto da_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(
      tiled_divide(a,Shape<_64,_16>{}));
  auto db_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(
      tiled_divide(b,Shape<_64,_16>{}));
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
    Op::fma(da,db,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],d[16],d[17],d[18],d[19],d[20],d[21],d[22],d[23],d[24],d[25],d[26],d[27],d[28],d[29],d[30],d[31]);
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
  int tid=threadIdx.x,rank=blockIdx.x&1,m0=blockIdx.y*BM,n0=(blockIdx.x/2)*BN;
  int products=rank==0 ? 4:3;
  if(tid==0){
    for(int j=0;j<Stages;++j){Barrier::init(&s.ready[j],1);Barrier::init(&s.free[j],4);}
    for(int j=0;j<2;++j){Barrier::init(&s.product_ready[j],128);Barrier::init(&s.product_free[j],256);}
    Barrier::init(&s.pack_ready,608);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
  constexpr auto lb=tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
  if(tid==0){
    for(int step=0;step<products;++step)for(int q=0;q<512/BK;++q){
      int p=product_for_step(step,rank);
      int sequence=step*(512/BK)+q,slot=sequence%Stages,k0=q*BK;
      if(sequence==(rank==0 ? 4:0)){
        wait(&s.pack_ready,0);
        auto* barrier=&cooperative_groups::details::get_grid_workspace()->barrier;
        uint32_t increment=(blockIdx.x==0&&blockIdx.y==0)?0x80000000u-127u:1u;
        uint32_t old,current;
        asm volatile("atom.add.acq_rel.gpu.u32 %0,[%1],%2;":"=r"(old):"l"(barrier),"r"(increment):"memory");
        do{asm volatile("ld.acquire.gpu.u32 %0,[%1];":"=r"(current):"l"(barrier):"memory");}while(!((old^current)&0x80000000u));
      }
      if(sequence>=Stages)wait(&s.free[slot],((sequence-Stages)/Stages)%2);
      bool alo=p==2 || p>=4,blo=p==3 || p>=4;
      // Seven-product experiment: only publish high operands; no residual loads.
      TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*(ASize+BSize));
      T* op=s.operand[slot];
      if(alo){
        int plane=p==2 ? 3:p-4;
        SM90_TMA_LOAD_4D::copy(&smap,&s.ready[slot],0,op,0,m0,k0/64,plane);
      }else{
        int col=(p==1 || p==3) ? 512:0,row=p==3 ? 512:0;
        SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],0,op,0,m0+row,(k0+col)/64);
      }
      if(blo){
        int plane=p==3 ? 3:p-4;
        SM90_TMA_LOAD_3D::copy(&tmap,&s.ready[slot],0,op+ASize,n0,k0,plane);
      }else{
        int col=p==2 ? 512:0,row=p==1 || p==2 ? 512:0;
        SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],0,op+ASize,n0+col,k0+row);
      }
    }
  }else if((tid>=32 && tid<128)||tid>=256){
    int worker=tid>=256 ? tid-160:tid-32;
    int tile=blockIdx.y*8+blockIdx.x/2;
    auto* flags=reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(packed)+(size_t(6)<<20));
    if(rank==1 && worker<256)flags[tile*256+worker]=0;
  for(int local=2*worker;local<2048;local+=2*608){
    int i=(blockIdx.y*gridDim.x+blockIdx.x)*2048+local;
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
    for(int j=0;j<8;++j)reinterpret_cast<uint32_t*>(packed)[(j*Plane+i)/2]=words[2*j];
  }

    asm volatile("fence.proxy.async.global;" ::: "memory");
    arrive(&s.pack_ready);
    if(worker<256){
      constexpr int Chunks=16;
      unsigned row=unsigned(worker)/BN,col=unsigned(worker)%BN;
      int base_even=product_index(row,col),base_odd=product_index(row+4,col)-256;
      float* out=D+(m0+row)*1024+n0+col;
      float first[Chunks],second[Chunks],current[Chunks];
      for(int step=0;step<products;++step){
        wait(&s.product_ready[step%2],(step/2)%2);
        #pragma unroll
        for(int j=0;j<Chunks;++j){
          int idx=((j&1)?base_odd:base_even)+j*256;
          current[j]=s.product[step%2][idx];
        }
        arrive(&s.product_free[step%2]);
        #pragma unroll
        for(int j=0;j<Chunks;++j){
          if(rank==0){
            if(step==0)first[j]=current[j];
            else if(step==1)out[j*4096]=__fadd_rn(first[j],current[j]);
            else if(step==2)first[j]=__fadd_rn(first[j],current[j]);
            else second[j]=__fadd_rn(first[j],current[j]);
          }else s.state[step][worker+j*256]=current[j];
        }
      }
      float* u2buf=reinterpret_cast<float*>(reinterpret_cast<char*>(packed)+PackedBytes)+tile*Elements;
      float* u3buf=u2buf+Plane;
      auto* flag=flags+tile*256+worker;
      if(rank==0){
        #pragma unroll
        for(int j=0;j<Chunks;++j){u2buf[worker+j*256]=first[j];u3buf[worker+j*256]=second[j];}
        asm volatile("st.release.gpu.u32 [%0],1;" :: "l"(flag) : "memory");
      }else{
        uint32_t ready;
        do{asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(ready) : "l"(flag) : "memory");}while(!ready);
        #pragma unroll
        for(int j=0;j<Chunks;++j){
          float u2=u2buf[worker+j*256],u3=u3buf[worker+j*256];
          float p5=s.state[0][worker+j*256],p3=s.state[1][worker+j*256],p4=s.state[2][worker+j*256];
          out[j*4096+512]=__fadd_rn(__fadd_rn(u2,p5),p3);
          out[j*4096+512*1024+512]=__fadd_rn(u3,p5);
          out[j*4096+512*1024]=__fsub_rn(u3,p4);
        }
      }
    }
  }else if(tid>=128 && tid<256){
    for(int step=0;step<products;++step){
      float acc[32]={};
      #pragma unroll
      for(int q=0;q<512/BK;++q){
        int sequence=step*(512/BK)+q,slot=sequence%Stages;
        wait(&s.ready[slot],(sequence/Stages)%2);
        T* op=s.operand[slot];
        auto ah=make_tensor(make_smem_ptr(op),la);
        auto bh=make_tensor(make_smem_ptr(op+ASize),lb);
        warpgroup_arrive();mma_term<T>(ah,bh,acc);
        warpgroup_commit_batch();warpgroup_wait<0>();
        __syncwarp();if((tid&31)==0)arrive(&s.free[slot]);
      }
      if(step>=2)wait(&s.product_free[step%2],((step-2)/2)%2);
      #pragma unroll
      for(int j=0;j<32;++j){
        int mn=GMMA::CLayout_64x64{}(tid-128,j);
        s.product[step%2][product_index(mn%64,mn/64)]=acc[j];
      }
      arrive(&s.product_ready[step%2]);
    }
  }
}
void check(cudaError_t e,const char* where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
void encode(CUtensorMap& map,const void* ptr,DType dtype,bool is_a,bool transformed){
  if(reinterpret_cast<uintptr_t>(ptr)%16)throw std::invalid_argument("SW TMA input must be16-byte aligned");
  if(is_a){
    // Expose K/64 as an outer tensor dimension. The shared box still packs
    // (64 K elements, BM rows, BK/64 chunks), matching the existing GMMA layout.
    uint64_t dims[4]={64,transformed ? 512ULL:1024ULL,transformed ? 8ULL:16ULL,4};
    uint64_t strides[3]={transformed ? 1024ULL:2048ULL,128,512ULL*512*2};
    uint32_t box[4]={64,BM,BK/64,1},element_strides[4]={1,1,1,1};
    auto result=cuTensorMapEncodeTiled(&map,dtype==DType::BF16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        transformed ? 4:3,const_cast<void*>(ptr),dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(result!=CUDA_SUCCESS)throw std::runtime_error("SW A tensor map status "+std::to_string(int(result)));
    return;
  }
  uint64_t dims[3]={transformed ? 512ULL:1024ULL,transformed ? 512ULL:1024ULL,4};
  uint64_t strides[2]={dims[0]*2,dims[0]*dims[1]*2};
  uint32_t box[3]={uint32_t(is_a ? 64:BN),uint32_t(is_a ? BM:BK),1},element_strides[3]={1,1,1};
  auto result=cuTensorMapEncodeTiled(&map,dtype==DType::BF16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
      transformed ? 3:2,const_cast<void*>(ptr),dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,
      is_a ? CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
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
  encode(s,workspace,state.dtype,true,true);encode(t,static_cast<char*>(workspace)+PackedBytes/2,state.dtype,false,true);
  void* args[]={&a,&b,&s,&t,&A,&B,&workspace,&D};dim3 grid(16,8);
  if(state.dtype==DType::BF16)
    check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(sw_cooperative_hilo<cute::bfloat16_t>),grid,Threads,args,sizeof(Shared<cute::bfloat16_t>),stream),"SW cooperative launch");
  else
    check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(sw_cooperative_hilo<cute::half_t>),grid,Threads,args,sizeof(Shared<cute::half_t>),stream),"SW cooperative launch");
}
} // namespace megafmm::detail

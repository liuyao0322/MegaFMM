#include "internal.h"
#include <cuda.h>
#include <cooperative_groups.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <stdexcept>
#include <type_traits>

namespace megafmm::diagnostic_trace {
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
  uint64_t ready[2],free[2],product_ready[2],product_free[2];
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

template<class T,int Role> __global__ __launch_bounds__(Threads,1)
void sw_cooperative_hilo(const __grid_constant__ CUtensorMap amap,
                         const __grid_constant__ CUtensorMap bmap,
                         const __grid_constant__ CUtensorMap smap,
                         const __grid_constant__ CUtensorMap tmap,
                         const T* A,const T* B,T* packed,float* D,Trace* traces,int sample){
  using namespace cute;
  extern __shared__ __align__(128) unsigned char buffer[];
  auto& s=*reinterpret_cast<Shared<T>*>(buffer);
  int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;
  uint64_t t0=0,t1=0,t2=0,t3=0,phase_begin=0,production_begin=0; uint32_t ready_ticks=0,issue_ticks=0,drain_ticks=0;
  if constexpr(Role==4)t0=stamp();
  if(tid==0){
    for(int j=0;j<2;++j){Barrier::init(&s.ready[j],1);Barrier::init(&s.free[j],4);}
    for(int j=0;j<2;++j){Barrier::init(&s.product_ready[j],128);Barrier::init(&s.product_free[j],352);}
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  // All CTAs cooperate in one input transformation per invocation, never across calls.
  for(int i=(blockIdx.y*gridDim.x+blockIdx.x)*Threads+tid;i<Plane;i+=gridDim.x*gridDim.y*Threads){
    int row=i/512,col=i%512,at=row*1024+col;
    float a11=float(A[at]),a12=float(A[at+512]),a21=float(A[at+512*1024]),a22=float(A[at+512*1024+512]);
    float b11=float(B[at]),b12=float(B[at+512]),b21=float(B[at+512*1024]),b22=float(B[at+512*1024+512]);
    float s1=__fadd_rn(a21,a22),s2=__fsub_rn(s1,a11);
    float t1=__fsub_rn(b12,b11),t2=__fsub_rn(b22,t1);
    float sv[4]={s1,s2,__fsub_rn(a11,a21),__fsub_rn(a12,s2)};
    float tv[4]={t1,t2,__fsub_rn(b22,b12),__fsub_rn(t2,b21)};
    #pragma unroll
    for(int j=0;j<4;++j){
      T sh=T(sv[j]),th=T(tv[j]);
      packed[(2*j)*Plane+i]=sh;
      packed[(2*j+1)*Plane+i]=T(__fsub_rn(sv[j],float(sh)));
      packed[(8+2*j)*Plane+i]=th;
      packed[(8+2*j+1)*Plane+i]=T(__fsub_rn(tv[j],float(th)));
    }
  }
  // Generic global stores become visible to TMA's async proxy before publication.
  asm volatile("fence.proxy.async.global;" ::: "memory");
  cooperative_groups::this_grid().sync();
  if constexpr(Role==4)t1=stamp();
  constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
  constexpr auto lb=tile_to_shape(GMMA::Layout_MN_SW64_Atom<T>{},Shape<_32,Int<BK>>{});
  if(tid==0){
    for(int step=0;step<7;++step)for(int q=0;q<512/BK;++q){
      int p=product_for_step(step);
      int sequence=step*(512/BK)+q,slot=sequence%2,k0=q*BK;
      if constexpr(Role==1)if(sequence==sample)t0=stamp();
      if(sequence>=2)wait(&s.free[slot],((sequence-2)/2)%2);
      if constexpr(Role==1)if(sequence==sample)t1=stamp();
      bool alo=p==2 || p>=4,blo=p==3 || p>=4;
      int bytes=2*(ASize+BSize)+(alo ? 2*ASize:0)+(blo ? 2*BSize:0);
      TxBarrier::arrive_and_expect_tx(&s.ready[slot],bytes);
      T* op=s.operand[slot];
      if(alo){
        int plane=p==2 ? 6:2*(p-4);
        for(int chunk=0;chunk<BK/64;++chunk){
          SM90_TMA_LOAD_3D::copy(&smap,&s.ready[slot],0,op+chunk*BM*64,k0+chunk*64,m0,plane);
          SM90_TMA_LOAD_3D::copy(&smap,&s.ready[slot],0,op+ASize+chunk*BM*64,k0+chunk*64,m0,plane+1);
        }
      }else{
        int col=(p==1 || p==3) ? 512:0,row=p==3 ? 512:0;
        for(int chunk=0;chunk<BK/64;++chunk)
          SM90_TMA_LOAD_2D::copy(&amap,&s.ready[slot],0,op+chunk*BM*64,k0+col+chunk*64,m0+row);
      }
      if(blo){
        int plane=p==3 ? 6:2*(p-4);
        SM90_TMA_LOAD_3D::copy(&tmap,&s.ready[slot],0,op+2*ASize,n0,k0,plane);
        SM90_TMA_LOAD_3D::copy(&tmap,&s.ready[slot],0,op+2*ASize+BSize,n0,k0,plane+1);
      }else{
        int col=p==2 ? 512:0,row=p==1 || p==2 ? 512:0;
        SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],0,op+2*ASize,n0+col,k0+row);
      }
      if constexpr(Role==1)if(sequence==sample){t2=stamp();t3=t2;}
    }
  }else if((tid>=32 && tid<128) || tid>=256){
    int worker=tid>=256 ? tid-160:tid-32;
    constexpr int Chunks=(Elements+351)/352;
    float first[Chunks],second[Chunks],current[Chunks];
    #pragma unroll
    for(int step=0;step<7;++step){
      if constexpr(Role==2)if(step==sample)t0=stamp();
      wait(&s.product_ready[step%2],(step/2)%2);
      if constexpr(Role==2)if(step==sample)t1=stamp();
      #pragma unroll
      for(int j=0;j<Chunks;++j){
        int i=worker+j*352;
        if(i<Elements)current[j]=s.product[step%2][product_index(i/BN,i%BN)];
      }
      // Release shared P immediately after reading; arithmetic/global stores can
      // overlap the Tensor warpgroup's reuse of this slot two steps later.
      if constexpr(Role==2)if(step==sample)t2=stamp();
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
      if constexpr(Role==2)if(step==sample)t3=stamp();
    }
  }else if(tid>=128 && tid<256){
    if constexpr(Role==5)production_begin=stamp();
    for(int step=0;step<7;++step){
      int p=product_for_step(step);
      float acc[16]={};
      for(int q=0;q<512/BK;++q){
        int sequence=step*(512/BK)+q,slot=sequence%2;
        if constexpr(Role==5)phase_begin=stamp();
        if constexpr(Role==0)if(sequence==sample)t0=stamp();
        wait(&s.ready[slot],(sequence/2)%2);
        if constexpr(Role==0)if(sequence==sample)t1=stamp();
        if constexpr(Role==5){uint64_t end=stamp();ready_ticks+=uint32_t(end-phase_begin);phase_begin=end;}
        T* op=s.operand[slot];
        auto ah=make_tensor(make_smem_ptr(op),la),al=make_tensor(make_smem_ptr(op+ASize),la);
        auto bh=make_tensor(make_smem_ptr(op+2*ASize),lb),bl=make_tensor(make_smem_ptr(op+2*ASize+BSize),lb);
        warpgroup_arrive();
        mma_term<T>(ah,bh,acc);
        if(p==2 || p>=4)mma_term<T>(al,bh,acc);
        if(p==3 || p>=4)mma_term<T>(ah,bl,acc);
        warpgroup_commit_batch();
        if constexpr(Role==0)if(sequence==sample)t2=stamp();
        if constexpr(Role==5){uint64_t end=stamp();issue_ticks+=uint32_t(end-phase_begin);phase_begin=end;}
        warpgroup_wait<0>();
        if constexpr(Role==0)if(sequence==sample)t3=stamp();
        if constexpr(Role==5){uint64_t end=stamp();drain_ticks+=uint32_t(end-phase_begin);}
        __syncwarp();
        if((tid&31)==0)arrive(&s.free[slot]);
      }
      if constexpr(Role==3)if(step==sample)t0=stamp();
      if(step>=2)wait(&s.product_free[step%2],((step-2)/2)%2);
      if constexpr(Role==3)if(step==sample)t1=stamp();
      #pragma unroll
      for(int j=0;j<16;++j){
        int mn=GMMA::CLayout_64x32{}(tid-128,j);
        s.product[step%2][product_index(mn%64,mn/64)]=acc[j];
      }
      if constexpr(Role==3)if(step==sample)t2=stamp();
      arrive(&s.product_ready[step%2]);
      if constexpr(Role==3)if(step==sample)t3=stamp();
    }
    if constexpr(Role==5){t0=ready_ticks;t1=issue_ticks;t2=drain_ticks;t3=stamp()-production_begin;}
  }
  if constexpr(Role==4)t2=stamp();
  __syncthreads();
  if constexpr(Role==4)t3=stamp();
  if((Role==0 || Role==3 || Role==4 || Role==5) ? tid==128:Role==1 ? tid==0:tid==32)
    traces[blockIdx.y*gridDim.x+blockIdx.x]={t0,t1,t2,t3};
}
void check(cudaError_t e,const char* where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
void encode(CUtensorMap& map,const void* ptr,DType dtype,bool is_a,bool transformed){
  if(reinterpret_cast<uintptr_t>(ptr)%16)throw std::invalid_argument("SW TMA input must be16-byte aligned");
  uint64_t dims[3]={transformed ? 512ULL:1024ULL,transformed ? 512ULL:1024ULL,8};
  uint64_t strides[2]={dims[0]*2,dims[0]*dims[1]*2};
  uint32_t box[3]={uint32_t(is_a ? 64:BN),uint32_t(is_a ? BM:BK),1},element_strides[3]={1,1,1};
  auto result=cuTensorMapEncodeTiled(&map,dtype==DType::BF16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
      transformed ? 3:2,const_cast<void*>(ptr),dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,
      is_a ? CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_64B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(result!=CUDA_SUCCESS)throw std::runtime_error("SW tensor map status "+std::to_string(int(result)));
}
} // namespace
} // namespace megafmm::diagnostic_trace

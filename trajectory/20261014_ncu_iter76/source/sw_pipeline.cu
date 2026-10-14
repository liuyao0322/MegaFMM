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
  float product[2][Elements];
  uint64_t ready[2],free[2],product_ready[2],product_free[2];
};
static_assert(sizeof(Shared<cute::half_t>)==213120,"shared budget");
__device__ void wait(const uint64_t* b,int phase){Barrier::wait(b,phase);asm volatile("" ::: "memory");}
__device__ void arrive(const uint64_t* b){asm volatile("" ::: "memory");Barrier::arrive(b);}

template<class T,class TA,class TB>
__device__ void mma_term(TA a,TB b,float (&d)[16]) {
  using namespace cute;
  using Op=std::conditional_t<std::is_same_v<T,bfloat16_t>,
      SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>,
      SM90_64x32x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::MN>>;
  #pragma unroll
  for(int kk=0;kk<BK/16;++kk){
    auto at=local_tile(a,Shape<_64,_16>{},make_coord(0,kk));
    auto bt=local_tile(b,Shape<_32,_16>{},make_coord(0,kk));
    uint64_t da=GMMA::make_gmma_desc<GMMA::Major::K>(at);
    uint64_t db=GMMA::make_gmma_desc<GMMA::Major::MN>(bt);
    Op::fma(da,db,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15]);
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
  constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
  constexpr auto lb=tile_to_shape(GMMA::Layout_MN_SW64_Atom<T>{},Shape<_32,Int<BK>>{});
  if(tid==0){
    for(int p=0;p<7;++p)for(int q=0;q<512/BK;++q){
      int sequence=p*(512/BK)+q,slot=sequence%2,k0=q*BK;
      if(sequence>=2)wait(&s.free[slot],((sequence-2)/2)%2);
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
    }
  }else if((tid>=32 && tid<128) || tid>=256){
    int worker=tid>=256 ? tid-160:tid-32;
    constexpr int Chunks=(Elements+351)/352;
    float values[7][Chunks];
    #pragma unroll
    for(int p=0;p<7;++p){
      wait(&s.product_ready[p%2],(p/2)%2);
      #pragma unroll
      for(int j=0;j<Chunks;++j){
        int i=worker+j*352;
        if(i<Elements)values[p][j]=s.product[p%2][i];
      }
      arrive(&s.product_free[p%2]);
    }
    #pragma unroll
    for(int j=0;j<Chunks;++j){
      int i=worker+j*352;
      if(i<Elements){
        float p1=values[0][j],p2=values[1][j],p3=values[2][j],p4=values[3][j],p5=values[4][j],p6=values[5][j],p7=values[6][j];
        float u1=__fadd_rn(p1,p2),u2=__fadd_rn(p1,p6),u3=__fadd_rn(u2,p7),u4=__fadd_rn(u2,p5);
        int row=m0+i/BN,col=n0+i%BN;
        D[row*1024+col]=u1;D[row*1024+col+512]=__fadd_rn(u4,p3);
        D[(row+512)*1024+col]=__fsub_rn(u3,p4);D[(row+512)*1024+col+512]=__fadd_rn(u3,p5);
      }
    }
  }else if(tid>=128 && tid<256){
    for(int p=0;p<7;++p){
      float acc[16]={};
      for(int q=0;q<512/BK;++q){
        int sequence=p*(512/BK)+q,slot=sequence%2;
        wait(&s.ready[slot],(sequence/2)%2);
        T* op=s.operand[slot];
        auto ah=make_tensor(make_smem_ptr(op),la),al=make_tensor(make_smem_ptr(op+ASize),la);
        auto bh=make_tensor(make_smem_ptr(op+2*ASize),lb),bl=make_tensor(make_smem_ptr(op+2*ASize+BSize),lb);
        warpgroup_arrive();
        mma_term<T>(ah,bh,acc);
        if(p==2 || p>=4)mma_term<T>(al,bh,acc);
        if(p==3 || p>=4)mma_term<T>(ah,bl,acc);
        warpgroup_commit_batch();warpgroup_wait<0>();
        __syncwarp();
        if((tid&31)==0)arrive(&s.free[slot]);
      }
      if(p>=2)wait(&s.product_free[p%2],((p-2)/2)%2);
      #pragma unroll
      for(int j=0;j<16;++j){int mn=GMMA::CLayout_64x32{}(tid-128,j);s.product[p%2][(mn%64)*BN+mn/64]=acc[j];}
      arrive(&s.product_ready[p%2]);
    }
  }
  __syncthreads();
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

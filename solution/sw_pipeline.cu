#include "internal.h"
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <stdexcept>
#include <type_traits>

namespace megafmm::detail {
namespace {
constexpr int BM=64, BN=32, BK=32, Elements=BM*BN;
constexpr int ASize=BM*BK, BSize=BK*BN;
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;

// One allocation; independent regions never alias a live asynchronous operand.
template<class T> struct alignas(128) Shared {
  T raw[2][4*ASize+4*BSize];
  float st[4*ASize+4*BSize];
  T operand[2][2*ASize+2*BSize];
  float product[7][Elements];
  float output[4][Elements];
  uint64_t raw_ready[2],raw_free[2],op_ready[2],op_free[2];
  uint64_t product_ready,product_free;
};
static_assert(sizeof(Shared<cute::half_t>)==213120, "recheck shared-memory budget");

__device__ void wait(const uint64_t* b,int phase) {
  Barrier::wait(b,phase); asm volatile("" ::: "memory");
}
__device__ void arrive(const uint64_t* b) {
  asm volatile("" ::: "memory"); Barrier::arrive(b);
}
__device__ void workers_sync() { asm volatile("bar.sync 1, 96;" ::: "memory"); }

template<class T> __device__ float left_operand(Shared<T>& s,int bank,int p,int i) {
  const T* raw=s.raw[bank];
  if(p==0)return float(raw[i]);
  if(p==1)return float(raw[ASize+i]);
  if(p==2)return s.st[3*ASize+i];
  if(p==3)return float(raw[3*ASize+i]);
  return s.st[(p-4)*ASize+i];
}
template<class T> __device__ float right_operand(Shared<T>& s,int bank,int p,int i) {
  const T* raw=s.raw[bank]+4*ASize;
  if(p==0)return float(raw[i]);
  if(p==1)return float(raw[2*BSize+i]);
  if(p==2)return float(raw[3*BSize+i]);
  if(p==3)return s.st[4*ASize+3*BSize+i];
  return s.st[4*ASize+(p-4)*BSize+i];
}

template<class T,class TA,class TB>
__device__ void mma_term(TA a,TB b,float (&d)[16]) {
  using namespace cute;
  using Op=std::conditional_t<std::is_same_v<T,bfloat16_t>,
      SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>,
      SM90_64x32x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::K>>;
  #pragma unroll
  for(int kk=0;kk<2;++kk) {
    auto at=local_tile(a,Shape<_64,_16>{},make_coord(0,kk));
    auto bt=local_tile(b,Shape<_32,_16>{},make_coord(0,kk));
    uint64_t da=GMMA::make_gmma_desc<GMMA::Major::K>(at);
    uint64_t db=GMMA::make_gmma_desc<GMMA::Major::K>(bt);
    Op::fma(da,db,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
            d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15]);
  }
}

template<class T>
__global__ __launch_bounds__(256,1)
void sw_unified_v1(const __grid_constant__ CUtensorMap map_a,
                   const __grid_constant__ CUtensorMap map_b,float* D) {
  using namespace cute;
  extern __shared__ __align__(128) unsigned char buffer[];
  auto& s=*reinterpret_cast<Shared<T>*>(buffer);
  int tid=threadIdx.x;
  int m0=blockIdx.y*BM,n0=blockIdx.x*BN;
  // Canonical interleaved layout avoids incompatible K-subtile swizzle origins.
  // Shared addresses are obtained from these exact layouts on both sides.
  constexpr auto la=tile_to_shape(GMMA::Layout_K_INTER_Atom<T>{},Shape<_64,_32>{});
  constexpr auto lb=tile_to_shape(GMMA::Layout_K_INTER_Atom<T>{},Shape<_32,_32>{});
  if(tid==0) {
    for(int j=0;j<2;++j) {
      Barrier::init(&s.raw_ready[j],1);
      Barrier::init(&s.raw_free[j],96);
      Barrier::init(&s.op_ready[j],96);
      Barrier::init(&s.op_free[j],128);
    }
    Barrier::init(&s.product_ready,128);
    Barrier::init(&s.product_free,96);
    cutlass::arch::fence_barrier_init();
  }
  for(int i=tid;i<4*Elements;i+=256) (&s.output[0][0])[i]=0;
  __syncthreads();

  if(tid==0) {
    // TMA warp leader: exactly eight source boxes per K slice.
    for(int q=0;q<16;++q) {
      int bank=q%2,k0=q*BK;
      if(q>=2)wait(&s.raw_free[bank],((q-2)/2)%2);
      TxBarrier::arrive_and_expect_tx(&s.raw_ready[bank],2*(4*ASize+4*BSize));
      #pragma unroll
      for(int quadrant=0;quadrant<4;++quadrant) {
        SM90_TMA_LOAD_2D::copy(&map_a,&s.raw_ready[bank],0,
            s.raw[bank]+quadrant*ASize,k0+(quadrant%2)*512,m0+(quadrant/2)*512);
        SM90_TMA_LOAD_2D::copy(&map_b,&s.raw_ready[bank],0,
            s.raw[bank]+4*ASize+quadrant*BSize,n0+(quadrant%2)*512,k0+(quadrant/2)*512);
      }
    }
  } else if(tid>=32 && tid<128) {
    int worker=tid-32;
    for(int q=0;q<16;++q) {
      int bank=q%2;
      wait(&s.raw_ready[bank],(q/2)%2);
      for(int i=worker;i<ASize;i+=96) {
        float a11=float(s.raw[bank][i]),a12=float(s.raw[bank][ASize+i]);
        float a21=float(s.raw[bank][2*ASize+i]),a22=float(s.raw[bank][3*ASize+i]);
        float s1=__fadd_rn(a21,a22),s2=__fsub_rn(s1,a11);
        s.st[i]=s1;s.st[ASize+i]=s2;
        s.st[2*ASize+i]=__fsub_rn(a11,a21);s.st[3*ASize+i]=__fsub_rn(a12,s2);
      }
      for(int i=worker;i<BSize;i+=96) {
        const T* b=s.raw[bank]+4*ASize;
        float b11=float(b[i]),b12=float(b[BSize+i]);
        float b21=float(b[2*BSize+i]),b22=float(b[3*BSize+i]);
        float t1=__fsub_rn(b12,b11),t2=__fsub_rn(b22,t1);
        s.st[4*ASize+i]=t1;s.st[4*ASize+BSize+i]=t2;
        s.st[4*ASize+2*BSize+i]=__fsub_rn(b22,b12);
        s.st[4*ASize+3*BSize+i]=__fsub_rn(t2,b21);
      }
      workers_sync();
      for(int p=0;p<7;++p) {
        int sequence=q*7+p,slot=sequence%2;
        if(sequence>=2)wait(&s.op_free[slot],((sequence-2)/2)%2);
        T* op=s.operand[slot];
        for(int i=worker;i<ASize;i+=96) {
          float v=left_operand(s,bank,p,i);T hi=T(v),lo=T(__fsub_rn(v,float(hi)));
          int offset=la(i/BK,i%BK);
          op[offset]=hi;op[ASize+offset]=lo;
        }
        for(int i=worker;i<BSize;i+=96) {
          float v=right_operand(s,bank,p,i);T hi=T(v),lo=T(__fsub_rn(v,float(hi)));
          int offset=lb(i%BN,i/BN);
          op[2*ASize+offset]=hi;op[2*ASize+BSize+offset]=lo;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        arrive(&s.op_ready[slot]);
      }
      arrive(&s.raw_free[bank]);
      wait(&s.product_ready,q%2);
      for(int i=worker;i<Elements;i+=96) {
        float p1=s.product[0][i],p2=s.product[1][i],p3=s.product[2][i];
        float p4=s.product[3][i],p5=s.product[4][i],p6=s.product[5][i],p7=s.product[6][i];
        float u1=__fadd_rn(p1,p2),u2=__fadd_rn(p1,p6);
        float u3=__fadd_rn(u2,p7),u4=__fadd_rn(u2,p5);
        s.output[0][i]=__fadd_rn(s.output[0][i],u1);
        s.output[1][i]=__fadd_rn(s.output[1][i],__fadd_rn(u4,p3));
        s.output[2][i]=__fadd_rn(s.output[2][i],__fsub_rn(u3,p4));
        s.output[3][i]=__fadd_rn(s.output[3][i],__fadd_rn(u3,p5));
      }
      arrive(&s.product_free);
    }
    for(int i=worker;i<Elements;i+=96) {
      int row=m0+i/BN,col=n0+i%BN;
      D[row*1024+col]=s.output[0][i];
      D[row*1024+col+512]=s.output[1][i];
      D[(row+512)*1024+col]=s.output[2][i];
      D[(row+512)*1024+col+512]=s.output[3][i];
    }
  } else if(tid>=128) {
    for(int q=0;q<16;++q) {
      if(q)wait(&s.product_free,(q-1)%2);
      for(int p=0;p<7;++p) {
        int sequence=q*7+p,slot=sequence%2;
        wait(&s.op_ready[slot],(sequence/2)%2);
        T* op=s.operand[slot];
        auto ah=make_tensor(make_smem_ptr(op),la);
        auto al=make_tensor(make_smem_ptr(op+ASize),la);
        auto bh=make_tensor(make_smem_ptr(op+2*ASize),lb);
        auto bl=make_tensor(make_smem_ptr(op+2*ASize+BSize),lb);
        float acc[16]={};
        warpgroup_arrive();
        mma_term<T>(ah,bh,acc);
        if(p==2 || p>=4)mma_term<T>(al,bh,acc);
        if(p==3 || p>=4)mma_term<T>(ah,bl,acc);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        #pragma unroll
        for(int j=0;j<16;++j) {
          int mn=GMMA::CLayout_64x32{}(tid-128,j);
          s.product[p][(mn%64)*BN+mn/64]=acc[j];
        }
        arrive(&s.op_free[slot]);
      }
      arrive(&s.product_ready);
    }
  }
  __syncthreads();
}

void check(cudaError_t e,const char* where) {
  if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
void encode(CUtensorMap& map,const void* ptr,DType dtype,int rows) {
  if(reinterpret_cast<uintptr_t>(ptr)%16)throw std::invalid_argument("SW TMA input must be 16-byte aligned");
  uint64_t dims[2]={1024,1024},strides[1]={2048};
  uint32_t box[2]={32,uint32_t(rows)},element_strides[2]={1,1};
  CUresult e=cuTensorMapEncodeTiled(&map,dtype==DType::BF16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
      2,const_cast<void*>(ptr),dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(e!=CUDA_SUCCESS)throw std::runtime_error("SW cuTensorMapEncodeTiled status "+std::to_string(int(e)));
}
} // namespace

struct SWState { DType dtype; };
std::shared_ptr<SWState> make_sw_pipeline(const Problem& p) {
  if(p.m!=1024 || p.n!=1024 || p.k!=1024 || p.trans_a || p.trans_b ||
     p.lda!=1024 || p.ldb!=1024 || p.ldd!=1024 || p.alpha!=1 || p.beta!=0)
    throw std::invalid_argument("SW v1 supports only contiguous non-transposed 1024^3, alpha=1, beta=0");
  cudaDeviceProp prop{};int device;
  check(cudaGetDevice(&device),"get SW device");check(cudaGetDeviceProperties(&prop,device),"get SW properties");
  if(prop.major!=9 || prop.minor!=0)throw std::invalid_argument("SW v1 requires Hopper sm90a");
  if(p.dtype==DType::BF16)
    check(cudaFuncSetAttribute(sw_unified_v1<cute::bfloat16_t>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<cute::bfloat16_t>)),"SW shared-memory opt-in");
  else
    check(cudaFuncSetAttribute(sw_unified_v1<cute::half_t>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<cute::half_t>)),"SW shared-memory opt-in");
  return std::make_shared<SWState>(SWState{p.dtype});
}
void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,cudaStream_t stream) {
  // Bind current pointers every call; CPU encoding is included in ordinary-call wall/event timing.
  // Graph replay retains descriptors but recomputes all input-dependent values.
  CUtensorMap a{},b{};encode(a,A,state.dtype,BM);encode(b,B,state.dtype,BK);
  dim3 grid(512/BN,512/BM);
  if(state.dtype==DType::BF16)
    sw_unified_v1<cute::bfloat16_t><<<grid,256,sizeof(Shared<cute::bfloat16_t>),stream>>>(a,b,D);
  else
    sw_unified_v1<cute::half_t><<<grid,256,sizeof(Shared<cute::half_t>),stream>>>(a,b,D);
  check(cudaGetLastError(),"launch SW unified v1");
}
} // namespace megafmm::detail

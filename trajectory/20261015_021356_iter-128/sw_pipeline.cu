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
constexpr int BM=64,BN=128,BK=64,Stages=4,ASize=BM*BK,BSize=BK*BN,Plane=512*512,Threads=256;
using Barrier=cutlass::arch::ClusterBarrier;
template<class T>struct alignas(128)Shared{T operand[Stages][ASize+BSize];uint64_t ready[Stages],free[Stages];};
static_assert(sizeof(Shared<cute::half_t>)==98432,"shared budget");
__device__ void wait(const uint64_t* b,int phase){Barrier::wait(b,phase);asm volatile("" ::: "memory");}
__device__ void arrive(const uint64_t* b){asm volatile("" ::: "memory");Barrier::arrive(b);}
// Preserve FP32 Winograd transforms and final input-dtype rounding exactly.
template<class T> __device__ __forceinline__ uint32_t operand_word(const T* input,int at,int p,bool is_a){
  const uint32_t* v=reinterpret_cast<const uint32_t*>(input);
  constexpr int Right=256,Down=512*1024/2;
  if(is_a){
    if(p==0)return v[at];
    if(p==1)return v[at+Right];
    if(p==3)return v[at+Right+Down];
  }else{
    if(p==0)return v[at];
    if(p==1)return v[at+Down];
    if(p==2)return v[at+Right+Down];
  }
  uint32_t w0=0,w1=0,w2=0,w3=0;
  if(is_a){
    w2=v[at+Down];
    if(p!=6)w3=v[at+Down+Right];
    if(p!=4)w0=v[at];
    if(p==2)w1=v[at+Right];
  }else{
    w1=v[at+Right];
    if(p!=6)w0=v[at];
    if(p!=4)w3=v[at+Down+Right];
    if(p==3)w2=v[at+Down];
  }
  uint32_t out=0;
  #pragma unroll
  for(int lane=0;lane<2;++lane){
    float a=float(T::bitcast(uint16_t(w0>>(16*lane))));
    float b=float(T::bitcast(uint16_t(w1>>(16*lane))));
    float c=float(T::bitcast(uint16_t(w2>>(16*lane))));
    float d=float(T::bitcast(uint16_t(w3>>(16*lane))));
    float x;
    if(is_a){
      if(p==6)x=__fsub_rn(a,c);
      else{float s1=__fadd_rn(c,d);if(p==4)x=s1;else{float s2=__fsub_rn(s1,a);x=p==5 ? s2:__fsub_rn(b,s2);}}
    }else{
      if(p==6)x=__fsub_rn(d,b);
      else{float t1=__fsub_rn(b,a);if(p==4)x=t1;else{float t2=__fsub_rn(d,t1);x=p==5 ? t2:__fsub_rn(t2,c);}}
    }
    out|=uint32_t(T(x).raw())<<(lane*16);
  }
  return out;
}
template<class T,class TA,class TB>
__device__ void mma_term(TA a,TB b,float (&d)[64]) {
  using namespace cute;
  using Op=std::conditional_t<std::is_same_v<T,bfloat16_t>,
      SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>,
      SM90_64x128x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::MN>>;
  // Build the canonical descriptor once; CuTe preserves the swizzled K offsets
  // in descriptor units instead of rebuilding address bitfields for each MMA.
  auto da_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(
      tiled_divide(a,Shape<_64,_16>{}));
  auto db_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(
      tiled_divide(b,Shape<_128,_16>{}));
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
    Op::fma(da,db,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],d[16],d[17],d[18],d[19],d[20],d[21],d[22],d[23],d[24],d[25],d[26],d[27],d[28],d[29],d[30],d[31],d[32],d[33],d[34],d[35],d[36],d[37],d[38],d[39],d[40],d[41],d[42],d[43],d[44],d[45],d[46],d[47],d[48],d[49],d[50],d[51],d[52],d[53],d[54],d[55],d[56],d[57],d[58],d[59],d[60],d[61],d[62],d[63]);
  }
}

template<class T> __global__ __launch_bounds__(Threads,2)
void sw_direct_leaf(const T* A,const T* B,float* products){
  using namespace cute;
  extern __shared__ __align__(128)unsigned char storage[];
  auto& s=*reinterpret_cast<Shared<T>*>(storage);
  int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN,p=blockIdx.z;
  if(tid==0){for(int j=0;j<Stages;++j){Barrier::init(&s.ready[j],128);Barrier::init(&s.free[j],4);}cutlass::arch::fence_barrier_init();}
  __syncthreads();
  constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
  constexpr auto lb=tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_128,Int<BK>>{},Step<_2,_1>{});
  if(tid<128){
    for(int q=0;q<512/BK;++q){
      int slot=q%Stages,k0=q*BK;
      if(q>=Stages)wait(&s.free[slot],((q-Stages)/Stages)%2);
      #pragma unroll 1
      for(int w=tid;w<ASize/2;w+=128){
        int r=(2*w)/BK,k=(2*w)%BK;
        int at=((m0+r)*1024+k0+k)/2;
        uint32_t word=operand_word(A,at,p,true);
        *reinterpret_cast<uint32_t*>(s.operand[slot]+la(r,k))=word;
      }
      #pragma unroll 1
      for(int w=tid;w<BSize/2;w+=128){
        int k=(2*w)/BN,n=(2*w)%BN;
        int at=((k0+k)*1024+n0+n)/2;
        uint32_t word=operand_word(B,at,p,false);
        *reinterpret_cast<uint32_t*>(s.operand[slot]+ASize+lb(n,k))=word;
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      arrive(&s.ready[slot]);
    }
  }else{
    float acc[64]={};
    #pragma unroll 1
    for(int q=0;q<512/BK;++q){
      int slot=q%Stages;
      wait(&s.ready[slot],(q/Stages)%2);
      auto ah=make_tensor(make_smem_ptr(s.operand[slot]),la);
      auto bh=make_tensor(make_smem_ptr(s.operand[slot]+ASize),lb);
      #pragma unroll
      for(int j=0;j<64;++j)warpgroup_fence_operand(acc[j]);
      warpgroup_arrive();mma_term<T>(ah,bh,acc);warpgroup_commit_batch();warpgroup_wait<0>();
      #pragma unroll
      for(int j=0;j<64;++j)warpgroup_fence_operand(acc[j]);
      __syncwarp();if((tid&31)==0)arrive(&s.free[slot]);
    }
    #pragma unroll
    for(int j=0;j<32;++j){
      int mn=GMMA::CLayout_64x128{}(tid-128,2*j);
      int at=p*Plane+(m0+mn%64)*512+n0+mn/64;
      *reinterpret_cast<float2*>(products+at)=make_float2(acc[2*j],acc[2*j+1]);
    }
  }
  __syncthreads();
}
__global__ void sw_recombine(const float* products,float* D){
  for(int i=2*(blockIdx.x*blockDim.x+threadIdx.x);i<Plane;i+=2*gridDim.x*blockDim.x){
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
} // namespace
struct SWState{DType dtype;};
std::shared_ptr<SWState> make_sw_pipeline(const Problem& p){
  if(p.m!=1024 || p.n!=1024 || p.k!=1024 || p.trans_a || p.trans_b || p.lda!=1024 || p.ldb!=1024 || p.ldd!=1024 || p.alpha!=1 || p.beta!=0)
    throw std::invalid_argument("SW requires contiguous non-transposed1024^3 alpha1 beta0");
  int device;cudaDeviceProp prop{};check(cudaGetDevice(&device),"SW device");check(cudaGetDeviceProperties(&prop,device),"SW properties");
  if(prop.major!=9 || prop.minor!=0)throw std::invalid_argument("SW needs Hopper");
  if(p.dtype==DType::BF16)check(cudaFuncSetAttribute(sw_direct_leaf<cute::bfloat16_t>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<cute::bfloat16_t>)),"SW shared opt-in");
  else check(cudaFuncSetAttribute(sw_direct_leaf<cute::half_t>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<cute::half_t>)),"SW shared opt-in");
  return std::make_shared<SWState>(SWState{p.dtype});
}
void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream){
  float* products=static_cast<float*>(workspace);
  if(state.dtype==DType::BF16)sw_direct_leaf<cute::bfloat16_t><<<dim3(4,8,7),Threads,sizeof(Shared<cute::bfloat16_t>),stream>>>(static_cast<const cute::bfloat16_t*>(A),static_cast<const cute::bfloat16_t*>(B),products);
  else sw_direct_leaf<cute::half_t><<<dim3(4,8,7),Threads,sizeof(Shared<cute::half_t>),stream>>>(static_cast<const cute::half_t*>(A),static_cast<const cute::half_t*>(B),products);
  check(cudaGetLastError(),"SW direct leaf launch");
  sw_recombine<<<128,256,0,stream>>>(products,D);check(cudaGetLastError(),"SW recombine launch");
}
} // namespace megafmm::detail

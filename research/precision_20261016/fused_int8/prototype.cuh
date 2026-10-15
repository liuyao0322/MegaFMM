#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include "megafmm/gemm.h"
#include "../int8_hardware/pow2.cuh"
#include <stdexcept>
#include <type_traits>

// Standalone fused candidate. No global transformed operands or production integration.
namespace fused_int8 {
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
constexpr size_t WorkBytes=128; // Compatibility with the independent comparator; unused by run.
void check(cudaError_t e,const char* where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
__device__ __forceinline__ void producer_sync(){asm volatile("bar.sync 1,128;" ::: "memory");}
struct alignas(1024) Slot {uint8_t bytes[49152];int eah,eal,ebh,ebl;};
struct alignas(1024) Shared {
 uint8_t a[4][8192],b[4][16384];Slot slot[2];
 uint64_t raw_ready,raw_free,ready[2],free[2];
 float maxima[4][4];int inside[2];
};
static_assert(sizeof(Shared)<232448);
template<class T> struct Traits {
 using Q=std::conditional_t<sizeof(T)==2,int8_t,T>;
 __host__ __device__ static constexpr auto ah(){return cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<T>{},cute::Shape<cute::_32,cute::_128>{});}
 __host__ __device__ static constexpr auto bh(){return cute::tile_to_shape(cute::GMMA::Layout_MN_SW128_Atom<T>{},cute::Shape<cute::_64,cute::_128>{});}
 __host__ __device__ static constexpr auto aq(){return cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<Q>{},cute::Shape<cute::_32,cute::_128>{});}
 __host__ __device__ static constexpr auto bq(){return cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<Q>{},cute::Shape<cute::_64,cute::_128>{});}
 // Flagged GMMA layouts move a byte-domain swizzle onto make_tensor's pointer.
 // Direct element indexing must first recast it to an element-domain swizzle.
 __host__ __device__ static int ah_index(int r,int k){return cute::as_position_independent_swizzle_layout(ah())(r,k);}
 __host__ __device__ static int bh_index(int n,int k){return cute::as_position_independent_swizzle_layout(bh())(n,k);}
};

template<class T> __device__ __forceinline__ float raw_a(Shared& s,int p,int r,int k){return float(reinterpret_cast<T*>(s.a[p])[Traits<T>::ah_index(r,k)]);}
template<class T> __device__ __forceinline__ float raw_b(Shared& s,int p,int n,int k){return float(reinterpret_cast<T*>(s.b[p])[Traits<T>::bh_index(n,k)]);}
template<class T,int Prod,bool SideA> __device__ __forceinline__ float transformed(Shared& s,int r,int k){
 float x[4];
 #pragma unroll
 for(int j=0;j<4;++j){
  float v=SideA?raw_a<T>(s,j,r,k):raw_b<T>(s,j,r,k);
  int e=s.inside[SideA?j%2:j/2];
  x[j]=int8_research::scale_pow2(v,SideA?-e:e);
 }
 if constexpr(SideA){
  if constexpr(Prod==3)return x[3];
  if constexpr(Prod==4)return __fadd_rn(x[2],x[3]);
  if constexpr(Prod==5)return __fsub_rn(__fadd_rn(x[2],x[3]),x[0]);
  if constexpr(Prod==6)return __fsub_rn(x[0],x[2]);
  return __fsub_rn(x[1],__fsub_rn(__fadd_rn(x[2],x[3]),x[0]));
 }else{
  if constexpr(Prod==2)return x[3];
  if constexpr(Prod==4)return __fsub_rn(x[1],x[0]);
  if constexpr(Prod==5)return __fsub_rn(x[3],__fsub_rn(x[1],x[0]));
  if constexpr(Prod==6)return __fsub_rn(x[3],x[1]);
  return __fsub_rn(__fsub_rn(x[3],__fsub_rn(x[1],x[0])),x[2]);
 }
}

__device__ __forceinline__ void reduce4(Shared& s,float* peak,int tid){
 #pragma unroll
 for(int j=0;j<4;++j){
  for(int d=16;d;d/=2)peak[j]=fmaxf(peak[j],__shfl_xor_sync(0xffffffff,peak[j],d));
  if((tid&31)==0)s.maxima[j][tid/32]=peak[j];
 }
 producer_sync();
}
template<class T> __device__ void choose_inside(Shared& s,int tid){
 float peak[4]={};
 for(int i=tid;i<4096;i+=128){int r=i/128,k=i%128;
  peak[0]=fmaxf(peak[0],fmaxf(fabsf(raw_a<T>(s,0,r,k)),fabsf(raw_a<T>(s,2,r,k))));
  peak[1]=fmaxf(peak[1],fmaxf(fabsf(raw_a<T>(s,1,r,k)),fabsf(raw_a<T>(s,3,r,k))));
 }
 for(int i=tid;i<8192;i+=128){int n=i%64,k=i/64;
  peak[2]=fmaxf(peak[2],fmaxf(fabsf(raw_b<T>(s,0,n,k)),fabsf(raw_b<T>(s,1,n,k))));
  peak[3]=fmaxf(peak[3],fmaxf(fabsf(raw_b<T>(s,2,n,k)),fabsf(raw_b<T>(s,3,n,k))));
 }
 reduce4(s,peak,tid);
 if(tid<2){
  float a=0,b=0;for(int w=0;w<4;++w){a=fmaxf(a,s.maxima[tid][w]);b=fmaxf(b,s.maxima[tid+2][w]);}
  s.inside[tid]=(a>0&&b>0)?__float2int_rn(.5f*(log2f(a)-log2f(b))):0;
 }
 producer_sync();
}
template<class T,int Prod> __device__ void pack_leaf(Shared& s,Slot& slot,int tid){
 float peak[4]={};
 auto ha=reinterpret_cast<T*>(slot.bytes+16384);auto hb=reinterpret_cast<T*>(slot.bytes);
 for(int i=tid;i<4096;i+=128){int r=i/128,k=i%128;
  float v=transformed<T,Prod,true>(s,r,k);T h=T(v);ha[Traits<T>::ah_index(r,k)]=h;peak[0]=fmaxf(peak[0],fabsf(float(h)));peak[2]=fmaxf(peak[2],fabsf(__fsub_rn(v,float(h))));
 }
 for(int i=tid;i<8192;i+=128){int n=i%64,k=i/64;
  float v=transformed<T,Prod,false>(s,n,k);T h=T(v);hb[Traits<T>::bh_index(n,k)]=h;peak[1]=fmaxf(peak[1],fabsf(float(h)));peak[3]=fmaxf(peak[3],fabsf(__fsub_rn(v,float(h))));
 }
 reduce4(s,peak,tid);
 if(tid<4){float a=0;for(int w=0;w<4;++w)a=fmaxf(a,s.maxima[tid][w]);int e=a>0?int(ceilf(log2f(a/127.f))):0;
  if(tid==0)slot.eah=e;else if(tid==1)slot.ebh=e;else if(tid==2)slot.eal=e;else slot.ebl=e;
 }
 producer_sync();
 auto qbh=reinterpret_cast<int8_t*>(slot.bytes+24576),qal=reinterpret_cast<int8_t*>(slot.bytes+32768);
 auto qbl=reinterpret_cast<int8_t*>(slot.bytes+36864),qah=reinterpret_cast<int8_t*>(slot.bytes+45056);
 for(int i=tid;i<4096;i+=128){int r=i/128,k=i%128;float h=float(ha[Traits<T>::ah_index(r,k)]);
  float l=__fsub_rn(transformed<T,Prod,true>(s,r,k),h);
  qah[Traits<T>::aq()(r,k)]=int8_t(max(-127,min(127,__float2int_rn(int8_research::scale_pow2(h,-slot.eah)))));
  qal[Traits<T>::aq()(r,k)]=int8_t(max(-127,min(127,__float2int_rn(int8_research::scale_pow2(l,-slot.eal)))));
 }
 for(int i=tid;i<8192;i+=128){int n=i%64,k=i/64;float h=float(hb[Traits<T>::bh_index(n,k)]);
  float l=__fsub_rn(transformed<T,Prod,false>(s,n,k),h);
  qbh[Traits<T>::bq()(n,k)]=int8_t(max(-127,min(127,__float2int_rn(int8_research::scale_pow2(h,-slot.ebh)))));
  qbl[Traits<T>::bq()(n,k)]=int8_t(max(-127,min(127,__float2int_rn(int8_research::scale_pow2(l,-slot.ebl)))));
 }
 asm volatile("fence.proxy.async.shared::cta;" ::: "memory");producer_sync();
}

template<class T> __device__ __forceinline__ void native_mma(uint64_t a,uint64_t b,float* d,cute::GMMA::ScaleOut scale){
 using namespace cute;
 if constexpr(std::is_same_v<T,cute::half_t>)
  SM90_64x32x16_F32F16F16_SS<GMMA::Major::MN,GMMA::Major::K>::fma(a,b,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],scale);
 else
  SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::MN,GMMA::Major::K>::fma(a,b,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],scale);
}
__device__ __forceinline__ void integer_mma(uint64_t a,uint64_t b,uint32_t* d,cute::GMMA::ScaleOut scale){
 cute::SM90_64x32x32_S32S8S8_SS_TN::fma(a,b,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],scale);
}
template<class T> __device__ __forceinline__ void native_product(uint8_t* ab,uint8_t* bb,float* d){
 using namespace cute;
 auto a=make_tensor(make_smem_ptr(reinterpret_cast<T*>(bb)),Traits<T>::bh());
 auto b=make_tensor(make_smem_ptr(reinterpret_cast<T*>(ab)),Traits<T>::ah());
 auto da=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(a,Shape<_64,_16>{}));
 auto db=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(b,Shape<_32,_16>{}));
 #pragma unroll
 for(int j=0;j<16;++j)warpgroup_fence_operand(d[j]);
 warpgroup_arrive();
 #pragma unroll
 for(int j=0;j<8;++j)native_mma<T>(da(0,0,j),db(0,0,j),d,j?GMMA::ScaleOut::One:GMMA::ScaleOut::Zero);
 warpgroup_commit_batch();warpgroup_wait<0>();
 #pragma unroll
 for(int j=0;j<16;++j)warpgroup_fence_operand(d[j]);
}
template<class T,int Prod,int Pair> __device__ __forceinline__ void correction(Slot& slot,float* d){
 using namespace cute;using Q=typename Traits<T>::Q;
 uint32_t i[16]={};
 #pragma unroll
 for(int j=0;j<16;++j)warpgroup_fence_operand(i[j]);
 warpgroup_arrive();
 constexpr bool low_a=Prod==2||(Prod!=3&&Pair==0);
 auto a=make_tensor(make_smem_ptr(reinterpret_cast<Q*>(slot.bytes+(low_a?24576:36864))),Traits<T>::bq());
 auto b=make_tensor(make_smem_ptr(reinterpret_cast<Q*>(slot.bytes+(low_a?32768:45056))),Traits<T>::aq());
 auto da=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(a,Shape<_64,_32>{}));
 auto db=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(b,Shape<_32,_32>{}));
 #pragma unroll
 for(int j=0;j<4;++j)integer_mma(da(0,0,j),db(0,0,j),i,j?GMMA::ScaleOut::One:GMMA::ScaleOut::Zero);
 warpgroup_commit_batch();warpgroup_wait<0>();
 int exponent=low_a?slot.eal+slot.ebh:slot.eah+slot.ebl;
 #pragma unroll
 for(int j=0;j<16;++j)warpgroup_fence_operand(i[j]);
 // One shared exponent per correction tile: keep the normal path contiguous.
 if(exponent>=-126&&exponent<=127){
  float scale=__int_as_float((exponent+127)<<23);
  #pragma unroll
  for(int j=0;j<16;++j)d[j]=__fmul_rn(float(static_cast<int32_t>(i[j])),scale);
 }else{
  #pragma unroll
  for(int j=0;j<16;++j)d[j]=int8_research::scale_pow2(float(static_cast<int32_t>(i[j])),exponent);
 }
}
template<int Prod> __device__ __forceinline__ void combine(float c[4][16],const float* d){
 #pragma unroll
 for(int j=0;j<16;++j){
  if constexpr(Prod==0){c[0][j]=__fadd_rn(c[0][j],d[j]);c[1][j]=__fadd_rn(c[1][j],d[j]);c[2][j]=__fadd_rn(c[2][j],d[j]);c[3][j]=__fadd_rn(c[3][j],d[j]);}
  if constexpr(Prod==1)c[0][j]=__fadd_rn(c[0][j],d[j]);
  if constexpr(Prod==5){c[1][j]=__fadd_rn(c[1][j],d[j]);c[2][j]=__fadd_rn(c[2][j],d[j]);c[3][j]=__fadd_rn(c[3][j],d[j]);}
  if constexpr(Prod==6){c[2][j]=__fadd_rn(c[2][j],d[j]);c[3][j]=__fadd_rn(c[3][j],d[j]);}
  if constexpr(Prod==4){c[1][j]=__fadd_rn(c[1][j],d[j]);c[3][j]=__fadd_rn(c[3][j],d[j]);}
  if constexpr(Prod==2)c[1][j]=__fadd_rn(c[1][j],d[j]);
  if constexpr(Prod==3)c[2][j]=__fsub_rn(c[2][j],d[j]);
 }
}
template<class T,int Prod> __device__ __forceinline__ void consume_leaf(Shared& s,int seq,int tid,float c[4][16]){
 int slot=seq%2;Barrier::wait(&s.ready[slot],seq/2%2);asm volatile("" ::: "memory");float d[16]={};
 native_product<T>(s.slot[slot].bytes+16384,s.slot[slot].bytes,d);combine<Prod>(c,d);
 correction<T,Prod,0>(s.slot[slot],d);combine<Prod>(c,d);
 if constexpr(Prod!=2&&Prod!=3){correction<T,Prod,1>(s.slot[slot],d);combine<Prod>(c,d);}
 asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");if((tid&31)==0)Barrier::arrive(&s.free[slot]);
}
template<class T> __global__ __launch_bounds__(256,1)
void kernel(const __grid_constant__ CUtensorMap A,const __grid_constant__ CUtensorMap B,float* D){
 using namespace cute;extern __shared__ __align__(1024) unsigned char buffer[];auto& s=*reinterpret_cast<Shared*>(buffer);
 int tid=threadIdx.x,m0=blockIdx.x*32,n0=blockIdx.y*64;
 if(tid==0){Barrier::init(&s.raw_ready,1);Barrier::init(&s.raw_free,4);for(int i=0;i<2;++i){Barrier::init(&s.ready[i],1);Barrier::init(&s.free[i],4);}cutlass::arch::fence_barrier_init();}
 __syncthreads();
 if(tid<128){
  for(int k=0;k<4;++k){
   if(tid==0){
    if(k){Barrier::wait(&s.raw_free,(k-1)%2);asm volatile("" ::: "memory");}
    TxBarrier::arrive_and_expect_tx(&s.raw_ready,98304);
    for(int j=0;j<4;++j){
     SM90_TMA_LOAD_3D::copy(&A,&s.raw_ready,0,s.a[j],0,m0+(j/2)*512,k*2+(j%2)*8);
     SM90_TMA_LOAD_2D::copy(&B,&s.raw_ready,0,s.b[j],n0+(j%2)*512,k*128+(j/2)*512);
    }
   }
   Barrier::wait(&s.raw_ready,k%2);asm volatile("" ::: "memory");choose_inside<T>(s,tid);
   #pragma unroll 1
   for(int leaf=0;leaf<5;++leaf){int seq=k*5+leaf,slot=seq%2;
    if(seq>=2){Barrier::wait(&s.free[slot],(seq-2)/2%2);asm volatile("" ::: "memory");}
    switch(leaf){
     case 0:pack_leaf<T,5>(s,s.slot[slot],tid);break;
     case 1:pack_leaf<T,6>(s,s.slot[slot],tid);break;
     case 2:pack_leaf<T,4>(s,s.slot[slot],tid);break;
     case 3:pack_leaf<T,2>(s,s.slot[slot],tid);break;
     default:pack_leaf<T,3>(s,s.slot[slot],tid);break;
    }
    if(tid==0)Barrier::arrive(&s.ready[slot]);
   }
  }
 }else{
  float c[4][16]={};
  for(int k=0;k<4;++k){
   Barrier::wait(&s.raw_ready,k%2);asm volatile("" ::: "memory");float d[16]={};
   native_product<T>(s.a[0],s.b[0],d);combine<0>(c,d);
   native_product<T>(s.a[1],s.b[2],d);combine<1>(c,d);
   asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");if((tid&31)==0)Barrier::arrive(&s.raw_free);
   consume_leaf<T,5>(s,k*5,tid,c);consume_leaf<T,6>(s,k*5+1,tid,c);consume_leaf<T,4>(s,k*5+2,tid,c);
   consume_leaf<T,2>(s,k*5+3,tid,c);consume_leaf<T,3>(s,k*5+4,tid,c);
  }
  #pragma unroll
  for(int j=0;j<16;++j){int mn=GMMA::CLayout_64x32{}(tid-128,j),at=(m0+mn/64)*1024+n0+mn%64;
   D[at]=c[0][j];D[at+512]=c[1][j];D[at+512*1024]=c[2][j];D[at+512*1024+512]=c[3][j];
  }
 }
}
void encode(CUtensorMap& map,const void* p,megafmm::DType dt,bool side_a){
 uint64_t dims[3]={side_a?64ULL:1024ULL,1024,16},strides[2]={2048,128};
 uint32_t box[3]={64,side_a?32U:128U,2},steps[3]={1,1,1};
 auto type=dt==megafmm::DType::BF16?CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
 auto e=cuTensorMapEncodeTiled(&map,type,side_a?3:2,const_cast<void*>(p),dims,strides,box,steps,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 if(e!=CUDA_SUCCESS)throw std::runtime_error("fused tensor map "+std::to_string(int(e)));
}
#ifndef FUSED_PROBE
void init(){
 check(cudaFuncSetAttribute(kernel<cute::half_t>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)),"fused FP shared");
 check(cudaFuncSetAttribute(kernel<cute::bfloat16_t>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)),"fused BF shared");
 check(cudaFuncSetAttribute(kernel<cute::half_t>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"fused FP carveout");
 check(cudaFuncSetAttribute(kernel<cute::bfloat16_t>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"fused BF carveout");
}
void run(megafmm::DType dt,const void* A,const void* B,float* D,void*,cudaStream_t stream,bool=true){
 CUtensorMap a,b;encode(a,A,dt,true);encode(b,B,dt,false);
 if(dt==megafmm::DType::BF16)kernel<cute::bfloat16_t><<<dim3(16,8),256,sizeof(Shared),stream>>>(a,b,D);
 else kernel<cute::half_t><<<dim3(16,8),256,sizeof(Shared),stream>>>(a,b,D);
 check(cudaPeekAtLastError(),"fused launch");
}
#endif
} // namespace fused_int8

#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include "megafmm/gemm.h"
#include <type_traits>
#include <stdexcept>
#include <string>
#include <cstdio>
#include "../mma.cuh"
#include "../inside_reference.cuh"

namespace native_fusion {
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
constexpr int Plane=512*512,RawBytes=49152,LeafBytes=135168,Slots=1;
constexpr size_t WorkBytes=1024*sizeof(int);
struct Work {int* inside;explicit Work(void* p):inside(static_cast<int*>(p)){};};
struct alignas(128) Shared {
 uint8_t raw[Slots][RawBytes],leaf[Slots][LeafBytes];
 uint64_t raw_ready[Slots],leaf_ready[Slots],leaf_free[Slots];
 int cached_inside[1024];
};
static_assert(sizeof(Shared)==188544);
inline void check(cudaError_t e,const char* at){if(e!=cudaSuccess)throw std::runtime_error(std::string(at)+": "+cudaGetErrorString(e));}
__host__ __device__ constexpr int product(int step){return step<2?step:step==2?5:step==3?6:step==4?4:step==5?2:3;}
__host__ __device__ constexpr bool low_a(int p){return p==2||p>=4;}
__host__ __device__ constexpr bool low_b(int p){return p==3||p>=4;}
__host__ __device__ constexpr int low_plane(int p){return p<4?10:p+3;}
static __device__ __noinline__ float general_pow2(float v,int e){return ldexpf(v,e);}
__device__ __forceinline__ float pow2(float v,int e){
 if(e>=-126&&e<=127)return __fmul_rn(v,__int_as_float((e+127)<<23));
 return general_pow2(v,e);
}
__device__ __forceinline__ void producer_sync(){asm volatile("bar.sync 1, 128;" ::: "memory");}
__device__ __forceinline__ void proxy_fence(){asm volatile("fence.proxy.async.shared::cta;" ::: "memory");}

// Byte-for-byte arithmetic/order of P47's exact eight-column inside stage.
template<class T> __device__ __forceinline__ void inside_columns(const T* A,const T* B,int* inside,Shared& s){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,k0=(blockIdx.y*16+blockIdx.x)*8;
 float am=0;
 #pragma unroll 4
 for(int row=tid/8;row<1024;row+=32)am=fmaxf(am,fabsf(float(A[row*1024+k0+tid%8])));
 am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,8));am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,16));
 float* ma=reinterpret_cast<float*>(s.raw[0]);float* mb=ma+64;
 if(lane<8)ma[warp*8+lane]=am;
 float bm=0;
 #pragma unroll 4
 for(int col=lane;col<1024;col+=32)bm=fmaxf(bm,fabsf(float(B[(k0+warp)*1024+col])));
 for(int step=16;step;step/=2)bm=fmaxf(bm,__shfl_xor_sync(0xffffffff,bm,step));
 if(lane==0)mb[warp]=bm;
 __syncthreads();
 if(tid<8){
  float x=0,y=mb[tid];
  #pragma unroll
  for(int w=0;w<8;++w)x=fmaxf(x,ma[w*8+tid]);
  inside[k0+tid]=(x>0&&y>0)?__float2int_rn(.5f*(log2f(x)-log2f(y))):0;
 }
}

__device__ __forceinline__ void load_raw(Shared& s,const CUtensorMap* a,const CUtensorMap* b,int k,int m0,int n0){
 using namespace cute;int slot=k%Slots;
 TxBarrier::arrive_and_expect_tx(&s.raw_ready[slot],RawBytes);
 #pragma unroll
 for(int q=0;q<4;++q){
  SM90_TMA_LOAD_2D::copy(a,&s.raw_ready[slot],0,s.raw[slot]+q*4096,k*64+(q%2)*512,n0+(q/2)*512);
  SM90_TMA_LOAD_2D::copy(b,&s.raw_ready[slot],0,s.raw[slot]+16384+q*8192,m0+(q%2)*512,k*64+(q/2)*512);
 }
}

template<bool SideA> __device__ __forceinline__ float transform(int p,float a,float b,float c,float d){
 if constexpr(SideA){
  float s1=__fadd_rn(c,d),s2=__fsub_rn(s1,a);
  switch(p){case 0:return a;case 1:return b;case 2:return __fsub_rn(b,s2);case 3:return d;case 4:return s1;case 5:return s2;default:return __fsub_rn(a,c);}
 }else{
  float t1=__fsub_rn(b,a),t2=__fsub_rn(d,t1);
  switch(p){case 0:return a;case 1:return c;case 2:return d;case 3:return __fsub_rn(t2,c);case 4:return t1;case 5:return t2;default:return __fsub_rn(d,b);}
 }
}
template<class T,int P,int L,int Count> __device__ __forceinline__ void store_native(float v,T* target,int at){
 T high=T(v);target[P*Count+at]=high;
 if constexpr(L<4){
  constexpr float scale=(std::is_same_v<T,cute::half_t> ? 2048.f : 256.f);
  target[(7+L)*Count+at]=T(__fmul_rn(__fsub_rn(v,float(high)),scale));
 }
}
template<class T,bool SideA> __device__ __forceinline__ void make_all(Shared& s,int k,const int* inside){
 using namespace cute;
 constexpr auto layout=[](){if constexpr(SideA)return tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_32,_64>{});else return tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,_64>{});}();
 constexpr int Count=SideA?2048:4096;
 const T* raw=reinterpret_cast<const T*>(s.raw[0]+(SideA?0:16384));
 T* target=reinterpret_cast<T*>(s.leaf[0]+(SideA?90112:0));
 #pragma unroll 4
 for(int i=threadIdx.x;i<Count;i+=128){
  int row=SideA?i/64:i%64,ki=SideA?i%64:i/64,at=as_position_independent_swizzle_layout(layout)(make_coord(row,ki));
  int e0=inside[k*64+ki],e1=inside[k*64+ki+512];if constexpr(SideA){e0=-e0;e1=-e1;}
  float a=pow2(float(raw[at]),e0),d=pow2(float(raw[3*Count+at]),e1);
  float b=pow2(float(raw[Count+at]),SideA?e1:e0),c=pow2(float(raw[2*Count+at]),SideA?e0:e1);
  store_native<T,0,8,Count>(a,target,at);store_native<T,1,8,Count>(SideA?b:c,target,at);
  if constexpr(SideA){
   float s1=__fadd_rn(c,d),s2=__fsub_rn(s1,a);
   store_native<T,3,8,Count>(d,target,at);store_native<T,4,0,Count>(s1,target,at);
   store_native<T,5,1,Count>(s2,target,at);store_native<T,6,2,Count>(__fsub_rn(a,c),target,at);
   store_native<T,2,3,Count>(__fsub_rn(b,s2),target,at);
  }else{
   float t1=__fsub_rn(b,a),t2=__fsub_rn(d,t1);
   store_native<T,2,8,Count>(d,target,at);store_native<T,4,0,Count>(t1,target,at);
   store_native<T,5,1,Count>(t2,target,at);store_native<T,6,2,Count>(__fsub_rn(d,b),target,at);
   store_native<T,3,3,Count>(__fsub_rn(t2,c),target,at);
  }
 }
}
template<class T> __device__ __forceinline__ void produce(Shared& s,const CUtensorMap* a,const CUtensorMap* b,const int* inside,int m0,int n0){
 if(threadIdx.x==0)load_raw(s,a,b,0,m0,n0);
 #pragma unroll 1
 for(int k=0;k<8;++k){
  Barrier::wait(&s.raw_ready[0],k%2);asm volatile("" ::: "memory");
  if(k){Barrier::wait(&s.leaf_free[0],(k-1)%2);asm volatile("" ::: "memory");}
  make_all<T,false>(s,k,inside);make_all<T,true>(s,k,inside);
  proxy_fence();Barrier::arrive(&s.leaf_ready[0]);
  // All raw reads complete before the issuer overwrites the single raw slot.
  proxy_fence();producer_sync();
  if(threadIdx.x==0&&k+1<8)load_raw(s,a,b,k+1,m0,n0);
 }
}

template<class T> __device__ __forceinline__ void consume(Shared& s,int m0,int n0,float* D){
 using namespace cute;
 constexpr auto la=tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,_64>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_32,_64>{});
 auto a=make_tensor(make_smem_ptr(reinterpret_cast<T*>(s.leaf[0])),la);
 auto b=make_tensor(make_smem_ptr(reinterpret_cast<T*>(s.leaf[0]+90112)),lb);
 auto da=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(a,Shape<_64,_16>{}));
 auto db=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(b,Shape<_32,_16>{}));
 float c[4][16]={};
 #pragma unroll 1
 for(int seq=0;seq<56;++seq){
  int p=product(seq%7);
  if(seq%7==0)Barrier::wait(&s.leaf_ready[0],(seq/7)%2);asm volatile("" ::: "memory");
  float d[16]={},r[16]={};
  #pragma unroll
  for(int j=0;j<16;++j){warpgroup_fence_operand(d[j]);warpgroup_fence_operand(r[j]);}
  // Orders both CPU reads/writes from the preceding leaf before reuse.
  warpgroup_arrive();uint64_t ao=p*8192/16,bo=p*4096/16;
  #pragma unroll
  for(int inner=0;inner<4;++inner)float_mma<T>(uint64_t(da(0,0,inner))+ao,uint64_t(db(0,0,inner))+bo,d,inner==0?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One);
  if(low_a(p)){
   #pragma unroll
   for(int inner=0;inner<4;++inner)float_mma<T>(uint64_t(da(0,0,inner))+ao,uint64_t(db(0,0,inner))+low_plane(p)*4096/16,r,inner==0?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One);
  }
  if(low_b(p)){
   #pragma unroll
   for(int inner=0;inner<4;++inner)float_mma<T>(uint64_t(da(0,0,inner))+low_plane(p)*8192/16,uint64_t(db(0,0,inner))+bo,r,(p==3&&inner==0)?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One);
  }
  warpgroup_commit_batch();warpgroup_wait<0>();
  #pragma unroll
  for(int j=0;j<16;++j){warpgroup_fence_operand(d[j]);warpgroup_fence_operand(r[j]);}
  asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
  if(seq%7==6&&(threadIdx.x&31)==0)Barrier::arrive(&s.leaf_free[0]);
  if(p>=2){
   constexpr float inverse=(std::is_same_v<T,cute::half_t> ? (1.f/2048.f) : (1.f/256.f));
   #pragma unroll
   for(int j=0;j<16;++j)d[j]=__fadd_rn(d[j],__fmul_rn(r[j],inverse));
  }
  if(p==0){
   #pragma unroll
   for(int j=0;j<16;++j){if(seq==0)c[0][j]=c[1][j]=c[2][j]=c[3][j]=d[j];else for(int q=0;q<4;++q)c[q][j]=__fadd_rn(c[q][j],d[j]);}
  }else if(p==1){
   #pragma unroll
   for(int j=0;j<16;++j)c[0][j]=__fadd_rn(c[0][j],d[j]);
  }else if(p==5){
   #pragma unroll
   for(int j=0;j<16;++j){c[1][j]=__fadd_rn(c[1][j],d[j]);c[2][j]=__fadd_rn(c[2][j],d[j]);c[3][j]=__fadd_rn(c[3][j],d[j]);}
  }else if(p==6){
   #pragma unroll
   for(int j=0;j<16;++j){c[2][j]=__fadd_rn(c[2][j],d[j]);c[3][j]=__fadd_rn(c[3][j],d[j]);}
  }else if(p==4){
   #pragma unroll
   for(int j=0;j<16;++j){c[1][j]=__fadd_rn(c[1][j],d[j]);c[3][j]=__fadd_rn(c[3][j],d[j]);}
  }else if(p==2){
   #pragma unroll
   for(int j=0;j<16;++j)c[1][j]=__fadd_rn(c[1][j],d[j]);
  }else{
   #pragma unroll
   for(int j=0;j<16;++j)c[2][j]=__fsub_rn(c[2][j],d[j]);
  }
 }
 #pragma unroll
 for(int j=0;j<16;++j){
  int mn=GMMA::CLayout_64x32{}(threadIdx.x-128,j),at=(n0+mn/64)*1024+m0+mn%64;
  D[at]=c[0][j];D[at+512]=c[1][j];D[at+512*1024]=c[2][j];D[at+512*1024+512]=c[3][j];
 }
}

template<class T> __device__ __forceinline__ void consume_probe(Shared& s,int m0,int n0,const T* a,const T* b,unsigned* errors){
 using namespace cute;
 constexpr auto la=tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,_64>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_32,_64>{});
 unsigned ae=0,be=0;int tid=threadIdx.x-128;
 #pragma unroll 1
 for(int seq=0;seq<56;++seq){
  int p=product(seq%7),k=seq/7;
  if(seq%7==0)Barrier::wait(&s.leaf_ready[0],(seq/7)%2);asm volatile("" ::: "memory");
  const uint16_t* ah=reinterpret_cast<const uint16_t*>(s.leaf[0]+90112+p*4096);
  const uint16_t* al=reinterpret_cast<const uint16_t*>(s.leaf[0]+90112+low_plane(p)*4096);
  const uint16_t* bh=reinterpret_cast<const uint16_t*>(s.leaf[0]+p*8192);
  const uint16_t* bl=reinterpret_cast<const uint16_t*>(s.leaf[0]+low_plane(p)*8192);
  const uint16_t* refa=reinterpret_cast<const uint16_t*>(a);const uint16_t* refb=reinterpret_cast<const uint16_t*>(b);
  for(int i=tid;i<2048;i+=128){
   int row=i/64,ki=i%64,at=as_position_independent_swizzle_layout(lb)(make_coord(row,ki)),g=(n0+row)*512+k*64+ki;
   ae+=ah[at]!=refa[p*Plane+g];if(low_a(p))ae+=al[at]!=refa[low_plane(p)*Plane+g];
  }
  for(int i=tid;i<4096;i+=128){
   int row=i%64,ki=i/64,at=as_position_independent_swizzle_layout(la)(make_coord(row,ki)),g=(k*64+ki)*512+m0+row;
   be+=bh[at]!=refb[p*Plane+g];if(low_b(p))be+=bl[at]!=refb[low_plane(p)*Plane+g];
  }
  asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
  if(seq%7==6&&(threadIdx.x&31)==0)Barrier::arrive(&s.leaf_free[0]);
 }
 if(ae)atomicAdd(errors,ae);if(be)atomicAdd(errors+1,be);
}

template<class T,bool Probe> __global__ __launch_bounds__(256,1)
void fused(const T* A,const T* B,const __grid_constant__ CUtensorMap map_a,const __grid_constant__ CUtensorMap map_b,
 int* inside,float* D,const T* ref_a,const T* ref_b,unsigned* errors){
 extern __shared__ __align__(128) unsigned char buffer[];auto& s=*reinterpret_cast<Shared*>(buffer);
 if(threadIdx.x==0){for(int j=0;j<Slots;++j){Barrier::init(&s.raw_ready[j],1);Barrier::init(&s.leaf_ready[j],128);Barrier::init(&s.leaf_free[j],4);}cutlass::arch::fence_barrier_init();}
 __syncthreads();inside_columns(A,B,inside,s);cooperative_groups::this_grid().sync();
 // All earlier generic reads/writes to reused raw scratch precede TMA writes.
 for(int k=threadIdx.x;k<1024;k+=256)s.cached_inside[k]=inside[k];
 proxy_fence();__syncthreads();
 int m0=blockIdx.y*64,n0=blockIdx.x*32;
 if(threadIdx.x<128)produce<T>(s,&map_a,&map_b,s.cached_inside,m0,n0);
 else {if constexpr(Probe)consume_probe<T>(s,m0,n0,ref_a,ref_b,errors);else consume<T>(s,m0,n0,D);}
}

inline void encode(CUtensorMap& map,const void* p,megafmm::DType dt,bool side_a){
 uint64_t dims[2]={1024,1024},strides[1]={2048};uint32_t box[2]={64,side_a?32u:64u},steps[2]={1,1};
 auto type=dt==megafmm::DType::BF16?CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
 auto e=cuTensorMapEncodeTiled(&map,type,2,const_cast<void*>(p),dims,strides,box,steps,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 if(e!=CUDA_SUCCESS)throw std::runtime_error("native raw tensor map "+std::to_string(int(e)));
}
template<class T,bool Probe> inline void init_one(){
 check(cudaFuncSetAttribute(fused<T,Probe>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)),"native shared");
 check(cudaFuncSetAttribute(fused<T,Probe>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"native carveout");
 int dev,supported,sms,active;check(cudaGetDevice(&dev),"native device");
 check(cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,dev),"native cooperative support");
 check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev),"native SM count");
 check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fused<T,Probe>,256,sizeof(Shared)),"native occupancy");
 printf("COOPERATIVE_INIT dtype=%s probe=%d supported=%d sm_count=%d active_blocks_per_sm=%d grid=128 shared_bytes=%zu\n",(std::is_same_v<T,cute::half_t> ? "fp16" : "bf16"),int(Probe),supported,sms,active,sizeof(Shared));
 if(!supported||active*sms<128)throw std::runtime_error("Native fusion requires128 resident CTAs");
}
inline void init(){init_one<cute::half_t,false>();init_one<cute::bfloat16_t,false>();}
template<class T,bool Probe> inline void launch(const void* A,const void* B,float* D,void* workspace,cudaStream_t stream,CUtensorMap ma,CUtensorMap mb,const T* ra=nullptr,const T* rb=nullptr,unsigned* error=nullptr){
 const T* a=static_cast<const T*>(A);const T* b=static_cast<const T*>(B);int* inside=Work(workspace).inside;
 void* args[]={&a,&b,&ma,&mb,&inside,&D,&ra,&rb,&error};
 check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(fused<T,Probe>),dim3(16,8),dim3(256),args,sizeof(Shared),stream),"native cooperative launch");
}
inline void run(megafmm::DType dt,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream,bool balance=true){
 if(!balance)throw std::invalid_argument("Native fusion requires exact inside balance");
 CUtensorMap ma,mb;encode(ma,A,dt,true);encode(mb,B,dt,false);
 if(dt==megafmm::DType::BF16)launch<cute::bfloat16_t,false>(A,B,D,workspace,stream,ma,mb);
 else launch<cute::half_t,false>(A,B,D,workspace,stream,ma,mb);
 check(cudaPeekAtLastError(),"native launch");
}
inline void diagnose_uniform_scales(const char* dtype,int pattern,int trial,Work,cudaStream_t,int stream_id=0){
 printf("NATIVE_FUSION dtype=%s pattern=%d trial=%d stream=%d diagnostic_only=1\n",dtype,pattern,trial,stream_id);
}
} // namespace native_fusion

#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <stdexcept>
#include <type_traits>
#include "pow2.cuh"

namespace int8_research {
constexpr int H=512, Plane=H*H, BK=128, BM=64, BN=32, Stages=8;
constexpr size_t HighBytes=7*Plane*2, QuantBytes=9*Plane;
// Seven main planes per side, nine INT8 planes per side, integer scale metadata.
constexpr size_t PartialBytes=2*1024*32*sizeof(float);
constexpr size_t WorkBytes=2*HighBytes+2*QuantBytes+(9*64+9*64+1024)*sizeof(int)+PartialBytes;
void check(cudaError_t e,const char* where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
struct Work {
 void *ha,*hb;int8_t *qa,*qb;int *ea,*eb,*inside;float* partial;
 explicit Work(void* w){
  auto p=static_cast<char*>(w);ha=p;p+=HighBytes;hb=p;p+=HighBytes;
  qa=reinterpret_cast<int8_t*>(p);p+=QuantBytes;qb=reinterpret_cast<int8_t*>(p);p+=QuantBytes;
  ea=reinterpret_cast<int*>(p);p+=9*64*sizeof(int);eb=reinterpret_cast<int*>(p);p+=9*64*sizeof(int);
  inside=reinterpret_cast<int*>(p);p+=1024*sizeof(int);partial=reinterpret_cast<float*>(p);
 }
};

// Reference-quality first hardware implementation. Its cost is included in run.
template<class T> __global__ void inside_scale(const T* A,const T* B,int* exponent){
 int k=blockIdx.x,tid=threadIdx.x;
 float amax=0,bmax=0;
 for(int r=tid;r<1024;r+=256){
  amax=fmaxf(amax,fabsf(float(A[r*1024+k])));
  bmax=fmaxf(bmax,fabsf(float(B[k*1024+r])));
 }
 __shared__ float ma[8],mb[8];
 for(int s=16;s;s/=2){amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,s));bmax=fmaxf(bmax,__shfl_xor_sync(0xffffffff,bmax,s));}
 if((tid&31)==0){ma[tid/32]=amax;mb[tid/32]=bmax;}__syncthreads();
 if(tid==0){
  float x=0,y=0;for(int j=0;j<8;++j){x=fmaxf(x,ma[j]);y=fmaxf(y,mb[j]);}
  exponent[k]=(x>0 && y>0)?__float2int_rn(.5f*(log2f(x)-log2f(y))):0;
 }
}

// Exact per-column maxima with coalesced input loads. The original inside_scale
// above remains a diagnostics-only reference; preprocess uses these two kernels.
template<class T> __global__ void inside_partial(const T* A,const T* B,float* partial){
 int slab=blockIdx.x,chunk=blockIdx.y,tid=threadIdx.x,lane=tid%32,warp=tid/32;
 float am=0;int k=slab*128+tid%128;
 for(int row=tid/128;row<32;row+=2)
  am=fmaxf(am,fabsf(float(A[(chunk*32+row)*1024+k])));
 __shared__ float aa[256],bb[128];aa[tid]=am;
 for(int kr=warp;kr<128;kr+=8){
  float bm=fabsf(float(B[(slab*128+kr)*1024+chunk*32+lane]));
  for(int step=16;step;step/=2)bm=fmaxf(bm,__shfl_xor_sync(0xffffffff,bm,step));
  if(lane==0)bb[kr]=bm;
 }
 __syncthreads();
 if(tid<128){int at=chunk*1024+slab*128+tid;partial[at]=fmaxf(aa[tid],aa[tid+128]);partial[1024*32+at]=bb[tid];}
}
__global__ void inside_finalize(const float* partial,int* exponent){
 int tid=threadIdx.x,k=blockIdx.x*8+tid/32,lane=tid%32;
 float am=partial[lane*1024+k],bm=partial[1024*32+lane*1024+k];
 for(int step=16;step;step/=2){am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,step));bm=fmaxf(bm,__shfl_xor_sync(0xffffffff,bm,step));}
 if(lane==0)exponent[k]=(am>0&&bm>0)?__float2int_rn(.5f*(log2f(am)-log2f(bm))):0;
}

// Input is a maximum of rounded native FP16/BF16 magnitudes, initialized at +0.
// Very tiny values and exceptional bit patterns retain the original expression.
__device__ __forceinline__ int native_high_exponent(float peak){
 unsigned u=__float_as_uint(peak);
 if(u==0)return 0;
 if(u>=0x04800000u&&u<=0x7f7fffffu)
  return int(u>>23)-133+int((u&0x7fffffu)>0x7e0000u);
 return peak>0?int(ceilf(log2f(peak/127.f))):0;
}

constexpr int PackThreads=768,PackBlocks=128;
struct alignas(128) DirectPackShared {
 union {
  struct {float ma[64],mb[8];} inside;
  struct {float maxima[5][24];int exponent[5];int8_t transpose[9][512*12];} pack;
 };
};
static_assert(sizeof(DirectPackShared)==55808);
static_assert(WorkBytes==12329472);
template<bool SideA> using PairPackShared=DirectPackShared;
static __device__ __noinline__ float pair_general_pow2(float x,int e){return ldexpf(x,e);}
__device__ __forceinline__ float pair_pow2(float x,int e){
 if(e>=-126&&e<=127)return __fmul_rn(x,__int_as_float((e+127)<<23));
 return pair_general_pow2(x,e);
}
struct Tuple {float a,b,c,d,u,v,w,z;};
template<class T,bool SideA,bool Balance> __device__ __forceinline__ Tuple tuple(
 const T* X,const int* ex,int r,int c){
 int at=r*1024+c;
 Tuple x{float(X[at]),float(X[at+512]),float(X[at+512*1024]),float(X[at+512*1024+512]),0,0,0,0};
 if constexpr(Balance){
  int e0=SideA?-ex[c]:ex[r],e1=SideA?-ex[c+512]:ex[r+512];
  x.a=pair_pow2(x.a,e0);x.d=pair_pow2(x.d,e1);
  if constexpr(SideA){x.b=pair_pow2(x.b,e1);x.c=pair_pow2(x.c,e0);}
  else{x.b=pair_pow2(x.b,e0);x.c=pair_pow2(x.c,e1);}
 }
 if constexpr(SideA){x.u=__fadd_rn(x.c,x.d);x.v=__fsub_rn(x.u,x.a);x.w=__fsub_rn(x.a,x.c);x.z=__fsub_rn(x.b,x.v);}
 else{x.u=__fsub_rn(x.b,x.a);x.v=__fsub_rn(x.d,x.u);x.w=__fsub_rn(x.d,x.b);x.z=__fsub_rn(x.v,x.c);}
 return x;
}
template<class T,bool SideA,int QP> __device__ __forceinline__ void quant_store(
 float v,int i,int r,int c,T* high,int8_t* q,DirectPackShared& s){
 constexpr int Bits=std::is_same_v<T,cute::half_t> ? 11:8;
 float hv=float(T(v));int eh=s.pack.exponent[QP<4?QP:4];
 int qh=max(-127,min(127,__float2int_rn(pair_pow2(hv,-eh))));
 int target=SideA?r*512+c:c*512+r;
 if constexpr(SideA)q[QP*Plane+target]=int8_t(qh);
 else s.pack.transpose[QP][(i/8)*12+i%8]=int8_t(qh);
 if constexpr(QP<4){
  int ql=max(-127,min(127,__float2int_rn(pair_pow2(__fsub_rn(v,hv),Bits-eh))));
  if constexpr(SideA)q[(QP+4)*Plane+target]=int8_t(ql);
  else s.pack.transpose[QP+4][(i/8)*12+i%8]=int8_t(ql);
 }
}
template<class T,bool SideA,bool Balance> __device__ __forceinline__ void direct_pack(
 const T* X,T* high,int8_t* q,int* exponent,const int* inside,DirectPackShared& s,int task){
 int tid=threadIdx.x;float peak[5]={};
 for(int i=tid;i<4096;i+=PackThreads){
  int r=SideA?task*8+i/512:i/8,c=SideA?i%512:task*8+i%8,at=r*512+c;
  Tuple x=tuple<T,SideA,Balance>(X,inside,r,c);
  high[0*Plane+at]=T(x.a);high[1*Plane+at]=T(SideA?x.b:x.c);high[(SideA?3:2)*Plane+at]=T(x.d);
  high[4*Plane+at]=T(x.u);high[5*Plane+at]=T(x.v);high[6*Plane+at]=T(x.w);high[(SideA?2:3)*Plane+at]=T(x.z);
  peak[0]=fmaxf(peak[0],fabsf(float(T(x.u))));peak[1]=fmaxf(peak[1],fabsf(float(T(x.v))));
  peak[2]=fmaxf(peak[2],fabsf(float(T(x.w))));peak[3]=fmaxf(peak[3],fabsf(float(T(x.z))));peak[4]=fmaxf(peak[4],fabsf(float(T(x.d))));
 }
 #pragma unroll
 for(int p=0;p<5;++p){
  for(int delta=16;delta;delta/=2)peak[p]=fmaxf(peak[p],__shfl_xor_sync(0xffffffff,peak[p],delta));
  if((tid&31)==0)s.pack.maxima[p][tid/32]=peak[p];
 }
 __syncthreads();
 if(tid<5){
  float p=0;for(int w=0;w<24;++w)p=fmaxf(p,s.pack.maxima[tid][w]);
  int e=native_high_exponent(p);s.pack.exponent[tid]=e;
  constexpr int Bits=std::is_same_v<T,cute::half_t> ? 11:8;
  if(tid<4){exponent[tid*64+task]=e;exponent[(tid+4)*64+task]=e-Bits;}
  else exponent[8*64+task]=e;
 }
 __syncthreads();
 for(int i=tid;i<4096;i+=PackThreads){
  int r=SideA?task*8+i/512:i/8,c=SideA?i%512:task*8+i%8;
  Tuple x=tuple<T,SideA,Balance>(X,inside,r,c);
  quant_store<T,SideA,0>(x.u,i,r,c,high,q,s);quant_store<T,SideA,1>(x.v,i,r,c,high,q,s);
  quant_store<T,SideA,2>(x.w,i,r,c,high,q,s);quant_store<T,SideA,3>(x.z,i,r,c,high,q,s);
  quant_store<T,SideA,8>(x.d,i,r,c,high,q,s);
 }
 if constexpr(!SideA){
  __syncthreads();
  for(int i=tid;i<4096;i+=PackThreads){
   int c=i/512,r=i%512;
   #pragma unroll
   for(int p=0;p<9;++p)q[p*Plane+(task*8+c)*512+r]=s.pack.transpose[p][r*12+c];
  }
 }
 asm volatile("fence.proxy.async.global;" ::: "memory");
}
template<class T,bool SideA,bool Balance> __global__ __launch_bounds__(768,1)
void pack(const T* X,T* high,int8_t* q,int* exponent,const int* inside){
 extern __shared__ __align__(128) unsigned char buffer[];
 direct_pack<T,SideA,Balance>(X,high,q,exponent,inside,*reinterpret_cast<DirectPackShared*>(buffer),blockIdx.x);
}
template<class T> __global__ __launch_bounds__(768,1)
void preprocess_cooperative_columns(const T* A,const T* B,T* ha,T* hb,int8_t* qa,int8_t* qb,
 int* ea,int* eb,int* inside,float* partial){
 extern __shared__ __align__(128) unsigned char buffer[];
 auto& s=*reinterpret_cast<DirectPackShared*>(buffer);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,k0=blockIdx.x*8;
 if(tid<256){
  float am=0;
  #pragma unroll 4
  for(int row=tid/8;row<1024;row+=32)am=fmaxf(am,fabsf(float(A[row*1024+k0+tid%8])));
  am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,8));am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,16));
  if(lane<8)s.inside.ma[warp*8+lane]=am;
  float bm=0;
  #pragma unroll 4
  for(int col=lane;col<1024;col+=32)bm=fmaxf(bm,fabsf(float(B[(k0+warp)*1024+col])));
  for(int delta=16;delta;delta/=2)bm=fmaxf(bm,__shfl_xor_sync(0xffffffff,bm,delta));
  if(lane==0)s.inside.mb[warp]=bm;
 }
 __syncthreads();
 if(tid<8){
  float am=0,bm=s.inside.mb[tid];for(int w=0;w<8;++w)am=fmaxf(am,s.inside.ma[w*8+tid]);
  inside[k0+tid]=(am>0&&bm>0)?__float2int_rn(.5f*(log2f(am)-log2f(bm))):0;
 }
 cooperative_groups::this_grid().sync();
 if(blockIdx.x<64)direct_pack<T,false,true>(B,hb,qb,eb,inside,s,blockIdx.x);
 else direct_pack<T,true,true>(A,ha,qa,ea,inside,s,blockIdx.x-64);
}
template<class T> void init_cooperative_preprocess(){
 check(cudaFuncSetAttribute(preprocess_cooperative_columns<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(DirectPackShared)),"direct fine prep shared");
 int dev=0,supported=0,sms=0,active=0;check(cudaGetDevice(&dev),"direct device");
 check(cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,dev),"cooperative support");
 check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev),"SM count");
 check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,preprocess_cooperative_columns<T>,768,sizeof(DirectPackShared)),"direct fine prep capacity");
 printf("COOPERATIVE_INIT dtype=%s supported=%d sm_count=%d active_blocks_per_sm=%d grid=128 threads=768 shared_bytes=%zu\n",std::is_same_v<T,cute::half_t> ? "fp16":"bf16",supported,sms,active,sizeof(DirectPackShared));
 if(!supported||active*sms<128)throw std::runtime_error("Insufficient direct fine cooperative capacity");
}
template<class T,bool Balance> void preprocess(const T* A,const T* B,Work w,cudaStream_t stream){
 if constexpr(Balance){
  T* ha=static_cast<T*>(w.ha);T* hb=static_cast<T*>(w.hb);
  void* args[]={&A,&B,&ha,&hb,&w.qa,&w.qb,&w.ea,&w.eb,&w.inside,&w.partial};
  check(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(preprocess_cooperative_columns<T>),dim3(128),dim3(768),args,sizeof(DirectPackShared),stream),"direct fine cooperative prep");
 }else{
  pack<T,true,false><<<64,768,sizeof(DirectPackShared),stream>>>(A,static_cast<T*>(w.ha),w.qa,w.ea,w.inside);
  pack<T,false,false><<<64,768,sizeof(DirectPackShared),stream>>>(B,static_cast<T*>(w.hb),w.qb,w.eb,w.inside);
 }
 check(cudaPeekAtLastError(),"direct fine prep launch");
}
} // namespace int8_research

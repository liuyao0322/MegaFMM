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
constexpr int H=512, Plane=H*H, BK=128, BM=64, BN=32, Stages=4;
constexpr size_t HighBytes=7*Plane*2, QuantBytes=9*Plane*2;
// Seven native main planes and nine BF16 correction planes per side.
// Historical integer metadata remains allocated but is neither read nor written.
constexpr size_t PartialBytes=2*1024*32*sizeof(float);
constexpr size_t WorkBytes=2*HighBytes+2*QuantBytes+(9*32+9*64+1024)*sizeof(int)+PartialBytes;
void check(cudaError_t e,const char* where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
struct Work {
 void *ha,*hb;cute::bfloat16_t *qa,*qb;int *ea,*eb,*inside;float* partial;
 explicit Work(void* w){
  auto p=static_cast<char*>(w);ha=p;p+=HighBytes;hb=p;p+=HighBytes;
  qa=reinterpret_cast<cute::bfloat16_t*>(p);p+=QuantBytes;qb=reinterpret_cast<cute::bfloat16_t*>(p);p+=QuantBytes;
  ea=reinterpret_cast<int*>(p);p+=9*32*sizeof(int);eb=reinterpret_cast<int*>(p);p+=9*64*sizeof(int);
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

template<class T,bool SideA,bool Balance,bool FastBalance=false> __device__ __forceinline__ float value(const T* X,const int* ex,int r,int c,int plane,float scale0=0,float scale1=0){
 int at=r*1024+c;
 float x0=float(X[at]),x1=float(X[at+512]),x2=float(X[at+512*1024]),x3=float(X[at+512*1024+512]);
 if constexpr(Balance){
  if constexpr(SideA){
   if constexpr(FastBalance){
    x0=__fmul_rn(x0,scale0);x2=__fmul_rn(x2,scale0);
    x1=__fmul_rn(x1,scale1);x3=__fmul_rn(x3,scale1);
   }else{
    x0=scale_pow2(x0,-ex[c]);x2=scale_pow2(x2,-ex[c]);
    x1=scale_pow2(x1,-ex[c+512]);x3=scale_pow2(x3,-ex[c+512]);
   }
  }else{
   if constexpr(FastBalance){
    x0=__fmul_rn(x0,scale0);x1=__fmul_rn(x1,scale0);
    x2=__fmul_rn(x2,scale1);x3=__fmul_rn(x3,scale1);
   }else{
    x0=scale_pow2(x0,ex[r]);x1=scale_pow2(x1,ex[r]);
    x2=scale_pow2(x2,ex[r+512]);x3=scale_pow2(x3,ex[r+512]);
   }
  }
 }
 if constexpr(SideA){
  float s1=__fadd_rn(x2,x3),s2=__fsub_rn(s1,x0);
  switch(plane){case 0:return x0;case 1:return x1;case 2:return __fsub_rn(x1,s2);case 3:return x3;
   case 4:return s1;case 5:return s2;default:return __fsub_rn(x0,x2);}
 }else{
  float t1=__fsub_rn(x1,x0),t2=__fsub_rn(x3,t1);
  switch(plane){case 0:return x0;case 1:return x2;case 2:return x3;case 3:return __fsub_rn(t2,x2);
   case 4:return t1;case 5:return t2;default:return __fsub_rn(x3,x1);}
 }
}

// Paired FP32 transforms stay shared. BF16 correction B is MN-major and
// correction A is K-major, so neither side needs a physical transpose.
template<bool SideA> struct PairPackShared {
 float values[2][SideA?8192:4096];
};
static __device__ __noinline__ float pair_general_pow2(float x,int e){return ldexpf(x,e);}
__device__ __forceinline__ float pair_pow2(float x,int e){
 if(e>=-126&&e<=127)return __fmul_rn(x,__int_as_float((e+127)<<23));
 return pair_general_pow2(x,e);
}

template<class T,bool SideA,int NativePlane,int QP> __device__ __forceinline__
void pair_store(int which,T* high,cute::bfloat16_t* q,int* exp,
 PairPackShared<SideA>& s,int tile,int r0,int c0){
 constexpr int Rows=SideA?64:128,Cols=SideA?128:32,Size=Rows*Cols,PT=Size/256;
 int tid=threadIdx.x;
 #pragma unroll 4
 for(int j=0;j<PT;++j){
  int i=tid+256*j,at=(r0+i/Cols)*512+c0+i%Cols;
  float v=s.values[which][i];T native=T(v);float high_value=float(native);
  high[NativePlane*Plane+at]=native;
  cute::bfloat16_t qh=cute::bfloat16_t(high_value);
  cute::bfloat16_t ql=cute::bfloat16_t(__fsub_rn(v,high_value));
  // BF16 supports MN-major B, so retain original row-major coordinates.
  q[QP*Plane+at]=qh;if constexpr(QP<4)q[(QP+4)*Plane+at]=ql;
 }

}
template<class T,bool SideA,int NP0,int QP0,int NP1,int QP1>
__device__ __forceinline__ void pair_store_both(T* high,cute::bfloat16_t* q,int* exp,
 PairPackShared<SideA>& s,int tile,int r0,int c0){
 pair_store<T,SideA,NP0,QP0>(0,high,q,exp,s,tile,r0,c0);
 pair_store<T,SideA,NP1,QP1>(1,high,q,exp,s,tile,r0,c0);
}

template<class T,bool SideA,bool Balance,int Group> __device__ __forceinline__
void pack_pair(const T* __restrict__ X,T* __restrict__ high,cute::bfloat16_t* __restrict__ q,int* __restrict__ exp,
 const int* __restrict__ inside,PairPackShared<SideA>& s){
 constexpr int Rows=SideA?64:128,Cols=SideA?128:32,PT=Rows*Cols/256,Tiles=(512/Rows)*(512/Cols);
 int tile=blockIdx.x%Tiles,tid=threadIdx.x,r0=tile/(512/Cols)*Rows,c0=tile%(512/Cols)*Cols;
 auto load=[&](auto fast,float sf0,float sf1){
  #pragma unroll 4
  for(int j=0;j<PT;++j){
   int i=tid+256*j,r=r0+i/Cols,c=c0+i%Cols,at=r*1024+c;
   float a=float(X[at]),b=float(X[at+512]),cc=float(X[at+512*1024]),d=float(X[at+512*1024+512]);
   if constexpr(Balance){
    if constexpr(decltype(fast)::value){
     float f0=sf0,f1=sf1;if constexpr(!SideA){f0=__shfl_sync(0xffffffff,sf0,j);f1=__shfl_sync(0xffffffff,sf1,j);}
     a=__fmul_rn(a,f0);d=__fmul_rn(d,f1);
     if constexpr(SideA){b=__fmul_rn(b,f1);cc=__fmul_rn(cc,f0);}else{b=__fmul_rn(b,f0);cc=__fmul_rn(cc,f1);}
    }else{
     int e0=SideA?-inside[c]:inside[r],e1=SideA?-inside[c+512]:inside[r+512];
     a=pair_pow2(a,e0);d=pair_pow2(d,e1);
     if constexpr(SideA){b=pair_pow2(b,e1);cc=pair_pow2(cc,e0);}else{b=pair_pow2(b,e0);cc=pair_pow2(cc,e1);}
    }
   }
   if constexpr(Group==2){
    high[r*512+c]=T(a);high[Plane+r*512+c]=T(SideA?b:cc);s.values[0][i]=d;
   }else if constexpr(SideA){
    float s1=__fadd_rn(cc,d),s2=__fsub_rn(s1,a);
    if constexpr(Group==0){s.values[0][i]=s1;s.values[1][i]=s2;}else{s.values[0][i]=__fsub_rn(a,cc);s.values[1][i]=__fsub_rn(b,s2);}
   }else{
    float t1=__fsub_rn(b,a),t2=__fsub_rn(d,t1);
    if constexpr(Group==0){s.values[0][i]=t1;s.values[1][i]=t2;}else{s.values[0][i]=__fsub_rn(d,b);s.values[1][i]=__fsub_rn(t2,cc);}
   }
  }
 };
 if constexpr(Balance){
  int pos=SideA?c0+tid%Cols:r0+tid/32+8*(tid%16);
  int e0=SideA?-inside[pos]:inside[pos],e1=SideA?-inside[pos+512]:inside[pos+512];
  bool normal=e0>=-126&&e0<=127&&e1>=-126&&e1<=127;if constexpr(!SideA)normal=__all_sync(0xffffffff,normal);
  if(normal)load(std::true_type{},__int_as_float((e0+127)<<23),__int_as_float((e1+127)<<23));else load(std::false_type{},0,0);
 }else load(std::false_type{},0,0);
 __syncthreads();
 if constexpr(Group==0)pair_store_both<T,SideA,4,0,5,1>(high,q,exp,s,tile,r0,c0);
 else if constexpr(Group==1)pair_store_both<T,SideA,6,2,(SideA?2:3),3>(high,q,exp,s,tile,r0,c0);
 else pair_store<T,SideA,(SideA?3:2),8>(0,high,q,exp,s,tile,r0,c0);
 asm volatile("fence.proxy.async.global;" ::: "memory");
}

template<class T,bool SideA,bool Balance> __global__ __launch_bounds__(256,1)
void pack(const T* __restrict__ X,T* __restrict__ high,cute::bfloat16_t* __restrict__ q,int* __restrict__ exp,const int* __restrict__ inside){
 extern __shared__ __align__(128) unsigned char scratch_buffer[];
 auto& scratch=*reinterpret_cast<PairPackShared<SideA>*>(scratch_buffer);
 int group=blockIdx.x/(SideA?32:64);
 if(group==0)pack_pair<T,SideA,Balance,0>(X,high,q,exp,inside,scratch);
 else if(group==1)pack_pair<T,SideA,Balance,1>(X,high,q,exp,inside,scratch);
 else pack_pair<T,SideA,Balance,2>(X,high,q,exp,inside,scratch);
}

// B occupies0..191, A192..287.192 is divisible by both tile counts,
// so pack_pair's blockIdx modulo Tiles still selects the original tile.
template<class T,bool Balance> __global__ __launch_bounds__(256,1)
void pack_joint(const T* __restrict__ A,const T* __restrict__ B,
 T* __restrict__ ha,T* __restrict__ hb,cute::bfloat16_t* __restrict__ qa,cute::bfloat16_t* __restrict__ qb,
 int* __restrict__ ea,int* __restrict__ eb,const int* __restrict__ inside){
 extern __shared__ __align__(128) unsigned char scratch_buffer[];
 if(blockIdx.x<192){
  auto& s=*reinterpret_cast<PairPackShared<false>*>(scratch_buffer);
  int group=blockIdx.x/64;
  if(group==0)pack_pair<T,false,Balance,0>(B,hb,qb,eb,inside,s);
  else if(group==1)pack_pair<T,false,Balance,1>(B,hb,qb,eb,inside,s);
  else pack_pair<T,false,Balance,2>(B,hb,qb,eb,inside,s);
 }else{
  auto& s=*reinterpret_cast<PairPackShared<true>*>(scratch_buffer);
  int group=(blockIdx.x-192)/32;
  if(group==0)pack_pair<T,true,Balance,0>(A,ha,qa,ea,inside,s);
  else if(group==1)pack_pair<T,true,Balance,1>(A,ha,qa,ea,inside,s);
  else pack_pair<T,true,Balance,2>(A,ha,qa,ea,inside,s);
 }
}
static_assert(sizeof(PairPackShared<true>)>=sizeof(PairPackShared<false>));


// All288CTAs participate in the grid barrier. The first128CTAs each finish
// all8column exponents before the barrier permits inside reads/scratch reuse.
template<class T> __global__ __launch_bounds__(256,1)
void preprocess_cooperative_columns(const T* __restrict__ A,const T* __restrict__ B,
 T* __restrict__ ha,T* __restrict__ hb,cute::bfloat16_t* __restrict__ qa,cute::bfloat16_t* __restrict__ qb,
 int* __restrict__ ea,int* __restrict__ eb,int* __restrict__ inside,float* __restrict__ partial){
 constexpr bool Balance=true;
 extern __shared__ __align__(128) unsigned char scratch_buffer[];
 auto grid=cooperative_groups::this_grid();
 if(blockIdx.x<128){
  int tid=threadIdx.x,lane=tid%32,warp=tid/32,k0=blockIdx.x*8;
  float am=0;
  #pragma unroll 4
  for(int row=tid/8;row<1024;row+=32)
   am=fmaxf(am,fabsf(float(A[row*1024+k0+tid%8])));
  am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,8));
  am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,16));
  float* ma=reinterpret_cast<float*>(scratch_buffer);float* mb=ma+64;
  if(lane<8)ma[warp*8+lane]=am;
  float bm=0;
  #pragma unroll 4
  for(int col=lane;col<1024;col+=32)
   bm=fmaxf(bm,fabsf(float(B[(k0+warp)*1024+col])));
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
 grid.sync();

 if(blockIdx.x<192){
  auto& s=*reinterpret_cast<PairPackShared<false>*>(scratch_buffer);
  int group=blockIdx.x/64;
  if(group==0)pack_pair<T,false,Balance,0>(B,hb,qb,eb,inside,s);
  else if(group==1)pack_pair<T,false,Balance,1>(B,hb,qb,eb,inside,s);
  else pack_pair<T,false,Balance,2>(B,hb,qb,eb,inside,s);
 }else{
  auto& s=*reinterpret_cast<PairPackShared<true>*>(scratch_buffer);
  int group=(blockIdx.x-192)/32;
  if(group==0)pack_pair<T,true,Balance,0>(A,ha,qa,ea,inside,s);
  else if(group==1)pack_pair<T,true,Balance,1>(A,ha,qa,ea,inside,s);
  else pack_pair<T,true,Balance,2>(A,ha,qa,ea,inside,s);
 }
}

template<class T> void init_cooperative_preprocess(){
 constexpr size_t smem=sizeof(PairPackShared<true>);
 check(cudaFuncSetAttribute(preprocess_cooperative_columns<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem),"cooperative shared");
 int device=0,supported=0,sms=0,active=0;
 check(cudaGetDevice(&device),"cooperative device");
 check(cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,device),"cooperative support");
 check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"cooperative SM count");
 check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,preprocess_cooperative_columns<T>,256,smem),"cooperative occupancy");
 printf("COOPERATIVE_INIT dtype=%s supported=%d sm_count=%d active_blocks_per_sm=%d grid=288 shared_bytes=%zu\n",
  (std::is_same_v<T,cute::half_t> ? "fp16" : "bf16"),supported,sms,active,smem);
 if(!supported||active*sms<288)throw std::runtime_error("Cooperative preprocess requires resident capacity for288CTAs");
}

template<class T,bool Balance> void preprocess(const T* A,const T* B,Work w,cudaStream_t stream){
 if constexpr(Balance){
  T* ha=static_cast<T*>(w.ha);T* hb=static_cast<T*>(w.hb);
  void* args[]={&A,&B,&ha,&hb,&w.qa,&w.qb,&w.ea,&w.eb,&w.inside,&w.partial};
  check(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(preprocess_cooperative_columns<T>),dim3(288),dim3(256),args,sizeof(PairPackShared<true>),stream),"cooperative preprocess launch");
 }else{
  pack_joint<T,false><<<288,256,sizeof(PairPackShared<true>),stream>>>(
   A,B,static_cast<T*>(w.ha),static_cast<T*>(w.hb),w.qa,w.qb,w.ea,w.eb,w.inside);
 }
 check(cudaPeekAtLastError(),"INT8 preprocess");
}
} // namespace int8_research

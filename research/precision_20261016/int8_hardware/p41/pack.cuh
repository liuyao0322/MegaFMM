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
constexpr size_t WorkBytes=2*HighBytes+2*QuantBytes+(9*32+9*64+1024)*sizeof(int)+PartialBytes;
void check(cudaError_t e,const char* where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
struct Work {
 void *ha,*hb;int8_t *qa,*qb;int *ea,*eb,*inside;float* partial;
 explicit Work(void* w){
  auto p=static_cast<char*>(w);ha=p;p+=HighBytes;hb=p;p+=HighBytes;
  qa=reinterpret_cast<int8_t*>(p);p+=QuantBytes;qb=reinterpret_cast<int8_t*>(p);p+=QuantBytes;
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

template<class T,bool SideA,bool Balance,int PlaneId> __device__ __forceinline__
void pack_plane(const T* __restrict__ X,T* __restrict__ high,int8_t* __restrict__ q,int* __restrict__ exp,const int* __restrict__ inside){
 constexpr int Rows=SideA?64:128,Cols=SideA?128:32,Size=Rows*Cols,PerThread=Size/256;
 constexpr int ColTiles=512/Cols,Tiles=(512/Rows)*ColTiles;
 constexpr int plane=PlaneId;
 int tile=blockIdx.x%Tiles,r0=(tile/ColTiles)*Rows,c0=(tile%ColTiles)*Cols,tid=threadIdx.x;
 constexpr int RawPlane=SideA?3:2;
 constexpr bool quant=plane>=4 || plane==(SideA?2:3) || plane==RawPlane;
 constexpr int qp=plane==RawPlane?8:plane>=4?plane-4:3;
 float vh[PerThread],vl[PerThread],hi=0,lo=0;
 auto load_values=[&](auto fast,float scale0,float scale1){
  #pragma unroll
  for(int j=0;j<PerThread;++j){
   int i=tid+256*j,r=r0+i/Cols,c=c0+i%Cols;
   float element_scale0=scale0,element_scale1=scale1;
   if constexpr(!SideA&&Balance&&decltype(fast)::value){
    // Each B warp visits rows warp+8*j; lane j owns that row's factors.
    element_scale0=__shfl_sync(0xffffffff,scale0,j);
    element_scale1=__shfl_sync(0xffffffff,scale1,j);
   }
   float v=value<T,SideA,Balance,decltype(fast)::value>(X,inside,r,c,plane,element_scale0,element_scale1);
   T h=T(v);high[plane*Plane+r*512+c]=h;
   vh[j]=float(h);vl[j]=__fsub_rn(v,float(h));
   hi=fmaxf(hi,fabsf(vh[j]));lo=fmaxf(lo,fabsf(vl[j]));
  }
 };
 if constexpr(SideA&&Balance){
  // Cols=128 and the per-thread stride256 keep c invariant over the element loop.
  int c=c0+tid%Cols,e0=-inside[c],e1=-inside[c+512];
  if(e0>=-126&&e0<=127&&e1>=-126&&e1<=127)
   load_values(std::true_type{},__int_as_float((e0+127)<<23),__int_as_float((e1+127)<<23));
  else load_values(std::false_type{},0,0);
 }else if constexpr(!SideA&&Balance){
  // Sixteen K rows per warp. Cache two factors per lane, rather than arrays
  // per thread, so register usage does not grow by 32 persistent factors.
  int row=r0+tid/32+8*(tid%16),e0=inside[row],e1=inside[row+512];
  bool normal=__all_sync(0xffffffff,e0>=-126&&e0<=127&&e1>=-126&&e1<=127);
  if(normal)load_values(std::true_type{},__int_as_float((e0+127)<<23),__int_as_float((e1+127)<<23));
  else load_values(std::false_type{},0,0);
 }else load_values(std::false_type{},0,0);
 if(!quant){asm volatile("fence.proxy.async.global;" ::: "memory");return;}
 __shared__ float maxima[2][8];__shared__ int exponents[2];
 for(int s=16;s;s/=2){hi=fmaxf(hi,__shfl_xor_sync(0xffffffff,hi,s));lo=fmaxf(lo,__shfl_xor_sync(0xffffffff,lo,s));}
 if((tid&31)==0){maxima[0][tid/32]=hi;maxima[1][tid/32]=lo;}__syncthreads();
 if(tid<2){
  // Relative residual scales make each two-sided correction pair share
  // one exponent. Both lanes independently use the same high maximum.
  float peak=0;for(int j=0;j<8;++j)peak=fmaxf(peak,maxima[0][j]);
  constexpr int LowBits=(std::is_same_v<T, cute::half_t> ? 11 : 8);
  int e=(peak>0?int(ceilf(log2f(peak/127.f))):0)-(tid?LowBits:0);
  exponents[tid]=e;if(!tid || qp<4)exp[(qp+(tid?4:0))*Tiles+tile]=e;
 }
 __syncthreads();
 // B's INT8 form is physically transposed so both integer MMA operands are K major.
 // A36-byte B row has9 words, coprime with32 shared banks.
 constexpr int TransposeStride=Cols+4;
 __shared__ int8_t transpose[SideA?1:2][SideA?1:Rows*TransposeStride];
 auto store_quant=[&](int j,int qh,int ql){
  int i=tid+256*j,r=r0+i/Cols,c=c0+i%Cols;
  qh=max(-127,min(127,qh));ql=max(-127,min(127,ql));
  if constexpr(SideA){q[qp*Plane+r*512+c]=int8_t(qh);if(qp<4)q[(qp+4)*Plane+r*512+c]=int8_t(ql);}
  else{int at=(i/Cols)*TransposeStride+i%Cols;transpose[0][at]=int8_t(qh);if(qp<4)transpose[1][at]=int8_t(ql);}
 };
 int eh=exponents[0],el=exponents[1];
 if(eh>=-127&&eh<=126&&el>=-127&&el<=126){
  float sh=__int_as_float((127-eh)<<23),sl=__int_as_float((127-el)<<23);
  #pragma unroll
  for(int j=0;j<PerThread;++j)store_quant(j,__float2int_rn(__fmul_rn(vh[j],sh)),__float2int_rn(__fmul_rn(vl[j],sl)));
 }else{
  #pragma unroll
  for(int j=0;j<PerThread;++j)store_quant(j,__float2int_rn(scale_pow2(vh[j],-eh)),__float2int_rn(scale_pow2(vl[j],-el)));
 }
 if constexpr(!SideA){
  __syncthreads();
  for(int i=tid;i<Size;i+=256){
   int c=i/Rows,r=i%Rows;
   q[qp*Plane+(c0+c)*512+r0+r]=transpose[0][r*TransposeStride+c];
   if(qp<4)q[(qp+4)*Plane+(c0+c)*512+r0+r]=transpose[1][r*TransposeStride+c];
  }
 }
 asm volatile("fence.proxy.async.global;" ::: "memory");
}

// Local absolute maxima start at +0 and contain no NaN. Unsigned binary32
// bit order equals value order here, including positive subnormals and +inf.
__device__ __forceinline__ float nonnegative_warp_max(float x){
 return __uint_as_float(__reduce_max_sync(0xffffffff,__float_as_uint(x)));
}

// Keep only two FP32 transformed planes live. All phases share one scratch area.
template<bool SideA> struct PairPackShared {float values[2][SideA?8192:4096];float maxima[8];int exponent;float maxima_second[8];int exponent_second;int8_t transpose[SideA?1:2][SideA?1:128*36];};
static __device__ __noinline__ float pair_general_pow2(float x,int e){return ldexpf(x,e);}
__device__ __forceinline__ float pair_pow2(float x,int e){
 if(e>=-126&&e<=127)return __fmul_rn(x,__int_as_float((e+127)<<23));
 return pair_general_pow2(x,e);
}

template<class T,bool SideA,int NativePlane,int QP> __device__ __forceinline__
void pair_store(int which,T* high,int8_t* q,int* exp,
 PairPackShared<SideA>& s,int tile,int r0,int c0){
 constexpr int Rows=SideA?64:128,Cols=SideA?128:32,Size=Rows*Cols,PT=Size/256;
 constexpr int Tiles=(512/Rows)*(512/Cols),Bits=(std::is_same_v<T,cute::half_t> ? 11 : 8);
 int tid=threadIdx.x;float hi=0;
 #pragma unroll 4
 for(int j=0;j<PT;++j){int i=tid+256*j;T h=T(s.values[which][i]);high[NativePlane*Plane+(r0+i/Cols)*512+c0+i%Cols]=h;hi=fmaxf(hi,fabsf(float(h)));}
 hi=nonnegative_warp_max(hi);
 if((tid&31)==0)s.maxima[tid/32]=hi;__syncthreads();
 if(tid==0){float peak=0;for(int w=0;w<8;++w)peak=fmaxf(peak,s.maxima[w]);
  int e=peak>0?int(ceilf(log2f(peak/127.f))):0;s.exponent=e;exp[QP*Tiles+tile]=e;
  if constexpr(QP<4)exp[(QP+4)*Tiles+tile]=e-Bits;
 }
 __syncthreads();int eh=s.exponent,el=eh-Bits;
 auto quantize=[&](auto fast,float sh,float sl){
  #pragma unroll 4
  for(int j=0;j<PT;++j){
   int i=tid+256*j;float v=s.values[which][i],h=float(T(v));
   int qh;if constexpr(decltype(fast)::value)qh=__float2int_rn(__fmul_rn(h,sh));else qh=__float2int_rn(pair_pow2(h,-eh));
   qh=max(-127,min(127,qh));int ql=0;
   if constexpr(QP<4){float low=__fsub_rn(v,h);
    if constexpr(decltype(fast)::value)ql=__float2int_rn(__fmul_rn(low,sl));else ql=__float2int_rn(pair_pow2(low,-el));
    ql=max(-127,min(127,ql));
   }
   if constexpr(SideA){q[QP*Plane+(r0+i/Cols)*512+c0+i%Cols]=int8_t(qh);if constexpr(QP<4)q[(QP+4)*Plane+(r0+i/Cols)*512+c0+i%Cols]=int8_t(ql);}
   else{int at=(i/Cols)*36+i%Cols;s.transpose[0][at]=int8_t(qh);if constexpr(QP<4)s.transpose[1][at]=int8_t(ql);}
  }
 };
 if(eh>=-127&&eh<=126&&el>=-127&&el<=126)
  quantize(std::true_type{},__int_as_float((127-eh)<<23),__int_as_float((127-el)<<23));
 else quantize(std::false_type{},0,0);
 if constexpr(!SideA){
  __syncthreads();
  for(int i=tid;i<Size;i+=256){int c=i/Rows,r=i%Rows;
   q[QP*Plane+(c0+c)*512+r0+r]=s.transpose[0][r*36+c];
   if constexpr(QP<4)q[(QP+4)*Plane+(c0+c)*512+r0+r]=s.transpose[1][r*36+c];
  }
 }
 // The next transformed plane reuses maxima/exponent and both transpose arrays.
 __syncthreads();
}

template<class T,bool SideA,int QP> __device__ __forceinline__
void pair_quant_store(int which,int eh,int8_t* q,
 PairPackShared<SideA>& s,int r0,int c0){
 constexpr int Rows=SideA?64:128,Cols=SideA?128:32,Size=Rows*Cols,PT=Size/256;
 constexpr int Bits=(std::is_same_v<T,cute::half_t> ? 11 : 8);
 int tid=threadIdx.x,el=eh-Bits;
 auto quantize=[&](auto fast,float sh,float sl){
  #pragma unroll 4
  for(int j=0;j<PT;++j){
   int i=tid+256*j;float v=s.values[which][i],h=float(T(v));
   int qh;if constexpr(decltype(fast)::value)qh=__float2int_rn(__fmul_rn(h,sh));else qh=__float2int_rn(pair_pow2(h,-eh));
   qh=max(-127,min(127,qh));int ql=0;
   if constexpr(QP<4){float low=__fsub_rn(v,h);
    if constexpr(decltype(fast)::value)ql=__float2int_rn(__fmul_rn(low,sl));else ql=__float2int_rn(pair_pow2(low,-el));
    ql=max(-127,min(127,ql));
   }
   if constexpr(SideA){q[QP*Plane+(r0+i/Cols)*512+c0+i%Cols]=int8_t(qh);if constexpr(QP<4)q[(QP+4)*Plane+(r0+i/Cols)*512+c0+i%Cols]=int8_t(ql);}
   else{int at=(i/Cols)*36+i%Cols;s.transpose[0][at]=int8_t(qh);if constexpr(QP<4)s.transpose[1][at]=int8_t(ql);}
  }
 };
 if(eh>=-127&&eh<=126&&el>=-127&&el<=126)
  quantize(std::true_type{},__int_as_float((127-eh)<<23),__int_as_float((127-el)<<23));
 else quantize(std::false_type{},0,0);
 if constexpr(!SideA){
  __syncthreads();
  for(int i=tid;i<Size;i+=256){int c=i/Rows,r=i%Rows;
   q[QP*Plane+(c0+c)*512+r0+r]=s.transpose[0][r*36+c];
   if constexpr(QP<4)q[(QP+4)*Plane+(c0+c)*512+r0+r]=s.transpose[1][r*36+c];
  }
 }
 // The next transformed plane reuses maxima/exponent and both transpose arrays.
 __syncthreads();
}

template<class T,bool SideA,int NP0,int QP0,int NP1,int QP1> __device__ __forceinline__
void pair_store_both(T* high,int8_t* q,int* exp,
 PairPackShared<SideA>& s,int tile,int r0,int c0){
 constexpr int Rows=SideA?64:128,Cols=SideA?128:32,PT=Rows*Cols/256;
 constexpr int Tiles=(512/Rows)*(512/Cols),Bits=(std::is_same_v<T,cute::half_t> ? 11 : 8);
 int tid=threadIdx.x;float hi0=0,hi1=0;
 #pragma unroll 4
 for(int j=0;j<PT;++j){
  int i=tid+256*j,at=(r0+i/Cols)*512+c0+i%Cols;
  T h0=T(s.values[0][i]),h1=T(s.values[1][i]);
  high[NP0*Plane+at]=h0;high[NP1*Plane+at]=h1;
  hi0=fmaxf(hi0,fabsf(float(h0)));hi1=fmaxf(hi1,fabsf(float(h1)));
 }
 hi0=nonnegative_warp_max(hi0);
 hi1=nonnegative_warp_max(hi1);
 if((tid&31)==0){s.maxima[tid/32]=hi0;s.maxima_second[tid/32]=hi1;}
 __syncthreads();
 if(tid==0){
  float peak0=0,peak1=0;
  for(int w=0;w<8;++w){peak0=fmaxf(peak0,s.maxima[w]);peak1=fmaxf(peak1,s.maxima_second[w]);}
  int e0=peak0>0?int(ceilf(log2f(peak0/127.f))):0;
  int e1=peak1>0?int(ceilf(log2f(peak1/127.f))):0;
  s.exponent=e0;s.exponent_second=e1;
  exp[QP0*Tiles+tile]=e0;exp[(QP0+4)*Tiles+tile]=e0-Bits;
  exp[QP1*Tiles+tile]=e1;exp[(QP1+4)*Tiles+tile]=e1-Bits;
 }
 __syncthreads();
 pair_quant_store<T,SideA,QP0>(0,s.exponent,q,s,r0,c0);
 pair_quant_store<T,SideA,QP1>(1,s.exponent_second,q,s,r0,c0);
}

template<class T,bool SideA,bool Balance,int Group> __device__ __forceinline__
void pack_pair(const T* __restrict__ X,T* __restrict__ high,int8_t* __restrict__ q,int* __restrict__ exp,
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
void pack(const T* __restrict__ X,T* __restrict__ high,int8_t* __restrict__ q,int* __restrict__ exp,const int* __restrict__ inside){
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
 T* __restrict__ ha,T* __restrict__ hb,int8_t* __restrict__ qa,int8_t* __restrict__ qb,
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
 T* __restrict__ ha,T* __restrict__ hb,int8_t* __restrict__ qa,int8_t* __restrict__ qb,
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
  bm=nonnegative_warp_max(bm);
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

#pragma once
#include <cuda_runtime.h>
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
 __shared__ float aa[256];aa[tid]=am;__syncthreads();
 if(tid<128)partial[((slab*128+tid)*32)+chunk]=fmaxf(aa[tid],aa[tid+128]);
 for(int kr=warp;kr<128;kr+=8){
  float bm=fabsf(float(B[(slab*128+kr)*1024+chunk*32+lane]));
  for(int step=16;step;step/=2)bm=fmaxf(bm,__shfl_xor_sync(0xffffffff,bm,step));
  if(lane==0)partial[1024*32+(slab*128+kr)*32+chunk]=bm;
 }
}
__global__ void inside_finalize(const float* partial,int* exponent){
 int tid=threadIdx.x,k=blockIdx.x*8+tid/32,lane=tid%32;
 float am=partial[k*32+lane],bm=partial[1024*32+k*32+lane];
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
   x0=scale_pow2(x0,ex[r]);x1=scale_pow2(x1,ex[r]);
   x2=scale_pow2(x2,ex[r+512]);x3=scale_pow2(x3,ex[r+512]);
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
   float v=value<T,SideA,Balance,decltype(fast)::value>(X,inside,r,c,plane,scale0,scale1);
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
 }else load_values(std::false_type{},0,0);
 if(!quant){asm volatile("fence.proxy.async.global;" ::: "memory");return;}
 __shared__ float maxima[2][8];__shared__ int exponents[2];
 for(int s=16;s;s/=2){hi=fmaxf(hi,__shfl_xor_sync(0xffffffff,hi,s));lo=fmaxf(lo,__shfl_xor_sync(0xffffffff,lo,s));}
 if((tid&31)==0){maxima[0][tid/32]=hi;maxima[1][tid/32]=lo;}__syncthreads();
 if(tid<2){
  float peak=0;for(int j=0;j<8;++j)peak=fmaxf(peak,maxima[tid][j]);
  int e=peak>0?int(ceilf(log2f(peak/127.f))):0;
  exponents[tid]=e;if(!tid || qp<4)exp[(qp+(tid?4:0))*Tiles+tile]=e;
 }
 __syncthreads();
 // B's INT8 form is physically transposed so both integer MMA operands are K major.
 __shared__ int8_t transpose[SideA?1:2][SideA?1:Size];
 auto store_quant=[&](int j,int qh,int ql){
  int i=tid+256*j,r=r0+i/Cols,c=c0+i%Cols;
  qh=max(-127,min(127,qh));ql=max(-127,min(127,ql));
  if constexpr(SideA){q[qp*Plane+r*512+c]=int8_t(qh);if(qp<4)q[(qp+4)*Plane+r*512+c]=int8_t(ql);}
  else{transpose[0][i]=int8_t(qh);if(qp<4)transpose[1][i]=int8_t(ql);}
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
   q[qp*Plane+(c0+c)*512+r0+r]=transpose[0][r*Cols+c];
   if(qp<4)q[(qp+4)*Plane+(c0+c)*512+r0+r]=transpose[1][r*Cols+c];
  }
 }
 asm volatile("fence.proxy.async.global;" ::: "memory");
}

template<class T,bool SideA,bool Balance> __global__ __launch_bounds__(256,1)
void pack(const T* __restrict__ X,T* __restrict__ high,int8_t* __restrict__ q,int* __restrict__ exp,const int* __restrict__ inside){
 int plane=blockIdx.x/(SideA?32:64);
 switch(plane){
  case 0:pack_plane<T,SideA,Balance,0>(X,high,q,exp,inside);break;
  case 1:pack_plane<T,SideA,Balance,1>(X,high,q,exp,inside);break;
  case 2:pack_plane<T,SideA,Balance,2>(X,high,q,exp,inside);break;
  case 3:pack_plane<T,SideA,Balance,3>(X,high,q,exp,inside);break;
  case 4:pack_plane<T,SideA,Balance,4>(X,high,q,exp,inside);break;
  case 5:pack_plane<T,SideA,Balance,5>(X,high,q,exp,inside);break;
  default:pack_plane<T,SideA,Balance,6>(X,high,q,exp,inside);break;
 }
}

template<class T,bool Balance> void preprocess(const T* A,const T* B,Work w,cudaStream_t stream){
 if constexpr(Balance){inside_partial<<<dim3(8,32),256,0,stream>>>(A,B,w.partial);inside_finalize<<<128,256,0,stream>>>(w.partial,w.inside);}
 pack<T,true,Balance><<<7*32,256,0,stream>>>(A,static_cast<T*>(w.ha),w.qa,w.ea,w.inside);
 pack<T,false,Balance><<<7*64,256,0,stream>>>(B,static_cast<T*>(w.hb),w.qb,w.eb,w.inside);
 check(cudaPeekAtLastError(),"INT8 preprocess");
}
} // namespace int8_research

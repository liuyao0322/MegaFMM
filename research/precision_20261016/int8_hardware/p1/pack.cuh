#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <stdexcept>
#include <type_traits>

namespace int8_research {
constexpr int H=512, Plane=H*H, BK=128, BM=64, BN=32, Stages=8;
constexpr size_t HighBytes=7*Plane*2, QuantBytes=9*Plane;
// Seven main planes per side, nine INT8 planes per side, integer scale metadata.
constexpr size_t WorkBytes=2*HighBytes+2*QuantBytes+(9*32+9*64+1024)*sizeof(int);
void check(cudaError_t e,const char* where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
struct Work {
 void *ha,*hb;int8_t *qa,*qb;int *ea,*eb,*inside;
 explicit Work(void* w){
  auto p=static_cast<char*>(w);ha=p;p+=HighBytes;hb=p;p+=HighBytes;
  qa=reinterpret_cast<int8_t*>(p);p+=QuantBytes;qb=reinterpret_cast<int8_t*>(p);p+=QuantBytes;
  ea=reinterpret_cast<int*>(p);p+=9*32*sizeof(int);eb=reinterpret_cast<int*>(p);p+=9*64*sizeof(int);
  inside=reinterpret_cast<int*>(p);
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

template<class T,bool SideA,bool Balance> __device__ float value(const T* X,const int* ex,int r,int c,int plane){
 int at=r*1024+c;
 float x0=float(X[at]),x1=float(X[at+512]),x2=float(X[at+512*1024]),x3=float(X[at+512*1024+512]);
 if constexpr(Balance){
  if constexpr(SideA){
   x0=ldexpf(x0,-ex[c]);x2=ldexpf(x2,-ex[c]);
   x1=ldexpf(x1,-ex[c+512]);x3=ldexpf(x3,-ex[c+512]);
  }else{
   x0=ldexpf(x0,ex[r]);x1=ldexpf(x1,ex[r]);
   x2=ldexpf(x2,ex[r+512]);x3=ldexpf(x3,ex[r+512]);
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

template<class T,bool SideA,bool Balance> __global__ __launch_bounds__(256,1)
void pack(const T* X,T* high,int8_t* q,int* exp,const int* inside){
 constexpr int Rows=SideA?64:128,Cols=SideA?128:32,Size=Rows*Cols,PerThread=Size/256;
 constexpr int ColTiles=512/Cols,Tiles=(512/Rows)*ColTiles;
 int plane=blockIdx.x/Tiles,tile=blockIdx.x%Tiles,r0=(tile/ColTiles)*Rows,c0=(tile%ColTiles)*Cols,tid=threadIdx.x;
 constexpr int RawPlane=SideA?3:2;
 bool quant=plane>=4 || plane==(SideA?2:3) || plane==RawPlane;
 int qp=plane==RawPlane?8:plane>=4?plane-4:3;
 float vh[PerThread],vl[PerThread],hi=0,lo=0;
 #pragma unroll
 for(int j=0;j<PerThread;++j){
  int i=tid+256*j,r=r0+i/Cols,c=c0+i%Cols;
  float v=value<T,SideA,Balance>(X,inside,r,c,plane);
  T h=T(v);high[plane*Plane+r*512+c]=h;
  vh[j]=float(h);vl[j]=__fsub_rn(v,float(h));
  hi=fmaxf(hi,fabsf(vh[j]));lo=fmaxf(lo,fabsf(vl[j]));
 }
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
 #pragma unroll
 for(int j=0;j<PerThread;++j){
  int i=tid+256*j,r=r0+i/Cols,c=c0+i%Cols;
  int qh=__float2int_rn(ldexpf(vh[j],-exponents[0]));qh=max(-127,min(127,qh));
  int ql=__float2int_rn(ldexpf(vl[j],-exponents[1]));ql=max(-127,min(127,ql));
  if constexpr(SideA){q[qp*Plane+r*512+c]=int8_t(qh);if(qp<4)q[(qp+4)*Plane+r*512+c]=int8_t(ql);}
  else{transpose[0][i]=int8_t(qh);if(qp<4)transpose[1][i]=int8_t(ql);}
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

template<class T,bool Balance> void preprocess(const T* A,const T* B,Work w,cudaStream_t stream){
 if constexpr(Balance)inside_scale<<<1024,256,0,stream>>>(A,B,w.inside);
 pack<T,true,Balance><<<7*32,256,0,stream>>>(A,static_cast<T*>(w.ha),w.qa,w.ea,w.inside);
 pack<T,false,Balance><<<7*64,256,0,stream>>>(B,static_cast<T*>(w.hb),w.qb,w.eb,w.inside);
 check(cudaPeekAtLastError(),"INT8 preprocess");
}
} // namespace int8_research

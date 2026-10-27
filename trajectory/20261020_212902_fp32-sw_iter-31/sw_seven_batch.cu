#include "sw_seven_batch.h"
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <algorithm>
#include <stdexcept>
namespace sw_seven_batch {
namespace {
using namespace cute;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void check(cutlass::Status e){if(e!=cutlass::Status::kSuccess)throw std::runtime_error(cutlassGetStatusString(e));}
// Form exactly the seven original Winograd operand pairs. Arithmetic transforms
// are FP32; every stored operand retains the original BF16 or FP16 dtype.
template<class T> __global__ void pack(const T* A,const T* B,T* pa,T* pb,int M,int N,int K,Shape z){
 for(int side=0;side<2;++side){
  int rows=side?z.kh:z.mh,cols=side?z.nh:z.kh,pr=side?z.kp:z.mp,pc=side?z.np:z.kp;
  size_t plane=size_t(pr)*pc;T* dst=side?pb:pa;const T* src=side?B:A;int ld=side?N:K;
  for(size_t at=size_t(blockIdx.x)*blockDim.x+threadIdx.x;at<plane;at+=size_t(blockDim.x)*gridDim.x){
   int r=int(at/pc),c=int(at%pc);float x[4]={};
   if(r<rows && c<cols)for(int q=0;q<4;++q){int rr=r+(q/2)*rows,cc=c+(q%2)*cols;
    if(rr<(side?K:M) && cc<(side?N:K))x[q]=float(src[size_t(rr)*ld+cc]);}
   float v[7];
   if(!side){float s1=__fadd_rn(x[2],x[3]),s2=__fsub_rn(s1,x[0]);
    v[0]=x[0];v[1]=x[1];v[2]=__fsub_rn(x[1],s2);v[3]=x[3];v[4]=s1;v[5]=s2;v[6]=__fsub_rn(x[0],x[2]);}
   else{float t1=__fsub_rn(x[1],x[0]),t2=__fsub_rn(x[3],t1);
    v[0]=x[0];v[1]=x[2];v[2]=x[3];v[3]=__fsub_rn(t2,x[2]);v[4]=t1;v[5]=t2;v[6]=__fsub_rn(x[3],x[1]);}
   #pragma unroll
   for(int q=0;q<7;++q)dst[size_t(q)*plane+at]=T(v[q]);
  }
 }
}
__global__ void combine(const float* products,float* D,int M,int N,Shape z){
 size_t plane=size_t(z.mp)*z.np;
 for(size_t at=size_t(blockIdx.x)*blockDim.x+threadIdx.x;at<size_t(z.mh)*z.nh;at+=size_t(blockDim.x)*gridDim.x){
  int r=int(at/z.nh),c=int(at%z.nh);size_t off=size_t(r)*z.np+c;float p[7];
  #pragma unroll
  for(int q=0;q<7;++q)p[q]=products[size_t(q)*plane+off];
  float u2=__fadd_rn(p[0],p[5]),u3=__fadd_rn(u2,p[6]),u4=__fadd_rn(u2,p[4]);
  D[size_t(r)*N+c]=__fadd_rn(p[0],p[1]);
  if(c+z.nh<N)D[size_t(r)*N+c+z.nh]=__fadd_rn(u4,p[2]);
  if(r+z.mh<M)D[size_t(r+z.mh)*N+c]=__fsub_rn(u3,p[3]);
  if(r+z.mh<M && c+z.nh<N)D[size_t(r+z.mh)*N+c+z.nh]=__fadd_rn(u3,p[4]);
 }
}
template<class T,int N,int CM,int Schedule>struct Leaf {
 using Tile=cute::Shape<_128,Int<N>,_64>;using Cluster=cute::Shape<Int<CM>,_1,_1>;
 using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,float,float,void,cutlass::layout::RowMajor,4,float,cutlass::layout::RowMajor,4,
  cute::conditional_t<Schedule==0,cutlass::epilogue::TmaWarpSpecializedCooperative,cutlass::epilogue::TmaWarpSpecialized>>::CollectiveOp;
 using Main=typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,T,cutlass::layout::RowMajor,8,T,cutlass::layout::RowMajor,8,float,Tile,Cluster,
  cutlass::gemm::collective::StageCountAutoCarveout<int(sizeof(typename Epi::SharedStorage))>,cute::conditional_t<Schedule==0,cutlass::gemm::KernelTmaWarpSpecializedCooperative,cutlass::gemm::KernelTmaWarpSpecializedPingpong>>::CollectiveOp;
 using Kernel=cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,Main,Epi>;
 using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
 static megafmm::detail::SWResources action(const megafmm::Problem& p,Config c,const void* A,const void* B,float* D,void* ws,int sms,cudaStream_t stream,bool init){
  if(init){check(cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,Kernel::SharedStorageSize));
   cudaFuncAttributes a{};int resident=0;check(cudaFuncGetAttributes(&a,cutlass::device_kernel<Kernel>));check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,cutlass::device_kernel<Kernel>,Kernel::MaxThreadsPerBlock,Kernel::SharedStorageSize));
   return {Kernel::SharedStorageSize+a.sharedSizeBytes,a.numRegs,resident};}
  auto z=shape(p);auto pa=static_cast<T*>(ws);auto pb=reinterpret_cast<T*>(static_cast<char*>(ws)+z.a);auto pc=reinterpret_cast<float*>(static_cast<char*>(ws)+z.a+z.b);
  typename Kernel::StrideA sa{z.kp,Int<1>{},int64_t(z.mp)*z.kp};
  typename Kernel::StrideB sb{Int<1>{},z.np,int64_t(z.kp)*z.np};
  typename Kernel::StrideC sc{z.np,Int<1>{},int64_t(z.mp)*z.np};
  typename Kernel::StrideD sd{z.np,Int<1>{},int64_t(z.mp)*z.np};
  // L=7 contains seven distinct transformed half-problems. No full-size AB call.
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kBatched,{z.mp,z.np,z.kp,7},{pa,sa,pb,sb},{{},nullptr,sc,pc,sd},{0,sms}};
  args.epilogue.thread.alpha=1;args.epilogue.thread.beta=0;args.scheduler.max_swizzle_size=1;
  using Raster=typename Kernel::TileScheduler::RasterOrderOptions;
  if(c.raster)args.scheduler.raster_order=c.raster==1?Raster::AlongM:Raster::AlongN;
  check(Gemm::can_implement(args));if(Gemm::get_workspace_size(args))throw std::runtime_error("unexpected seven-batch leaf scratch");
  int blocks=int(std::min<size_t>(c.pack_cap,(std::max(size_t(z.mp)*z.kp,size_t(z.kp)*z.np)+255)/256));
  pack<<<blocks,256,0,stream>>>(static_cast<const T*>(A),static_cast<const T*>(B),pa,pb,int(p.m),int(p.n),int(p.k),z);check(cudaPeekAtLastError());
  auto params=Kernel::to_underlying_arguments(args,nullptr);check(Gemm::run(params,stream));
  combine<<<std::min<size_t>(512,(size_t(z.mh)*z.nh+255)/256),256,0,stream>>>(pc,D,int(p.m),int(p.n),z);check(cudaPeekAtLastError());return {};
 }
};
template<class T>megafmm::detail::SWResources dispatch(const megafmm::Problem& p,Config c,const void* A,const void* B,float* D,void* ws,int sms,cudaStream_t stream,bool init){
 switch(c.id){
 case 0:return Leaf<T,128,1,0>::action(p,c,A,B,D,ws,sms,stream,init);
 case 1:return Leaf<T,256,1,0>::action(p,c,A,B,D,ws,sms,stream,init);
 case 2:return Leaf<T,128,2,0>::action(p,c,A,B,D,ws,sms,stream,init);
 case 3:return Leaf<T,128,1,2>::action(p,c,A,B,D,ws,sms,stream,init);
 default:throw std::invalid_argument("unknown seven-batch config");}
}
}
Shape shape(const megafmm::Problem& p){
 if(p.m<=0 || p.n<=0 || p.k<=0 || p.m>32768 || p.n>32768 || p.k>32768 || p.trans_a || p.trans_b || p.lda!=p.k || p.ldb!=p.n || p.ldd!=p.n || p.alpha!=1 || p.beta!=0)throw std::invalid_argument("unsupported seven-batch problem");
 auto align=[](int x){return (x+15)/16*16;};Shape z{};z.mh=(p.m+1)/2;z.nh=(p.n+1)/2;z.kh=(p.k+1)/2;z.mp=align(z.mh);z.np=align(z.nh);z.kp=align(z.kh);
 z.a=7*size_t(z.mp)*z.kp*2;z.b=7*size_t(z.kp)*z.np*2;z.c=7*size_t(z.mp)*z.np*4;z.bytes=z.a+z.b+z.c;return z;
}
megafmm::detail::SWResources setup(const megafmm::Problem& p,Config c){return p.dtype==megafmm::DType::BF16?dispatch<cutlass::bfloat16_t>(p,c,nullptr,nullptr,nullptr,nullptr,0,nullptr,true):dispatch<cutlass::half_t>(p,c,nullptr,nullptr,nullptr,nullptr,0,nullptr,true);}
void run(const megafmm::Problem& p,Config c,const void* A,const void* B,float* D,void* ws,int sms,cudaStream_t stream){
 if(p.dtype==megafmm::DType::BF16)dispatch<cutlass::bfloat16_t>(p,c,A,B,D,ws,sms,stream,false);else dispatch<cutlass::half_t>(p,c,A,B,D,ws,sms,stream,false);
}
}

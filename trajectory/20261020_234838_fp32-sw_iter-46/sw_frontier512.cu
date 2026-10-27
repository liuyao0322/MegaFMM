#include "sw_frontier512.h"
#include "sw_frontier512_prefix.inc"
#include "sw_frontier512_compute.inc"
#include "sw_frontier512_encode.inc"
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T,int Threads>
SWResources action(const Problem& p,Config c,const void* A,const void* B,float* D,void* ws,cudaStream_t stream,bool init){
 constexpr int N=8,K=256,S=4;
 constexpr size_t smem=sizeof(Shared<T,N,K,S>);
 auto kernel=compute<T,N,K,S,true,1>;
 if(init){check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  check(cudaFuncSetAttribute(kernel,cudaFuncAttributePreferredSharedMemoryCarveout,100));
  check(cudaFuncSetAttribute(transform_wide<T,Threads,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
  cudaFuncAttributes attr{};int resident=0;check(cudaFuncGetAttributes(&attr,kernel));
  check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,256,smem));return {smem+attr.sharedSizeBytes,attr.numRegs,resident};}
 int mh=p.m/2,nh=p.n/2,kh=p.k/2;CUtensorMap amap{},bmap{},smap{},tmap{};
 auto tbase=static_cast<char*>(ws)+size_t(4)*mh*kh*2;
 encode(amap,A,p.dtype,true,false,mh,nh,kh,N,K);encode(bmap,B,p.dtype,false,false,mh,nh,kh,N,K);
 encode(smap,ws,p.dtype,true,true,mh,nh,kh,N,K);encode(tmap,tbase,p.dtype,false,true,mh,nh,kh,N,K);
 int blocks=std::min<int64_t>(c.pack_cap/2,(std::max(int64_t(mh)*kh,int64_t(kh)*nh)+8*Threads-1)/(8*Threads));
 transform_wide<T,Threads,true><<<dim3(blocks,2),Threads,0,stream>>>(static_cast<const T*>(A),static_cast<const T*>(B),static_cast<T*>(ws),mh,nh,kh);check(cudaPeekAtLastError());
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(mh/N,nh/64,1);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=smem;cfg.stream=stream;cfg.attrs=&attr;cfg.numAttrs=1;
 check(cudaLaunchKernelEx(&cfg,kernel,amap,bmap,smap,tmap,mh,nh,kh,D,int(p.m),int(p.n),0,c.raster));
 return {};
}
template<class T>SWResources dispatch(const Problem& p,Config c,const void* A,const void* B,float* D,void* ws,cudaStream_t stream,bool init){
 if(c.id==0)return action<T,256>(p,c,A,B,D,ws,stream,init);
 if(c.id==1)return action<T,192>(p,c,A,B,D,ws,stream,init);
 throw std::invalid_argument("unknown frontier512 config");
}
} // anonymous
WorkspaceShape shape(const Problem& p,Config c){
 if(c.id<0 || c.id>1 || p.m!=512 || p.n!=512 || p.k!=512 || c.pack_cap<2 || c.raster!=0 || p.trans_a || p.trans_b || p.lda!=p.k || p.ldb!=p.n || p.ldd!=p.n || p.alpha!=1 || p.beta!=0 || (p.dtype!=DType::BF16 && p.dtype!=DType::FP16))throw std::invalid_argument("unsupported frontier512 shape");
 return {size_t(2)*p.k*(p.m+p.n)};
}
SWResources setup(const Problem& p,Config c){shape(p,c);return p.dtype==DType::BF16?dispatch<cute::bfloat16_t>(p,c,nullptr,nullptr,nullptr,nullptr,nullptr,true):dispatch<cute::half_t>(p,c,nullptr,nullptr,nullptr,nullptr,nullptr,true);}
void run(const Problem& p,Config c,const void* A,const void* B,float* D,void* ws,int,cudaStream_t stream){shape(p,c);if(p.dtype==DType::BF16)dispatch<cute::bfloat16_t>(p,c,A,B,D,ws,stream,false);else dispatch<cute::half_t>(p,c,A,B,D,ws,stream,false);}
}

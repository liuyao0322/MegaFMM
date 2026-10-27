#include "sw_small_native.h"
#include "sw_small_prefix.inc"
#include "sw_small_compute.inc"
#include "sw_small_split.inc"
#include "sw_small_encode.inc"
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void reconstruct(const float* P,float* D,int mh,int nh){
 size_t count=size_t(mh)*nh;
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x){
  int r=i/nh,c=i%nh;float p[7];
  #pragma unroll
  for(int j=0;j<7;++j)p[j]=P[size_t(j)*count+i];
  float u2=__fadd_rn(p[0],p[5]),u3=__fadd_rn(u2,p[6]),u4=__fadd_rn(u2,p[4]);
  size_t o=size_t(r)*2*nh+c;
  D[o]=__fadd_rn(p[0],p[1]);D[o+nh]=__fadd_rn(u4,p[2]);
  D[o+size_t(mh)*2*nh]=__fsub_rn(u3,p[3]);D[o+size_t(mh)*2*nh+nh]=__fadd_rn(u3,p[4]);
 }
}
template<class T,int N,int K,int S,bool Short,bool Split>
SWResources action(const Problem& p,Config c,const void* A,const void* B,float* D,void* ws,cudaStream_t stream,bool init){
 constexpr size_t smem=sizeof(Shared<T,N,K,S>);
 auto kernel=[](){if constexpr(Split)return split_compute<T,N,K,S,Short>;else return compute<T,N,K,S,Short>;}();
 // Selecting the pointer with if constexpr avoids instantiating unsupported consumers.
 if(init){check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  cudaFuncAttributes attr{};int resident=0;check(cudaFuncGetAttributes(&attr,kernel));
  check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,256,smem));return {smem+attr.sharedSizeBytes,attr.numRegs,resident};}
 int mh=p.m/2,nh=p.n/2,kh=p.k/2;CUtensorMap amap{},bmap{},smap{},tmap{};
 auto tbase=static_cast<char*>(ws)+size_t(4)*mh*kh*2;
 encode(amap,A,p.dtype,true,false,mh,nh,kh,N,K);encode(bmap,B,p.dtype,false,false,mh,nh,kh,N,K);
 encode(smap,ws,p.dtype,true,true,mh,nh,kh,N,K);encode(tmap,tbase,p.dtype,false,true,mh,nh,kh,N,K);
 int blocks=std::min<int64_t>(c.pack_cap,(std::max(int64_t(mh)*kh,int64_t(kh)*nh)+511)/512);
 transform<T><<<blocks,256,0,stream>>>(static_cast<const T*>(A),static_cast<const T*>(B),static_cast<T*>(ws),mh,nh,kh);check(cudaPeekAtLastError());
 float* target=D;
 if constexpr(Split)target=reinterpret_cast<float*>(static_cast<char*>(ws)+size_t(2)*p.k*(p.m+p.n));
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(mh/N,nh/64,Split?7:1);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=smem;cfg.stream=stream;cfg.attrs=&attr;cfg.numAttrs=1;
 check(cudaLaunchKernelEx(&cfg,kernel,amap,bmap,smap,tmap,mh,nh,kh,target,int(p.m),int(p.n),0,c.raster));
 if constexpr(Split){reconstruct<<<std::min<size_t>(128,(size_t(mh)*nh+255)/256),256,0,stream>>>(target,D,mh,nh);check(cudaPeekAtLastError());}
 return {};
}
template<class T>SWResources dispatch(const Problem& p,Config c,const void* A,const void* B,float* D,void* ws,cudaStream_t stream,bool init){switch(c.id){
 case 0:return action<T,16,128,8,false,false>(p,c,A,B,D,ws,stream,init);
 case 1:return action<T,16,128,4,false,false>(p,c,A,B,D,ws,stream,init);
 case 2:return action<T,16,64,4,false,false>(p,c,A,B,D,ws,stream,init);
 case 3:return action<T,16,128,8,true,false>(p,c,A,B,D,ws,stream,init);
 case 4:return action<T,16,128,4,true,false>(p,c,A,B,D,ws,stream,init);
 case 5:return action<T,32,128,8,true,false>(p,c,A,B,D,ws,stream,init);
 case 6:return action<T,16,128,4,true,true>(p,c,A,B,D,ws,stream,init);
 case 7:return action<T,32,128,4,true,true>(p,c,A,B,D,ws,stream,init);
 default:throw std::invalid_argument("unknown small native config");}}
} // anonymous
WorkspaceShape shape(const Problem& p,Config c){
 if(c.id<0 || c.id>7 || p.m<=0 || p.n<=0 || p.k<=0 || p.m>32768 || p.n>32768 || p.k>32768 || p.m%64 || p.n%128 || p.k%256 || (c.id>=3 && p.k!=512) || p.trans_a || p.trans_b || p.lda!=p.k || p.ldb!=p.n || p.ldd!=p.n || p.alpha!=1 || p.beta!=0)throw std::invalid_argument("unsupported small native shape");
 return {size_t(2)*p.k*(p.m+p.n)+(c.id>=6?size_t(7)*(p.m/2)*(p.n/2)*4:0)};
}
SWResources setup(const Problem& p,Config c){shape(p,c);return p.dtype==DType::BF16?dispatch<cute::bfloat16_t>(p,c,nullptr,nullptr,nullptr,nullptr,nullptr,true):dispatch<cute::half_t>(p,c,nullptr,nullptr,nullptr,nullptr,nullptr,true);}
void run(const Problem& p,Config c,const void* A,const void* B,float* D,void* ws,int,cudaStream_t stream){if(p.dtype==DType::BF16)dispatch<cute::bfloat16_t>(p,c,A,B,D,ws,stream,false);else dispatch<cute::half_t>(p,c,A,B,D,ws,stream,false);}
} // sw_small_native

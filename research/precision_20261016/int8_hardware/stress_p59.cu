// Generated from prior extended validation; original reference unchanged.
#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#include "p59/corrected.cuh"
#include <cuda_profiler_api.h>

template<class T> __global__ void varied_fill(T* a,T* b,double* ad,double* bd,int pattern,int seed){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Count;i+=gridDim.x*blockDim.x){
  int r=i/N,c=i%N;
  int ai=(pattern==2||pattern==13)?r*N+c%512:i,bi=(pattern==2||pattern==13)?(r%512)*N+c:i;
  float x=float(hash(uint64_t(ai)^seed)>>40)*(2.f/16777216.f)-1;
  float y=float(hash(uint64_t(bi)^seed^0x632be59bd9b4e019ULL)>>40)*(2.f/16777216.f)-1;
  if(pattern==0){
   constexpr bool bf=std::is_same_v<T,__nv_bfloat16>;
   x=ldexpf(x,bf?60:10);y=ldexpf(y,bf?-60:-10);
  }
  if(pattern==1){x=ldexpf(x,r<512?6:-6);y=ldexpf(y,c<512?-6:6);}
  if(pattern==2||pattern==13){if(c>=512)x=-x+1.f/256;}
  if(pattern==3){x=ldexpf(x,c<512?0:-12);y=ldexpf(y,r<512?-12:0);}
  if(pattern==4){x=ldexpf(x,int(hash(i+seed)%17)-8);y=ldexpf(y,int(hash(i+seed+1)%17)-8);}
  if(pattern==5){x=(r==c)?1:0;y=y*2;}
  if(pattern==6){bool af=((c%2)==1)^(c>=512),bf=((r%2)==1)^(r>=512);x=ldexpf(x,af?-12:0);y=ldexpf(y,bf?0:-12);}
  // Different K128 restoration exponents force the original fallback.
  if(pattern==7){x=ldexpf(x,4*((c%512)/128)-6);y=ldexpf(y,4*((r%512)/128)-6);}
  // BF16 restoration exponents below the common-path guard; FP16 stays representable.
  if(pattern==8){constexpr bool bf=std::is_same_v<T,__nv_bfloat16>;x=ldexpf(x,bf?-60:-10);y=ldexpf(y,bf?-60:-10);}
  if(pattern==9||pattern==10){
   constexpr bool bf=std::is_same_v<T,__nv_bfloat16>;
   int hi=bf?127:14,lo=bf?-132:-24;
   x=ldexpf(x,pattern==9?hi:lo);y=ldexpf(y,pattern==9?lo:hi);
  }
  if(pattern==11){
   constexpr bool bf=std::is_same_v<T,__nv_bfloat16>;
   x=ldexpf(x,(c%64)<32?(bf?127:14):0);
   y=ldexpf(y,(r%64)<32?(bf?-132:-24):0);
  }
  if(pattern==12||pattern==13){int e[4]={-4,0,4,-2};x=ldexpf(x,e[(r/8)%4]);y=ldexpf(y,e[(c/8)%4]);}
  T xx=rounded<T>(x),yy=rounded<T>(y);
  a[i]=xx;b[i]=yy;ad[i]=float(xx);bd[i]=float(yy);
 }
}
__global__ void guard_fill(float* d){for(int i=threadIdx.x;i<Count+256;i+=blockDim.x)d[i]=1234567.f;}

template<class T> int validate_extra(megafmm::DType dtype,bool quick){
 Stream ss[2];Blas bb0(ss[0].s),bb1(ss[1].s);cublasHandle_t blas[2]={bb0.h,bb1.h};
 Buffer a[2]={Buffer(Count*2),Buffer(Count*2)},b[2]={Buffer(Count*2),Buffer(Count*2)};
 Buffer a64[2]={Buffer(Count*8),Buffer(Count*8)},b64[2]={Buffer(Count*8),Buffer(Count*8)};
 Buffer ref[2]={Buffer(Count*8),Buffer(Count*8)},d[2]={Buffer((Count+256)*4),Buffer((Count+256)*4)};
 Buffer base[2]={Buffer(Count*4),Buffer(Count*4)},scratch[2]={Buffer(128*sizeof(Metric)),Buffer(128*sizeof(Metric))};
 Graph graph[2];
 Buffer workspace[2]={Buffer(int8_research::WorkBytes+512),Buffer(int8_research::WorkBytes+512)};
 for(int s=0;s<2;++s){
  check(cudaStreamBeginCapture(ss[s].s,cudaStreamCaptureModeThreadLocal));
  int8_research::run(dtype,a[s].p,b[s].p,d[s].as<float>()+128,(char*)workspace[s].p+256,ss[s].s,true);
  check(cudaStreamEndCapture(ss[s].s,&graph[s].g));check(cudaGraphInstantiate(&graph[s].x,graph[s].g,nullptr,nullptr,0));
 }
 int failures=0;
 for(int pat=0;pat<14;++pat)for(int trial=0;trial<(quick?1:3);++trial){
  for(int s=0;s<2;++s){
   varied_fill<<<512,256,0,ss[s].s>>>(a[s].as<T>(),b[s].as<T>(),a64[s].as<double>(),b64[s].as<double>(),pat,135239+trial*541+s*293+pat*149+(dtype==megafmm::DType::FP16?307:0));
   guard_fill<<<1,256,0,ss[s].s>>>(d[s].as<float>());
   double one=1,zero=0;
   check(cublasDgemm(blas[s],CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&one,b64[s].as<double>(),N,a64[s].as<double>(),N,&zero,ref[s].as<double>(),N));
   float fone=1,fzero=0;
   check(cublasGemmEx(blas[s],CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&fone,b[s].p,
    dtype==megafmm::DType::BF16?CUDA_R_16BF:CUDA_R_16F,N,a[s].p,
    dtype==megafmm::DType::BF16?CUDA_R_16BF:CUDA_R_16F,N,&fzero,base[s].p,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  }
  for(int s=0;s<2;++s)check(cudaMemsetAsync(workspace[s].p,0xa5,int8_research::WorkBytes+512,ss[s].s));
  // Deliberately interleave Graph and direct submissions on independent streams.
  for(int repeat=0;repeat<(quick?2:8);++repeat)for(int s=0;s<2;++s){
   if((repeat+s)%2)check(cudaGraphLaunch(graph[s].x,ss[s].s));
   else int8_research::run(dtype,a[s].p,b[s].p,d[s].as<float>()+128,(char*)workspace[s].p+256,ss[s].s,true);
  }
  for(int s=0;s<2;++s){
   int8_research::diagnose_uniform_scales(dtype==megafmm::DType::BF16?"bf16":"fp16",pat,trial,int8_research::Work((char*)workspace[s].p+256),ss[s].s,s);
   Metric dm=metric(base[s].as<float>(),ref[s].as<double>(),scratch[s],ss[s].s);
   Metric cm=metric(d[s].as<float>()+128,ref[s].as<double>(),scratch[s],ss[s].s);
   double limit=3*ef(dm)+(dtype==megafmm::DType::BF16?1e-3:1e-4);
   float guards[256];
   check(cudaMemcpy(guards,d[s].p,128*4,cudaMemcpyDeviceToHost));
   check(cudaMemcpy(guards+128,d[s].as<float>()+128+Count,128*4,cudaMemcpyDeviceToHost));
   bool safe=true;for(auto x:guards)safe=safe&&x==1234567.f;
   uint8_t wg[512];check(cudaMemcpy(wg,workspace[s].p,256,cudaMemcpyDeviceToHost));
   check(cudaMemcpy(wg+256,(char*)workspace[s].p+256+int8_research::WorkBytes,256,cudaMemcpyDeviceToHost));
   for(auto x:wg)safe=safe&&x==0xa5;
   bool fixed=ef(cm)<=(dtype==megafmm::DType::BF16?1e-3:1e-4);
   bool pass=safe&&dm.nonfinite==0&&cm.nonfinite==0&&ef(cm)<=limit&&fixed;
   failures+=!pass;
   printf("EXTRA dtype=%s pattern=%d trial=%d stream=%d ef=%.12g dense=%.12g gate=%.12g guards=%d fixed_floor=%d nonfinite=%llu pass=%d\n",dtype==megafmm::DType::BF16?"bf16":"fp16",pat,trial,s,ef(cm),ef(dm),limit,safe,fixed,cm.nonfinite,pass);fflush(stdout);
  }
 }
 return failures;
}

int main(int argc,char** argv){try{
 bool quick=argc==2&&std::string(argv[1])=="--sanitizer";
 if(argc!=1&&!quick)throw std::invalid_argument("usage: stress [--sanitizer]");
 int8_research::init();
 int failures=validate_extra<__nv_bfloat16>(megafmm::DType::BF16,quick)+validate_extra<__half>(megafmm::DType::FP16,quick);
 return failures?2:0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}

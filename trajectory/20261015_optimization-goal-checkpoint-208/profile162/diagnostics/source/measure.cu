#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <string>
struct Trace {uint64_t t0,t1,t2,t3;};
__device__ __forceinline__ uint64_t stamp(){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}
#include "baseline.cuh"
#include "instrumented.cuh"
namespace base=megafmm::diagnostic_base;
namespace diag=megafmm::diagnostic_trace;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Memory {void* p=nullptr;explicit Memory(size_t n){ck(cudaMalloc(&p,n));}~Memory(){cudaFree(p);}};
__global__ void clock_probe(Trace* t){uint64_t a=stamp(),b=stamp(),c=stamp(),d=stamp();t[threadIdx.x]={a,b,c,d};}
template<class T> void measure(const char* name,megafmm::DType dtype,std::ofstream& records,std::ofstream& timing,int first_role){
 constexpr int Count=1024*1024,Reps=10;
 Memory ma(Count*2),mb(Count*2),mw(4194304),md(Count*4),mt(Reps*128*sizeof(Trace));
 T* A=static_cast<T*>(ma.p);T* B=static_cast<T*>(mb.p);T* W=static_cast<T*>(mw.p);float* D=static_cast<float*>(md.p);
 std::vector<T> a(Count),b(Count);std::vector<float> reference(Count),output(Count);std::vector<Trace> trace(Reps*128);
 cudaStream_t stream;ck(cudaStreamCreate(&stream));cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));
 CUtensorMap am{},bm{},sm{},tm{};base::encode(am,A,dtype,true,false);base::encode(bm,B,dtype,false,false);base::encode(sm,W,dtype,true,true);base::encode(tm,reinterpret_cast<char*>(W)+2097152,dtype,false,true);
 auto fbase=base::sw_cooperative_hilo<T>;
 const void* funcs[]={reinterpret_cast<void*>(diag::sw_cooperative_hilo<T,5>),reinterpret_cast<void*>(diag::sw_cooperative_hilo<T,5>),reinterpret_cast<void*>(diag::sw_cooperative_hilo<T,5>),reinterpret_cast<void*>(diag::sw_cooperative_hilo<T,5>),reinterpret_cast<void*>(diag::sw_cooperative_hilo<T,5>),reinterpret_cast<void*>(diag::sw_cooperative_hilo<T,5>)};
 size_t shared=sizeof(base::Shared<T>);ck(cudaFuncSetAttribute(fbase,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
 cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));
 for(int role=-1;role<6;++role){
  const void* f=role<0?reinterpret_cast<void*>(fbase):funcs[role];
  ck(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));int blocks=0;ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,f,768,shared));
  cudaFuncAttributes attrs;ck(cudaFuncGetAttributes(&attrs,f));
  std::printf("RESOURCE dtype=%s role=%d registers=%d local=%zu active_blocks=%d shared=%zu\n",name,role,attrs.numRegs,attrs.localSizeBytes,blocks,shared);
  if(blocks*prop.multiProcessorCount<128)throw std::runtime_error("cooperative capacity insufficient");
 }
 auto launch=[&](int role,int sample,int rep){
  Trace* tr=static_cast<Trace*>(mt.p)+rep*128;
  void* args[]={&am,&bm,&sm,&tm,&A,&B,&W,&D,&tr,&sample};
  ck(cudaLaunchCooperativeKernel(role<0?reinterpret_cast<void*>(fbase):funcs[role],dim3(16,8),dim3(768),args,shared,stream));
 };
 auto graph=[&](int role,int sample){
  cudaGraph_t g;cudaGraphExec_t ex;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  for(int r=0;r<Reps;++r)launch(role,sample,r);
  ck(cudaStreamEndCapture(stream,&g));ck(cudaGraphInstantiate(&ex,g,0));
  for(int i=0;i<2;++i)ck(cudaGraphLaunch(ex,stream));ck(cudaStreamSynchronize(stream));
  ck(cudaEventRecord(start,stream));ck(cudaGraphLaunch(ex,stream));ck(cudaEventRecord(stop,stream));ck(cudaEventSynchronize(stop));float ms;ck(cudaEventElapsedTime(&ms,start,stop));
  ck(cudaGraphExecDestroy(ex));ck(cudaGraphDestroy(g));return double(ms)*1000/Reps;
 };
 for(int dataset=0;dataset<3;++dataset){
  uint64_t rng=123456789+dataset*987654321ULL;
  auto next=[&](){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return float(int(rng&65535)-32768)/32768;};
  for(int i=0;i<Count;++i){a[i]=T(next());b[i]=T(next());}
  ck(cudaMemcpy(A,a.data(),Count*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(B,b.data(),Count*2,cudaMemcpyHostToDevice));
  double bt=graph(-1,0);ck(cudaMemcpy(reference.data(),D,Count*4,cudaMemcpyDeviceToHost));
  timing<<name<<','<<dataset<<",-1,0,"<<bt<<",1\n";
  for(int role=first_role;role<6;++role){
   int samples=role<2?14:role<4?7:1;
   // Alternate stage traversal across fresh datasets to expose ordering effects.
   for(int index=0;index<samples;++index){int sample=dataset%2?samples-1-index:index;
    double us=graph(role,sample);ck(cudaMemcpy(output.data(),D,Count*4,cudaMemcpyDeviceToHost));
    bool exact=std::memcmp(reference.data(),output.data(),Count*4)==0;
    timing<<name<<','<<dataset<<','<<role<<','<<sample<<','<<us<<','<<exact<<'\n';
    if(!exact)throw std::runtime_error("instrumented output differs from baseline");
    ck(cudaMemcpy(trace.data(),mt.p,trace.size()*sizeof(Trace),cudaMemcpyDeviceToHost));
    for(int r=0;r<Reps;++r)for(int cta=0;cta<128;++cta){auto& t=trace[r*128+cta];
     if(role!=5 && (t.t0==0 || t.t1<t.t0 || t.t2<t.t1 || t.t3<t.t2))throw std::runtime_error("invalid trace timestamps");
     records<<name<<','<<dataset<<','<<role<<','<<sample<<','<<r<<','<<cta<<','<<t.t0<<','<<t.t1<<','<<t.t2<<','<<t.t3<<'\n';
    }
   }
   std::printf("MEASURED dtype=%s dataset=%d role=%d baseline_us=%.3f exact=1\n",name,dataset,role,bt);std::fflush(stdout);
  }
 }
 ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));ck(cudaStreamDestroy(stream));
}
int main(int argc,char** argv){try{
 int first_role=argc>2?std::atoi(argv[2]):0;
 std::string dir=argc>1?argv[1]:"results/production_chain";
 std::ofstream records(dir+"/records.csv"),timing(dir+"/timing.csv");
 records<<"dtype,dataset,role,sample,repeat,cta,t0,t1,t2,t3\n";timing<<"dtype,dataset,role,sample,event_us,bit_exact\n";
 Memory probe(32*sizeof(Trace));clock_probe<<<1,32>>>(static_cast<Trace*>(probe.p));ck(cudaDeviceSynchronize());Trace ht[32];ck(cudaMemcpy(ht,probe.p,sizeof(ht),cudaMemcpyDeviceToHost));
 for(int i=0;i<3;++i)std::printf("STAMP_PAIR_NS=%llu\n",static_cast<unsigned long long>((&ht[0].t0)[i+1]-(&ht[0].t0)[i]));
 measure<cute::bfloat16_t>("bf16",megafmm::DType::BF16,records,timing,first_role);
 measure<cute::half_t>("fp16",megafmm::DType::FP16,records,timing,first_role);
 return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

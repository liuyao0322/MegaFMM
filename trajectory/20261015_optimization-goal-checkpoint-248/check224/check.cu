#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstring>
#include <stdexcept>
#include "baseline.cuh"
#include "candidate.cuh"
namespace base=megafmm::diagnostic_base;
namespace cand=megafmm::diagnostic_candidate;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Memory{void* p;Memory(size_t n){ck(cudaMalloc(&p,n));}~Memory(){cudaFree(p);}};
template<class T>void check(const char* name,megafmm::DType dtype,bool quick){
 constexpr int N=1024*1024;
 Memory aa(2*N),bb(2*N),wb(8388608),wc(4194304),db(4*N),dc(4*N);
 cudaStream_t stream;ck(cudaStreamCreate(&stream));
 megafmm::Problem p;p.m=p.n=p.k=p.lda=p.ldb=p.ldc=p.ldd=1024;p.dtype=dtype;
 auto bs=base::make_sw_pipeline(p);auto cs=cand::make_sw_pipeline(p);
 std::vector<T>a(N),b(N);std::vector<float>x(N),y(N);int count=0;
 for(int pattern=0;pattern<(quick?1:6);++pattern)for(int dataset=0;dataset<(quick?1:3);++dataset){
  uint64_t rng=0x9e3779b97f4a7c15ULL+dataset*123456789ULL+pattern*987654321ULL;
  auto next=[&](){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return float(int(rng&65535)-32768)/32768;};
  for(int i=0;i<N;++i){float av=next(),bv=next();
   if(pattern==1){av=(i&1?1.f:-1.f)*(1.f+av/256);bv=((i/1024)&1?1.f:-1.f)*(1.f+bv/256);}
   if(pattern==2){av=std::ldexp(av,(i%13)-6);bv=std::ldexp(bv,6-(i%13));}
   if(pattern==3){av*=0.000244140625f;bv*=0.000244140625f;}
   if(pattern==4)av=bv=0;
   if(pattern==5){if(i%31)av=0;if(i%37)bv=0;}
   a[i]=T(av);b[i]=T(bv);
  }
  ck(cudaMemcpyAsync(aa.p,a.data(),2*N,cudaMemcpyHostToDevice,stream));ck(cudaMemcpyAsync(bb.p,b.data(),2*N,cudaMemcpyHostToDevice,stream));
  for(int graph=0;graph<2;++graph){
   ck(cudaMemsetAsync(wb.p,0xff,8388608,stream));ck(cudaMemsetAsync(wc.p,0xff,4194304,stream));ck(cudaMemsetAsync(db.p,0xff,4*N,stream));ck(cudaMemsetAsync(dc.p,0xff,4*N,stream));ck(cudaStreamSynchronize(stream));
   auto run=[&](){base::run_sw_pipeline(*bs,aa.p,bb.p,static_cast<float*>(db.p),wb.p,stream);cand::run_sw_pipeline(*cs,aa.p,bb.p,static_cast<float*>(dc.p),wc.p,stream);};
   if(graph){cudaGraph_t g;cudaGraphExec_t ex;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));run();ck(cudaStreamEndCapture(stream,&g));ck(cudaGraphInstantiate(&ex,g,0));for(int r=0;r<3;++r)ck(cudaGraphLaunch(ex,stream));ck(cudaStreamSynchronize(stream));ck(cudaGraphExecDestroy(ex));ck(cudaGraphDestroy(g));}
   else{run();ck(cudaStreamSynchronize(stream));}
   ck(cudaMemcpy(x.data(),db.p,4*N,cudaMemcpyDeviceToHost));ck(cudaMemcpy(y.data(),dc.p,4*N,cudaMemcpyDeviceToHost));
   if(std::memcmp(x.data(),y.data(),4*N))throw std::runtime_error("bitwise mismatch");
   for(int i=0;i<N;++i)if(!std::isfinite(y[i]))throw std::runtime_error("nonfinite output");
   ++count;std::printf("EXACT dtype=%s pattern=%d dataset=%d graph=%d\n",name,pattern,dataset,graph);std::fflush(stdout);
  }
 }
 std::printf("SUMMARY dtype=%s checks=%d failures=0\n",name,count);ck(cudaStreamDestroy(stream));
}
int main(int argc,char**){try{check<cute::bfloat16_t>("bf16",megafmm::DType::BF16,argc>1);check<cute::half_t>("fp16",megafmm::DType::FP16,argc>1);return 0;}catch(const std::exception&e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

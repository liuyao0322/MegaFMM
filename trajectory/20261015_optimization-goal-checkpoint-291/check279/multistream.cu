#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstring>
#include <stdexcept>
#include "baseline.cuh"
#include "candidate.cuh"
namespace base=megafmm::diagnostic_base;
namespace cand=megafmm::diagnostic_candidate;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Mem{void*p;Mem(size_t n){ck(cudaMalloc(&p,n));}~Mem(){cudaFree(p);}};
template<class T>void test(megafmm::DType dtype,const char* name){
 constexpr size_t N=1024*1024;
 Mem a0(2*N),a1(2*N),b0(2*N),b1(2*N),db0(4*N),db1(4*N),dc0(4*N),dc1(4*N),wb(8388608),w0(4194304),w1(4194304);
 cudaStream_t streams[2];for(auto& st:streams)ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
 void* as[2]={a0.p,a1.p};void* bs[2]={b0.p,b1.p};void* ds[2]={dc0.p,dc1.p};void* refs[2]={db0.p,db1.p};void* ws[2]={w0.p,w1.p};
 megafmm::Problem p;p.m=p.n=p.k=p.lda=p.ldb=p.ldc=p.ldd=1024;p.dtype=dtype;
 auto ref=base::make_sw_pipeline(p);auto candidate=cand::make_sw_pipeline(p);
 std::vector<T>a(N),b(N);uint64_t rng=0x123456789abcdefULL;
 auto next=[&](){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return float(int(rng&65535)-32768)/32768;};
 std::vector<float> expected[2],got(N);
 for(int j=0;j<2;++j){for(size_t i=0;i<N;++i){a[i]=T(next());b[i]=T(next());}
  ck(cudaMemcpyAsync(as[j],a.data(),2*N,cudaMemcpyHostToDevice,streams[j]));ck(cudaMemcpyAsync(bs[j],b.data(),2*N,cudaMemcpyHostToDevice,streams[j]));ck(cudaStreamSynchronize(streams[j]));
  base::run_sw_pipeline(*ref,as[j],bs[j],static_cast<float*>(refs[j]),wb.p,streams[j]);ck(cudaStreamSynchronize(streams[j]));expected[j].resize(N);ck(cudaMemcpy(expected[j].data(),refs[j],4*N,cudaMemcpyDeviceToHost));
 }
 for(int graph=0;graph<2;++graph){
  cudaGraph_t graphs[2]{};cudaGraphExec_t ex[2]{};
  if(graph)for(int j=0;j<2;++j){ck(cudaStreamBeginCapture(streams[j],cudaStreamCaptureModeThreadLocal));cand::run_sw_pipeline(*candidate,as[j],bs[j],static_cast<float*>(ds[j]),ws[j],streams[j]);ck(cudaStreamEndCapture(streams[j],&graphs[j]));ck(cudaGraphInstantiate(&ex[j],graphs[j],0));}
  for(int repeat=0;repeat<32;++repeat)for(int j=0;j<2;++j){
   ck(cudaMemsetAsync(ws[j],0xff,4194304,streams[j]));ck(cudaMemsetAsync(ds[j],0xff,4*N,streams[j]));
   if(graph)ck(cudaGraphLaunch(ex[j],streams[j]));else cand::run_sw_pipeline(*candidate,as[j],bs[j],static_cast<float*>(ds[j]),ws[j],streams[j]);
  }
  for(int j=0;j<2;++j){ck(cudaStreamSynchronize(streams[j]));ck(cudaMemcpy(got.data(),ds[j],4*N,cudaMemcpyDeviceToHost));if(std::memcmp(got.data(),expected[j].data(),4*N))throw std::runtime_error("multistream bitwise mismatch");if(graph){ck(cudaGraphExecDestroy(ex[j]));ck(cudaGraphDestroy(graphs[j]));}}
  std::printf("MULTISTREAM dtype=%s graph=%d streams=2 launches=64 failures=0\n",name,graph);
 }
 for(auto st:streams)ck(cudaStreamDestroy(st));
}
int main(){try{test<cute::bfloat16_t>(megafmm::DType::BF16,"bf16");test<cute::half_t>(megafmm::DType::FP16,"fp16");}catch(const std::exception&e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

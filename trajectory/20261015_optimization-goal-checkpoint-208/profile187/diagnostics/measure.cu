#include "instrumented.cuh"
#include <vector>
#include <cstdio>
#include <random>
using namespace megafmm;
template<class T>void measure(DType dtype){
 Problem prob{};prob.m=prob.n=prob.k=prob.lda=prob.ldb=prob.ldd=1024;prob.dtype=dtype;prob.alpha=1;prob.beta=0;
 auto state=detail::make_sw_pipeline(prob);
 T *a,*b;float* d;void* w;cudaMalloc(&a,2097152);cudaMalloc(&b,2097152);cudaMalloc(&d,4194304);cudaMalloc(&w,12582912);
 cudaStream_t stream;cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
 cudaEvent_t start,end;cudaEventCreate(&start);cudaEventCreate(&end);
 for(int dataset=0;dataset<3;++dataset){
  std::mt19937 rng(123+dataset);std::uniform_real_distribution<float> u(-1,1);std::vector<T> h(1048576);
  for(auto&x:h)x=T(u(rng));cudaMemcpy(a,h.data(),2097152,cudaMemcpyHostToDevice);
  for(auto&x:h)x=T(u(rng));cudaMemcpy(b,h.data(),2097152,cudaMemcpyHostToDevice);
  for(int k=0;k<20;++k)detail::run_sw_pipeline(*state,a,b,d,w,stream);
  cudaEventRecord(start,stream);detail::run_sw_pipeline(*state,a,b,d,w,stream);cudaEventRecord(end,stream);cudaEventSynchronize(end);
  float ms;cudaEventElapsedTime(&ms,start,end);fprintf(stderr,"dtype%d dataset%d instrumented_event_us=%f\n",int(dtype),dataset,ms*1000);
  std::vector<uint64_t> trace(224*10);cudaMemcpy(trace.data(),static_cast<char*>(w)+(size_t(11)<<20)+4096,trace.size()*8,cudaMemcpyDeviceToHost);
  for(int c=0;c<224;++c){printf("%d,%d,%d",int(dtype),dataset,c);for(int j=0;j<10;++j)printf(",%lu",trace[c*10+j]);puts("");}
 }
 cudaFree(a);cudaFree(b);cudaFree(d);cudaFree(w);cudaStreamDestroy(stream);cudaEventDestroy(start);cudaEventDestroy(end);
}
int main(){puts("dtype,dataset,cta,t0,t1,t2,t3,t4,t5,t6,t7,t8,t9");measure<cute::bfloat16_t>(DType::BF16);measure<cute::half_t>(DType::FP16);}

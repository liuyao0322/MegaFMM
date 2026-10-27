// Profile a fixed candidate; never retune while NCU is attached.
#include "internal.h"
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(int argc,char** argv) {try {
  int id=argc>1?atoi(argv[1]):0, n=argc>2?atoi(argv[2]):1024;
  if(id<0 || id>=megafmm::detail::HopperCandidateCount || n<=0) return 1;
  megafmm::Problem p;p.m=p.n=p.k=n;p.lda=p.ldb=p.ldc=p.ldd=n;p.dtype=megafmm::DType::FP16;
  cudaDeviceProp prop;int device;check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
  void *a=nullptr,*b=nullptr,*d=nullptr,*w=nullptr;
  check(cudaMalloc(&a,size_t(n)*n*2));check(cudaMalloc(&b,size_t(n)*n*2));check(cudaMalloc(&d,size_t(n)*n*4));
  const size_t bytes=megafmm::detail::hopper_workspace(p,id);if(bytes)check(cudaMalloc(&w,bytes));
  check(cudaMemset(a,0,size_t(n)*n*2));check(cudaMemset(b,0,size_t(n)*n*2));
  megafmm::detail::hopper_dense(id>=5 && id<=8?id-5:id,p,nullptr,nullptr,nullptr,nullptr,nullptr,prop.multiProcessorCount,nullptr,2);
  for(int i=0;i<5;++i)
    if(!megafmm::detail::hopper_dense(id,p,a,b,nullptr,static_cast<float*>(d),w,prop.multiProcessorCount,nullptr,1))
      throw std::runtime_error("fixed candidate does not support shape");
  check(cudaDeviceSynchronize());check(cudaProfilerStart());
  megafmm::detail::hopper_dense(id,p,a,b,nullptr,static_cast<float*>(d),w,prop.multiProcessorCount,nullptr,1);
  check(cudaDeviceSynchronize());check(cudaProfilerStop());
  printf("fixed_candidate=%s shape=%d dtype=fp16\n",megafmm::detail::hopper_candidate_name(id),n);
  cudaFree(a);cudaFree(b);cudaFree(d);if(w)cudaFree(w);return 0;
}catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

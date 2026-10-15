#include "megafmm/gemm.h"
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(int argc,char**argv){try{
 megafmm::Problem p;p.m=argc>1?atoll(argv[1]):8192;p.n=argc>2?atoll(argv[2]):p.m;p.k=argc>3?atoll(argv[3]):p.m;
 p.lda=p.k;p.ldb=p.ldc=p.ldd=p.n;p.dtype=megafmm::DType::FP16;
 megafmm::Policy policy;policy.algorithm=argc>4?megafmm::Algorithm::Tuned:megafmm::Algorithm::Dense;policy.tuning_input_sets=2;
 auto plan=megafmm::create_plan(p,policy);void*a=nullptr,*b=nullptr,*w=nullptr;float*d=nullptr;
 ck(cudaMalloc(&a,size_t(p.m)*p.k*2));ck(cudaMalloc(&b,size_t(p.k)*p.n*2));ck(cudaMalloc(&d,size_t(p.m)*p.n*4));if(plan->workspace_bytes())ck(cudaMalloc(&w,plan->workspace_bytes()));
 ck(cudaMemset(a,0,size_t(p.m)*p.k*2));ck(cudaMemset(b,0,size_t(p.k)*p.n*2));
 // Tuning is outside the profiler range. Diagnostic zero inputs match the fixed-candidate harness; not acceptance timing.
 megafmm::tune(*plan,a,b,nullptr,d,w);printf("profile_route=%s config=%s\n",plan->algorithm_name().c_str(),plan->lt_algorithm_config(a,b,nullptr,d,w).c_str());fflush(stdout);
 for(int i=0;i<5;++i)megafmm::run(*plan,a,b,nullptr,d,w);
 ck(cudaDeviceSynchronize());ck(cudaProfilerStart());megafmm::run(*plan,a,b,nullptr,d,w);ck(cudaDeviceSynchronize());ck(cudaProfilerStop());
 ck(cudaFree(a));ck(cudaFree(b));ck(cudaFree(d));if(w)ck(cudaFree(w));return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

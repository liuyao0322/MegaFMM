// Diagnostic complete-call paired timing against the tuned dispatcher.
// This tool does not perform the independent FP64 acceptance suite.
#include "internal.h"
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer{void* p=nullptr;explicit Buffer(size_t n){if(n)check(cudaMalloc(&p,n));}~Buffer(){if(p)cudaFree(p);}};
template<class T>T rounded(float v);
template<>__half rounded(float v){return __float2half_rn(v);}
template<>__nv_bfloat16 rounded(float v){return __float2bfloat16_rn(v);}
template<class T>int compare(int id,int m,int n,int k,megafmm::DType dtype){
  using namespace megafmm;
  Problem p;p.m=m;p.n=n;p.k=k;p.lda=k;p.ldb=p.ldc=p.ldd=n;p.dtype=dtype;
  cudaDeviceProp prop{};int device;check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
  Policy policy;policy.algorithm=Algorithm::Tuned;policy.tuning_input_sets=2;
  auto plan=create_plan(p,policy);Buffer workspace(plan->workspace_bytes()),scratch(detail::hopper_workspace(p,id,prop.multiProcessorCount)),out(size_t(m)*n*4);
  Buffer a0(size_t(m)*k*2),a1(size_t(m)*k*2),b0(size_t(k)*n*2),b1(size_t(k)*n*2);
  void* aa[]={a0.p,a1.p};void* bb[]={b0.p,b1.p};
  std::mt19937 rng(357);std::uniform_real_distribution<float> dist(-1,1);
  auto fill=[&](void* ptr,size_t count){std::vector<T> host(count);for(auto& x:host)x=rounded<T>(dist(rng));check(cudaMemcpy(ptr,host.data(),count*2,cudaMemcpyHostToDevice));};
  for(int set=0;set<2;++set){fill(aa[set],size_t(m)*k);fill(bb[set],size_t(k)*n);}
  tune(*plan,aa[0],bb[0],nullptr,static_cast<float*>(out.p),workspace.p,nullptr);
  printf("diagnostic_only=1 fp64_acceptance=unchecked shape=%d,%d,%d dtype=%s candidate=%s baseline_route=%s\n",m,n,k,dtype_name(dtype),detail::hopper_candidate_name(id),plan->algorithm_name().c_str());fflush(stdout);
  auto launch=[&](int route,int set){if(route==0)run(*plan,aa[set],bb[set],nullptr,static_cast<float*>(out.p),workspace.p,nullptr);else if(!detail::hopper_dense(id,p,aa[set],bb[set],nullptr,static_cast<float*>(out.p),scratch.p,prop.multiProcessorCount,nullptr,1))throw std::runtime_error("candidate rejected");};
  detail::hopper_dense(id,p,nullptr,nullptr,nullptr,nullptr,nullptr,prop.multiProcessorCount,nullptr,2);
  for(int i=0;i<20;++i){launch(0,i%2);launch(1,i%2);}check(cudaDeviceSynchronize());
  check(cudaProfilerStart());
  launch(0,0);launch(1,0);
  check(cudaDeviceSynchronize());check(cudaProfilerStop());
  puts("profile_only=1 same_original_random_inputs=1 timing_acceptance=unchecked");return 0;
}
int main(int argc,char**argv){try{if(argc!=6)throw std::runtime_error("usage: compare_candidate ID M N K bf16|fp16");int id=atoi(argv[1]),m=atoi(argv[2]),n=atoi(argv[3]),k=atoi(argv[4]);if(id<0 || id>=megafmm::detail::HopperCandidateCount || m<=0 || n<=0 || k<=0)throw std::runtime_error("invalid dimension or candidate");if(std::string(argv[5])=="bf16")return compare<__nv_bfloat16>(id,m,n,k,megafmm::DType::BF16);return compare<__half>(id,m,n,k,megafmm::DType::FP16);}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

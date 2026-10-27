#include "internal.h"
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>
using namespace megafmm;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer{void* p=nullptr;explicit Buffer(size_t n){check(cudaMalloc(&p,n));}~Buffer(){cudaFree(p);}};
int main(int argc,char** argv){try{
  int id=argc>1?atoi(argv[1]):52;
  Problem p;p.m=p.n=p.k=p.lda=p.ldb=p.ldc=p.ldd=4096;p.dtype=DType::BF16;
  auto configs=detail::sw_configs(p);auto c=configs.at(id);if(c.id!=id)throw std::runtime_error("registry ID mismatch");
  auto res=detail::setup_sw_config(p,c);
  Buffer a(p.m*p.k*2),b(p.k*p.n*2),a2(p.m*p.k*2),b2(p.k*p.n*2),d(p.m*p.n*4),ws(detail::sw_config_workspace(p,c));
  std::mt19937 gen(1234567);std::normal_distribution<float> normal(0,1);std::vector<__nv_bfloat16> data(p.m*p.k);
  for(void* ptr:{a.p,b.p,a2.p,b2.p}){for(auto& x:data)x=__float2bfloat16_rn(normal(gen));check(cudaMemcpy(ptr,data.data(),data.size()*2,cudaMemcpyHostToDevice));}
  cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  for(int i=0;i<100;++i)detail::run_sw_config(p,c,i%2?a2.p:a.p,i%2?b2.p:b.p,static_cast<float*>(d.p),ws.p,stream);
  check(cudaStreamSynchronize(stream));
  printf("PROFILE id=%d registers=%d resident=%d config=%s\n",id,res.registers,res.resident,detail::sw_config_name(c).c_str());fflush(stdout);
  check(cudaProfilerStart());detail::run_sw_config(p,c,a.p,b.p,static_cast<float*>(d.p),ws.p,stream);check(cudaStreamSynchronize(stream));check(cudaProfilerStop());
  check(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

#include "internal.h"
#include <cmath>
#include <stdexcept>
namespace megafmm::detail {
namespace {
struct Norm { double error,reference; unsigned nonfinite; };
__global__ void difference(Problem p,const float* d,const float* ref,Norm* norm) {
  double e=0,r=0;unsigned bad=0;
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<p.m*p.n;i+=int64_t(gridDim.x)*blockDim.x) {
    const int64_t at=(i/p.n)*p.ldd+i%p.n;
    double x=d[at],y=ref[at];
    if(!isfinite(x) || !isfinite(y)) ++bad;
    else {e+=(x-y)*(x-y);r+=y*y;}
  }
  __shared__ Norm values[256];values[threadIdx.x]={e,r,bad};__syncthreads();
  for(int stride=128;stride;stride/=2) {
    if(threadIdx.x<stride) {
      values[threadIdx.x].error+=values[threadIdx.x+stride].error;
      values[threadIdx.x].reference+=values[threadIdx.x+stride].reference;
      values[threadIdx.x].nonfinite+=values[threadIdx.x+stride].nonfinite;
    }
    __syncthreads();
  }
  if(threadIdx.x==0) {atomicAdd(&norm->error,values[0].error);atomicAdd(&norm->reference,values[0].reference);atomicAdd(&norm->nonfinite,values[0].nonfinite);}
}
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(std::string("strategy validation: ")+cudaGetErrorString(e));}
}
bool dense_agreement(const Problem& p,const float* D,const float* reference,cudaStream_t stream) {
  Norm* gpu=nullptr;Norm host{};
  check(cudaMalloc(reinterpret_cast<void**>(&gpu),sizeof(Norm)));
  try {
    check(cudaMemsetAsync(gpu,0,sizeof(Norm),stream));
    difference<<<128,256,0,stream>>>(p,D,reference,gpu);check(cudaPeekAtLastError());
    check(cudaMemcpyAsync(&host,gpu,sizeof(Norm),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
  } catch(...) {cudaFree(gpu);throw;}
  check(cudaFree(gpu));
  return host.nonfinite==0 && std::isfinite(host.error) && std::isfinite(host.reference) &&
      host.error<=1e-10*host.reference;
}
}

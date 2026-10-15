#include "internal.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace megafmm::detail {
namespace {
struct Norm { double error,reference; unsigned nonfinite; };
template<class Reference>
__global__ void difference(Problem p,const float* d,const Reference* ref,Norm* norm,
                           bool compact_reference=false) {
  double e=0,r=0;unsigned bad=0;
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<p.m*p.n;i+=int64_t(gridDim.x)*blockDim.x) {
    const int64_t at=(i/p.n)*p.ldd+i%p.n;
    double x=d[at],y=ref[compact_reference?i:at];
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
void check(cublasStatus_t e) {if(e!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("strategy FP64 reference cuBLAS status "+std::to_string(int(e)));}
struct Buffer {
  void* data=nullptr;
  explicit Buffer(size_t bytes){check(cudaMalloc(&data,bytes));}
  ~Buffer(){cudaFree(data);}
  Buffer(const Buffer&)=delete;
  Buffer& operator=(const Buffer&)=delete;
  template<class T>T* as()const{return static_cast<T*>(data);}
};
struct ReferenceBlas {
  cublasHandle_t handle=nullptr;
  ReferenceBlas(){check(cublasCreate(&handle));}
  ~ReferenceBlas(){cublasDestroy(handle);}
};
template<class T>
__global__ void pack_double(const T* input,double* output,int64_t rows,int64_t columns,
                            int64_t stride,bool transpose) {
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*columns;i+=int64_t(gridDim.x)*blockDim.x) {
    const auto r=i/columns,c=i%columns;
    output[i]=double(float(input[transpose?c*stride+r:r*stride+c]));
  }
}
template<class T>
bool original_fp64_agreement_typed(const Problem& p,const void* A,const void* B,
                                   const float* D,const float* dense,cudaStream_t stream) {
  // These allocations and synchronization occur only in explicit tuning.
  Buffer a64(size_t(p.m)*p.k*sizeof(double)),b64(size_t(p.k)*p.n*sizeof(double));
  Buffer reference(size_t(p.m)*p.n*sizeof(double)),norms(2*sizeof(Norm));
  pack_double<<<256,256,0,stream>>>(static_cast<const T*>(A),a64.as<double>(),p.m,p.k,p.lda,p.trans_a);
  pack_double<<<256,256,0,stream>>>(static_cast<const T*>(B),b64.as<double>(),p.k,p.n,p.ldb,p.trans_b);
  check(cudaPeekAtLastError());
  ReferenceBlas blas;
  check(cublasSetStream(blas.handle,stream));
  const double alpha=p.alpha,zero=0;
  check(cublasDgemm(blas.handle,CUBLAS_OP_N,CUBLAS_OP_N,int(p.n),int(p.m),int(p.k),
      &alpha,b64.as<double>(),int(p.n),a64.as<double>(),int(p.k),&zero,reference.as<double>(),int(p.n)));
  check(cudaMemsetAsync(norms.data,0,2*sizeof(Norm),stream));
  difference<<<128,256,0,stream>>>(p,D,reference.as<double>(),norms.as<Norm>(),true);
  difference<<<128,256,0,stream>>>(p,dense,reference.as<double>(),norms.as<Norm>()+1,true);
  check(cudaPeekAtLastError());
  Norm host[2]{};
  check(cudaMemcpyAsync(host,norms.data,sizeof(host),cudaMemcpyDeviceToHost,stream));
  check(cudaStreamSynchronize(stream));
  for(const auto& n:host)
    if(n.nonfinite || !std::isfinite(n.error) || !std::isfinite(n.reference) || n.error<0 || n.reference<0)return false;
  const double candidate_error=std::sqrt(host[0].error/std::max(host[0].reference,1e-300));
  const double dense_error=std::sqrt(host[1].error/std::max(host[1].reference,1e-300));
  const double floor=p.dtype==DType::BF16?1e-3:1e-4;
  return std::isfinite(candidate_error) && std::isfinite(dense_error) && candidate_error<=3*dense_error+floor;
}
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
bool original_fp64_agreement(const Problem& p,const void* A,const void* B,
                             const float* D,const float* dense,cudaStream_t stream) {
  if(p.beta!=0 || p.m<=0 || p.n<=0 || p.k<=0 || std::max({p.m,p.n,p.k})>32768)return false;
  if(p.dtype==DType::BF16)return original_fp64_agreement_typed<__nv_bfloat16>(p,A,B,D,dense,stream);
  return original_fp64_agreement_typed<__half>(p,A,B,D,dense,stream);
}
}

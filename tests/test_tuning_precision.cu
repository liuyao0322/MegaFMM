#include "internal.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
struct Buffer {
  void* p=nullptr;explicit Buffer(size_t bytes){check(cudaMalloc(&p,bytes));}
  ~Buffer(){cudaFree(p);}
};
template<class T>T rounded(float value);
template<>__half rounded(float value){return __float2half_rn(value);}
template<>__nv_bfloat16 rounded(float value){return __float2bfloat16_rn(value);}
template<class T>void test(megafmm::DType dtype,bool transpose){
  using namespace megafmm;
  Problem p;p.m=3;p.n=8;p.k=16;p.lda=24;p.ldb=24;p.ldc=p.ldd=12;
  p.dtype=dtype;p.alpha=0.5f;p.beta=0;p.trans_a=p.trans_b=transpose;
  std::vector<T> a(size_t(transpose?p.k:p.m)*p.lda,rounded<T>(1));
  std::vector<T> b(size_t(transpose?p.n:p.k)*p.ldb,rounded<T>(1));
  std::vector<float> output(p.m*p.ldd,-777),dense(output);
  Buffer da(a.size()*2),db(b.size()*2),dd(output.size()*4),dr(output.size()*4);
  check(cudaMemcpy(da.p,a.data(),a.size()*2,cudaMemcpyHostToDevice));
  check(cudaMemcpy(db.p,b.data(),b.size()*2,cudaMemcpyHostToDevice));
  auto upload=[&](){check(cudaMemcpy(dd.p,output.data(),output.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(dr.p,dense.data(),dense.size()*4,cudaMemcpyHostToDevice));};
  auto gate=[&](){return detail::original_fp64_agreement(p,da.p,db.p,static_cast<float*>(dd.p),static_cast<float*>(dr.p),nullptr);};
  auto set_values=[&](float candidate,float baseline){
    for(int64_t r=0;r<p.m;++r)for(int64_t c=0;c<p.n;++c){output[r*p.ldd+c]=candidate;dense[r*p.ldd+c]=baseline;}
    upload();
  };
  // A more accurate candidate must survive disagreement with rounded Dense.
  set_values(8,8*(1+2e-5f));
  require(!detail::dense_agreement(p,static_cast<float*>(dd.p),static_cast<float*>(dr.p),nullptr),"fixture did not reproduce cheap-screen rejection");
  require(gate(),"FP64 guard rejected exact output when Dense disagrees");
  const float floor=dtype==DType::BF16?1e-3f:1e-4f;
  set_values(8*(1+0.75f*floor),8);require(gate(),"FP64 guard rejected error below floor");
  set_values(8*(1+1.5f*floor),8);require(!gate(),"FP64 guard accepted error above floor");
  set_values(8*(1+floor+4e-5f),8*(1+2e-5f));require(gate(),"FP64 guard omitted three-times-Dense allowance");
  set_values(8*(1+floor+8e-5f),8*(1+2e-5f));require(!gate(),"FP64 guard exceeded original three-times-Dense gate");
  set_values(8.08f,8);
  require(!gate(),"FP64 guard accepted one-percent corruption");
  set_values(std::numeric_limits<float>::quiet_NaN(),8);
  require(!gate(),"FP64 guard accepted nonfinite candidate");
  set_values(8,std::numeric_limits<float>::infinity());
  require(!gate(),"FP64 guard accepted nonfinite Dense baseline");
  // Recompute from the current inputs; a previously correct output becomes stale.
  for(auto& x:a)x=rounded<T>(2);
  check(cudaMemcpy(da.p,a.data(),a.size()*2,cudaMemcpyHostToDevice));
  set_values(8,16);require(!gate(),"FP64 guard reused stale input values");
  set_values(16,16);require(gate(),"FP64 guard rejected updated input/output");
  p.alpha=0;set_values(0,0);require(gate(),"FP64 guard rejected zero reference");
  set_values(1e-20f,0);require(!gate(),"FP64 guard accepted error against zero reference");
  p.beta=1;require(!gate(),"beta-zero guard accepted nonzero beta");
  printf("PASS FP64 tuning guard dtype=%s transpose=%d alpha=checked strides=checked fresh_inputs=checked corruption=reject nonfinite=reject zero_reference=checked\n",dtype_name(dtype),int(transpose));
}
}
int main(){try{
  for(bool transpose:{false,true}){test<__nv_bfloat16>(megafmm::DType::BF16,transpose);test<__half>(megafmm::DType::FP16,transpose);}
  return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

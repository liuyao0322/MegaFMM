#include "megafmm/gemm.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t status) { if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
void require(bool condition,const std::string& message) { if (!condition) throw std::runtime_error(message); }
struct Buffer {
  void* data=nullptr;
  explicit Buffer(size_t bytes) { if (bytes) check(cudaMalloc(&data,bytes)); }
  ~Buffer() { if(data) cudaFree(data); }
  Buffer(const Buffer&)=delete;
  Buffer& operator=(const Buffer&)=delete;
  template<class T> T* as() const { return static_cast<T*>(data); }
};
struct Stream {
  cudaStream_t value=nullptr;
  Stream() { check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking)); }
  ~Stream() { if(value)cudaStreamDestroy(value); }
};
struct Graph {
  cudaGraph_t value=nullptr;cudaGraphExec_t executable=nullptr;
  ~Graph() { if(executable)cudaGraphExecDestroy(executable);if(value)cudaGraphDestroy(value); }
};
template<class T> T round_value(float v);
template<> __half round_value(float v) { return __float2half_rn(v); }
template<> __nv_bfloat16 round_value(float v) { return __float2bfloat16_rn(v); }
template<class T> double double_value(T v) { return double(float(v)); }
template<> double double_value(__nv_bfloat16 v) { return double(__bfloat162float(v)); }
struct Metrics { double maxabs=0, ef=0, scaled=0, normalized=0, nearzero=0; size_t nonfinite=0; };
Metrics metrics(const std::vector<float>& actual,const std::vector<double>& reference,
                int m,int n,int ldd,double floor,double rtol,double normab) {
  Metrics out; double err2=0,ref2=0;
  for(int i=0;i<m;++i) for(int j=0;j<n;++j) {
    double a=actual[size_t(i)*ldd+j],r=reference[size_t(i)*n+j];
    if(!std::isfinite(a) || !std::isfinite(r)) { ++out.nonfinite; continue; }
    double e=std::abs(a-r); err2+=e*e; ref2+=r*r;
    out.maxabs=std::max(out.maxabs,e);
    out.scaled=std::max(out.scaled,e/(floor+rtol*std::abs(r)));
    if(std::abs(r)<=floor) out.nearzero=std::max(out.nearzero,e);
  }
  out.ef=std::sqrt(err2/std::max(ref2,std::numeric_limits<double>::min()));
  out.normalized=std::sqrt(err2)/std::max(normab,std::numeric_limits<double>::min());
  return out;
}
enum class Pattern { Integers, Zero, Identity, Cancellation, Uniform, Positive, Range, HighHalf };
const char* pattern_name(Pattern p) {
  switch(p) {
    case Pattern::Integers:return "integers";case Pattern::Zero:return "zero";
    case Pattern::Identity:return "identity";case Pattern::Cancellation:return "cancellation";
    case Pattern::Uniform:return "uniform";case Pattern::Positive:return "positive";
    case Pattern::Range:return "range";case Pattern::HighHalf:return "fp16_upper_range";
  } return "unknown";
}
std::string algorithm_name(megafmm::Algorithm algorithm) {
  switch(algorithm) {
    case megafmm::Algorithm::Auto:return "auto";case megafmm::Algorithm::Dense:return "dense";
    case megafmm::Algorithm::WinogradLt:return "lt";case megafmm::Algorithm::WinogradHopper:return "hopper";
    case megafmm::Algorithm::Megakernel:return "mega";
  } return "unknown";
}
const char* warp_mma_name(int variant,const megafmm::Problem& p) {
  if(variant==14 || variant==15 || variant==16) {
    bool full=p.m%32==0 && p.n%16==0 && p.k%32==0 && p.lda%8==0 && p.ldb%8==0;
    if(!full)return "megakernel_warp_mma_packed";
    if(p.ldd%2!=0 || (p.beta!=0.0f && p.ldc%2!=0))return "megakernel_warp_mma_16x8_vec4";
    if(variant>=15 && p.alpha==1.0f && p.beta==0.0f)
      return variant==16 ? "megakernel_warp_mma_16x8_prefetch":"megakernel_warp_mma_16x8_identity";
    return "megakernel_warp_mma_16x8_epilogue";
  }
  if(variant==13)return p.m%32==0 && p.n%32==0 && p.k%32==0 &&
                         p.lda%8==0 && p.ldb%8==0 && p.ldc%8==0 && p.ldd%8==0 ?
                         "megakernel_product_cluster":"megakernel_product_warps";
  if(variant==11 || variant==12)return p.m%32==0 && p.n%16==0 && p.k%32==0 && p.lda%8==0 && p.ldb%8==0 ?
                         (variant==11 ? "megakernel_warp_mma_16x8":"megakernel_warp_mma_16x8_vec4"):
                         "megakernel_warp_mma_packed";
  if(variant==10)return p.m%32==0 && p.n%32==0 && p.k%32==0 &&
                         p.lda%8==0 && p.ldb%8==0 && p.ldc%8==0 && p.ldd%8==0 ?
                         "megakernel_product_warps_fast":"megakernel_product_warps";
  if(variant==9)return "megakernel_product_warps";
  if(variant<=6)return "megakernel_warp_mma";
  if(variant==8 && p.m%32==0 && p.n%32==0 && p.k%32==0)return "megakernel_warp_mma_fast";
  return "megakernel_warp_mma_packed";
}
template<class T> Metrics run_case(megafmm::Problem p,megafmm::Algorithm algorithm,
                                  Pattern pattern,bool alias,bool certified,uint32_t seed=1234567,int variant=0,
                                  bool do_tune=true,size_t workspace_limit=size_t(8)<<30,
                                  const char* expected_route=nullptr) {
  int ar=p.trans_a ? p.k:p.m, ac=p.trans_a ? p.m:p.k;
  int br=p.trans_b ? p.n:p.k, bc=p.trans_b ? p.k:p.n;
  std::vector<T> a(size_t(ar)*p.lda),b(size_t(br)*p.ldb);
  std::vector<float> c(size_t(p.m)*p.ldc,0),result(size_t(p.m)*p.ldd,-777.0f);
  std::mt19937 rng(seed);
  auto value=[&](int i,int j,bool left) {
    if(pattern==Pattern::Zero) return 0.0f;
    if(pattern==Pattern::Identity && left) return i==j ? 1.0f:0.0f;
    if(pattern==Pattern::Integers || pattern==Pattern::Identity) return float(int(rng()%5)-2);
    if(pattern==Pattern::Cancellation) {
      // Exact dyadic perturbations and opposite neighboring terms exercise near-zero outputs.
      return (j%2 ? -1.0f:1.0f)*(1.0f+float((i+j)%3)/32.0f);
    }
    float v=float(rng()>>8)/16777216.0f;
    if(pattern==Pattern::Positive) return v;
    if(pattern==Pattern::Range) return (2*v-1)*16;
    if(pattern==Pattern::HighHalf) return (left ? 65504.0f:0.00006103515625f)*(j%2 ? -1.0f:1.0f);
    return 2*v-1;
  };
  for(int i=0;i<p.m;++i) for(int j=0;j<p.k;++j) {
    size_t at=p.trans_a ? size_t(j)*p.lda+i:size_t(i)*p.lda+j;
    a[at]=round_value<T>(value(i,j,true));
  }
  for(int i=0;i<p.k;++i) for(int j=0;j<p.n;++j) {
    size_t at=p.trans_b ? size_t(j)*p.ldb+i:size_t(i)*p.ldb+j;
    b[at]=round_value<T>(value(i,j,false));
  }
  for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j) c[size_t(i)*p.ldc+j]=float(int(rng()%9)-4)/4;
  double anorm=0,bnorm=0;
  for(int i=0;i<p.m;++i) for(int l=0;l<p.k;++l) {
    double v=double_value(a[p.trans_a ? size_t(l)*p.lda+i:size_t(i)*p.lda+l]);anorm+=v*v;
  }
  for(int l=0;l<p.k;++l) for(int j=0;j<p.n;++j) {
    double v=double_value(b[p.trans_b ? size_t(j)*p.ldb+l:size_t(l)*p.ldb+j]);bnorm+=v*v;
  }
  std::vector<double> reference(size_t(p.m)*p.n);
  // FP64 plain AB from the original rounded inputs, with no transformed operands.
  for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j) {
    double sum=0;
    for(int l=0;l<p.k;++l)
      sum+=double_value(a[p.trans_a ? size_t(l)*p.lda+i:size_t(i)*p.lda+l]) *
           double_value(b[p.trans_b ? size_t(j)*p.ldb+l:size_t(l)*p.ldb+j]);
    reference[size_t(i)*p.n+j]=double(p.alpha)*sum+double(p.beta)*c[size_t(i)*p.ldc+j];
  }
  Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dc(c.size()*sizeof(float)),dd(result.size()*sizeof(float));
  check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
  check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
  check(cudaMemcpy(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
  check(cudaMemcpy(dd.data,result.data(),result.size()*sizeof(float),cudaMemcpyHostToDevice));
  megafmm::Policy policy;policy.algorithm=algorithm;policy.certified_range=certified;
  if(algorithm==megafmm::Algorithm::WinogradHopper)policy.hopper_variant=variant;
  if(algorithm==megafmm::Algorithm::Megakernel)policy.mega_variant=variant;
  if(algorithm==megafmm::Algorithm::WinogradLt)policy.lt_variant=variant;
  policy.workspace_limit=workspace_limit;
  auto plan=megafmm::create_plan(p,policy);
  if(expected_route)require(plan->algorithm_name()==expected_route,
                           std::string("unexpected route: expected ")+expected_route+", got "+plan->algorithm_name());
  if(expected_route && (std::string(expected_route).find("megakernel_warp_mma")==0 ||
                        std::string(expected_route).find("megakernel_product_warps")==0 ||
                        std::string(expected_route)=="megakernel_product_cluster"))
    require(plan->workspace_bytes()==0,"warp MMA unexpectedly requires workspace");
  Buffer workspace(plan->workspace_bytes());
  float* output=alias ? dc.as<float>():dd.as<float>();
  require(!alias || p.ldc==p.ldd,"test alias needs equal C/D strides");
  if(do_tune) {
    megafmm::tune(*plan,da.data,db.data,p.beta==0 ? nullptr:dc.as<float>(),output,workspace.data);
    if(alias && pattern==Pattern::Integers) {
      check(cudaMemcpy(result.data(),output,result.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
        require(double(result[size_t(i)*p.ldd+j])==reference[size_t(i)*p.n+j],
                "tuned in-place alpha/beta output incorrect");
    }
  }
  // Restore C0 after tuning when C0 and D share storage.
  if(alias) check(cudaMemcpy(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
  for(int repeat=0;repeat<2;++repeat) {
    if(alias) check(cudaMemcpy(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
    megafmm::run(*plan,da.data,db.data,p.beta==0 ? nullptr:dc.as<float>(),output,workspace.data);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(result.data(),output,result.size()*sizeof(float),cudaMemcpyDeviceToHost));
  }
  double floor=p.dtype==megafmm::DType::BF16 ? 1e-3:1e-4,rtol=p.dtype==megafmm::DType::BF16 ? 1e-2:1e-3;
  Metrics e=metrics(result,reference,p.m,p.n,p.ldd,floor,rtol,std::sqrt(anorm*bnorm));
  std::cout << "CASE " << algorithm_name(algorithm) << " variant=" << variant
            << " actual=" << plan->algorithm_name() << ' '
            << megafmm::dtype_name(p.dtype) << ' ' << pattern_name(pattern) << ' '
            << p.m << ',' << p.n << ',' << p.k << " maxabs=" << e.maxabs << " eF=" << e.ef
            << " scaled=" << e.scaled << " normalized=" << e.normalized << " nonfinite=" << e.nonfinite << '\n';
  for(int i=0;i<p.m;++i) for(int j=p.n;j<p.ldd;++j)
    require(result[size_t(i)*p.ldd+j]==(alias ? c[size_t(i)*p.ldc+j]:-777.0f),"output padding overwritten");
  return e;
}
megafmm::Problem problem(int m,int n,int k,megafmm::DType dtype) {
  megafmm::Problem p;p.m=m;p.n=n;p.k=k;p.lda=k;p.ldb=n;p.ldc=n;p.ldd=n;p.dtype=dtype;return p;
}
template<class T> void structural_suite(megafmm::DType dtype) {
  const std::vector<megafmm::Algorithm> algorithms={megafmm::Algorithm::Dense,megafmm::Algorithm::WinogradLt,
    megafmm::Algorithm::WinogradHopper,megafmm::Algorithm::Megakernel,megafmm::Algorithm::Auto};
  for(auto algo:algorithms) {
    for(auto pat:{Pattern::Integers,Pattern::Zero,Pattern::Identity,Pattern::Cancellation}) {
      auto e=run_case<T>(problem(128,128,128,dtype),algo,pat,false,true);
      require(e.nonfinite==0 && e.maxabs==0,"exact structural DAG test failed");
    }
    for(auto dims:{std::vector<int>{65,97,33},std::vector<int>{1,1,1},std::vector<int>{128,64,256},std::vector<int>{130,66,258}}) {
      auto e=run_case<T>(problem(dims[0],dims[1],dims[2],dtype),algo,Pattern::Integers,false,true);
      require(e.nonfinite==0 && e.maxabs==0,"rectangular/boundary structural test failed");
    }
    auto p=problem(31,47,29,dtype);p.lda+=3;p.ldb+=5;p.ldc+=7;p.ldd+=9;p.alpha=.5f;p.beta=-.25f;
    auto e=run_case<T>(p,algo,Pattern::Integers,false,true);
    require(e.nonfinite==0 && e.maxabs==0,"stride alpha/beta test failed");
    p.ldc=p.ldd;
    e=run_case<T>(p,algo,Pattern::Integers,true,true);
    require(e.nonfinite==0 && e.maxabs==0,"C0/D alias test failed");
    // These aligned shapes reach forced SW/Hopper/mega rather than relying on boundary fallback.
    p=problem(128,128,128,dtype);p.alpha=.5f;p.beta=-.25f;
    e=run_case<T>(p,algo,Pattern::Integers,false,true);
    require(e.nonfinite==0 && e.maxabs==0,"fast-path alpha/beta test failed");
    e=run_case<T>(p,algo,Pattern::Integers,true,true);
    require(e.nonfinite==0 && e.maxabs==0,"fast-path C0/D alias test failed");
    p.lda+=8;p.ldb+=8;p.ldc+=8;p.ldd+=8;
    e=run_case<T>(p,algo,Pattern::Integers,false,true);
    require(e.nonfinite==0 && e.maxabs==0,"aligned padded-stride alpha/beta test failed");
    e=run_case<T>(p,algo,Pattern::Integers,true,true);
    require(e.nonfinite==0 && e.maxabs==0,"aligned padded-stride C0/D alias test failed");
    for(bool ta:{false,true}) for(bool tb:{false,true}) {
      p=problem(17,23,19,dtype);p.trans_a=ta;p.trans_b=tb;
      p.lda=(ta ? p.m:p.k)+5;p.ldb=(tb ? p.k:p.n)+7;p.alpha=-.5f;p.beta=.25f;
      e=run_case<T>(p,algo,Pattern::Integers,false,true);
      require(e.nonfinite==0 && e.maxabs==0,"transpose fallback test failed");
    }
    // Ordinary random and larger magnitudes exercise dense fallback under an unknown range declaration.
    for(auto pat:{Pattern::Uniform,Pattern::Positive,Pattern::Range,Pattern::HighHalf}) {
      p=problem(37,43,31,dtype);
      auto reference_dense=run_case<T>(p,megafmm::Algorithm::Dense,pat,false,false,9876543);
      e=run_case<T>(p,algo,pat,false,false,9876543);
      double floor=dtype==megafmm::DType::BF16 ? 1e-3:1e-4;
      require(e.nonfinite==0 && e.ef<=3*reference_dense.ef+floor,"unknown-range dense fallback numerical gate failed");
    }
  }
  for(auto candidate:{std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,1},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,2},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,3},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,4},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,5},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,6},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,7},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,8},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,9},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,10},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,11},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,12},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,13},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,14},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,15},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,16},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,1},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,2}}) {
    auto e=run_case<T>(problem(256,256,256,dtype),candidate.first,Pattern::Integers,false,true,1234567,candidate.second);
    require(e.nonfinite==0 && e.maxabs==0,"candidate variant structural test failed");
  }
  for(int variant:{2,3}) {
    auto p=problem(256,256,256,dtype);
    p.alpha=.5f;p.beta=-.25f;
    auto e=run_case<T>(p,megafmm::Algorithm::Megakernel,Pattern::Integers,true,true,1234567,variant);
    require(e.nonfinite==0 && e.maxabs==0,"streaming megakernel alpha/beta alias test failed");
  }
}
template<class T> void batched_suite(megafmm::DType dtype,bool lt_only=false) {
  for(auto candidate:{std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradLt,1},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,5},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,6}}) {
  const auto algorithm=candidate.first;const int variant=candidate.second;
  if(lt_only && algorithm!=megafmm::Algorithm::WinogradLt)continue;
  auto exact=[dtype,algorithm,variant](megafmm::Problem p,Pattern pattern,bool alias=false,bool certified=true,
                    bool expect_batched=true,size_t workspace_limit=size_t(8)<<30,bool do_tune=false) {
    const char* route=!expect_batched ? "dense_cublaslt":
                     algorithm==megafmm::Algorithm::WinogradLt ? "winograd_lt_batched":"winograd_sm80_batched";
    std::cout << "BEGIN_BATCHED dtype=" << megafmm::dtype_name(dtype)
              << " algorithm=" << algorithm_name(algorithm) << " variant=" << variant
              << " pattern=" << pattern_name(pattern) << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
              << " certified_range=" << certified << " workspace_limit=" << workspace_limit
              << " expected=" << route << std::endl;
    auto e=run_case<T>(p,algorithm,pattern,alias,certified,1234567,variant,
                       do_tune,workspace_limit,route);
    require(e.nonfinite==0 && e.maxabs==0,"batched structural/alias/fallback test failed");
  };
  // Small exact operands exercise all seven slot mappings and the fixed U DAG.
  // Tune one case; the remaining scenarios also cover the untuned route.
  for(auto pattern:{Pattern::Integers,Pattern::Zero,Pattern::Identity,Pattern::Cancellation})
    exact(problem(128,128,128,dtype),pattern,false,true,true,size_t(8)<<30,pattern==Pattern::Integers);
  exact(problem(128,64,256,dtype),Pattern::Integers);
  exact(problem(256,256,256,dtype),Pattern::Integers);
  auto p=problem(128,128,128,dtype);p.alpha=.5f;p.beta=-.25f;
  exact(p,Pattern::Integers);
  // Verify tune's output before restoring the original C0 for repeated run calls.
  exact(p,Pattern::Integers,true,true,true,size_t(8)<<30,true);
  p.lda+=8;p.ldb+=8;p.ldc+=8;p.ldd+=8;
  exact(p,Pattern::Integers);
  exact(p,Pattern::Integers,true);
  // These cases must select Dense even though a batched route was requested.
  exact(problem(128,128,128,dtype),Pattern::Integers,false,false,false);
  exact(problem(65,97,33,dtype),Pattern::Integers,false,true,false);
  for(auto dims:{std::vector<int>{130,128,128},std::vector<int>{128,130,128},std::vector<int>{128,128,130}}) {
    p=problem(dims[0],dims[1],dims[2],dtype);p.lda=(p.lda+7)/8*8;p.ldb=(p.ldb+7)/8*8;
    exact(p,Pattern::Integers,false,true,false);
  }
  exact(problem(128,128,128,dtype),Pattern::Integers,false,true,false,0);
  p=problem(128,128,128,dtype);p.lda+=1;
  exact(p,Pattern::Integers,false,true,false);
  p=problem(128,128,128,dtype);p.ldb+=1;
  exact(p,Pattern::Integers,false,true,false);
  p=problem(128,128,128,dtype);p.trans_a=true;
  exact(p,Pattern::Integers,false,true,false);
  p=problem(128,128,128,dtype);p.trans_b=true;
  exact(p,Pattern::Integers,false,true,false);
  }
}
template<class T> void warp_mma_suite(megafmm::DType dtype) {
  for(int variant=4;variant<=16;++variant) {
    auto exact=[dtype,variant](megafmm::Problem p,Pattern pattern,bool alias=false,bool certified=true,
                              bool expect_warp=true,bool do_tune=false) {
      const char* route=expect_warp ? warp_mma_name(variant,p):"dense_cublaslt";
      std::cout << "BEGIN_WARP_MMA dtype=" << megafmm::dtype_name(dtype) << " variant=" << variant
                << " pattern=" << pattern_name(pattern) << " shape=" << p.m << ',' << p.n << ',' << p.k
                << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
                << " certified_range=" << certified << " expected=" << route << std::endl;
      auto e=run_case<T>(p,megafmm::Algorithm::Megakernel,pattern,alias,certified,1234567,variant,
                         do_tune,0,route);
      require(e.nonfinite==0 && e.maxabs==0,"warp MMA structural/alias/fallback test failed");
    };
    for(auto pattern:{Pattern::Integers,Pattern::Zero,Pattern::Identity,Pattern::Cancellation})
      exact(problem(128,128,128,dtype),pattern);
    if(variant>=7) {
      exact(problem(256,256,256,dtype),Pattern::Integers);
      auto full=problem(128,128,128,dtype);full.alpha=.5f;full.beta=-.25f;
      full.lda+=8;full.ldb+=8;full.ldc+=8;full.ldd+=8;
      exact(full,Pattern::Integers,true,true,true,true);
      if(variant==10 || variant==13 || variant==14 || variant==15 || variant==16) {
        auto output_padded=problem(128,128,128,dtype);output_padded.alpha=.5f;output_padded.beta=-.25f;
        output_padded.ldc+=1;output_padded.ldd+=1;
        exact(output_padded,Pattern::Integers,true,true,true,true);
      }
      if(variant==14 || variant==15 || variant==16) {
        auto output_padded=problem(128,128,128,dtype);output_padded.alpha=.5f;output_padded.beta=-.25f;
        output_padded.ldd+=1;
        exact(output_padded,Pattern::Integers);
        output_padded=problem(128,128,128,dtype);output_padded.alpha=.5f;output_padded.beta=-.25f;
        output_padded.ldc+=1;
        exact(output_padded,Pattern::Integers);
        output_padded=problem(128,128,128,dtype);output_padded.ldc+=1;
        exact(output_padded,Pattern::Integers);
      }
      if(variant==15 || variant==16) {
        auto identity=problem(128,128,128,dtype);
        identity.lda+=8;identity.ldb+=8;identity.ldc+=8;identity.ldd+=8;
        exact(identity,Pattern::Integers,false,true,true,true);
        auto alpha_only=problem(128,128,128,dtype);alpha_only.alpha=.5f;
        exact(alpha_only,Pattern::Integers);
        auto negative_zero=problem(128,128,128,dtype);negative_zero.beta=-0.0f;negative_zero.ldc+=1;
        exact(negative_zero,Pattern::Integers);
      }
    }
    // Partial output/K tiles, including a final CTA with inactive owner warps.
    exact(problem(80,80,48,dtype),Pattern::Integers);
    auto p=problem(80,112,48,dtype);
    exact(p,Pattern::Integers);
    p.alpha=.5f;p.beta=-.25f;
    exact(p,Pattern::Integers);
    exact(p,Pattern::Integers,true,true,true,true);
    p.lda+=8;p.ldb+=8;p.ldc+=8;p.ldd+=8;
    exact(p,Pattern::Integers);
    exact(p,Pattern::Integers,true);
    exact(problem(128,128,128,dtype),Pattern::Integers,false,false,false);
    exact(problem(65,97,33,dtype),Pattern::Integers,false,true,false);
    for(auto dims:{std::vector<int>{130,128,128},std::vector<int>{128,130,128},std::vector<int>{128,128,130}}) {
      p=problem(dims[0],dims[1],dims[2],dtype);p.lda=(p.lda+7)/8*8;p.ldb=(p.ldb+7)/8*8;
      exact(p,Pattern::Integers,false,true,false);
    }
    p=problem(128,128,128,dtype);p.lda+=1;
    exact(p,Pattern::Integers,false,true,false);
    p=problem(128,128,128,dtype);p.ldb+=1;
    exact(p,Pattern::Integers,false,true,false);
    p=problem(128,128,128,dtype);p.trans_a=true;
    exact(p,Pattern::Integers,false,true,false);
    p=problem(128,128,128,dtype);p.trans_b=true;
    exact(p,Pattern::Integers,false,true,false);
  }
}
template<class T> void prefetch_short_k_suite(megafmm::DType dtype) {
  // Half-K holds one, two, or three MMA tiles: the next-bank prefetch executes
  // zero, one, or two times, including the final iteration that must not load.
  for(auto mn:{std::pair<int,int>{128,128},std::pair<int,int>{32,16},std::pair<int,int>{32,48}})
  for(int k:{32,64,96}) {
    auto p=problem(mn.first,mn.second,k,dtype);
    std::cout << "BEGIN_PREFETCH_SHORT_K dtype=" << megafmm::dtype_name(dtype)
              << " variant=16 shape=" << p.m << ',' << p.n << ',' << p.k << std::endl;
    auto e=run_case<T>(p,megafmm::Algorithm::Megakernel,Pattern::Integers,false,true,
                       1234567,16,false,0,"megakernel_warp_mma_16x8_prefetch");
    require(e.nonfinite==0 && e.maxabs==0,"prefetch short-K final-bank test failed");
  }
}
template<class T> void warp_mma_bitwise_suite(megafmm::DType dtype) {
  for(auto dims:{std::vector<int>{128,128,128},std::vector<int>{80,112,48},
                std::vector<int>{32,16,32},std::vector<int>{32,48,32}})
  for(int scenario=0;scenario<3;++scenario) {
    auto p=problem(dims[0],dims[1],dims[2],dtype);
    bool alias=scenario==2;
    if(scenario!=0) {p.alpha=.375f;p.beta=-.625f;}
    uint32_t seed=424242u+uint32_t(p.m*31+p.n*17+p.k*7+scenario*131)+
                  (dtype==megafmm::DType::BF16 ? 0u:918273u);
    std::mt19937 rng(seed);
    std::vector<T> a(size_t(p.m)*p.lda),b(size_t(p.k)*p.ldb);
    std::vector<float> c(size_t(p.m)*p.ldc),actual(size_t(p.m)*p.ldd),baseline(actual.size());
    // Fresh exact dyadic inputs retain nontrivial rounding in transformed S/T.
    for(auto& value:a)value=round_value<T>(float(int(rng()%2049)-1024)/1024.0f);
    for(auto& value:b)value=round_value<T>(float(int(rng()%2049)-1024)/1024.0f);
    for(auto& value:c)value=float(int(rng()%257)-128)/64.0f;
    Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dc(c.size()*sizeof(float)),
           dd(actual.size()*sizeof(float));
    check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
    std::cout << "BEGIN_MMA_BITWISE dtype=" << megafmm::dtype_name(dtype)
              << " shape=" << p.m << ',' << p.n << ',' << p.k << " alpha=" << p.alpha
              << " beta=" << p.beta << " alias=" << alias << " seed=" << seed << std::endl;
    for(int variant:{4,7,8,9,10,11,12,13,14,15,16}) {
      megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
      policy.certified_range=true;policy.mega_variant=variant;
      auto plan=megafmm::create_plan(p,policy);
      require(plan->algorithm_name()==warp_mma_name(variant,p),"MMA bitwise test selected another route");
      require(plan->workspace_bytes()==0,"MMA bitwise test unexpectedly requires workspace");
      check(cudaMemcpy(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
      if(!alias)check(cudaMemset(dd.data,0xff,actual.size()*sizeof(float)));
      float* output=alias ? dc.as<float>():dd.as<float>();
      megafmm::run(*plan,da.data,db.data,p.beta==0 ? nullptr:dc.as<float>(),output,nullptr);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(float value:actual)require(std::isfinite(value),"MMA bitwise test returned nonfinite output");
      if(variant==4)baseline=actual;
      else require(std::memcmp(baseline.data(),actual.data(),actual.size()*sizeof(float))==0,
                   "Mega4 vs Mega"+std::to_string(variant)+" output bits differ on fresh dyadic inputs");
    }
    std::cout << "PASS_MMA_BITWISE dtype=" << megafmm::dtype_name(dtype)
              << " shape=" << p.m << ',' << p.n << ',' << p.k << " alpha=" << p.alpha
              << " beta=" << p.beta << " alias=" << alias << " variants=4,7,8,9,10,11,12,13,14,15,16\n";
  }
}
template<class T> void cluster_graph_suite(megafmm::DType dtype) {
  for(auto dims:{std::vector<int>{32,32,32},std::vector<int>{96,96,96},std::vector<int>{80,112,48}}) {
    auto p=problem(dims[0],dims[1],dims[2],dtype);p.alpha=.5f;p.beta=-.25f;
    p.lda+=8;p.ldb+=8;p.ldc+=8;p.ldd+=8;
    std::mt19937 rng(998877u+uint32_t(p.m*31+p.n*17+p.k*7));
    std::vector<T> a(size_t(p.m)*p.lda),b(size_t(p.k)*p.ldb);
    std::vector<float> c(size_t(p.m)*p.ldc,-777.0f),actual(c.size());
    for(int i=0;i<p.m;++i) for(int l=0;l<p.k;++l)
      a[size_t(i)*p.lda+l]=round_value<T>(float(int(rng()%5)-2));
    for(int l=0;l<p.k;++l) for(int j=0;j<p.n;++j)
      b[size_t(l)*p.ldb+j]=round_value<T>(float(int(rng()%5)-2));
    std::vector<double> reference_ab(size_t(p.m)*p.n);
    // Independent FP64 plain AB uses the original rounded inputs.
    for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
      for(int l=0;l<p.k;++l)
        reference_ab[size_t(i)*p.n+j]+=double_value(a[size_t(i)*p.lda+l])*double_value(b[size_t(l)*p.ldb+j]);
    Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dc(c.size()*sizeof(float));
    check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
    check(cudaMemcpy(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaDeviceSynchronize());
    megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
    policy.certified_range=true;policy.mega_variant=13;
    auto plan=megafmm::create_plan(p,policy);
    require(plan->algorithm_name()==warp_mma_name(13,p),"cluster graph selected another route");
    require(plan->workspace_bytes()==0,"cluster graph unexpectedly requires workspace");
    std::cout << "BEGIN_CLUSTER_GRAPH dtype=" << megafmm::dtype_name(dtype)
              << " shape=" << p.m << ',' << p.n << ',' << p.k << " actual=" << plan->algorithm_name()
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=1" << std::endl;
    Stream stream;Graph graph;
    check(cudaStreamSynchronize(stream.value));
    check(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));
    megafmm::run(*plan,da.data,db.data,dc.as<float>(),dc.as<float>(),nullptr,stream.value);
    check(cudaStreamEndCapture(stream.value,&graph.value));
    check(cudaGraphInstantiate(&graph.executable,graph.value,0));
    for(int replay=0;replay<3;++replay) {
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
        c[size_t(i)*p.ldc+j]=float((i+j+3*replay)%9-4)/4.0f;
      // Restore a fresh C0 on the same stream before every graph launch.
      check(cudaMemcpyAsync(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice,stream.value));
      check(cudaGraphLaunch(graph.executable,stream.value));
      check(cudaStreamSynchronize(stream.value));
      check(cudaMemcpy(actual.data(),dc.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j) {
        double reference=double(p.alpha)*reference_ab[size_t(i)*p.n+j]+double(p.beta)*c[size_t(i)*p.ldc+j];
        require(double(actual[size_t(i)*p.ldd+j])==reference,"cluster graph replay alpha/beta output incorrect");
      }
      for(int i=0;i<p.m;++i) for(int j=p.n;j<p.ldd;++j)
        require(actual[size_t(i)*p.ldd+j]==-777.0f,"cluster graph replay overwrote output padding");
    }
    std::cout << "PASS_CLUSTER_GRAPH dtype=" << megafmm::dtype_name(dtype)
              << " shape=" << p.m << ',' << p.n << ',' << p.k << " actual=" << plan->algorithm_name()
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=1 replays=3\n";
  }
}
template<class T> void epilogue_output_rebind_suite(megafmm::DType dtype,int variant=14) {
  for(bool identity:{false,true}) {
  if(identity && variant!=15 && variant!=16)continue;
  auto p=problem(128,128,128,dtype);
  if(!identity) {p.alpha=.5f;p.beta=-.25f;}
  std::mt19937 rng(776655u);
  std::vector<T> a(size_t(p.m)*p.lda),b(size_t(p.k)*p.ldb);
  std::vector<float> c(size_t(p.m)*p.ldc),actual(c.size());
  for(auto& value:a)value=round_value<T>(float(int(rng()%5)-2));
  for(auto& value:b)value=round_value<T>(float(int(rng()%5)-2));
  for(auto& value:c)value=float(int(rng()%9)-4)/4.0f;
  std::vector<double> reference(size_t(p.m)*p.n);
  for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j) {
    double sum=0;
    for(int l=0;l<p.k;++l)
      sum+=double_value(a[size_t(i)*p.lda+l])*double_value(b[size_t(l)*p.ldb+j]);
    reference[size_t(i)*p.n+j]=double(p.alpha)*sum+double(p.beta)*c[size_t(i)*p.ldc+j];
  }
  Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dc((c.size()+1)*sizeof(float)),
         dd((actual.size()+1)*sizeof(float));
  check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
  check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
  policy.certified_range=true;policy.mega_variant=variant;
  auto plan=megafmm::create_plan(p,policy);
  require(plan->algorithm_name()==warp_mma_name(variant,p),"output rebind selected another route");
  require(plan->workspace_bytes()==0,"output rebind unexpectedly requires workspace");
  for(int mode=0;mode<3;++mode) {
    if(identity && mode==1)continue;
    std::cout << "BEGIN_EPILOGUE_OUTPUT_REBIND dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " alpha=" << p.alpha << " beta=" << p.beta
              << " mode=" << mode << " alias=" << (mode==2) << std::endl;
    for(int shift:{0,1,0}) {
      // D only, active C0 only, and exact C0=D each lose 8-byte alignment.
      float* source=dc.as<float>()+(mode==0 ? 0:shift);
      float* output=mode==2 ? source:dd.as<float>()+(mode==0 ? shift:0);
      check(cudaMemset(dc.data,0xff,(c.size()+1)*sizeof(float)));
      check(cudaMemset(dd.data,0xff,(actual.size()+1)*sizeof(float)));
      check(cudaMemcpy(source,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
      megafmm::run(*plan,da.data,db.data,p.beta==0.0f ? nullptr:source,output,nullptr);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(size_t i=0;i<actual.size();++i)
        require(double(actual[i])==reference[i],"same-plan 4-byte output/C0 rebind returned incorrect alpha/beta output");
    }
    std::cout << "PASS_EPILOGUE_OUTPUT_REBIND dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " alpha=" << p.alpha << " beta=" << p.beta
              << " mode=" << mode << " alias=" << (mode==2)
              << " aligned_then_float_aligned_then_aligned=1\n";
  }
  }
}
template<class T> void rebind_suite(megafmm::DType dtype,bool batched_only=false) {
  // Reuse a tuned plan with changed addresses: an aligned preferred SW route must
  // safely choose Dense when the new input views have only element alignment.
  auto p=problem(256,256,256,dtype);
  std::vector<T> a(size_t(p.m)*p.k),b(size_t(p.k)*p.n);
  for(int i=0;i<p.m;++i) for(int l=0;l<p.k;++l)
    a[size_t(i)*p.k+l]=round_value<T>(float(i%5-2));
  for(int l=0;l<p.k;++l) for(int j=0;j<p.n;++j)
    b[size_t(l)*p.n+j]=round_value<T>(float(j%7-3));
  Buffer da((a.size()+1)*sizeof(T)),db((b.size()+1)*sizeof(T)),dd(size_t(p.m)*p.n*sizeof(float));
  std::vector<float> actual(size_t(p.m)*p.n);
  for(auto candidate:{std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Dense,0},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradLt,0},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradLt,1},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,0},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,5},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,6},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,3},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,4},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,5},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,6},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,7},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,8},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,9},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,10},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,11},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,12},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,13},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,14},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,15},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,16}}) {
    auto algo=candidate.first;int variant=candidate.second;
    bool is_batched=(algo==megafmm::Algorithm::WinogradLt && variant==1) ||
                    (algo==megafmm::Algorithm::WinogradHopper && variant>=5);
    if(batched_only && !(algo==megafmm::Algorithm::WinogradLt && variant==1))continue;
    megafmm::Policy policy;policy.algorithm=algo;policy.certified_range=true;
    if(algo==megafmm::Algorithm::WinogradLt)policy.lt_variant=variant;
    if(algo==megafmm::Algorithm::WinogradHopper)policy.hopper_variant=variant;
    if(algo==megafmm::Algorithm::Megakernel)policy.mega_variant=variant;
    auto plan=megafmm::create_plan(p,policy);Buffer workspace(plan->workspace_bytes());
    if(is_batched)require(plan->algorithm_name()==(algo==megafmm::Algorithm::WinogradLt ?
                         "winograd_lt_batched":"winograd_sm80_batched"),"batched rebind test selected Dense");
    if(algo==megafmm::Algorithm::Megakernel && variant>=4) {
      require(plan->algorithm_name()==warp_mma_name(variant,p),"warp MMA rebind test selected another route");
      require(plan->workspace_bytes()==0,"warp MMA rebind unexpectedly requires workspace");
    }
    check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
    megafmm::tune(*plan,da.data,db.data,nullptr,dd.as<float>(),workspace.data);
    for(bool shifted:{false,true,false}) {
      T* left=da.as<T>()+int(shifted),*right=db.as<T>()+int(shifted);
      check(cudaMemcpy(left,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
      check(cudaMemcpy(right,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
      megafmm::run(*plan,left,right,nullptr,dd.as<float>(),workspace.data);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),dd.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
        require(actual[size_t(i)*p.n+j]==float(p.k*(i%5-2)*(j%7-3)),
                "same-plan pointer alignment rebind returned an incorrect product");
    }
    std::cout << "PASS_POINTER_REBIND dtype=" << megafmm::dtype_name(dtype)
              << " algorithm=" << algorithm_name(algo) << " variant=" << variant
              << " aligned_then_element_aligned_then_aligned=1\n";
  }
}
void batched_policy_rejection_suite(bool lt_only=false) {
  for(auto algorithm:{megafmm::Algorithm::Auto,megafmm::Algorithm::Dense,megafmm::Algorithm::WinogradLt,
                      megafmm::Algorithm::WinogradHopper,megafmm::Algorithm::Megakernel})
  for(bool empty:{false,true}) for(int family=0;family<3;++family) for(int variant:{-1,family==0 ? 2:family==1 ? 7:17}) {
    if(lt_only && family!=0)continue;
    megafmm::Policy policy;policy.algorithm=algorithm;policy.certified_range=true;
    if(family==0)policy.lt_variant=variant;
    else if(family==1)policy.hopper_variant=variant;
    else policy.mega_variant=variant;
    bool rejected=false;
    try { auto plan=megafmm::create_plan(problem(empty ? 0:128,128,128,megafmm::DType::BF16),policy); }
    catch(const std::exception&) { rejected=true; }
    require(rejected,family==0 ? "invalid Lt variant accepted":
                     family==1 ? "invalid Hopper variant accepted":"invalid Mega variant accepted");
  }
}
void degenerate_suite() {
  for(auto algo:{megafmm::Algorithm::Dense,megafmm::Algorithm::WinogradLt,megafmm::Algorithm::WinogradHopper,
                 megafmm::Algorithm::Megakernel,megafmm::Algorithm::Auto}) {
    megafmm::Policy policy;policy.algorithm=algo;policy.certified_range=true;
    for(int zero_dim:{0,1}) {
      auto p=problem(zero_dim==0 ? 0:9,zero_dim==1 ? 0:11,17,megafmm::DType::BF16);
      auto plan=megafmm::create_plan(p,policy);
      megafmm::run(*plan,nullptr,nullptr,nullptr,nullptr,nullptr);
      check(cudaDeviceSynchronize());
    }
    for(bool alpha_zero:{false,true}) for(float beta:{0.0f,.5f,-1.0f}) {
      auto p=problem(7,11,alpha_zero ? 13:0,megafmm::DType::FP16);p.alpha=alpha_zero ? 0:1;p.beta=beta;
      std::vector<float> c(size_t(p.m)*p.n),out(c.size());
      for(size_t i=0;i<c.size();++i)c[i]=float(int(i%9)-4)/4;
      Buffer dc(c.size()*sizeof(float)),dd(c.size()*sizeof(float));
      check(cudaMemcpy(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
      auto plan=megafmm::create_plan(p,policy);Buffer workspace(plan->workspace_bytes());
      megafmm::run(*plan,nullptr,nullptr,beta==0 ? nullptr:dc.as<float>(),dd.as<float>(),workspace.data);
      check(cudaDeviceSynchronize());check(cudaMemcpy(out.data(),dd.data,out.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(size_t i=0;i<c.size();++i)require(out[i]==beta*c[i],"degenerate beta C0 failed");
    }
  }
}
void rejection_suite() {
  auto rejected=[](megafmm::Problem p) {
    try { auto plan=megafmm::create_plan(p); } catch(const std::exception&) { return true; }
    return false;
  };
  auto p=problem(16,16,16,megafmm::DType::BF16);p.m=-1;require(rejected(p),"negative dimension accepted");
  p=problem(16,16,16,megafmm::DType::BF16);p.lda=15;require(rejected(p),"short lda accepted");
  p=problem(16,16,16,megafmm::DType::BF16);p.ldb=15;require(rejected(p),"short ldb accepted");
  p=problem(16,16,16,megafmm::DType::BF16);p.ldd=15;require(rejected(p),"short ldd accepted");
  p=problem(16,16,16,megafmm::DType::BF16);p.beta=1;p.ldc=15;require(rejected(p),"short ldc accepted");
  p=problem(16,16,16,megafmm::DType::BF16);p.lda=std::numeric_limits<int64_t>::max();require(rejected(p),"address overflow accepted");
  p=problem(16,16,16,megafmm::DType::BF16);
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::WinogradLt;policy.certified_range=true;
  auto plan=megafmm::create_plan(p,policy);
  Buffer memory(16384),workspace(plan->workspace_bytes());
  bool overlap_rejected=false;
  try { megafmm::run(*plan,memory.data,memory.as<char>()+4096,nullptr,memory.as<float>(),workspace.data); }
  catch(const std::exception&) { overlap_rejected=true; }
  require(overlap_rejected,"A/D overlap accepted");
  if(plan->workspace_bytes()) {
    overlap_rejected=false;
    try { megafmm::run(*plan,workspace.data,memory.data,nullptr,memory.as<float>()+2048,workspace.data); }
    catch(const std::exception&) { overlap_rejected=true; }
    require(overlap_rejected,"A/workspace overlap accepted");
  }
  p.beta=1;plan=megafmm::create_plan(p,policy);
  Buffer beta_workspace(plan->workspace_bytes());
  overlap_rejected=false;
  try { megafmm::run(*plan,memory.data,memory.as<char>()+4096,memory.as<float>()+2048,
                     memory.as<float>()+2049,beta_workspace.data); }
  catch(const std::exception&) { overlap_rejected=true; }
  require(overlap_rejected,"partial C0/D overlap accepted");
  p.ldc=17;plan=megafmm::create_plan(p,policy);
  Buffer unequal_stride_workspace(plan->workspace_bytes());
  overlap_rejected=false;
  try { megafmm::run(*plan,memory.data,memory.as<char>()+4096,memory.as<float>()+2048,
                     memory.as<float>()+2048,unequal_stride_workspace.data); }
  catch(const std::exception&) { overlap_rejected=true; }
  require(overlap_rejected,"exact C0/D alias with unequal strides accepted");
  p.ldc=p.ldd;plan=megafmm::create_plan(p,policy);
  Buffer c_overlap_workspace(plan->workspace_bytes());
  if(plan->workspace_bytes()) {
    overlap_rejected=false;
    try { megafmm::run(*plan,memory.data,memory.as<char>()+4096,c_overlap_workspace.as<float>(),
                       memory.as<float>()+2048,c_overlap_workspace.data); }
    catch(const std::exception&) { overlap_rejected=true; }
    require(overlap_rejected,"C0/workspace overlap accepted");
  }
  overlap_rejected=false;
  try {
    const void* overflowing_a=reinterpret_cast<const void*>(std::numeric_limits<uintptr_t>::max()-1);
    megafmm::run(*plan,overflowing_a,memory.as<char>()+4096,memory.as<float>()+1024,
                 memory.as<float>()+2048,c_overlap_workspace.data);
  } catch(const std::exception&) { overlap_rejected=true; }
  require(overlap_rejected,"runtime pointer range overflow accepted");
}
template<class T> bool strict_numerics(megafmm::DType dtype) {
  bool pass=true;double floor=dtype==megafmm::DType::BF16 ? 1e-3:1e-4;
  struct Candidate { megafmm::Algorithm algorithm;int variant; };
  std::vector<Candidate> candidates={{megafmm::Algorithm::WinogradLt,0},{megafmm::Algorithm::WinogradLt,1}};
  for(int variant=0;variant<=6;++variant)candidates.push_back({megafmm::Algorithm::WinogradHopper,variant});
  for(int variant=0;variant<=16;++variant)candidates.push_back({megafmm::Algorithm::Megakernel,variant});
  for(auto pat:{Pattern::Uniform,Pattern::Positive,Pattern::Range,Pattern::Cancellation,Pattern::HighHalf}) {
    std::vector<std::vector<double>> ef(candidates.size());
    std::vector<double> dense_ef;
    std::vector<size_t> failures(candidates.size(),0),nonfinite(candidates.size(),0);
    // Near the FP16 limit is outside the certified fast-path range and must use fallback.
    bool certified=pat!=Pattern::HighHalf;
    for(int trial=0;trial<3;++trial) {
      // 256 cubed gives variant 3 complete output tiles and therefore exercises
      // its new beta=0 fast kernel rather than only its general fallback.
      auto p=problem(256,256,256,dtype);uint32_t seed=1234567+trial*918273;
      auto dense=run_case<T>(p,megafmm::Algorithm::Dense,pat,false,certified,seed);
      dense_ef.push_back(dense.ef);
      for(size_t candidate=0;candidate<candidates.size();++candidate) {
        const auto algo=candidates[candidate].algorithm;
        const int variant=candidates[candidate].variant;
        // Kernel numerical coverage does not require retuning seven leaves for
        // every input pattern. Dense is still tuned once per shared reference.
        auto sw=run_case<T>(p,algo,pat,false,certified,seed,variant,false);
        bool finite=dense.nonfinite==0 && sw.nonfinite==0;
        bool gate=finite && sw.ef<=3*dense.ef+floor;
        pass=gate && pass;failures[candidate]+=!gate;nonfinite[candidate]+=dense.nonfinite+sw.nonfinite;
        ef[candidate].push_back(sw.ef);
        std::cout << "NUMERICAL_GATE dtype=" << megafmm::dtype_name(dtype)
                  << " algorithm=" << algorithm_name(algo) << " variant=" << variant
                  << " pattern=" << pattern_name(pat)
                  << " trial=" << trial << " certified_range=" << certified << " pass=" << gate
                  << " finite=" << finite << " nonfinite=" << dense.nonfinite+sw.nonfinite
                  << " dense_ef=" << dense.ef << " solution_ef=" << sw.ef
                  << " maxabs=" << sw.maxabs << " scaled=" << sw.scaled
                  << " normalized=" << sw.normalized << " nearzero_abs=" << sw.nearzero
                  << " floor=" << floor << " limit=" << 3*dense.ef+floor << '\n';
      }
    }
    std::sort(dense_ef.begin(),dense_ef.end());
    for(size_t candidate=0;candidate<candidates.size();++candidate) {
      auto& errors=ef[candidate];std::sort(errors.begin(),errors.end());
      std::cout << "NUMERICAL_DISTRIBUTION dtype=" << megafmm::dtype_name(dtype)
                << " algorithm=" << algorithm_name(candidates[candidate].algorithm)
                << " variant=" << candidates[candidate].variant << " pattern=" << pattern_name(pat)
                << " trials=" << errors.size() << " failures=" << failures[candidate]
                << " nonfinite=" << nonfinite[candidate]
                << " ef_min=" << errors.front() << " ef_median=" << errors[errors.size()/2]
                << " ef_max=" << errors.back() << " dense_ef_median=" << dense_ef[dense_ef.size()/2]
                << " gate_floor=" << floor << '\n';
    }
  }
  return pass;
}
} // namespace
int main(int argc,char** argv) {
  try {
    std::cout << std::unitbuf;std::cerr << std::unitbuf;
    const std::string usage="usage: test_correctness [--strict-numerics] [--smoke-batched | --smoke-candidates [--mega-only] [--mega-variant 0..16 | --hopper-variant 0..6]]";
    bool numeric=false, candidates_only=false,mega_only=false,batched_only=false;
    int hopper_filter=-1,mega_filter=-1;
    for(int i=1;i<argc;++i) {
      if(std::string(argv[i])=="--strict-numerics")numeric=true;
      else if(std::string(argv[i])=="--smoke-batched")batched_only=true;
      else if(std::string(argv[i])=="--smoke-candidates")candidates_only=true;
      else if(std::string(argv[i])=="--mega-only")mega_only=true;
      else if(std::string(argv[i])=="--mega-variant") {
        require(i+1<argc,usage);
        std::string value=argv[++i];size_t consumed=0;
        mega_filter=std::stoi(value,&consumed);
        require(consumed==value.size() && mega_filter>=0 && mega_filter<=16,usage);
      }
      else if(std::string(argv[i])=="--hopper-variant") {
        require(i+1<argc,usage);
        std::string value=argv[++i];size_t consumed=0;
        hopper_filter=std::stoi(value,&consumed);
        require(consumed==value.size() && hopper_filter>=0 && hopper_filter<=6,usage);
      } else throw std::runtime_error(usage);
    }
    require((!mega_only && hopper_filter<0 && mega_filter<0) || candidates_only,
            "candidate filters require --smoke-candidates; "+usage);
    require(!mega_only || hopper_filter<0,"--mega-only and --hopper-variant cannot be combined; "+usage);
    require(mega_filter<0 || hopper_filter<0,"--mega-variant and --hopper-variant cannot be combined; "+usage);
    require(!batched_only || !candidates_only,"--smoke-batched and --smoke-candidates cannot be combined; "+usage);
    if(batched_only) {
      batched_suite<__nv_bfloat16>(megafmm::DType::BF16,true);
      batched_suite<__half>(megafmm::DType::FP16,true);
      rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,true);
      rebind_suite<__half>(megafmm::DType::FP16,true);
      batched_policy_rejection_suite(true);
      std::cout << "PASS batched Lt slot mappings, rectangles, strides, alpha/beta alias, Dense fallback, rebind and policy rejection\n";
      return 0;
    }
    if(candidates_only) {
      auto check_candidates=[mega_only,hopper_filter,mega_filter](auto input_tag,megafmm::DType dtype) {
        using T=decltype(input_tag);
        for(auto candidate:{std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,0},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,1},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,2},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,3},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,4},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,5},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,6},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,7},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,8},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,9},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,10},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,11},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,12},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,13},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,14},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,15},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,16},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,0},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,1},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,2},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,3},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,4},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,5},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,6}}) {
          if(mega_only && candidate.first!=megafmm::Algorithm::Megakernel)continue;
          if(hopper_filter>=0 && (candidate.first!=megafmm::Algorithm::WinogradHopper ||
                                  candidate.second!=hopper_filter))continue;
          if(mega_filter>=0 && (candidate.first!=megafmm::Algorithm::Megakernel ||
                                candidate.second!=mega_filter))continue;
          bool warp_mma=candidate.first==megafmm::Algorithm::Megakernel && candidate.second>=4;
          for(int scenario=0;scenario<15;++scenario) {
            if(scenario==2 && candidate.first!=megafmm::Algorithm::Megakernel && candidate.second<5)continue;
            if(scenario>=3 && !warp_mma)continue;
            if(scenario==6 && candidate.second<7)continue;
            if(scenario==7 && candidate.second!=10 && candidate.second!=13 && candidate.second!=14 && candidate.second!=15 && candidate.second!=16)continue;
            if((scenario==8 || scenario==9) && candidate.second!=11 && candidate.second!=12 && candidate.second!=14 && candidate.second!=15 && candidate.second!=16)continue;
            if((scenario==10 || scenario==11) && candidate.second!=13)continue;
            if(scenario>=12 && candidate.second!=14 && candidate.second!=15 && candidate.second!=16)continue;
            auto p=scenario==2 ? problem(128,64,256,dtype):problem(256,256,256,dtype);
            if(scenario==3 || scenario==4)p=problem(80,112,48,dtype);
            if(scenario==5)p=problem(80,80,48,dtype);
            if(scenario==6 || scenario==7)p=problem(128,128,128,dtype);
            if(scenario==8)p=problem(32,16,32,dtype);
            if(scenario==9)p=problem(32,48,32,dtype);
            if(scenario==10)p=problem(32,32,32,dtype);
            if(scenario==11)p=problem(96,96,96,dtype);
            if(scenario>=12)p=problem(128,128,128,dtype);
            if(scenario==4 || scenario==6) {p.lda+=8;p.ldb+=8;p.ldc+=8;p.ldd+=8;}
            if(scenario==7) {p.ldc+=1;p.ldd+=1;}
            if(scenario==12 || scenario==14)p.ldc+=1;
            if(scenario==13)p.ldd+=1;
            bool alias=scenario==1 || scenario==4 || scenario==6 || scenario==7 || scenario==9;
            if(alias) {p.alpha=.5f;p.beta=-.25f;}
            if(scenario==12 || scenario==13) {p.alpha=.5f;p.beta=-.25f;}
            std::cout << "BEGIN_CANDIDATE dtype=" << megafmm::dtype_name(dtype)
                      << " algorithm=" << algorithm_name(candidate.first) << " variant=" << candidate.second
                      << " shape=" << p.m << ',' << p.n << ',' << p.k
                      << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias << std::endl;
            megafmm::Policy policy;
            policy.algorithm=candidate.first;policy.certified_range=true;
            if(candidate.first==megafmm::Algorithm::Megakernel)policy.mega_variant=candidate.second;
            if(candidate.first==megafmm::Algorithm::WinogradHopper)policy.hopper_variant=candidate.second;
            auto planned=megafmm::create_plan(p,policy);
            require(planned->algorithm_name().find("dense")==std::string::npos,
                    "supported candidate unexpectedly fell back to Dense");
            const char* route=warp_mma ? warp_mma_name(candidate.second,p):
                              candidate.first==megafmm::Algorithm::WinogradHopper && candidate.second>=5 ?
                                "winograd_sm80_batched":nullptr;
            auto e=run_case<T>(p,candidate.first,Pattern::Integers,alias,true,1234567,candidate.second,
                               true,size_t(8)<<30,route);
            require(e.nonfinite==0 && e.maxabs==0,"candidate beta0/alias/rectangular/partial CTA smoke failed");
          }
        }
      };
      check_candidates(__nv_bfloat16{},megafmm::DType::BF16);
      check_candidates(__half{},megafmm::DType::FP16);
      if(mega_filter==13) {
        cluster_graph_suite<__nv_bfloat16>(megafmm::DType::BF16);
        cluster_graph_suite<__half>(megafmm::DType::FP16);
      }
      if(mega_filter==14 || mega_filter==15 || mega_filter==16) {
        epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
      }
      if(mega_filter==16) {
        prefetch_short_k_suite<__nv_bfloat16>(megafmm::DType::BF16);
        prefetch_short_k_suite<__half>(megafmm::DType::FP16);
      }
      std::cout << "PASS selected candidate kernel paths, beta0, alpha/beta alias and rectangular fallback\n";
      return 0;
    }
    structural_suite<__nv_bfloat16>(megafmm::DType::BF16);
    structural_suite<__half>(megafmm::DType::FP16);
    batched_suite<__nv_bfloat16>(megafmm::DType::BF16);
    batched_suite<__half>(megafmm::DType::FP16);
    warp_mma_suite<__nv_bfloat16>(megafmm::DType::BF16);
    warp_mma_suite<__half>(megafmm::DType::FP16);
    prefetch_short_k_suite<__nv_bfloat16>(megafmm::DType::BF16);
    prefetch_short_k_suite<__half>(megafmm::DType::FP16);
    warp_mma_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16);
    warp_mma_bitwise_suite<__half>(megafmm::DType::FP16);
    cluster_graph_suite<__nv_bfloat16>(megafmm::DType::BF16);
    cluster_graph_suite<__half>(megafmm::DType::FP16);
    rebind_suite<__nv_bfloat16>(megafmm::DType::BF16);
    rebind_suite<__half>(megafmm::DType::FP16);
    epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16);
    epilogue_output_rebind_suite<__half>(megafmm::DType::FP16);
    epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,15);
    epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,15);
    epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,16);
    epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,16);
    degenerate_suite();rejection_suite();batched_policy_rejection_suite();
    std::cout << "PASS structural DAG, fallback numerics, strides/transposes, alias, degenerate and rejection tests\n";
    bool pass=true;
    if(numeric) {
      bool bf16_pass=strict_numerics<__nv_bfloat16>(megafmm::DType::BF16);
      bool fp16_pass=strict_numerics<__half>(megafmm::DType::FP16);
      pass=bf16_pass && fp16_pass;
    }
    else std::cout << "Forced SW mixed-precision certification is a separate gate: use --strict-numerics or megafmm_bench\n";
    return pass ? 0:2;
  } catch(const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n';return 1; }
}

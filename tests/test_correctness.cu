#include "megafmm/gemm.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
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
template<class T> Metrics run_case(megafmm::Problem p,megafmm::Algorithm algorithm,
                                  Pattern pattern,bool alias,bool certified,uint32_t seed=1234567,int variant=0) {
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
  policy.hopper_variant=variant;policy.mega_variant=variant;
  auto plan=megafmm::create_plan(p,policy);
  Buffer workspace(plan->workspace_bytes());
  float* output=alias ? dc.as<float>():dd.as<float>();
  require(!alias || p.ldc==p.ldd,"test alias needs equal C/D strides");
  megafmm::tune(*plan,da.data,db.data,p.beta==0 ? nullptr:dc.as<float>(),output,workspace.data);
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
  std::cout << "CASE " << algorithm_name(algorithm) << " actual=" << plan->algorithm_name() << ' '
            << megafmm::dtype_name(p.dtype) << ' ' << pattern_name(pattern) << ' '
            << p.m << ',' << p.n << ',' << p.k << " maxabs=" << e.maxabs << " eF=" << e.ef
            << " scaled=" << e.scaled << " normalized=" << e.normalized << " nonfinite=" << e.nonfinite << '\n';
  if(!alias) for(int i=0;i<p.m;++i) for(int j=p.n;j<p.ldd;++j)
    require(result[size_t(i)*p.ldd+j]==-777.0f,"output padding overwritten");
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
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,1},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,2}}) {
    auto e=run_case<T>(problem(256,256,256,dtype),candidate.first,Pattern::Integers,false,true,1234567,candidate.second);
    require(e.nonfinite==0 && e.maxabs==0,"candidate variant structural test failed");
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
}
template<class T> bool strict_numerics(megafmm::DType dtype) {
  bool pass=true;double floor=dtype==megafmm::DType::BF16 ? 1e-3:1e-4;
  for(auto algo:{megafmm::Algorithm::WinogradLt,megafmm::Algorithm::WinogradHopper,megafmm::Algorithm::Megakernel}) {
    for(auto pat:{Pattern::Uniform,Pattern::Positive,Pattern::Range,Pattern::Cancellation,Pattern::HighHalf}) {
      std::vector<double> ef,dense_ef;
      size_t failures=0,nonfinite=0;
      // Near the FP16 limit is outside the certified fast-path range and must use fallback.
      bool certified=pat!=Pattern::HighHalf;
      for(int trial=0;trial<3;++trial) {
        auto p=problem(128,128,128,dtype);uint32_t seed=1234567+trial*918273;
        auto dense=run_case<T>(p,megafmm::Algorithm::Dense,pat,false,certified,seed);
        auto sw=run_case<T>(p,algo,pat,false,certified,seed);
        bool finite=dense.nonfinite==0 && sw.nonfinite==0;
        bool gate=finite && sw.ef<=3*dense.ef+floor;
        pass=gate && pass;failures+=!gate;nonfinite+=dense.nonfinite+sw.nonfinite;
        ef.push_back(sw.ef);dense_ef.push_back(dense.ef);
        std::cout << "NUMERICAL_GATE dtype=" << megafmm::dtype_name(dtype)
                  << " algorithm=" << algorithm_name(algo) << " pattern=" << pattern_name(pat)
                  << " trial=" << trial << " certified_range=" << certified << " pass=" << gate
                  << " finite=" << finite << " nonfinite=" << dense.nonfinite+sw.nonfinite
                  << " dense_ef=" << dense.ef << " solution_ef=" << sw.ef
                  << " maxabs=" << sw.maxabs << " scaled=" << sw.scaled
                  << " normalized=" << sw.normalized << " nearzero_abs=" << sw.nearzero
                  << " floor=" << floor << " limit=" << 3*dense.ef+floor << '\n';
      }
      std::sort(ef.begin(),ef.end());std::sort(dense_ef.begin(),dense_ef.end());
      std::cout << "NUMERICAL_DISTRIBUTION dtype=" << megafmm::dtype_name(dtype)
                << " algorithm=" << algorithm_name(algo) << " pattern=" << pattern_name(pat)
                << " trials=" << ef.size() << " failures=" << failures << " nonfinite=" << nonfinite
                << " ef_min=" << ef.front() << " ef_median=" << ef[ef.size()/2]
                << " ef_max=" << ef.back() << " dense_ef_median=" << dense_ef[dense_ef.size()/2]
                << " gate_floor=" << floor << '\n';
    }
  }
  return pass;
}
} // namespace
int main(int argc,char** argv) {
  try {
    bool numeric=false;
    for(int i=1;i<argc;++i) {
      if(std::string(argv[i])=="--strict-numerics")numeric=true;
      else throw std::runtime_error("usage: test_correctness [--strict-numerics]");
    }
    structural_suite<__nv_bfloat16>(megafmm::DType::BF16);
    structural_suite<__half>(megafmm::DType::FP16);
    degenerate_suite();rejection_suite();
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

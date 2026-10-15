#include "megafmm/gemm.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
bool tuned_mode=false;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
struct Buffer {
  void* data = nullptr;
  explicit Buffer(size_t bytes) { if (bytes) check(cudaMalloc(&data, bytes)); }
  ~Buffer() { if (data) cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  template<class T> T* as() const { return static_cast<T*>(data); }
};
struct Stream {
  cudaStream_t value = nullptr;
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
struct Graph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  ~Graph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
};
template<class T> T rounded(float value);
template<> __half rounded(float value) { return __float2half_rn(value); }
template<> __nv_bfloat16 rounded(float value) { return __float2bfloat16_rn(value); }
template<class T> double exact(T value) { return double(float(value)); }
template<> double exact(__nv_bfloat16 value) { return double(__bfloat162float(value)); }

template<class T>
void dense_case(megafmm::DType dtype, int m, int n, int k, bool ta, bool tb,
                bool alias, bool graph_mode, bool element_aligned) {
  megafmm::Problem p;
  p.m=m; p.n=n; p.k=k; p.dtype=dtype; p.trans_a=ta; p.trans_b=tb;
  p.lda=(ta ? m:k)+8; p.ldb=(tb ? k:n)+8; p.ldc=n+3; p.ldd=n+3;
  if(tuned_mode && m>=128) p.ldc=p.ldd=n+8;
  p.alpha=0.5f; p.beta=alias ? -0.25f:0.0f;
  const size_t ac=size_t(ta ? k:m)*p.lda, bc=size_t(tb ? n:k)*p.ldb;
  const size_t dc=size_t(m)*p.ldd;
  std::vector<T> a(ac),b(bc);
  std::vector<float> c(dc),out(dc);
  Buffer da((ac+1)*sizeof(T)),db((bc+1)*sizeof(T)),dd(dc*sizeof(float)),d0(dc*sizeof(float));
  T* ap=da.as<T>()+(element_aligned ? 1:0);
  T* bp=db.as<T>()+(element_aligned ? 1:0);
  Stream stream;
  megafmm::Policy policy;
  policy.algorithm=tuned_mode ? megafmm::Algorithm::Tuned : graph_mode ? megafmm::Algorithm::Auto:megafmm::Algorithm::Dense;
  policy.workspace_limit=size_t(32)<<20;
  auto plan=megafmm::create_plan(p,policy);
  Buffer workspace(plan->workspace_bytes());
  require(tuned_mode ? plan->algorithm_name().find("tuned_")==0 : plan->algorithm_name()=="dense_cublaslt", "unexpected Dense route");
  require(plan->workspace_bytes()<=policy.workspace_limit,"workspace budget exceeded");
  Graph graph;
  std::mt19937 rng(1234567);
  for(int trial=0;trial<3;++trial) {
    for(auto& v:a)v=rounded<T>(float(int(rng()%2049)-1024)/1024);
    for(auto& v:b)v=rounded<T>(float(int(rng()%2049)-1024)/1024);
    for(auto& v:c)v=float(int(rng()%257)-128)/64;
    check(cudaMemcpyAsync(ap,a.data(),ac*sizeof(T),cudaMemcpyHostToDevice,stream.value));
    check(cudaMemcpyAsync(bp,b.data(),bc*sizeof(T),cudaMemcpyHostToDevice,stream.value));
    check(cudaMemcpyAsync(d0.data,c.data(),dc*sizeof(float),cudaMemcpyHostToDevice,stream.value));
    float* output=alias ? d0.as<float>():dd.as<float>();
    if(trial==0) {
      megafmm::tune(*plan,ap,bp,alias ? d0.as<float>():nullptr,output,workspace.data,stream.value);
      check(cudaMemcpyAsync(d0.data,c.data(),dc*sizeof(float),cudaMemcpyHostToDevice,stream.value));
      const auto config=plan->lt_algorithm_config(ap,bp,alias ? d0.as<float>():nullptr,output,workspace.data);
      require(config.find("tuned=1")!=std::string::npos,"explicit Dense tuning metadata missing");
      if(graph_mode) {
        check(cudaStreamSynchronize(stream.value));
        check(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));
        megafmm::run(*plan,ap,bp,alias ? d0.as<float>():nullptr,output,workspace.data,stream.value);
        check(cudaStreamEndCapture(stream.value,&graph.graph));
        check(cudaGraphInstantiate(&graph.executable,graph.graph,nullptr,nullptr,0));
      }
    }
    if(graph_mode)check(cudaGraphLaunch(graph.executable,stream.value));
    else megafmm::run(*plan,ap,bp,alias ? d0.as<float>():nullptr,output,workspace.data,stream.value);
    check(cudaMemcpyAsync(out.data(),output,dc*sizeof(float),cudaMemcpyDeviceToHost,stream.value));
    check(cudaStreamSynchronize(stream.value));
    double err2=0,ref2=0;
    for(int i=0;i<m;++i)for(int j=0;j<n;++j) {
      double sum=0;
      for(int l=0;l<k;++l)
        sum+=exact(a[ta ? size_t(l)*p.lda+i:size_t(i)*p.lda+l])*
             exact(b[tb ? size_t(j)*p.ldb+l:size_t(l)*p.ldb+j]);
      const double reference=double(p.alpha)*sum+double(p.beta)*c[size_t(i)*p.ldc+j];
      const double actual=out[size_t(i)*p.ldd+j];
      require(std::isfinite(actual),"Dense produced nonfinite output");
      err2+=(actual-reference)*(actual-reference);ref2+=reference*reference;
    }
    const double ef=std::sqrt(err2/std::max(ref2,std::numeric_limits<double>::min()));
    const double floor=dtype==megafmm::DType::BF16 ? 1e-3:1e-4;
    require(ef<=floor,"Dense error exceeds baseline floor");
    std::cout << "PASS_DENSE dtype=" << megafmm::dtype_name(dtype) << " shape=" << m << ',' << n << ',' << k
              << " transpose=" << ta << tb << " alias=" << alias << " graph=" << graph_mode
              << " element_aligned=" << element_aligned << " trial=" << trial << " eF=" << ef << '\n';
  }
}
void policy_and_binding_cases() {
  megafmm::Problem p; p.m=p.n=p.k=16;
  auto rejected=[](megafmm::Problem value,megafmm::Policy policy={}) {
    try { auto plan=megafmm::create_plan(value,policy); return false; }
    catch(const std::invalid_argument&) { return true; }
  };
  auto invalid=p;invalid.m=-1;require(rejected(invalid),"negative dimension accepted");
  invalid=p;invalid.lda=15;require(rejected(invalid),"short stride accepted");
  invalid=p;invalid.lda=std::numeric_limits<int64_t>::max();require(rejected(invalid),"overflow accepted");
  megafmm::Policy policy;policy.algorithm=static_cast<megafmm::Algorithm>(2);
  require(rejected(p,policy),"removed SW algorithm accepted");
  p.m=0;require(rejected(p,policy),"removed SW algorithm accepted before empty dispatch");
  p.m=16;policy={};policy.heuristic_candidates=0;require(rejected(p,policy),"invalid candidate budget accepted");
  p.m=0;auto empty=megafmm::create_plan(p);megafmm::run(*empty,nullptr,nullptr,nullptr,nullptr,nullptr);
  require(empty->algorithm_name()=="empty","empty route missing");
  p.m=16;p.alpha=0;p.beta=-0.5f;
  auto scale=megafmm::create_plan(p);Buffer d(256*sizeof(float));std::vector<float> host(256,2);
  check(cudaMemcpy(d.data,host.data(),host.size()*sizeof(float),cudaMemcpyHostToDevice));
  megafmm::run(*scale,nullptr,nullptr,d.as<float>(),d.as<float>(),nullptr);
  check(cudaMemcpy(host.data(),d.data,host.size()*sizeof(float),cudaMemcpyDeviceToHost));
  for(float v:host)require(v==-1,"degenerate in-place scale failed");
  p.alpha=1;p.k=0;p.beta=0;scale=megafmm::create_plan(p);
  megafmm::run(*scale,nullptr,nullptr,nullptr,d.as<float>(),nullptr);
  check(cudaMemcpy(host.data(),d.data,host.size()*sizeof(float),cudaMemcpyDeviceToHost));
  for(float v:host)require(v==0,"K0 beta0 did not zero output");
  p.k=16;p.alpha=1;p.beta=0;auto plan=megafmm::create_plan(p);Buffer w(plan->workspace_bytes()),a(512),b(512);
  bool failed=false;
  try {megafmm::run(*plan,a.data,b.data,nullptr,a.as<float>(),w.data);}catch(const std::invalid_argument&){failed=true;}
  require(failed,"output/input overlap accepted");
  failed=false;
  try {megafmm::run(*plan,nullptr,b.data,nullptr,d.as<float>(),w.data);}catch(const std::invalid_argument&){failed=true;}
  require(failed,"null A accepted");
  if(plan->workspace_bytes()) {
    failed=false;
    try {megafmm::run(*plan,a.data,b.data,nullptr,d.as<float>(),nullptr);}catch(const std::invalid_argument&){failed=true;}
    require(failed,"missing workspace accepted");
  }
  std::cout << "PASS Dense policy, binding, empty, scale, overflow and overlap checks\n";
}
template<class T> void suite(megafmm::DType dtype) {
  for(bool ta:{false,true})for(bool tb:{false,true})
    dense_case<T>(dtype,17,23,19,ta,tb,false,false,false);
  dense_case<T>(dtype,32,48,64,false,false,true,false,false);
  dense_case<T>(dtype,128,128,128,false,false,false,true,false);
  dense_case<T>(dtype,32,32,32,false,false,true,true,false);
  dense_case<T>(dtype,17,23,19,false,false,false,false,true);
}
}
int main(int argc,char** argv) {
  try {
    if(argc>2 || (argc==2 && std::string(argv[1])!="--strict-numerics" && std::string(argv[1])!="--tuned"))
      throw std::invalid_argument("usage: test_correctness [--strict-numerics|--tuned]");
    tuned_mode=argc==2 && std::string(argv[1])=="--tuned";
    policy_and_binding_cases();
    suite<__nv_bfloat16>(megafmm::DType::BF16);
    suite<__half>(megafmm::DType::FP16);
    std::cout << "PASS GEMM policy with fresh rounded inputs and original-input FP64 references\n";
    return 0;
  } catch(const std::exception& e) {std::cerr << "FAIL " << e.what() << '\n';return 1;}
}

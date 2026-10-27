// Public API integration checks, compiled separately against the frozen measured library.
#include "megafmm/gemm.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace megafmm;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
struct Buffer{void* p=nullptr;explicit Buffer(size_t n){check(cudaMalloc(&p,n));}~Buffer(){cudaFree(p);}};
struct Stream{cudaStream_t s;Stream(){check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){cudaStreamDestroy(s);}};
template<class T>T cast(float x);
template<>__nv_bfloat16 cast(float x){return __float2bfloat16_rn(x);}
template<>__half cast(float x){return __float2half_rn(x);}
template<class F>void rejects(F f,const char* label){bool threw=false;try{f();}catch(const std::invalid_argument&){threw=true;}require(threw,label);}
template<class T>void test(DType dtype){
  Problem p;p.m=p.n=p.k=p.lda=p.ldb=p.ldc=p.ldd=256;p.dtype=dtype;
  Policy policy;policy.algorithm=Algorithm::SWTuned;policy.workspace_limit=262144;policy.sw_tuning_candidates=1;
  auto plan=create_plan(p,policy);require(plan->workspace_bytes()==262144,"workspace filter did not exclude padded candidates");
  auto insufficient=policy;--insufficient.workspace_limit;rejects([&]{create_plan(p,insufficient);},"insufficient workspace accepted");
  auto invalid=policy;invalid.sw_tuning_candidates=-1;rejects([&]{create_plan(p,invalid);},"negative shortlist accepted");
  auto trans=p;trans.trans_a=true;rejects([&]{create_plan(trans,policy);},"unsupported transpose accepted");
  auto beta=p;beta.beta=1;rejects([&]{create_plan(beta,policy);},"unsupported beta accepted");
  Buffer a(256*256*2+256),b(256*256*2),a2(256*256*2),d(256*256*4),d2(256*256*4),ws(plan->workspace_bytes()),ws2(plan->workspace_bytes());
  Stream first,second;
  auto fill=[&](void* ptr,float value){std::vector<T> v(256*256,cast<T>(value));check(cudaMemcpy(ptr,v.data(),v.size()*2,cudaMemcpyHostToDevice));check(cudaDeviceSynchronize());};
  auto verify=[&](void* ptr,float expected){std::vector<float> v(256*256);check(cudaMemcpy(v.data(),ptr,v.size()*4,cudaMemcpyDeviceToHost));require(std::all_of(v.begin(),v.end(),[&](float x){return x==expected;}),"fresh/concurrent output differs from original AB");};
  fill(a.p,1);fill(b.p,1);
  tune(*plan,a.p,b.p,nullptr,static_cast<float*>(d.p),ws.p,first.s);
  int measured=0,confirmed=0;
  for(const auto& r:plan->sw_tuning_results()){
    require(r.workspace_bytes<=policy.workspace_limit,"workspace-ineligible configuration leaked into Plan");
    measured+=!r.ranking_ms.empty();confirmed+=!r.confirmation_ms.empty();
    if(!r.ranking_ms.empty())require(r.numerical_pass && r.ranking_ms.size()==5 && r.confirmation_ms.size()==5,"shortlist winner lacks validation/timings");
  }
  require(measured==1 && confirmed==1,"positive shortlist budget was ignored");
  verify(d.p,256);
  rejects([&]{run(*plan,static_cast<char*>(a.p)+2,b.p,nullptr,static_cast<float*>(d.p),ws.p,first.s);},"unaligned rebound silently fell back");
  // One selected Plan can serve independent workspaces, outputs and streams.
  fill(a.p,2);fill(a2.p,3);
  run(*plan,a.p,b.p,nullptr,static_cast<float*>(d.p),ws.p,first.s);
  run(*plan,a2.p,b.p,nullptr,static_cast<float*>(d2.p),ws2.p,second.s);
  check(cudaStreamSynchronize(first.s));check(cudaStreamSynchronize(second.s));verify(d.p,512);verify(d2.p,768);
  // Capturing run must preserve fresh-input semantics without retuning.
  cudaGraph_t g=nullptr;check(cudaStreamBeginCapture(first.s,cudaStreamCaptureModeThreadLocal));
  run(*plan,a.p,b.p,nullptr,static_cast<float*>(d.p),ws.p,first.s);
  check(cudaStreamEndCapture(first.s,&g));cudaGraphExec_t e=nullptr;check(cudaGraphInstantiate(&e,g,0));
  fill(a.p,4);check(cudaGraphLaunch(e,first.s));check(cudaStreamSynchronize(first.s));verify(d.p,1024);
  check(cudaGraphExecDestroy(e));check(cudaGraphDestroy(g));
  // Explicit tune is prohibited during capture, before any candidate is executed.
  check(cudaStreamBeginCapture(first.s,cudaStreamCaptureModeThreadLocal));
  rejects([&]{tune(*plan,a.p,b.p,nullptr,static_cast<float*>(d.p),ws.p,first.s);},"tuning inside capture accepted");
  check(cudaStreamEndCapture(first.s,&g));check(cudaGraphDestroy(g));
  // Diagnostic mode may retain finite precision failures, never NaNs/Infs.
  fill(a.p,std::numeric_limits<float>::quiet_NaN());
  auto diagnosis_policy=policy;diagnosis_policy.sw_tuning_allow_inaccurate=true;
  auto diagnosis=create_plan(p,diagnosis_policy);bool nonfinite_rejected=false;
  try{tune(*diagnosis,a.p,b.p,nullptr,static_cast<float*>(d.p),ws.p,first.s);}catch(const std::runtime_error&){nonfinite_rejected=true;}
  require(nonfinite_rejected,"diagnostic tuning accepted nonfinite input/output");
  int nonfinite_records=0;
  for(const auto& r:diagnosis->sw_tuning_results())nonfinite_records+=r.status=="nonfinite_rejected";
  require(nonfinite_records==1,"nonfinite rejection was not recorded");
  printf("PASS public SW API %s budget=1 workspace_filter invalid_policy layouts rebinding dual_stream fresh_Graph tune_capture_reject diagnostic_nonfinite_reject\n",dtype_name(dtype));
}
int main(){try{test<__nv_bfloat16>(DType::BF16);test<__half>(DType::FP16);return 0;}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

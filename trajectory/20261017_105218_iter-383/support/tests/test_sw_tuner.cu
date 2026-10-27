#include "internal.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
using namespace megafmm;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
struct Buffer{void* p=nullptr;explicit Buffer(size_t n){if(n)check(cudaMalloc(&p,n));}~Buffer(){if(p)cudaFree(p);}};
struct Stream{cudaStream_t s;Stream(){check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){cudaStreamDestroy(s);}};
template<class T>T rounded(float x);
template<>__half rounded(float x){return __float2half_rn(x);}
template<>__nv_bfloat16 rounded(float x){return __float2bfloat16_rn(x);}
Problem problem(int m,int n,int k,DType dtype){Problem p;p.m=m;p.n=n;p.k=k;p.lda=k;p.ldb=n;p.ldd=p.ldc=n;p.dtype=dtype;return p;}
template<class T>void fixture(Problem p,Buffer& a,Buffer& b,int seed,bool random=false,float scale=1){
  std::vector<T> av(p.m*p.k),bv(p.k*p.n);std::mt19937 gen(seed);std::uniform_real_distribution<float> dist(-1,1);
  for(int i=0;i<p.m;++i)for(int j=0;j<p.k;++j)av[i*p.k+j]=rounded<T>(scale*(random?dist(gen):float(((i+seed)%3-1)*((j+2*seed)%3-1))));
  for(int i=0;i<p.k;++i)for(int j=0;j<p.n;++j)bv[i*p.n+j]=rounded<T>(scale*(random?dist(gen):float((i*5+j+seed)%7-3)));
  check(cudaMemcpy(a.p,av.data(),av.size()*2,cudaMemcpyHostToDevice));check(cudaMemcpy(b.p,bv.data(),bv.size()*2,cudaMemcpyHostToDevice));check(cudaDeviceSynchronize());
}
std::string candidate_context;
void verify(Problem p,Buffer& d,int seed,float scale=1){
  std::vector<float> out(p.m*p.n+64);check(cudaMemcpy(out.data(),d.p,out.size()*4,cudaMemcpyDeviceToHost));
  std::vector<int> sums(p.n);
  for(int k=0;k<p.k;++k)for(int j=0;j<p.n;++j)sums[j]+=((k+2*seed)%3-1)*((k*5+j+seed)%7-3);
  for(int i=0;i<p.m;++i)for(int j=0;j<p.n;++j)if(out[i*p.n+j]!=float(((i+seed)%3-1)*sums[j])*scale*scale)throw std::runtime_error(candidate_context+" row="+std::to_string(i)+" col="+std::to_string(j)+" seed="+std::to_string(seed)+" actual="+std::to_string(out[i*p.n+j])+" expected="+std::to_string(((i+seed)%3-1)*sums[j]));
  for(size_t i=p.m*p.n;i<out.size();++i)require(out[i]==-12345,"SW candidate overwrote output guard");
}
void guard(Problem p,Buffer& d){std::vector<float> out(p.m*p.n+64,-12345);check(cudaMemcpy(d.p,out.data(),out.size()*4,cudaMemcpyHostToDevice));check(cudaDeviceSynchronize());}
template<class T>void candidate_sweep(DType dtype,bool quick=false){
  Stream stream;
  for(auto shape:std::vector<std::vector<int>>{{1024,1024,1024},{514,770,258},{512,1024,512},{512,512,2048}}){
    auto p=problem(shape[0],shape[1],shape[2],dtype);auto configs=detail::sw_configs(p);require(!configs.empty(),"missing SW candidates");
    size_t bytes=0;for(auto c:configs)bytes=std::max(bytes,detail::sw_config_workspace(p,c));
    Buffer a(p.m*p.k*2),b(p.k*p.n*2),d((p.m*p.n+64)*4),ws(bytes+256);
    int tested=0;
    for(auto c:configs){
      if(quick && (c.pack_cap!=128 || c.raster!=2))continue;
      ++tested;candidate_context=detail::sw_config_name(c);
      auto resource=detail::setup_sw_config(p,c);require(resource.resident>0,"illegal resident CTA count");
      for(int seed:{1,2}){
        // BF16 range regression: these exact values exceed FP16 range.
        float scale=dtype==DType::BF16 && seed==2?1048576.0f:1.0f;
        fixture<T>(p,a,b,seed,false,scale);guard(p,d);
        detail::run_sw_config(p,c,a.p,b.p,static_cast<float*>(d.p),ws.p,stream.s);check(cudaStreamSynchronize(stream.s));verify(p,d,seed,scale);
      }
      cudaGraph_t g;check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));
      detail::run_sw_config(p,c,a.p,b.p,static_cast<float*>(d.p),ws.p,stream.s);
      check(cudaStreamEndCapture(stream.s,&g));cudaGraphExec_t e;check(cudaGraphInstantiate(&e,g,0));
      float graph_scale=dtype==DType::BF16?1048576.0f:1.0f;
      fixture<T>(p,a,b,3,false,graph_scale);guard(p,d);check(cudaGraphLaunch(e,stream.s));check(cudaStreamSynchronize(stream.s));verify(p,d,3,graph_scale);
      cudaGraphExecDestroy(e);cudaGraphDestroy(g);
    }
    printf("PASS native SW candidates %dx%dx%d %s count=%zu fresh=3 launch+graph\n",shape[0],shape[1],shape[2],dtype_name(dtype),size_t(tested));fflush(stdout);
  }
}
template<class T>void tune_test(DType dtype,bool graph){
  int dim=graph?256:1024;auto p=problem(dim,dim,dim,dtype);Policy policy;policy.algorithm=Algorithm::SWTuned;policy.tuning_input_sets=2;policy.sw_tuning_graph=graph;policy.sw_tuning_allow_inaccurate=false;
  auto plan=create_plan(p,policy);Buffer a(p.m*p.k*2),b(p.k*p.n*2),d((p.m*p.n+64)*4),ws(plan->workspace_bytes());Stream stream;
  fixture<T>(p,a,b,1);guard(p,d);
  bool rejected=false;try{run(*plan,a.p,b.p,nullptr,static_cast<float*>(d.p),ws.p,stream.s);}catch(const std::runtime_error&){rejected=true;}
  require(rejected,"untuned SW plan ran");
  tune(*plan,a.p,b.p,nullptr,static_cast<float*>(d.p),ws.p,graph?nullptr:stream.s);verify(p,d,1);
  int confirmed=0;float best=INFINITY;
  for(const auto& r:plan->sw_tuning_results())if(!r.confirmation_ms.empty()){
    require(r.numerical_pass && r.ranking_ms.size()==5 && r.confirmation_ms.size()==5,"missing correctness/timing record");
    auto times=r.confirmation_ms;std::sort(times.begin(),times.end());best=std::min(best,times[2]);++confirmed;
  }
  require(confirmed==4 && best>0,"finalists were not confirmed");
  auto metadata=plan->lt_algorithm_config(a.p,b.p,nullptr,static_cast<float*>(d.p),ws.p);
  auto at=metadata.find("selected_id=");require(at!=std::string::npos,"missing selected candidate id");
  int selected=std::stoi(metadata.substr(at+12));bool selected_best=false,restored=false;
  for(const auto& r:plan->sw_tuning_results()){
    restored|=r.id==-2;
    if(r.id==selected){auto times=r.confirmation_ms;std::sort(times.begin(),times.end());selected_best=times.size()==5 && times[2]==best;}
  }
  require(selected_best,"SW tuner did not choose lowest confirmed median");
  int batch=0,native=0;
  for(const auto& r:plan->sw_tuning_results()) {
    if(r.id>=100000 && r.id<100004){++batch;require(r.config.find("batch_count=7")!=std::string::npos,"batch metadata missing");}
    else if(r.id>=0)++native;
  }
  require(batch==4 && native>0,"batched leaves replaced native candidates or are absent");
  int small=0;for(const auto& r:plan->sw_tuning_results())if(r.id>=500000 && r.id<500006)++small;
  require(small==3,"missing native narrow candidates at1024");
  if(!graph) {
    Policy limited=policy;limited.workspace_limit=detail::SW1024WorkspaceBytes;
    auto small=create_plan(p,limited);bool preserved=false;
    for(const auto& r:small->sw_tuning_results()) {
      require(r.id<100000 || r.id>=500000,"batch candidate ignored workspace budget");preserved|=r.id==-2;
    }
    require(preserved && small->workspace_bytes()<=limited.workspace_limit,"workspace filtering removed restored1024");
  }
  if(!graph)require(restored,"1024 restored SW is absent from candidate search");
  require(plan->lt_algorithm_config(a.p,b.p,nullptr,static_cast<float*>(d.p),ws.p).find("status=precision_pass")!=std::string::npos,"strict SW tuner not marked valid");
  fixture<T>(p,a,b,4);guard(p,d);run(*plan,a.p,b.p,nullptr,static_cast<float*>(d.p),ws.p,stream.s);check(cudaStreamSynchronize(stream.s));verify(p,d,4);
  // Retuning on fresh random inputs invalidates every precision-failing candidate.
  fixture<T>(p,a,b,19,true);rejected=false;
  try{tune(*plan,a.p,b.p,nullptr,static_cast<float*>(d.p),ws.p,graph?nullptr:stream.s);}catch(const std::runtime_error&){rejected=true;}
  require(rejected,"strict SW tuning accepted known uncompensated random error");
  rejected=false;try{run(*plan,a.p,b.p,nullptr,static_cast<float*>(d.p),ws.p,stream.s);}catch(const std::runtime_error&){rejected=true;}
  require(rejected,"failed retune retained stale SW winner");
  require(Policy{}.sw_tuning_allow_inaccurate,"default SW policy still requires old precision gate");
  policy.sw_tuning_allow_inaccurate=true;auto diagnosis=create_plan(p,policy);Buffer ws2(diagnosis->workspace_bytes());
  tune(*diagnosis,a.p,b.p,nullptr,static_cast<float*>(d.p),ws2.p,graph?nullptr:stream.s);
  require(diagnosis->lt_algorithm_config(a.p,b.p,nullptr,static_cast<float*>(d.p),ws2.p).find("status=error_reported")!=std::string::npos,"error reporting tune silently certified historical precision");
  int measured=0,finite_candidates=0;
  for(const auto& r:diagnosis->sw_tuning_results()){
    if(r.status!="nonfinite_rejected" && r.status!="model_not_sampled")++finite_candidates;
    if(!r.ranking_ms.empty()){require(r.ranking_ms.size()==5,"incomplete performance ranking");++measured;}
  }
  require(measured==finite_candidates && measured>4,"performance mode excluded finite candidates");
  printf("PASS SW tuning %s graph=%d strict_valid/no_valid/retune/diagnostic/fresh_inputs\n",dtype_name(dtype),int(graph));fflush(stdout);
}
}
int main(int argc,char** argv){try{
  bool quick=argc>1 && std::string(argv[1])=="--sanitizer";
  bool sweep=quick || (argc>1 && std::string(argv[1])=="--candidates");
  for(auto dtype:{DType::BF16,DType::FP16}){
    if(sweep){if(dtype==DType::BF16)candidate_sweep<__nv_bfloat16>(dtype,quick);else candidate_sweep<__half>(dtype,quick);}
    else for(bool graph:{false,true}){if(dtype==DType::BF16)tune_test<__nv_bfloat16>(dtype,graph);else tune_test<__half>(dtype,graph);}
  }return 0;
}catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

// Compare every registered packed template against frozen production controls. Explicit setup is outside timing; every call packs fresh inputs.
#include "sw_fp32_batch.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" size_t megafmm_baseline_workspace(const megafmm::Problem*);
extern "C" void* megafmm_baseline_prepare(const megafmm::Problem*,void*);
extern "C" void megafmm_baseline_destroy(void*);
extern "C" void megafmm_baseline_run(const megafmm::Problem*,const float*,const float*,float*,void*,cudaStream_t,void*);
extern "C" size_t megafmm_previous_workspace(const megafmm::Problem*);
extern "C" void* megafmm_previous_prepare(const megafmm::Problem*,void*);
extern "C" void megafmm_previous_destroy(void*);
extern "C" void megafmm_previous_run(const megafmm::Problem*,const float*,const float*,float*,void*,cudaStream_t,void*);
extern "C" size_t megafmm_archived_workspace(const megafmm::Problem*);
extern "C" void* megafmm_archived_prepare(const megafmm::Problem*,void*);
extern "C" void megafmm_archived_destroy(void*);
extern "C" void megafmm_archived_run(const megafmm::Problem*,const float*,const float*,float*,void*,cudaStream_t,void*);


namespace {
using Clock=std::chrono::steady_clock;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {
  void* data=nullptr;
  explicit Buffer(size_t bytes){if(bytes)check(cudaMalloc(&data,bytes));}
  ~Buffer(){if(data)cudaFree(data);}
  float* f()const{return static_cast<float*>(data);}
};
struct Stream {
  cudaStream_t value;
  Stream(){check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
  ~Stream(){cudaStreamDestroy(value);}
};
struct Event {
  cudaEvent_t value;
  Event(){check(cudaEventCreate(&value));}
  ~Event(){cudaEventDestroy(value);}
};
__device__ uint64_t hash64(uint64_t v){
  v+=0x9e3779b97f4a7c15ULL;v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;
  v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);
}
__global__ void fill(float* p,int64_t count,uint64_t seed){
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x)
    p[i]=float(hash64(uint64_t(i)^seed)>>40)*(2.0f/16777216.0f)-1.0f;
}
__global__ void bit_difference(const float* a,const float* b,int64_t count,unsigned* bad){
  unsigned local=0;
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x)
    local+=__float_as_uint(a[i])!=__float_as_uint(b[i]);
  if(local)atomicAdd(bad,local);
}
struct Options {
  std::vector<std::vector<int>> shapes;
  int warmup=20,trials=100,groups=5,correct=3,rotate=2;
  int packed_config=-1; // -1 measures the entire template registry
  size_t workspace=size_t(8)<<30;
  std::string output;
};
Options parse(int argc,char** argv){
  Options o;
  for(int i=1;i<argc;++i){
    const std::string flag=argv[i];
    if(++i==argc)throw std::runtime_error("missing value for "+flag);
    const std::string value=argv[i];
    if(flag=="--shapes"){
      std::istringstream all(value);std::string item;
      while(std::getline(all,item,';')){
        std::replace(item.begin(),item.end(),',',' ');std::istringstream one(item);
        int m,n,k;std::string extra;
        if(!(one>>m>>n>>k)||(one>>extra)||std::min({m,n,k})<=0)throw std::runtime_error("invalid shape");
        o.shapes.push_back({m,n,k});
      }
    }else if(flag=="--packed-config")o.packed_config=value=="all"?-1:std::stoi(value);
    else if(flag=="--warmup")o.warmup=std::stoi(value);
    else if(flag=="--trials")o.trials=std::stoi(value);
    else if(flag=="--groups")o.groups=std::stoi(value);
    else if(flag=="--correct-trials")o.correct=std::stoi(value);
    else if(flag=="--rotate-inputs")o.rotate=std::stoi(value);
    else if(flag=="--workspace-limit")o.workspace=std::stoull(value);
    else if(flag=="--output")o.output=value;
    else if(flag=="--dtype"){if(value!="fp32")throw std::runtime_error("packed comparison requires fp32");}
    else if(flag=="--algorithm"){if(value!="sw-tuned-strict")throw std::runtime_error("packed comparison requires strict SW");}
    else throw std::runtime_error("unknown option "+flag);
  }
  if(o.packed_config < -1 || o.packed_config>=int(megafmm::detail::sw_fp32_batch::configurations.size()))throw std::runtime_error("invalid packed config");
  if(o.shapes.empty()||o.warmup<0||o.trials<1||o.groups<1||o.correct<1||o.rotate<1)throw std::runtime_error("invalid comparison options");
  return o;
}
double median(std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];}
bool measure(const Options& o,int m,int n,int k,std::ostream& csv){
  using namespace megafmm;
  namespace packed=detail::sw_fp32_batch;
  Stream stream;
  Problem p;p.m=m;p.n=n;p.k=k;p.lda=k;p.ldb=n;p.ldc=p.ldd=n;p.dtype=DType::FP32;
  if(!packed::problem(p))throw std::runtime_error("shape unsupported by packed kernel");
  auto setup_start=Clock::now();
  std::vector<int> ids;
  std::vector<detail::SWResources> resources;
  for(const auto& c:packed::configurations)if(o.packed_config<0 || o.packed_config==c.id) {
    ids.push_back(c.id);resources.push_back(packed::setup(c.id));
  }
  const int archived_case=3+int(ids.size()),cases=archived_case+1;
  Policy policy;policy.algorithm=Algorithm::Dense;policy.workspace_limit=o.workspace;policy.tuning_input_sets=o.rotate;
  auto dense=create_plan(p,policy);
  // Experimental packing policies can require more than the default seven
  // planes. The shared workspace must cover every configuration actually run.
  size_t packed_required=packed::shape(p).bytes;
  for(int id:ids)packed_required=std::max(packed_required,packed::workspace_size(p,id));
  const size_t required=std::max({dense->workspace_bytes(),packed_required,megafmm_baseline_workspace(&p),megafmm_previous_workspace(&p),megafmm_archived_workspace(&p)});
  if(required>o.workspace)throw std::runtime_error("workspace limit exceeded");
  std::cerr<<"WORKSPACE "<<m<<','<<n<<','<<k<<" allocated="<<required<<" packed_required="<<packed_required<<" dense_required="<<dense->workspace_bytes()<<'\n';
  Buffer workspace(required),baseline(size_t(m)*n*4),dd(size_t(m)*n*4),previous(size_t(m)*n*4),archived(size_t(m)*n*4),bad(sizeof(unsigned));
  std::vector<std::unique_ptr<Buffer>> aa,bb,outputs;
  for(int id:ids)outputs.emplace_back(new Buffer(size_t(m)*n*4));
  for(int i=0;i<o.rotate;++i){aa.emplace_back(new Buffer(size_t(m)*k*4));bb.emplace_back(new Buffer(size_t(k)*n*4));}
  auto fill_set=[&](int at,uint64_t seed){
    fill<<<1024,256,0,stream.value>>>(aa[at]->f(),int64_t(m)*k,seed);
    fill<<<1024,256,0,stream.value>>>(bb[at]->f(),int64_t(k)*n,seed^0x632be59bd9b4e019ULL);
    check(cudaGetLastError());
  };
  for(int i=0;i<o.rotate;++i)fill_set(i,1234567+uint64_t(i)*0xbf58476d1ce4e5b9ULL);
  std::vector<packed::Maps> maps;
  for(int id:ids)maps.push_back(packed::prepare(p,id,workspace.data));
  std::unique_ptr<void,void(*)(void*)> old_maps(megafmm_baseline_prepare(&p,workspace.data),megafmm_baseline_destroy);
  std::unique_ptr<void,void(*)(void*)> previous_maps(megafmm_previous_prepare(&p,workspace.data),megafmm_previous_destroy);
  std::unique_ptr<void,void(*)(void*)> archived_maps(megafmm_archived_prepare(&p,workspace.data),megafmm_archived_destroy);
  tune(*dense,aa[0]->data,bb[0]->data,nullptr,dd.f(),workspace.data,stream.value);
  check(cudaStreamSynchronize(stream.value));
  const double setup_ms=std::chrono::duration<double,std::milli>(Clock::now()-setup_start).count();
  for(size_t i=0;i<ids.size();++i)std::cerr<<"CONFIG "<<m<<','<<n<<','<<k<<" id="<<ids[i]
    <<" new="<<packed::name(ids[i])<<" shared="<<resources[i].shared<<" registers="<<resources[i].registers
    <<" resident="<<resources[i].resident<<" dense="
    <<dense->lt_algorithm_config(aa[0]->data,bb[0]->data,nullptr,dd.f(),workspace.data)<<'\n';
  // Keep historical case numbers 0=Lt,1=original,2=current[0],3=previous.
  const auto case_for=[](int index){return index==0?2:index+3;};
  auto run=[&](int which,int i){
    const int at=i%o.rotate;
    if(which==0)megafmm::run(*dense,aa[at]->data,bb[at]->data,nullptr,dd.f(),workspace.data,stream.value);
    else if(which==1)megafmm_baseline_run(&p,aa[at]->f(),bb[at]->f(),baseline.f(),workspace.data,stream.value,old_maps.get());
    else if(which==3)megafmm_previous_run(&p,aa[at]->f(),bb[at]->f(),previous.f(),workspace.data,stream.value,previous_maps.get());
    else if(which==archived_case)megafmm_archived_run(&p,aa[at]->f(),bb[at]->f(),archived.f(),workspace.data,stream.value,archived_maps.get());
    else {
      const int index=which==2?0:which-3;
      packed::run_prepared(p,ids[index],aa[at]->f(),bb[at]->f(),nullptr,outputs[index]->f(),workspace.data,stream.value,maps[index]);
    }
  };
  std::vector<bool> passes(ids.size(),true),bitwise(ids.size(),true);
  std::vector<double> errors(ids.size(),0);
  double old_ef=0,dense_ef=0,gate=0,previous_ef=0,archived_ef=0;
  for(int trial=0;trial<o.correct;++trial){
    fill_set(0,1234567+uint64_t(trial)*0x9e3779b97f4a7c15ULL);
    for(int which=0;which<cases;++which)run(which,0);
    auto ref=detail::make_sw_reference(p,aa[0]->data,bb[0]->data,dd.f(),stream.value);
    const auto old_check=detail::check_sw_reference(*ref,baseline.f(),stream.value);
    const auto previous_check=detail::check_sw_reference(*ref,previous.f(),stream.value);
    const auto archived_check=detail::check_sw_reference(*ref,archived.f(),stream.value);
    archived_ef=std::max(archived_ef,archived_check.relative_error);
    old_ef=std::max(old_ef,old_check.relative_error);previous_ef=std::max(previous_ef,previous_check.relative_error);
    for(size_t i=0;i<ids.size();++i) {
      const auto current=detail::check_sw_reference(*ref,outputs[i]->f(),stream.value);
      check(cudaMemsetAsync(bad.data,0,sizeof(unsigned),stream.value));
      bit_difference<<<128,256,0,stream.value>>>(outputs[i]->f(),baseline.f(),int64_t(m)*n,static_cast<unsigned*>(bad.data));
      bit_difference<<<128,256,0,stream.value>>>(outputs[i]->f(),previous.f(),int64_t(m)*n,static_cast<unsigned*>(bad.data));
      bit_difference<<<128,256,0,stream.value>>>(outputs[i]->f(),archived.f(),int64_t(m)*n,static_cast<unsigned*>(bad.data));
      unsigned mismatches=0;check(cudaMemcpyAsync(&mismatches,bad.data,sizeof(unsigned),cudaMemcpyDeviceToHost,stream.value));
      check(cudaStreamSynchronize(stream.value));
      passes[i]=passes[i]&&current.pass&&old_check.pass&&previous_check.pass&&archived_check.pass;bitwise[i]=bitwise[i]&&mismatches==0;
      errors[i]=std::max(errors[i],current.relative_error);
      dense_ef=std::max(dense_ef,current.dense_error);gate=std::max(gate,current.limit);
      std::cerr<<"CHECK "<<m<<','<<n<<','<<k<<" id="<<ids[i]<<" trial="<<trial<<" eF="<<current.relative_error
        <<" old_eF="<<old_check.relative_error<<" previous_eF="<<previous_check.relative_error<<" dense_eF="<<current.dense_error<<" gate="<<current.limit
        <<" bit_mismatches="<<mismatches<<" pass="<<(current.pass&&old_check.pass&&previous_check.pass&&archived_check.pass)<<'\n';
    }
  }
  for(int i=0;i<o.warmup;++i)for(int which=0;which<cases;++which)run(which,i);
  check(cudaStreamSynchronize(stream.value));Event start,end;
  std::vector<std::vector<double>> times(cases);
  for(int group=0;group<o.groups;++group)for(int step=0;step<cases;++step){
    const int which=group%2?cases-1-step:step;
    check(cudaEventRecord(start.value,stream.value));for(int i=0;i<o.trials;++i)run(which,i);
    check(cudaEventRecord(end.value,stream.value));check(cudaEventSynchronize(end.value));float ms;
    check(cudaEventElapsedTime(&ms,start.value,end.value));times[which].push_back(ms/o.trials);
    std::cerr<<"TIME "<<m<<','<<n<<','<<k<<" group="<<group<<" case="<<which<<" ms="<<ms/o.trials<<'\n';
  }
  const double lt=median(times[0]),old=median(times[1]),prior=median(times[3]);
  bool pass=true;
  for(size_t i=0;i<ids.size();++i) {
    const double now=median(times[case_for(int(i))]);
    const auto& resource=resources[i];const auto& cfg=packed::configurations[ids[i]];
    csv<<std::setprecision(12)<<m<<','<<n<<','<<k<<",fp32,"<<old<<','<<now<<','<<lt<<','<<old/now<<','<<lt/now
      <<','<<int(passes[i])<<','<<int(bitwise[i])<<','<<errors[i]<<','<<old_ef<<','<<dense_ef<<','<<gate<<','<<o.warmup<<','<<o.groups
      <<','<<o.trials<<','<<o.correct<<','<<o.rotate<<','<<setup_ms<<','<<packed::workspace_size(p,ids[i])<<','<<resource.shared
      <<','<<resource.registers<<','<<resource.resident<<','<<prior<<','<<prior/now<<','<<previous_ef
      <<','<<ids[i]<<','<<cfg.bm<<','<<cfg.bn<<','<<cfg.bk<<','<<cfg.tm<<','<<cfg.tn<<','<<cfg.warps_m<<','<<cfg.warps_n
      <<','<<cfg.threads<<','<<cfg.compute_k<<','<<cfg.group_m
      <<','<<packed::strategy_name(cfg.strategy)<<','<<int(packed::uses_direct_b(p,ids[i],aa[0]->f(),bb[0]->f(),maps[i]))
      <<','<<int(packed::uses_cooperative(p,ids[i],aa[0]->f(),bb[0]->f(),maps[i]))
      <<','<<maps[i].cooperative_max_blocks<<','<<median(times[archived_case])<<','<<archived_ef<<'\n';csv.flush();
    std::cerr<<"RESULT "<<m<<','<<n<<','<<k<<" id="<<ids[i]<<" new_ms="<<now<<" old_ms="<<old<<" lt_ms="<<lt
      <<" previous_ms="<<prior<<" vs_previous="<<prior/now<<" vs_old="<<old/now<<" vs_lt="<<lt/now
      <<" numerical_pass="<<passes[i]<<" bitwise="<<bitwise[i]<<'\n';
    pass=pass&&passes[i]&&bitwise[i];
  }
  return pass;
}
}
int main(int argc,char** argv){try{
  const auto o=parse(argc,argv);std::ofstream file;
  if(!o.output.empty()){file.open(o.output);if(!file)throw std::runtime_error("cannot open CSV");}
  auto& csv=o.output.empty()?std::cout:file;
  csv<<"m,n,k,dtype,baseline_ms,solution_ms,dense_ms,vs_baseline,speedup,numerical_pass,bitwise_match,solution_ef,baseline_ef,dense_ef,gate,warmup,groups,trials,correct_trials,rotate_inputs,setup_ms,solution_workspace,shared_bytes,registers,resident_ctas,previous_ms,vs_previous,previous_ef,configuration_id,bm,bn,bk,tm,tn,warps_m,warps_n,threads,compute_k,group_m,strategy,direct_b,cooperative,cooperative_max_blocks,archived_ms,archived_ef\n";
  bool pass=true;for(const auto& s:o.shapes)pass=measure(o,s[0],s[1],s[2],csv)&&pass;
  return pass?0:1;
}catch(const std::exception& e){std::cerr<<"ERROR "<<e.what()<<'\n';return 1;}}

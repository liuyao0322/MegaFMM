#include "sw_fp32_batch.h"
#include <cuda_profiler_api.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
  void* p=nullptr;
  explicit Buffer(size_t n) {if(n)check(cudaMalloc(&p,n));}
  ~Buffer(){if(p)cudaFree(p);}
  float* f() const {return static_cast<float*>(p);}
};
__device__ uint64_t hash64(uint64_t v) {
  v+=0x9e3779b97f4a7c15ULL;v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;
  v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);
}
__global__ void fill(float* p,int64_t count,uint64_t seed) {
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x)
    p[i]=float(hash64(uint64_t(i)^seed)>>40)*(2.0f/16777216.0f)-1.0f;
}
int main(int argc,char** argv) {try {
  if(argc<8 || argc>9)throw std::runtime_error("usage: profile_fp32 M N K sw|dense capture_calls event_trials config_id [auto|raw|fallback]");
  const int m=std::stoi(argv[1]),n=std::stoi(argv[2]),k=std::stoi(argv[3]),calls=std::stoi(argv[5]),trials=std::stoi(argv[6]);
  const bool sw=std::string(argv[4])=="sw";
  if(!sw && std::string(argv[4])!="dense")throw std::runtime_error("invalid target");
  if(m<=0 || n<=0 || k<=0 || std::max({m,n,k})>32768)throw std::runtime_error("invalid dimensions");
  const int config_id=std::stoi(argv[7]);
  megafmm::Problem p;p.m=m;p.n=n;p.k=p.lda=k;p.ldb=p.ldc=p.ldd=n;p.dtype=megafmm::DType::FP32;
  cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  Buffer a(size_t(m)*k*4),b(size_t(k)*n*4),a2(size_t(m)*k*4),b2(size_t(k)*n*4),d(size_t(m)*n*4);
  float* aa[2]={a.f(),a2.f()},*bb[2]={b.f(),b2.f()};
  for(int i=0;i<2;++i) {
    uint64_t seed=i?1234567+uint64_t(i)*0xbf58476d1ce4e5b9ULL:1234567;
    fill<<<1024,256,0,stream>>>(aa[i],int64_t(m)*k,seed);
    fill<<<1024,256,0,stream>>>(bb[i],int64_t(k)*n,seed^0x632be59bd9b4e019ULL);
  }
  check(cudaGetLastError());check(cudaStreamSynchronize(stream));
  std::unique_ptr<megafmm::Plan> dense;
  if(!sw){megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Dense;policy.tuning_input_sets=2;dense=megafmm::create_plan(p,policy);}
  Buffer workspace(sw?megafmm::detail::sw_fp32_batch::workspace_size(p,config_id):dense->workspace_bytes());
  megafmm::detail::sw_fp32_batch::Maps maps;
  if(sw) {
    // Match the benchmark: initialize all candidate attributes before selecting.
    for(const auto& candidate:megafmm::detail::sw_fp32_batch::configurations)
      megafmm::detail::sw_fp32_batch::setup(candidate.id);
    std::cout<<"SETUP_ALL "<<megafmm::detail::sw_fp32_batch::configurations.size()<<'\n';
    const auto resource=megafmm::detail::sw_fp32_batch::setup(config_id);
    maps=megafmm::detail::sw_fp32_batch::prepare(p,config_id,workspace.p);
    const std::string mode=argc==9?argv[8]:"auto";
    if(mode=="fallback"){maps.cooperative_max_blocks=0;maps.raw_b_available=false;}
    else if(mode!="raw" && mode!="auto")throw std::runtime_error("invalid route mode");
    if(mode!="auto" && (megafmm::detail::sw_fp32_batch::configurations.at(config_id).strategy==megafmm::detail::sw_fp32_batch::ExecutionStrategy::DirectBMaterialized || megafmm::detail::sw_fp32_batch::configurations.at(config_id).fused_raw_b) && megafmm::detail::sw_fp32_batch::uses_direct_b(p,config_id,a.f(),b.f(),maps)!=(mode=="raw"))throw std::runtime_error("profile directB route mismatch");
    const auto z=megafmm::detail::sw_fp32_batch::shape(p);
    const auto cfg=megafmm::detail::sw_fp32_batch::configurations.at(config_id);
    int sms=0;check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));
    std::cout<<"GEOMETRY mp="<<z.mp<<" np="<<z.np<<" kp="<<z.kp<<" tiles_m="<<((m+1)/2+cfg.bm-1)/cfg.bm<<" tiles_n="<<((n+1)/2+cfg.bn-1)/cfg.bn<<" capacity="<<maps.cooperative_max_blocks<<" sms="<<sms<<'\n';
    std::cout<<"MODE "<<mode<<'\n';
    std::cout<<"RESOURCE shared="<<resource.shared<<" regs="<<resource.registers<<" resident="<<resource.resident<<'\n';
    std::cout<<"COOPERATIVE "<<megafmm::detail::sw_fp32_batch::uses_cooperative(p,config_id,a.f(),b.f(),maps)<<"\nDIRECT_B "<<megafmm::detail::sw_fp32_batch::uses_direct_b(p,config_id,a.f(),b.f(),maps)<<"\nCONFIG "<<megafmm::detail::sw_fp32_batch::name(config_id)<<'\n';
  } else {
    megafmm::tune(*dense,a.p,b.p,nullptr,d.f(),workspace.p,stream);
    std::cout<<"CONFIG "<<dense->lt_algorithm_config(a.p,b.p,nullptr,d.f(),workspace.p)<<'\n';
  }
  auto run=[&](int i){if(sw)megafmm::detail::sw_fp32_batch::run_prepared(p,config_id,aa[i%2],bb[i%2],nullptr,d.f(),workspace.p,stream,maps);
    else megafmm::run(*dense,aa[i%2],bb[i%2],nullptr,d.f(),workspace.p,stream);};
  for(int i=0;i<20;++i)run(i);check(cudaStreamSynchronize(stream));
  cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
  if(trials>0)for(int group=0;group<5;++group) {
    check(cudaEventRecord(start,stream));for(int i=0;i<trials;++i)run(i);
    check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));float ms;
    check(cudaEventElapsedTime(&ms,start,end));
    std::cout<<"TIME n="<<n<<" target="<<argv[4]<<" group="<<group<<" trials="<<trials<<" ms="<<ms/trials<<'\n';
  }
  check(cudaProfilerStart());for(int i=0;i<calls;++i)run(i);
  check(cudaStreamSynchronize(stream));check(cudaProfilerStop());
  std::cout<<"DONE n="<<n<<" target="<<argv[4]<<" capture_calls="<<calls<<" workspace="<<(sw?megafmm::detail::sw_fp32_batch::workspace_size(p,config_id):dense->workspace_bytes())<<'\n';
  check(cudaEventDestroy(start));check(cudaEventDestroy(end));check(cudaStreamDestroy(stream));
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

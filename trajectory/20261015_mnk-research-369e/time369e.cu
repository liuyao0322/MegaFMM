// Research only: vary persistent grid size while retaining existing kernel code.
#include "internal.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
using namespace megafmm;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer{void* p=nullptr;explicit Buffer(size_t n){if(n)ck(cudaMalloc(&p,n));}~Buffer(){if(p)cudaFree(p);}};
template<class T>T rounded(float v);
template<>__half rounded(float v){return __float2half_rn(v);}
template<>__nv_bfloat16 rounded(float v){return __float2bfloat16_rn(v);}
constexpr int ids[]={61,70,90,92};
constexpr int targets[]={112,120,128,132};
void initialize(const Problem& p){
  int dev;cudaDeviceProp prop{};ck(cudaGetDevice(&dev));ck(cudaGetDeviceProperties(&prop,dev));
  if(prop.multiProcessorCount!=132)throw std::runtime_error("research targets require the assigned 132-SM H100");
  for(int id:ids)detail::hopper_dense(id,p,nullptr,nullptr,nullptr,nullptr,nullptr,132,nullptr,2);
}
void launch(int id,int target,const Problem& p,const void* a,const void* b,float* d){
  if(!detail::hopper_dense(id,p,a,b,nullptr,d,nullptr,target,nullptr,1))throw std::runtime_error("fixed grid candidate rejected");
}
template<class T>void preflight(DType dtype){
  Problem p;p.m=257;p.n=520;p.k=1032;p.lda=1040;p.ldb=p.ldc=p.ldd=528;p.alpha=.5f;p.beta=0;p.dtype=dtype;
  initialize(p);
  constexpr int max_m=12289;
  std::vector<T>a(size_t(max_m)*1040),b(size_t(1032)*528);
  std::vector<float>d(size_t(max_m)*528),base(d.size());
  Buffer aa(a.size()*2),bb(b.size()*2),dd(d.size()*4);
  for(int trial=0;trial<14;++trial){
    constexpr int tails[]={1032,8,40,72,104,136,168,256,288,320,352,384,8,1032};
    p.m=trial<12?257:max_m;p.k=tails[trial];p.lda=p.k+8;
    double sum=0;for(int q=0;q<p.k;++q)sum+=(1+q%3)*(1+q%5);
    std::fill(a.begin(),a.end(),rounded<T>(-11));std::fill(b.begin(),b.end(),rounded<T>(-13));
    for(int r=0;r<p.m;++r)for(int q=0;q<p.k;++q)a[size_t(r)*p.lda+q]=rounded<T>(float((r%7-3)*(1+q%3)));
    for(int q=0;q<p.k;++q)for(int c=0;c<p.n;++c)b[size_t(q)*p.ldb+c]=rounded<T>(float((c%5-2)*(1+q%5)));
    ck(cudaMemcpy(aa.p,a.data(),a.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(bb.p,b.data(),b.size()*2,cudaMemcpyHostToDevice));
    const size_t count=size_t(p.m)*p.ldd;
    for(int id:ids)for(int target:targets){
      std::fill(d.begin(),d.begin()+count,-777.f);ck(cudaMemcpy(dd.p,d.data(),count*4,cudaMemcpyHostToDevice));
      launch(id,target,p,aa.p,bb.p,static_cast<float*>(dd.p));ck(cudaDeviceSynchronize());
      ck(cudaMemcpy(d.data(),dd.p,count*4,cudaMemcpyDeviceToHost));
      for(int r=0;r<p.m;++r)for(int c=0;c<p.ldd;++c){
        const double expected=c<p.n ? .5*(r%7-3)*(c%5-2)*sum:-777;
        if(d[size_t(r)*p.ldd+c]!=expected)throw std::runtime_error("original AB FP64 closed-form or padding mismatch");
      }
    }
  }
  printf("PASS dtype=%s four kernels x four grid targets, twelve K tails plus persistent reuse K8/K1032, exact original-AB FP64 and padding\n",dtype_name(dtype));fflush(stdout);
  p.m=257;p.k=1032;p.lda=1040;
  const size_t count=size_t(p.m)*p.ldd;
  std::mt19937 rng(3695);std::uniform_real_distribution<float> random(-1.f,1.f);
  for(int trial=0;trial<3;++trial){
    for(size_t i=0;i<size_t(p.m)*p.lda;++i)a[i]=rounded<T>(random(rng));for(auto&v:b)v=rounded<T>(random(rng));
    ck(cudaMemcpy(aa.p,a.data(),size_t(p.m)*p.lda*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(bb.p,b.data(),b.size()*2,cudaMemcpyHostToDevice));
    for(int id:ids){
      std::fill(d.begin(),d.begin()+count,-777.f);ck(cudaMemcpy(dd.p,d.data(),count*4,cudaMemcpyHostToDevice));
      launch(id,132,p,aa.p,bb.p,static_cast<float*>(dd.p));ck(cudaDeviceSynchronize());ck(cudaMemcpy(base.data(),dd.p,count*4,cudaMemcpyDeviceToHost));
      for(int target:targets){
        std::fill(d.begin(),d.begin()+count,-777.f);ck(cudaMemcpy(dd.p,d.data(),count*4,cudaMemcpyHostToDevice));
        launch(id,target,p,aa.p,bb.p,static_cast<float*>(dd.p));ck(cudaDeviceSynchronize());ck(cudaMemcpy(d.data(),dd.p,count*4,cudaMemcpyDeviceToHost));
        if(std::memcmp(base.data(),d.data(),count*4))throw std::runtime_error("fresh fractional input grid bitwise mismatch");
      }
    }
  }
  printf("PASS dtype=%s three fresh fractional input trials bitwise against each matching132-grid original\n",dtype_name(dtype));fflush(stdout);
}
template<class T>int measure(int m,DType dtype){
  Problem p;p.m=p.n=p.k=m;p.lda=p.ldb=p.ldc=p.ldd=m;p.dtype=dtype;initialize(p);
  Policy policy;policy.algorithm=Algorithm::Tuned;policy.tuning_input_sets=2;
  auto plan=create_plan(p,policy);Buffer workspace(plan->workspace_bytes()),out(size_t(m)*m*4);
  Buffer a0(size_t(m)*m*2),a1(size_t(m)*m*2),b0(size_t(m)*m*2),b1(size_t(m)*m*2);
  void* aa[]={a0.p,a1.p};void* bb[]={b0.p,b1.p};
  std::mt19937 rng(3695);std::uniform_real_distribution<float> random(-1.f,1.f);
  auto fill=[&](void* dst){std::vector<T>host(size_t(m)*m);for(auto&v:host)v=rounded<T>(random(rng));ck(cudaMemcpy(dst,host.data(),host.size()*2,cudaMemcpyHostToDevice));};
  for(int set=0;set<2;++set){fill(aa[set]);fill(bb[set]);}
  tune(*plan,aa[0],bb[0],nullptr,static_cast<float*>(out.p),workspace.p,nullptr);
  printf("diagnostic_only=1 fp64_acceptance=unchecked shape=%d,%d,%d dtype=%s baseline_route=%s\n",m,m,m,dtype_name(dtype),plan->algorithm_name().c_str());fflush(stdout);
  auto call=[&](int side,int id,int target,int set){
    if(side==0)run(*plan,aa[set],bb[set],nullptr,static_cast<float*>(out.p),workspace.p,nullptr);
    else launch(id,side==1?target:132,p,aa[set],bb[set],static_cast<float*>(out.p));
  };
  cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
  for(int id:ids)for(int target:targets){
    for(int warm=0;warm<20;++warm)for(int side=0;side<3;++side)call(side,id,target,warm%2);
    ck(cudaDeviceSynchronize());std::vector<float>gains;
    for(int group=0;group<5;++group){float ms[3]{};
      for(int order=0;order<3;++order){int side=group%2?2-order:order;ck(cudaEventRecord(begin));
        for(int i=0;i<100;++i)call(side,id,target,i%2);
        ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));ck(cudaEventElapsedTime(&ms[side],begin,end));ms[side]/=100;
      }
      gains.push_back(ms[0]/ms[1]);printf("candidate=%d target=%d group=%d baseline_ms=%.9f custom_ms=%.9f speedup=%.6f original_ms=%.9f original_speedup=%.6f\n",id,target,group,ms[0],ms[1],gains.back(),ms[2],ms[2]/ms[1]);fflush(stdout);
    }
    std::sort(gains.begin(),gains.end());printf("candidate=%d target=%d diagnostic_median=%.6f diagnostic_min=%.6f acceptance=unchecked\n",id,target,gains[2],gains[0]);fflush(stdout);
  }
  ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));return 0;
}
int main(int argc,char**argv){try{
  if(argc==2 && std::string(argv[1])=="--check"){preflight<__half>(DType::FP16);preflight<__nv_bfloat16>(DType::BF16);return 0;}
  if(argc!=3)throw std::runtime_error("usage: time369e --check | 4096|8192 bf16|fp16");
  int m=std::stoi(argv[1]);if(m!=4096 && m!=8192)throw std::runtime_error("unsupported research shape");
  if(std::string(argv[2])=="bf16")return measure<__nv_bfloat16>(m,DType::BF16);
  if(std::string(argv[2])=="fp16")return measure<__half>(m,DType::FP16);
  throw std::runtime_error("invalid dtype");
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

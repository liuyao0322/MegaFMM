// Research signal only: fresh rotating inputs and complete-call paired timing.
// This does not replace the original FP64 formal benchmark or stable certification.
#include "bitwise369d_prefix.cuh"
#include <algorithm>
#include <random>
#include <vector>
#include <cstdio>
#include <stdexcept>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer{void* p=nullptr;explicit Buffer(size_t n){if(n)ck(cudaMalloc(&p,n));}~Buffer(){if(p)cudaFree(p);}};
template<class T,int CM,bool Nine>
using Research=megafmm::detail::Candidate<T,128,256,CM,0,0,64,false,1,1,true,Nine,false,1,1>;
template<class T,int CM,bool Nine>
using Original=megafmm::detail::Candidate<T,128,256,CM,0,0,64,false,1,1,true,Nine,false,1,0>;
template<class T,int CM,bool Nine>void resources(const megafmm::Problem& p){
  using C=Research<T,CM,Nine>;using K=typename C::Kernel;
  C::run(p,nullptr,nullptr,nullptr,nullptr,nullptr,132,nullptr,2);
  cudaFuncAttributes a{};ck(cudaFuncGetAttributes(&a,cutlass::device_kernel<K>));
  int blocks=0;ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,cutlass::device_kernel<K>,K::MaxThreadsPerBlock,K::SharedStorageSize));
  printf("tile_n=%d tile_k=%d stages=%d registers=%d local=%zu resident_blocks=%d\n",256,64,int(K::DispatchPolicy::Stages),a.numRegs,a.localSizeBytes,blocks);
  if(blocks<1 || a.localSizeBytes!=0)throw std::runtime_error("one-CTA no-spill guard failed");
}
template<class T>int measure(int m,int n,int k,megafmm::DType dtype){
  using namespace megafmm;
  Problem p;p.m=m;p.n=n;p.k=k;p.lda=k;p.ldb=p.ldc=p.ldd=n;p.dtype=dtype;
  resources<T,1,false>(p);
  resources<T,2,false>(p);
  resources<T,1,true>(p);
  resources<T,2,true>(p);
  Original<T,1,false>::run(p,nullptr,nullptr,nullptr,nullptr,nullptr,132,nullptr,2);
  Original<T,2,false>::run(p,nullptr,nullptr,nullptr,nullptr,nullptr,132,nullptr,2);
  Original<T,1,true>::run(p,nullptr,nullptr,nullptr,nullptr,nullptr,132,nullptr,2);
  Original<T,2,true>::run(p,nullptr,nullptr,nullptr,nullptr,nullptr,132,nullptr,2);
  int dev;cudaDeviceProp prop{};ck(cudaGetDevice(&dev));ck(cudaGetDeviceProperties(&prop,dev));
  Policy policy;policy.algorithm=Algorithm::Tuned;policy.tuning_input_sets=2;
  auto plan=create_plan(p,policy);Buffer workspace(plan->workspace_bytes()),out(size_t(m)*n*4);
  Buffer a0(size_t(m)*k*2),a1(size_t(m)*k*2),b0(size_t(k)*n*2),b1(size_t(k)*n*2);
  void* aa[]={a0.p,a1.p};void* bb[]={b0.p,b1.p};
  std::mt19937 rng(369);std::uniform_real_distribution<float> dist(-1,1);
  auto fill=[&](void* dst,size_t count){std::vector<T> host(count);for(auto& v:host)v=T(dist(rng));ck(cudaMemcpy(dst,host.data(),count*sizeof(T),cudaMemcpyHostToDevice));};
  for(int i=0;i<2;++i){fill(aa[i],size_t(m)*k);fill(bb[i],size_t(k)*n);}
  tune(*plan,aa[0],bb[0],nullptr,static_cast<float*>(out.p),workspace.p,nullptr);
  printf("diagnostic_only=1 fp64_acceptance=unchecked shape=%d,%d,%d dtype=%s baseline_route=%s\n",m,n,k,dtype_name(dtype),plan->algorithm_name().c_str());fflush(stdout);
  auto launch=[&](int route,int set){
    if(route==0)run(*plan,aa[set],bb[set],nullptr,static_cast<float*>(out.p),workspace.p,nullptr);
    else {
      bool ok=false;
      switch(route){
        case 1:ok=Research<T,1,false>::run(p,aa[set],bb[set],nullptr,static_cast<float*>(out.p),nullptr,prop.multiProcessorCount,nullptr,1,8);break;
        case 2:ok=Research<T,2,false>::run(p,aa[set],bb[set],nullptr,static_cast<float*>(out.p),nullptr,prop.multiProcessorCount,nullptr,1,8);break;
        case 3:ok=Research<T,1,true>::run(p,aa[set],bb[set],nullptr,static_cast<float*>(out.p),nullptr,prop.multiProcessorCount,nullptr,1,8);break;
        case 4:ok=Research<T,2,true>::run(p,aa[set],bb[set],nullptr,static_cast<float*>(out.p),nullptr,prop.multiProcessorCount,nullptr,1,8);break;
        case 5:ok=Original<T,1,false>::run(p,aa[set],bb[set],nullptr,static_cast<float*>(out.p),nullptr,prop.multiProcessorCount,nullptr,1,8);break;
        case 6:ok=Original<T,2,false>::run(p,aa[set],bb[set],nullptr,static_cast<float*>(out.p),nullptr,prop.multiProcessorCount,nullptr,1,8);break;
        case 7:ok=Original<T,1,true>::run(p,aa[set],bb[set],nullptr,static_cast<float*>(out.p),nullptr,prop.multiProcessorCount,nullptr,1,8);break;
        case 8:ok=Original<T,2,true>::run(p,aa[set],bb[set],nullptr,static_cast<float*>(out.p),nullptr,prop.multiProcessorCount,nullptr,1,8);break;
      }
      if(!ok)throw std::runtime_error("research candidate rejected");
    }
  };
  for(int i=0;i<20;++i)for(int route=0;route<9;++route)launch(route,i%2);
  ck(cudaDeviceSynchronize());cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
  const int tile_n[]={256,256,256,256};
  const int tile_k[]={64,64,64,64};
  for(int route=1;route<5;++route){std::vector<float> gains;
    for(int group=0;group<5;++group){float ms[3]{};
      for(int order=0;order<3;++order){int side=(group&1)?2-order:order,which=side==0?0:(side==1?route:route+4);ck(cudaEventRecord(begin));
        for(int i=0;i<100;++i)launch(which,i%2);
        ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));ck(cudaEventElapsedTime(&ms[side],begin,end));ms[side]/=100;
      }
      gains.push_back(ms[0]/ms[1]);printf("variant=%d tile_n=%d tile_k=%d group=%d baseline_ms=%.9f custom_ms=%.9f speedup=%.6f original_ms=%.9f original_speedup=%.6f\n",route,tile_n[route-1],tile_k[route-1],group,ms[0],ms[1],gains.back(),ms[2],ms[2]/ms[1]);fflush(stdout);
    }
    std::sort(gains.begin(),gains.end());printf("variant=%d tile_n=%d tile_k=%d diagnostic_median=%.6f diagnostic_min=%.6f acceptance=unchecked\n",route,tile_n[route-1],tile_k[route-1],gains[2],gains[0]);fflush(stdout);
  }
  ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));return 0;
}
int main(int argc,char**argv){try{
  if(argc!=5)throw std::runtime_error("usage: time369d M N K bf16|fp16");
  int m=std::stoi(argv[1]),n=std::stoi(argv[2]),k=std::stoi(argv[3]);
  if(m<=0 || n<=0 || k<=0 || n%8 || k%8)throw std::runtime_error("invalid shape");
  if(std::string(argv[4])=="bf16")return measure<cutlass::bfloat16_t>(m,n,k,megafmm::DType::BF16);
  if(std::string(argv[4])=="fp16")return measure<cutlass::half_t>(m,n,k,megafmm::DType::FP16);
  throw std::runtime_error("invalid dtype");
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

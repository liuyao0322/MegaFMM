#include "probe369b_prefix.cuh"
#include <vector>
#include <cstdio>
#include <stdexcept>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<int N,int Stages,bool Direct=false,int CM=1> int probe(){
  using Candidate=megafmm::detail::Candidate<cutlass::half_t,128,N,CM,0,Stages,32,Direct,1,2,true,true>;
  using Kernel=typename Candidate::Kernel;
  megafmm::Problem p;p.m=257;p.n=520;p.k=1032;p.lda=1040;p.ldb=p.ldc=p.ldd=528;p.alpha=1;p.beta=0;p.dtype=megafmm::DType::FP16;
  Candidate::run(p,nullptr,nullptr,nullptr,nullptr,nullptr,132,nullptr,2);
  cudaFuncAttributes attr{};ck(cudaFuncGetAttributes(&attr,cutlass::device_kernel<Kernel>));
  int blocks=0;ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,cutlass::device_kernel<Kernel>,Kernel::MaxThreadsPerBlock,Kernel::SharedStorageSize));
  printf("tile=128x%d stages=%d threads=%d registers=%d shared=%d local=%zu block_limit=%d required_blocks=%d\n",N,Stages,int(Kernel::MaxThreadsPerBlock),attr.numRegs,int(Kernel::SharedStorageSize),attr.localSizeBytes,blocks,int(Kernel::MinBlocksPerMultiprocessor));
  printf("actual_stages=%d tile_k=32 inflight_groups=%d workers_per_sm=2\n",int(Kernel::DispatchPolicy::Stages),PROBE_VARIANT<2?1:3);fflush(stdout);
  // Static register allocation: verify the actual residency before launch.
  if(blocks<2 || attr.localSizeBytes!=0)return 2;
  int device=0;cudaDeviceProp prop{};ck(cudaGetDevice(&device));ck(cudaGetDeviceProperties(&prop,device));
  std::vector<cutlass::half_t>a(p.m*p.lda,cutlass::half_t(1)),b(p.k*p.ldb,cutlass::half_t(1));
  std::vector<float>out(p.m*p.ldd,-777);
  void *da=nullptr,*db=nullptr;float*dd=nullptr;
  ck(cudaMalloc(&da,a.size()*2));ck(cudaMalloc(&db,b.size()*2));ck(cudaMalloc(&dd,out.size()*4));
  ck(cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),b.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dd,out.data(),out.size()*4,cudaMemcpyHostToDevice));
  if(!Candidate::run(p,da,db,nullptr,dd,nullptr,prop.multiProcessorCount,nullptr,1))throw std::runtime_error("candidate rejected preflight");
  ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),dd,out.size()*4,cudaMemcpyDeviceToHost));
  for(int r=0;r<p.m;++r)for(int c=0;c<p.ldd;++c)if(out[r*p.ldd+c]!=(c<p.n?1032.f:-777.f))throw std::runtime_error("preflight result or padding mismatch");
  // Non-symmetric data is essential to verify operand exchange and column-major D.
  for(int trial=0;trial<12;++trial) {
    constexpr int test_k[]={1032,8,40,72,104,256,288,320,352,384,136,168};p.k=test_k[trial];p.lda=p.k+8;
    p.alpha=0.5f;
    for(size_t i=0;i<a.size();++i)a[i]=cutlass::half_t(float(int((i*13+trial*3)%7)-3));
    for(size_t i=0;i<b.size();++i)b[i]=cutlass::half_t(float(int((i*11+trial*5)%7)-3));
    std::fill(out.begin(),out.end(),-777.f);
    ck(cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),b.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dd,out.data(),out.size()*4,cudaMemcpyHostToDevice));
    if(!Candidate::run(p,da,db,nullptr,dd,nullptr,prop.multiProcessorCount,nullptr,1))throw std::runtime_error("fresh candidate rejected");
    ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),dd,out.size()*4,cudaMemcpyDeviceToHost));
    for(int r=0;r<p.m;++r)for(int c=0;c<p.ldd;++c) {
      double ref=-777;
      if(c<p.n){ref=0;for(int q=0;q<p.k;++q)ref+=double(float(a[r*p.lda+q]))*double(float(b[q*p.ldb+c]));ref*=p.alpha;}
      if(out[r*p.ldd+c]!=ref)throw std::runtime_error("fresh FP64 original-AB or padding mismatch");
    }
  }
  puts("PASS twelve fresh non-symmetric original-AB FP64 references, alpha=0.5, strided tails/padding, K=1032/8/40/72/104/256/288/320/352/384/136/168 consumer/batch-tail coverage");
  // More output tiles than the two-worker-per-SM grid forces persistent CTAs to reuse pipeline state.
  // The K-dependent separable inputs have a full original-AB closed-form FP64
  // oracle, keeping this large-output correctness check cheap on the host.
  p.m=12289;
  ck(cudaFree(da));ck(cudaFree(dd));
  a.resize(size_t(p.m)*1040);out.resize(size_t(p.m)*p.ldd);
  ck(cudaMalloc(&da,a.size()*2));ck(cudaMalloc(&dd,out.size()*4));
  for(int kcase:{8,1032}) {
    p.k=kcase;p.lda=p.k+8;p.alpha=0.5f;
    std::fill(a.begin(),a.end(),cutlass::half_t(-11));
    std::fill(b.begin(),b.end(),cutlass::half_t(-13));
    std::fill(out.begin(),out.end(),-777.f);
    double sum=0;
    for(int q=0;q<p.k;++q)sum+=(1+q%3)*(1+q%5);
    for(int r=0;r<p.m;++r)for(int q=0;q<p.k;++q)a[r*p.lda+q]=cutlass::half_t(float((r%7-3)*(1+q%3)));
    for(int q=0;q<p.k;++q)for(int c=0;c<p.n;++c)b[q*p.ldb+c]=cutlass::half_t(float((c%5-2)*(1+q%5)));
    ck(cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),b.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dd,out.data(),out.size()*4,cudaMemcpyHostToDevice));
    if(!Candidate::run(p,da,db,nullptr,dd,nullptr,prop.multiProcessorCount,nullptr,1))throw std::runtime_error("persistent preflight rejected");
    ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),dd,out.size()*4,cudaMemcpyDeviceToHost));
    for(int r=0;r<p.m;++r)for(int c=0;c<p.ldd;++c) {
      double ref=c<p.n ? p.alpha*(r%7-3)*(c%5-2)*sum:-777;
      if(out[r*p.ldd+c]!=ref)throw std::runtime_error("persistent original-AB or padding mismatch");
    }
  }
  puts("PASS persistent pipeline reuse12289x520 K8/K1032 original-AB FP64 closed-form and padding");
  ck(cudaFree(da));ck(cudaFree(db));ck(cudaFree(dd));puts("PASS resource preflight and exact all-ones tail computation; not production acceptance");return 0;
}
#ifndef PROBE_VARIANT
#define PROBE_VARIANT 0
#endif
int main(){try{
#if (PROBE_VARIANT % 2) == 0
 return probe<128,5>();
#else
 return probe<160,5>();
#endif
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

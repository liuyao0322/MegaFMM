#include "probe362_prefix.cuh"
#include <vector>
#include <cstdio>
#include <stdexcept>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<int N,int Stages,bool Direct=false,int CM=1> int probe(){
  using Candidate=megafmm::detail::Candidate<cutlass::half_t,128,N,CM,0,Stages,64,Direct,1,1,true,false,true>;
  using Kernel=typename Candidate::Kernel;
  megafmm::Problem p;p.m=257;p.n=520;p.k=1032;p.lda=1040;p.ldb=p.ldc=p.ldd=528;p.alpha=1;p.beta=0;p.dtype=megafmm::DType::FP16;
  Candidate::run(p,nullptr,nullptr,nullptr,nullptr,nullptr,132,nullptr,2);
  cudaFuncAttributes attr{};ck(cudaFuncGetAttributes(&attr,cutlass::device_kernel<Kernel>));
  int blocks=0;ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,cutlass::device_kernel<Kernel>,Kernel::MaxThreadsPerBlock,Kernel::SharedStorageSize));
  printf("tile=128x%d stages=%d threads=%d registers=%d shared=%d local=%zu block_limit=%d required_blocks=%d\n",N,Stages,int(Kernel::MaxThreadsPerBlock),attr.numRegs,int(Kernel::SharedStorageSize),attr.localSizeBytes,blocks,int(Kernel::MinBlocksPerMultiprocessor));fflush(stdout);
  // Static register allocation: verify the actual residency before launch.
  if(blocks<int(Kernel::MinBlocksPerMultiprocessor))return 2;
  int device=0;cudaDeviceProp prop{};ck(cudaGetDevice(&device));ck(cudaGetDeviceProperties(&prop,device));
  std::vector<cutlass::half_t>a(p.m*p.lda,cutlass::half_t(1)),b(p.k*p.ldb,cutlass::half_t(1));
  std::vector<float>out(p.m*p.ldd,-777);
  void *da=nullptr,*db=nullptr,*workspace=nullptr;float*dd=nullptr;
  const auto workspace_bytes=Candidate::workspace_bytes(p,prop.multiProcessorCount);ck(cudaMalloc(&workspace,workspace_bytes));printf("splits=%d workspace_bytes=%zu\n",megafmm::detail::probe_splits,workspace_bytes);
  ck(cudaMalloc(&da,a.size()*2));ck(cudaMalloc(&db,b.size()*2));ck(cudaMalloc(&dd,out.size()*4));
  ck(cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),b.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dd,out.data(),out.size()*4,cudaMemcpyHostToDevice));
  if(!Candidate::run(p,da,db,nullptr,dd,workspace,prop.multiProcessorCount,nullptr,1))throw std::runtime_error("candidate rejected preflight");
  ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),dd,out.size()*4,cudaMemcpyDeviceToHost));
  for(int r=0;r<p.m;++r)for(int c=0;c<p.ldd;++c)if(out[r*p.ldd+c]!=(c<p.n?1032.f:-777.f))throw std::runtime_error("preflight result or padding mismatch");
  ck(cudaFree(da));ck(cudaFree(db));ck(cudaFree(dd));ck(cudaFree(workspace));puts("PASS resource preflight and exact all-ones tail computation; not production acceptance");return 0;
}
int main(int argc,char**argv){try{
 megafmm::detail::probe_splits=argc>1?atoi(argv[1]):2;
 const int cm=argc>2?atoi(argv[2]):1;
 if(megafmm::detail::probe_splits<2 || megafmm::detail::probe_splits>4)return 1;
 return cm==2?probe<256,0,false,2>():probe<256,0>();
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

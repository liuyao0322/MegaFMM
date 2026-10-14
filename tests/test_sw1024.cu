#include "megafmm/gemm.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int N=1024,Count=N*N;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void check(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS status "+std::to_string(int(e)));}
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Buffer {
  void* p=nullptr;
  explicit Buffer(size_t n){if(n)check(cudaMalloc(&p,n));}
  ~Buffer(){if(p)cudaFree(p);}
  template<class T>T* as(){return static_cast<T*>(p);}
};
struct Stream {cudaStream_t s;Stream(){check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){cudaStreamDestroy(s);}};
struct Graph {cudaGraph_t g=nullptr;cudaGraphExec_t x=nullptr;~Graph(){if(x)cudaGraphExecDestroy(x);if(g)cudaGraphDestroy(g);}};
struct Blas {cublasHandle_t h;explicit Blas(cudaStream_t s){check(cublasCreate(&h));check(cublasSetStream(h,s));}~Blas(){cublasDestroy(h);}};
__device__ uint64_t hash(uint64_t v){v+=0x9e3779b97f4a7c15ULL;v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}
template<class T>__device__ T rounded(float v);
template<>__device__ __half rounded(float v){return __float2half_rn(v);}
template<>__device__ __nv_bfloat16 rounded(float v){return __float2bfloat16_rn(v);}
template<class T>__global__ void fill(T* a,T* b,double* a64,double* b64,int pattern,uint64_t seed){
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Count;i+=gridDim.x*blockDim.x){
    int r=i/N,c=i%N;
    int ai=pattern==3 ? r*N+c%512:i;
    int bi=pattern==3 ? (r%512)*N+c:i;
    uint64_t ha=hash(uint64_t(ai)^seed),hb=hash(uint64_t(bi)^seed^0x632be59bd9b4e019ULL);
    float av=float(ha>>40)*(2.0f/16777216.0f)-1,bv=float(hb>>40)*(2.0f/16777216.0f)-1;
    if(pattern==1){av=fabsf(av);bv=fabsf(bv);}
    if(pattern==2){av=ldexpf(av,int(ha%9)-4);bv=ldexpf(bv,int(hb%9)-4);}
    if(pattern==3 && c>=512)av=-av+1.0f/256;
    if(pattern==4)av=0;
    if(pattern==5)av=r==c ? 1:0;
    T aa=rounded<T>(av),bb=rounded<T>(bv);a[i]=aa;b[i]=bb;
    a64[i]=double(float(aa));b64[i]=double(float(bb));
  }
}
struct Metric {double e2,r2;unsigned long long nonfinite;};
__global__ void measure_error(const float* d,const double* r,Metric* output){
  __shared__ Metric sums[256];Metric m{};
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Count;i+=gridDim.x*blockDim.x){
    double x=d[i],y=r[i];if(!isfinite(x)||!isfinite(y)){++m.nonfinite;continue;}
    m.e2+=(x-y)*(x-y);m.r2+=y*y;
  }
  sums[threadIdx.x]=m;__syncthreads();
  for(int s=128;s;s/=2){if(threadIdx.x<s){sums[threadIdx.x].e2+=sums[threadIdx.x+s].e2;sums[threadIdx.x].r2+=sums[threadIdx.x+s].r2;sums[threadIdx.x].nonfinite+=sums[threadIdx.x+s].nonfinite;}__syncthreads();}
  if(threadIdx.x==0)output[blockIdx.x]=sums[0];
}
Metric metric(float* d,double* ref,Buffer& scratch,cudaStream_t stream){
  measure_error<<<128,256,0,stream>>>(d,ref,scratch.as<Metric>());check(cudaGetLastError());
  std::vector<Metric> values(128);check(cudaMemcpyAsync(values.data(),scratch.p,128*sizeof(Metric),cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
  Metric sum{};for(auto v:values){sum.e2+=v.e2;sum.r2+=v.r2;sum.nonfinite+=v.nonfinite;}return sum;
}
double ef(Metric m){return m.r2 ? sqrt(m.e2/m.r2):m.e2==0 ? 0:INFINITY;}

template<class T>int validate(megafmm::DType dtype,bool quick){
  Stream stream;Blas blas(stream.s);Graph graph;
  Buffer a(Count*sizeof(T)),b(Count*sizeof(T)),a64(Count*8),b64(Count*8),ref(Count*8),d(Count*4),base(Count*4),scratch(128*sizeof(Metric));
  megafmm::Problem p;p.m=p.n=p.k=N;p.dtype=dtype;
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::SWPipeline;
  if(quick)policy.heuristic_candidates=1; // Sanitizer scope, never used for performance.
  auto sw=megafmm::create_plan(p,policy);
  require(sw->algorithm_name().find("sw_unified_v1_hilo_")==0,"not native SW");
  require(sw->workspace_bytes()==0,"SW should not allocate global intermediates");
  policy.algorithm=megafmm::Algorithm::Auto;auto dense=megafmm::create_plan(p,policy);
  require(dense->algorithm_name()=="dense_cublaslt","Auto was incorrectly promoted");
  Buffer workspace(dense->workspace_bytes());
  // Rejections use the same 1024 size and never execute a second matrix size.
  policy.algorithm=megafmm::Algorithm::SWPipeline;
  auto invalid=p;invalid.trans_a=true;bool rejected=false;
  try{auto x=megafmm::create_plan(invalid,policy);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"SW accepted transpose");
  rejected=false;try{megafmm::run(*sw,static_cast<char*>(a.p)+2,b.p,nullptr,d.as<float>(),nullptr,stream.s);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"TMA accepted misaligned input");
  fill<<<512,256,0,stream.s>>>(a.as<T>(),b.as<T>(),a64.as<double>(),b64.as<double>(),0,1234);
  megafmm::tune(*dense,a.p,b.p,nullptr,base.as<float>(),workspace.p,stream.s);
  megafmm::tune(*sw,a.p,b.p,nullptr,d.as<float>(),nullptr,stream.s);
  check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));
  megafmm::run(*sw,a.p,b.p,nullptr,d.as<float>(),nullptr,stream.s);
  check(cudaStreamEndCapture(stream.s,&graph.g));check(cudaGraphInstantiate(&graph.x,graph.g,nullptr,nullptr,0));
  const char* patterns[]={"uniform","positive","range16","cancellation","zero","identity"};
  int failures=0,checks=0;
  for(int pattern=0;pattern<(quick ? 1:6);++pattern)for(int trial=0;trial<(quick ? 1:3);++trial){
    fill<<<512,256,0,stream.s>>>(a.as<T>(),b.as<T>(),a64.as<double>(),b64.as<double>(),pattern,1234567ULL+trial*0x9e3779b97f4a7c15ULL+pattern*101);
    check(cudaGetLastError());double one=1,zero=0;
    check(cublasDgemm(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&one,b64.as<double>(),N,a64.as<double>(),N,&zero,ref.as<double>(),N));
    megafmm::run(*dense,a.p,b.p,nullptr,base.as<float>(),workspace.p,stream.s);
    Metric dm=metric(base.as<float>(),ref.as<double>(),scratch,stream.s);
    double gate=3*ef(dm)+(dtype==megafmm::DType::BF16 ? 1e-3:1e-4);
    for(int mode=0;mode<2;++mode){
      // Poison output and replay the SAME captured graph on each freshly refilled input.
      check(cudaMemsetAsync(d.p,0xff,Count*4,stream.s));
      if(mode)check(cudaGraphLaunch(graph.x,stream.s));else megafmm::run(*sw,a.p,b.p,nullptr,d.as<float>(),nullptr,stream.s);
      Metric sm=metric(d.as<float>(),ref.as<double>(),scratch,stream.s);
      bool pass=dm.nonfinite==0 && sm.nonfinite==0 && ef(sm)<=gate;++checks;if(!pass)++failures;
      printf("CHECK size=1024 dtype=%s pattern=%s trial=%d mode=%s dense_ef=%.12g sw_ef=%.12g gate=%.12g nonfinite=%llu pass=%d\n",megafmm::dtype_name(dtype),patterns[pattern],trial,mode ? "graph":"launch",ef(dm),ef(sm),gate,sm.nonfinite,pass);
    }
  }
  printf("SUMMARY dtype=%s checks=%d failures=%d\n",megafmm::dtype_name(dtype),checks,failures);
  return failures;
}
}
int main(int argc,char** argv){try{
  bool quick=argc==2 && std::string(argv[1])=="--quick";
  if(argc>1 && !quick)throw std::invalid_argument("usage: test_sw1024 [--quick]");
  int failures=validate<__nv_bfloat16>(megafmm::DType::BF16,quick)+validate<__half>(megafmm::DType::FP16,quick);
  return failures ? 2:0;
}catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

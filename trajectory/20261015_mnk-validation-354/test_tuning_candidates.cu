#include "internal.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {
  void* p=nullptr;explicit Buffer(size_t n){if(n)check(cudaMalloc(&p,n));}~Buffer(){if(p)cudaFree(p);}
};
struct Execution {
  cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
  Execution(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
  ~Execution(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);cudaStreamDestroy(stream);}
};
template<class T>T rounded(float x);
template<>__half rounded(float x){return __float2half_rn(x);}
template<>__nv_bfloat16 rounded(float x){return __float2bfloat16_rn(x);}
template<class T>
void candidate(int id,megafmm::DType dtype,int sms,int layout=0) {
  using namespace megafmm;
  Problem p;p.dtype=dtype;p.alpha=0.5f;p.beta=id>=60?0.0f:-0.25f;
  if(id<4 || id>=9) {p.m=129;p.n=256;p.k=128;p.lda=136;p.ldb=264;p.ldc=p.ldd=264;}
  else {p.m=id==4?17:129;p.n=id==4?23:137;p.k=id==4?19:65;
        p.trans_a=p.trans_b=true;p.lda=p.m+3;p.ldb=p.k+5;p.ldc=p.ldd=p.n+7;}
  if(id>=9) {p.m=769;p.n=1536;p.k=16;p.lda=24;p.ldb=1544;p.ldc=p.ldd=1544;}
  if(id>=26) {p.m=129;p.n=256;p.k=520;p.lda=528;p.ldb=264;p.ldc=p.ldd=264;}
  if((id>=39 && id<=42) || (id>=70 && id<=75)) {p.m=1025;p.n=520;p.k=128;p.lda=136;p.ldb=528;p.ldc=p.ldd=528;}
  if(layout==1) {p.m=129;p.n=137;p.k=65;p.trans_a=p.trans_b=true;p.lda=132;p.ldb=70;p.ldc=p.ldd=144;}
  if(layout==2) {p.m=129;p.n=138;p.k=66;p.lda=68;p.ldb=140;p.ldc=p.ldd=144;}
  if(layout==3) {p.m=3;p.n=6;p.k=2;p.lda=4;p.ldb=8;p.ldc=p.ldd=8;}
  const size_t ac=(p.trans_a?p.k:p.m)*p.lda,bc=(p.trans_b?p.n:p.k)*p.ldb,dc=p.m*p.ldd;
  std::vector<T>a(ac),b(bc);std::vector<float>initial(dc),result(dc);
  Buffer da(ac*2),db(bc*2),out0(dc*4),out1(dc*4);
  Buffer work0(detail::hopper_workspace(p,id)),work1(detail::hopper_workspace(p,id));
  Execution execution[2];void* outputs[]={out0.p,out1.p};void* workspaces[]={work0.p,work1.p};
  const int base=id>=5 && id<=8?id-5:id;
  detail::hopper_dense(base,p,nullptr,nullptr,nullptr,nullptr,nullptr,sms,nullptr,2);
  if(id==37 || id==38 || (id>=43 && id<=55)) {
    Problem rejected=p;rejected.trans_a=true;
    if(detail::hopper_dense(id,rejected,da.p,db.p,static_cast<float*>(out0.p),static_cast<float*>(out0.p),work0.p,sms,nullptr,0))
      throw std::runtime_error("even MMA accepted transpose");
    rejected=p;rejected.lda=(1<<20)+2;
    if(detail::hopper_dense(id,rejected,da.p,db.p,static_cast<float*>(out0.p),static_cast<float*>(out0.p),work0.p,sms,nullptr,0))
      throw std::runtime_error("even MMA accepted unsafe 32-bit offset range");
    if(detail::hopper_dense(id,p,da.p,db.p,static_cast<float*>(out0.p),static_cast<float*>(out0.p)+1,work0.p,sms,nullptr,0))
      throw std::runtime_error("even MMA accepted misaligned vector output");
  }
  if(id>=60) {
    Problem rejected=p;rejected.beta=1;
    if(detail::hopper_dense(id,rejected,da.p,db.p,static_cast<float*>(out0.p),static_cast<float*>(out0.p),work0.p,sms,nullptr,0))
      throw std::runtime_error("zero-beta candidate accepted nonzero beta");
  }
  // Capture only after one setup; both executions bind distinct output/scratch.
  check(cudaStreamBeginCapture(execution[0].stream,cudaStreamCaptureModeThreadLocal));
  if(!detail::hopper_dense(id,p,da.p,db.p,static_cast<float*>(out0.p),static_cast<float*>(out0.p),work0.p,sms,execution[0].stream,1))
    throw std::runtime_error("candidate rejected supported test shape");
  check(cudaStreamEndCapture(execution[0].stream,&execution[0].graph));
  check(cudaGraphInstantiate(&execution[0].executable,execution[0].graph,nullptr,nullptr,0));
  for(int trial=0;trial<2;++trial) {
    for(size_t i=0;i<ac;++i)a[i]=rounded<T>(float(int((i*13+trial*7)%7)-3));
    for(size_t i=0;i<bc;++i)b[i]=rounded<T>(float(int((i*19+trial*11)%7)-3));
    for(int64_t r=0;r<p.m;++r)for(int64_t c=0;c<p.ldd;++c)
      initial[r*p.ldd+c]=c<p.n ? float(int((r*p.n+c)%5)-2):-12345;
    check(cudaMemcpy(da.p,a.data(),ac*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.p,b.data(),bc*2,cudaMemcpyHostToDevice));
    for(int mode=0;mode<2;++mode) {
      auto stream=execution[mode].stream;auto output=static_cast<float*>(outputs[mode]);
      check(cudaMemcpyAsync(output,initial.data(),dc*4,cudaMemcpyHostToDevice,stream));
      if(mode==0)check(cudaGraphLaunch(execution[0].executable,stream));
      else if(!detail::hopper_dense(id,p,da.p,db.p,output,output,workspaces[mode],sms,stream,1))
        throw std::runtime_error("candidate launch rejected");
    }
    for(int mode=0;mode<2;++mode) {
      check(cudaMemcpyAsync(result.data(),outputs[mode],dc*4,cudaMemcpyDeviceToHost,execution[mode].stream));
      check(cudaStreamSynchronize(execution[mode].stream));
      for(int64_t r=0;r<p.m;++r)for(int64_t c=0;c<p.ldd;++c) {
        double expected=initial[r*p.ldd+c];
        if(c<p.n) {
          double sum=0;
          for(int64_t q=0;q<p.k;++q)
            sum+=double(float(a[p.trans_a?q*p.lda+r:r*p.lda+q]))*float(b[p.trans_b?c*p.ldb+q:q*p.ldb+c]);
          expected=p.alpha*sum+p.beta*expected;
        }
        if(!std::isfinite(result[r*p.ldd+c]) || result[r*p.ldd+c]!=expected) {
          fprintf(stderr,"id=%d dtype=%s trial=%d mode=%d at=%lld,%lld expected=%g actual=%g\n",id,dtype_name(dtype),trial,mode,(long long)r,(long long)c,expected,result[r*p.ldd+c]);
          throw std::runtime_error("candidate FP64 or padding mismatch");
        }
      }
    }
  }
  printf("PASS candidate=%s dtype=%s fresh_trials=2 concurrent_graph_launch=2 in_place=1 padding=checked\n",detail::hopper_candidate_name(id),dtype_name(dtype));
}
}
int main(int argc,char** argv){try{
  int first=0,last=megafmm::detail::HopperCandidateCount;
  bool only_wmma=false;
  if(argc==2 && std::string(argv[1])=="--wmma") {first=24;only_wmma=true;}
  else if(argc==2 && std::string(argv[1])=="--new") first=33;
  else if(argc==3 && std::string(argv[1])=="--candidate") {
    first=std::stoi(argv[2]);
    if(first<0 || first>=megafmm::detail::HopperCandidateCount)throw std::runtime_error("candidate out of range");
    last=first+1;
  }
  else if(argc!=1) throw std::runtime_error("usage: test_tuning_candidates [--wmma|--new|--candidate ID]");
  int device;cudaDeviceProp prop;check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
  for(int id=first;id<last;++id) {
    if(only_wmma && id!=24 && id!=25 && (id<33 || (id>38 && id<43))) continue;
    candidate<__nv_bfloat16>(id,megafmm::DType::BF16,prop.multiProcessorCount);
    candidate<__half>(id,megafmm::DType::FP16,prop.multiProcessorCount);
    if(id==24 || id==25 || (id>=33 && id<=36)) {
      candidate<__nv_bfloat16>(id,megafmm::DType::BF16,prop.multiProcessorCount,1);
      candidate<__half>(id,megafmm::DType::FP16,prop.multiProcessorCount,1);
      candidate<__nv_bfloat16>(id,megafmm::DType::BF16,prop.multiProcessorCount,2);
      candidate<__half>(id,megafmm::DType::FP16,prop.multiProcessorCount,2);
    }
    if(id==37 || id==38 || (id>=43 && id<=55)) {
      candidate<__nv_bfloat16>(id,megafmm::DType::BF16,prop.multiProcessorCount,2);
      candidate<__half>(id,megafmm::DType::FP16,prop.multiProcessorCount,2);
      candidate<__nv_bfloat16>(id,megafmm::DType::BF16,prop.multiProcessorCount,3);
      candidate<__half>(id,megafmm::DType::FP16,prop.multiProcessorCount,3);
    }
  }
  return 0;
}catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

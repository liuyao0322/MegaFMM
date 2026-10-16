#include "megafmm/gemm.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool ok,const char* msg) { if(!ok) throw std::runtime_error(msg); }
struct Buffer {
  void* p=nullptr;
  explicit Buffer(size_t bytes) { if(bytes) check(cudaMalloc(&p,bytes)); }
  ~Buffer() { if(p) cudaFree(p); }
};
struct Graph {
  cudaGraph_t graph=nullptr; cudaGraphExec_t exec=nullptr;
  ~Graph() { if(exec) cudaGraphExecDestroy(exec); if(graph) cudaGraphDestroy(graph); }
};
template<class T> T quantize(float x);
template<> __half quantize(float x) { return __float2half_rn(x); }
template<> __nv_bfloat16 quantize(float x) { return __float2bfloat16_rn(x); }

// Independent scalar oracle: logical even padding, FP32 transforms, exactly
// one cast per transformed operand, FP64 leaf sums, and Winograd recombination.
// Fractional tests compare to this oracle because uncorrected SW has a known
// rounding floor. Integer tests also compare directly to original-input AB.
template<class T>
float oracle(const megafmm::Problem& p,const std::vector<T>& a,const std::vector<T>& b,int r,int c) {
  int mh=(p.m+1)/2, nh=(p.n+1)/2, kh=(p.k+1)/2;
  int ar=r%mh, bc=c%nh;
  auto A=[&](int i,int j) { return i<p.m && j<p.k ? float(a[p.trans_a ? j*p.lda+i : i*p.lda+j]) : 0.0f; };
  auto B=[&](int i,int j) { return i<p.k && j<p.n ? float(b[p.trans_b ? j*p.ldb+i : i*p.ldb+j]) : 0.0f; };
  double products[7]{};
  for(int q=0;q<kh;++q) {
    float a0=A(ar,q),a1=A(ar,q+kh),a2=A(ar+mh,q),a3=A(ar+mh,q+kh);
    float b0=B(q,bc),b1=B(q,bc+nh),b2=B(q+kh,bc),b3=B(q+kh,bc+nh);
    float s1=a2+a3,s2=s1-a0,s3=a0-a2,s4=a1-s2;
    float t1=b1-b0,t2=b3-t1,t3=b3-b1,t4=t2-b2;
    float av[]={a0,a1,s4,a3,s1,s2,s3}, bv[]={b0,b2,b3,t4,t1,t2,t3};
    for(int j=0;j<7;++j) products[j]+=double(float(quantize<T>(av[j])))*float(quantize<T>(bv[j]));
  }
  float v[7]; for(int i=0;i<7;++i) v[i]=float(products[i]);
  float u2=v[0]+v[5],u3=u2+v[6],u4=u2+v[4];
  return r<mh ? (c<nh ? v[0]+v[1] : u4+v[2]) : (c<nh ? u3-v[3] : u3+v[4]);
}

template<class T>
void exercise(megafmm::DType dtype,int m,int n,int k,bool ta=false,bool tb=false,
              bool strided=false,float alpha=1,float beta=0,bool inplace=false,bool fractional=false,
              bool diagonal=false,bool dedicated=false,bool unaligned=false,bool hopper=false) {
  megafmm::Problem p;p.m=m;p.n=n;p.k=k;p.trans_a=ta;p.trans_b=tb;p.dtype=dtype;
  p.lda=(ta?m:k)+(strided?3:0);p.ldb=(tb?k:n)+(strided?5:0);
  p.ldc=n+(strided?7:0);p.ldd=inplace?p.ldc:n+(strided?9:0);p.alpha=alpha;p.beta=beta;
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::SWPipeline;policy.workspace_limit=hopper ? size_t(8)<<30 : dedicated ? size_t(4)<<20 : 0;
  auto plan=megafmm::create_plan(p,policy);
  require(plan->workspace_bytes()==(hopper ? size_t(2)*k*(m+n) : dedicated ? size_t(4)<<20 : 0),"SW workspace dispatch failed");
  Buffer scratch(plan->workspace_bytes());
  require(plan->algorithm_name()==(hopper ? "sw_hopper" : dedicated ? "sw1024_pdl" : "sw_pipeline"),"generic SW dispatch failed");
  if(diagonal) require(m==k && !fractional,"diagonal reference requires square integer A");
  size_t na=(ta?k:m)*p.lda,nb=(tb?n:k)*p.ldb,nd=m*p.ldd,nc=m*p.ldc;
  std::vector<T> a(na),b(nb);std::vector<float> c(nc),d(nd);
  Buffer da((na+1)*sizeof(T)),db((nb+1)*sizeof(T)),dc(nc*4),dd(nd*4);
  // Exercise valid element alignment without relying on cudaMalloc alignment.
  size_t offset=(dedicated || hopper) && !unaligned ? 0 : sizeof(T);
  void* ap=static_cast<char*>(da.p)+offset;void* bp=static_cast<char*>(db.p)+offset;
  float* out=static_cast<float*>(inplace?dc.p:dd.p);
  auto config=plan->lt_algorithm_config(ap,bp,beta?static_cast<float*>(dc.p):nullptr,out,scratch.p);
  require(config.find(!unaligned && hopper ? "sw_hopper;" : dedicated && !unaligned ? "sw1024_pdl;" : "sw_pipeline;")==0,"bound SW dispatch metadata failed");
  if(dedicated || hopper) {
    bool rejected=false;
    try { megafmm::run(*plan,ap,bp,nullptr,out,nullptr); }
    catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"dedicated SW accepted missing workspace");
  }
  cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  {
    Graph graph;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    megafmm::run(*plan,ap,bp,beta?static_cast<float*>(dc.p):nullptr,out,scratch.p,stream);
    check(cudaStreamEndCapture(stream,&graph.graph));
    check(cudaGraphInstantiate(&graph.exec,graph.graph,nullptr,nullptr,0));
    for(int trial=0;trial<2;++trial) {
      for(size_t i=0;i<na;++i) a[i]=quantize<T>(float(int((i*13+trial*7)%7)-3)*(fractional?0.137f:1.0f));
      // A diagonal input makes a full 1024^3 regression cheap to reference on CPU.
      if(diagonal) {
        std::fill(a.begin(),a.end(),quantize<T>(0));
        for(int r=0;r<m;++r) a[r*p.lda+r]=quantize<T>((r+trial)%2 ? -1.0f:1.0f);
      }
      for(size_t i=0;i<nb;++i) b[i]=quantize<T>(float(int((i*17+trial*11)%7)-3)*(fractional?0.193f:1.0f));
      for(size_t i=0;i<nc;++i) c[i]=float(int(i%5)-2);
      check(cudaMemcpyAsync(ap,a.data(),na*sizeof(T),cudaMemcpyHostToDevice,stream));
      check(cudaMemcpyAsync(bp,b.data(),nb*sizeof(T),cudaMemcpyHostToDevice,stream));
      if(trial==0) {
        check(cudaMemcpyAsync(dc.p,c.data(),nc*4,cudaMemcpyHostToDevice,stream));
        megafmm::tune(*plan,ap,bp,beta?static_cast<float*>(dc.p):nullptr,out,scratch.p,stream);
      }
      for(int mode=0;mode<2;++mode) {
        check(cudaMemcpyAsync(dc.p,c.data(),nc*4,cudaMemcpyHostToDevice,stream));
        if(!inplace) { std::fill(d.begin(),d.end(),-12345);check(cudaMemcpyAsync(dd.p,d.data(),nd*4,cudaMemcpyHostToDevice,stream)); }
        if(scratch.p) check(cudaMemsetAsync(scratch.p,0xa5,plan->workspace_bytes(),stream));
        if(mode) check(cudaGraphLaunch(graph.exec,stream));
        else megafmm::run(*plan,ap,bp,beta?static_cast<float*>(dc.p):nullptr,out,scratch.p,stream);
        check(cudaMemcpyAsync(d.data(),out,nd*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
        for(int r=0;r<m;++r) {
          for(int col=0;col<n;++col) {
            double ref=0;
            if(fractional) ref=oracle(p,a,b,r,col);
            else if(diagonal) ref=double(float(a[r*p.lda+r]))*float(b[tb?col*p.ldb+r:r*p.ldb+col]);
            else for(int q=0;q<k;++q) ref+=double(float(a[ta?q*p.lda+r:r*p.lda+q]))*float(b[tb?col*p.ldb+q:q*p.ldb+col]);
            ref=alpha*ref+(beta?beta*c[r*p.ldc+col]:0);
            float actual=d[r*p.ldd+col];
            if(!std::isfinite(actual) || std::abs(actual-ref)>(fractional?2e-5*(1+std::abs(ref)):0)) {
              fprintf(stderr,"mismatch %dx%dx%d dtype=%s trial=%d graph=%d at %d,%d actual=%g ref=%.12g\n",m,n,k,megafmm::dtype_name(dtype),trial,mode,r,col,actual,ref);
              throw std::runtime_error("multisize reference mismatch");
            }
          }
          for(int col=n;col<p.ldd;++col)
            require(d[r*p.ldd+col]==(inplace?c[r*p.ldc+col]:-12345),"output padding overwritten");
        }
      }
    }
  }
  check(cudaStreamDestroy(stream));
  printf("PASS dtype=%s shape=%dx%dx%d ta=%d tb=%d stride=%d alpha=%g beta=%g inplace=%d fractional=%d dedicated=%d unaligned=%d hopper=%d fresh_launch_graph=4\n",megafmm::dtype_name(dtype),m,n,k,ta,tb,strided,alpha,beta,inplace,fractional,dedicated,unaligned,hopper);
}

// Two independently bound calls share one immutable Plan. Sparse integer A
// gives an exact original-AB oracle without a cubic CPU reference.
template<class T> void concurrent_hopper(megafmm::DType dtype,int m,int n,int k) {
  megafmm::Problem p;p.m=m;p.n=n;p.k=k;p.dtype=dtype;
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::SWPipeline;
  auto plan=megafmm::create_plan(p,policy);
  require(plan->algorithm_name()=="sw_hopper","concurrency did not select Hopper SW");
  const size_t ac=size_t(m)*k,bc=size_t(k)*n,dc=size_t(m)*n,ws=plan->workspace_bytes();
  Buffer a0(ac*sizeof(T)),a1(ac*sizeof(T)),b0(bc*sizeof(T)),b1(bc*sizeof(T));
  Buffer d0((dc+64)*4),d1((dc+64)*4),w0(ws+512),w1(ws+512);
  void* ap[2]={a0.p,a1.p};void* bp[2]={b0.p,b1.p};
  float* out[2]={static_cast<float*>(d0.p)+32,static_cast<float*>(d1.p)+32};
  void* work[2]={static_cast<char*>(w0.p)+256,static_cast<char*>(w1.p)+256};
  void* outbase[2]={d0.p,d1.p};void* workbase[2]={w0.p,w1.p};
  cudaStream_t streams[2];Graph graphs[2];
  for(int set=0;set<2;++set) {
    check(cudaStreamCreateWithFlags(&streams[set],cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(streams[set],cudaStreamCaptureModeThreadLocal));
    megafmm::run(*plan,ap[set],bp[set],nullptr,out[set],work[set],streams[set]);
    check(cudaStreamEndCapture(streams[set],&graphs[set].graph));
    check(cudaGraphInstantiate(&graphs[set].exec,graphs[set].graph,nullptr,nullptr,0));
  }
  std::vector<T> a[2],b[2];
  for(int trial=0;trial<3;++trial) {
    for(int set=0;set<2;++set) {
      a[set].assign(ac,quantize<T>(0));b[set].resize(bc);
      for(int r=0;r<m;++r) {
        int q=(r*13+trial*7+set*11)%k;
        a[set][size_t(r)*k+q]=quantize<T>((r+trial+set)%2?-1:1);
      }
      for(size_t i=0;i<bc;++i)b[set][i]=quantize<T>(float(int((i*17+trial*7+set)%7)-3));
      check(cudaMemcpyAsync(ap[set],a[set].data(),ac*sizeof(T),cudaMemcpyHostToDevice,streams[set]));
      check(cudaMemcpyAsync(bp[set],b[set].data(),bc*sizeof(T),cudaMemcpyHostToDevice,streams[set]));
    }
    for(int mode=0;mode<2;++mode) {
      for(int set=0;set<2;++set) {
        check(cudaMemsetAsync(outbase[set],0x5a,(dc+64)*4,streams[set]));
        check(cudaMemsetAsync(workbase[set],0xa5,ws+512,streams[set]));
        if(mode)check(cudaGraphLaunch(graphs[set].exec,streams[set]));
        else megafmm::run(*plan,ap[set],bp[set],nullptr,out[set],work[set],streams[set]);
      }
      for(int set=0;set<2;++set) {
        std::vector<float> result(dc);uint32_t og[64],wg[128];
        check(cudaMemcpyAsync(result.data(),out[set],dc*4,cudaMemcpyDeviceToHost,streams[set]));
        check(cudaMemcpyAsync(og,outbase[set],128,cudaMemcpyDeviceToHost,streams[set]));
        check(cudaMemcpyAsync(og+32,out[set]+dc,128,cudaMemcpyDeviceToHost,streams[set]));
        check(cudaMemcpyAsync(wg,workbase[set],256,cudaMemcpyDeviceToHost,streams[set]));
        check(cudaMemcpyAsync(wg+64,static_cast<char*>(work[set])+ws,256,cudaMemcpyDeviceToHost,streams[set]));
        check(cudaStreamSynchronize(streams[set]));
        for(uint32_t x:og)require(x==0x5a5a5a5a,"dual stream output guard overwritten");
        for(uint32_t x:wg)require(x==0xa5a5a5a5,"dual stream workspace guard overwritten");
        for(int r=0;r<m;++r)for(int c=0;c<n;++c) {
          int q=(r*13+trial*7+set*11)%k;
          float expected=((r+trial+set)%2?-1:1)*float(b[set][size_t(q)*n+c]);
          require(result[size_t(r)*n+c]==expected,"dual stream fresh original AB mismatch");
        }
      }
    }
  }
  for(auto stream:streams)check(cudaStreamDestroy(stream));
  printf("PASS shared_plan_dual_stream dtype=%s shape=%dx%dx%d fresh_launch_graph=12 guards=pass\n",megafmm::dtype_name(dtype),m,n,k);
}

template<class T> void suite(megafmm::DType dtype) {
  for(auto shape:std::vector<std::vector<int>>{{1,1,1},{2,3,5},{17,31,33},{63,65,127},{128,256,64},{512,512,512},{1025,65,33},{2048,17,31},{4096,3,9}})
    exercise<T>(dtype,shape[0],shape[1],shape[2]);
  for(bool ta:{false,true}) for(bool tb:{false,true})
    exercise<T>(dtype,35,49,67,ta,tb,true,0.5f,-0.25f,false,true);
  exercise<T>(dtype,33,17,65,true,true,true,-1,0.5f,true,true);
  exercise<T>(dtype,1024,1024,1024,false,false,false,1,0,false,false,true);
  exercise<T>(dtype,1024,1024,1024,true,true,true,0.5f,-0.25f,true,false,true);
  // Runtime K loops, independent rectangular S/T planes, short and wrapped rings.
  for(auto shape:std::vector<std::vector<int>>{{256,128,256},{512,256,512},{1536,128,768},{2048,256,256},{1024,256,1536}})
    exercise<T>(dtype,shape[0],shape[1],shape[2],false,false,false,1,0,false,false,false,false,false,true);
  exercise<T>(dtype,256,128,256,false,false,false,1,0,false,true,false,false,false,true);
  // More than one wave of CTAs with an inexpensive exact original-AB oracle.
  exercise<T>(dtype,4096,2048,4096,false,false,false,1,0,false,false,true,false,false,true);
  exercise<T>(dtype,512,256,512,false,false,false,1,0,false,false,false,false,true,true);
  concurrent_hopper<T>(dtype,1536,128,768);
  concurrent_hopper<T>(dtype,2048,256,1536);
  // Restore the fast specialization while retaining zero-budget and unaligned SW.
  exercise<T>(dtype,1024,1024,1024,false,false,false,1,0,false,false,true,true);
  exercise<T>(dtype,1024,1024,1024,false,false,false,1,0,false,false,true,true,true);
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::SWPipeline;policy.workspace_limit=0;
  megafmm::Problem p;p.dtype=dtype;p.m=3;p.n=5;p.k=0;p.beta=2;
  auto scale=megafmm::create_plan(p,policy);require(scale->algorithm_name()=="degenerate_scale","K=0 dispatch");
  Buffer d(15*4);std::vector<float> v(15,3);check(cudaMemcpy(d.p,v.data(),15*4,cudaMemcpyHostToDevice));
  megafmm::run(*scale,nullptr,nullptr,static_cast<float*>(d.p),static_cast<float*>(d.p),nullptr);
  check(cudaMemcpy(v.data(),d.p,15*4,cudaMemcpyDeviceToHost));for(float x:v)require(x==6,"K=0 scaling");
  p.k=7;p.alpha=0;auto zero_alpha=megafmm::create_plan(p,policy);
  megafmm::run(*zero_alpha,nullptr,nullptr,static_cast<float*>(d.p),static_cast<float*>(d.p),nullptr);
  check(cudaMemcpy(v.data(),d.p,15*4,cudaMemcpyDeviceToHost));for(float x:v)require(x==12,"alpha=0 scaling");
  p.m=0;auto empty=megafmm::create_plan(p,policy);megafmm::run(*empty,nullptr,nullptr,nullptr,nullptr,nullptr);
}
} // namespace
int main(int argc,char** argv) { try {
  if(argc==2 && std::string(argv[1])=="--hopper-only") {
    concurrent_hopper<__nv_bfloat16>(megafmm::DType::BF16,256,128,256);
    concurrent_hopper<__nv_bfloat16>(megafmm::DType::BF16,2048,256,1536);
    concurrent_hopper<__half>(megafmm::DType::FP16,256,128,256);
    concurrent_hopper<__half>(megafmm::DType::FP16,2048,256,1536);
    return 0;
  }
  if(argc!=1)throw std::invalid_argument("usage: test_sw_pipeline [--hopper-only]");
  suite<__nv_bfloat16>(megafmm::DType::BF16);suite<__half>(megafmm::DType::FP16);
  puts("PASS SW pipeline implementation checks (original-input precision gate remains separate)");return 0;
} catch(const std::exception& e) { fprintf(stderr,"ERROR %s\n",e.what());return 1; } }

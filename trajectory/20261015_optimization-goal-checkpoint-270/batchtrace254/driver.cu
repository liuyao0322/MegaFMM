#include <vector>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "base.cuh"
#include "trace.cuh"
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>void run(megafmm::DType dt,const char* name){
 constexpr int N=1024*1024,Batch=10;void* A;void* B;void* W;float* D;float* E;uint64_t* tr;cudaStream_t st;ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
 ck(cudaMalloc(&A,N*4));ck(cudaMalloc(&B,N*4));ck(cudaMalloc(&W,4194304));ck(cudaMalloc(&D,N*8));ck(cudaMalloc(&E,N*8));ck(cudaMalloc(&tr,Batch*128*8*8));
 megafmm::Problem p;p.m=p.n=p.k=p.lda=p.ldb=p.ldc=p.ldd=1024;p.dtype=dt;
 auto bs=megafmm::base254::make_sw_pipeline(p);auto ts=megafmm::trace254::make_sw_pipeline(p);
 std::vector<T>a(2*N),b(2*N);std::vector<float>x(2*N),y(2*N);std::vector<uint64_t>traces(Batch*128*8);
 for(int trial=0;trial<3;++trial){for(int i=0;i<2*N;++i){a[i]=T(float((i*17+trial*43)%127-63)/64);b[i]=T(float((i*13+trial*29)%131-65)/64);}
 ck(cudaMemcpy(A,a.data(),N*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(B,b.data(),N*4,cudaMemcpyHostToDevice));
 for(int i=0;i<20;++i)megafmm::base254::run_sw_pipeline(*bs,static_cast<T*>(A)+(i%2)*N,static_cast<T*>(B)+(i%2)*N,D+(i%2)*N,W,st);ck(cudaStreamSynchronize(st));
 cudaGraph_t g;cudaGraphExec_t ge;ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));for(int j=0;j<Batch;++j)megafmm::trace254::run_sw_pipeline(*ts,static_cast<T*>(A)+(j%2)*N,static_cast<T*>(B)+(j%2)*N,E+(j%2)*N,W,st,tr+j*128*8);ck(cudaStreamEndCapture(st,&g));ck(cudaGraphInstantiate(&ge,g,0));
 for(int graph=0;graph<2;++graph)for(int rep=0;rep<5;++rep){if(graph)ck(cudaGraphLaunch(ge,st));else for(int j=0;j<Batch;++j)megafmm::trace254::run_sw_pipeline(*ts,static_cast<T*>(A)+(j%2)*N,static_cast<T*>(B)+(j%2)*N,E+(j%2)*N,W,st,tr+j*128*8);ck(cudaStreamSynchronize(st));
 ck(cudaMemcpy(x.data(),D,N*8,cudaMemcpyDeviceToHost));ck(cudaMemcpy(y.data(),E,N*8,cudaMemcpyDeviceToHost));if(memcmp(x.data(),y.data(),N*8))throw std::runtime_error("trace mismatch");
 ck(cudaMemcpy(traces.data(),tr,traces.size()*8,cudaMemcpyDeviceToHost));for(int invocation=0;invocation<Batch;++invocation)for(int c=0;c<128;++c){printf("%s,%d,%d,%d,%d,%d",name,trial,graph,rep,invocation,c);for(int j=0;j<7;++j)printf(",%llu",(unsigned long long)traces[(invocation*128+c)*8+j]);puts("");}
 }cudaGraphExecDestroy(ge);cudaGraphDestroy(g);}
 cudaFree(A);cudaFree(B);cudaFree(W);cudaFree(D);cudaFree(E);cudaFree(tr);cudaStreamDestroy(st);
}
int main(){puts("dtype,trial,graph,rep,invocation,cta,pack_start,pack_lane0_finish,compute_entry,dep_before,dep_after,tc_start,tc_end");run<cute::bfloat16_t>(megafmm::DType::BF16,"bf16");run<cute::half_t>(megafmm::DType::FP16,"fp16");}

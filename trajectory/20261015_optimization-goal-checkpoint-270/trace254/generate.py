from pathlib import Path
import subprocess
p=Path('results/opt_goal/trace254')
s=subprocess.check_output(['git','show','13340d1:solution/sw_pipeline.cu'],text=True).replace('namespace megafmm::detail','namespace megafmm::base254')
(p/'base.cuh').write_text(s)
s=s.replace('namespace megafmm::base254','namespace megafmm::trace254')
s=s.replace('void sw_transform_pdl(const T* A,const T* B,T* packed){','void sw_transform_pdl(const T* A,const T* B,T* packed,uint64_t* trace){')
s=s.replace('const T* A,const T* B,T* packed,float* D){','const T* A,const T* B,T* packed,float* D,uint64_t* trace){')
a=s.index('template<class T> __global__');s=s[:a]+'''__device__ __forceinline__ void stamp254(uint64_t* trace,int phase,int cta){uint64_t t;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t));trace[cta*8+phase]=t;}
'''+s[a:]
s=s.replace('  cudaTriggerProgrammaticLaunchCompletion();','  if(threadIdx.x==0)stamp254(trace,0,blockIdx.x);\n  cudaTriggerProgrammaticLaunchCompletion();')
s=s.replace('  asm volatile("fence.proxy.async.global;" ::: "memory");','  asm volatile("fence.proxy.async.global;" ::: "memory");\n  if(threadIdx.x==0)stamp254(trace,1,blockIdx.x);')
s=s.replace('  int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;','  int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;\n  if(tid==0)stamp254(trace,2,blockIdx.y*16+blockIdx.x);')
s=s.replace('      if(sequence==4)cudaGridDependencySynchronize();','      if(sequence==4){stamp254(trace,3,blockIdx.y*16+blockIdx.x);cudaGridDependencySynchronize();stamp254(trace,4,blockIdx.y*16+blockIdx.x);}')
s=s.replace('    tensor_chain(s,tid,m0,n0,D);','    if(tid==128)stamp254(trace,5,blockIdx.y*16+blockIdx.x);\n    tensor_chain(s,tid,m0,n0,D);\n    if(tid==128)stamp254(trace,6,blockIdx.y*16+blockIdx.x);')
s=s.replace('void* workspace,cudaStream_t stream){','void* workspace,cudaStream_t stream,uint64_t* trace){')
s=s.replace('static_cast<cute::bfloat16_t*>(workspace));','static_cast<cute::bfloat16_t*>(workspace),trace);').replace('static_cast<cute::half_t*>(workspace));','static_cast<cute::half_t*>(workspace),trace);')
s=s.replace('workspace),D),"SW PDL compute launch"','workspace),D,trace),"SW PDL compute launch"')
(p/'trace.cuh').write_text(s)
(p/'driver.cu').write_text('''#include <vector>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "base.cuh"
#include "trace.cuh"
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>void run(megafmm::DType dt,const char* name){
 constexpr int N=1024*1024;void* A;void* B;void* W;float* D;float* E;uint64_t* tr;cudaStream_t st;ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
 ck(cudaMalloc(&A,N*2));ck(cudaMalloc(&B,N*2));ck(cudaMalloc(&W,4194304));ck(cudaMalloc(&D,N*4));ck(cudaMalloc(&E,N*4));ck(cudaMalloc(&tr,128*8*8));
 megafmm::Problem p;p.m=p.n=p.k=p.lda=p.ldb=p.ldc=p.ldd=1024;p.dtype=dt;
 auto bs=megafmm::base254::make_sw_pipeline(p);auto ts=megafmm::trace254::make_sw_pipeline(p);
 std::vector<T>a(N),b(N);std::vector<float>x(N),y(N);std::vector<uint64_t>traces(128*8);
 for(int trial=0;trial<3;++trial){for(int i=0;i<N;++i){a[i]=T(float((i*17+trial*43)%127-63)/64);b[i]=T(float((i*13+trial*29)%131-65)/64);}
 ck(cudaMemcpy(A,a.data(),N*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(B,b.data(),N*2,cudaMemcpyHostToDevice));
 for(int i=0;i<20;++i)megafmm::base254::run_sw_pipeline(*bs,A,B,D,W,st);ck(cudaStreamSynchronize(st));
 cudaGraph_t g;cudaGraphExec_t ge;ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));megafmm::trace254::run_sw_pipeline(*ts,A,B,E,W,st,tr);ck(cudaStreamEndCapture(st,&g));ck(cudaGraphInstantiate(&ge,g,0));
 for(int graph=0;graph<2;++graph)for(int rep=0;rep<5;++rep){if(graph)ck(cudaGraphLaunch(ge,st));else megafmm::trace254::run_sw_pipeline(*ts,A,B,E,W,st,tr);ck(cudaStreamSynchronize(st));
 ck(cudaMemcpy(x.data(),D,N*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(y.data(),E,N*4,cudaMemcpyDeviceToHost));if(memcmp(x.data(),y.data(),N*4))throw std::runtime_error("trace mismatch");
 ck(cudaMemcpy(traces.data(),tr,traces.size()*8,cudaMemcpyDeviceToHost));for(int c=0;c<128;++c){printf("%s,%d,%d,%d,%d",name,trial,graph,rep,c);for(int j=0;j<7;++j)printf(",%llu",(unsigned long long)traces[c*8+j]);puts("");}
 }cudaGraphExecDestroy(ge);cudaGraphDestroy(g);}
 cudaFree(A);cudaFree(B);cudaFree(W);cudaFree(D);cudaFree(E);cudaFree(tr);cudaStreamDestroy(st);
}
int main(){puts("dtype,trial,graph,rep,cta,pack_start,pack_lane0_finish,compute_entry,dep_before,dep_after,tc_start,tc_end");run<cute::bfloat16_t>(megafmm::DType::BF16,"bf16");run<cute::half_t>(megafmm::DType::FP16,"fp16");}
''')

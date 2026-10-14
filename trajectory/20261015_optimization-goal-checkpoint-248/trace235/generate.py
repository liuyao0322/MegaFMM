from pathlib import Path
import subprocess
p=Path('results/opt_goal/trace235')
s=subprocess.check_output(['git','show','92569ff:solution/sw_pipeline.cu'],text=True).replace('namespace megafmm::detail','namespace megafmm::base235')
(p/'base.cuh').write_text(s)
s=s.replace('namespace megafmm::base235','namespace megafmm::trace235')
s=s.replace('const T* A,const T* B,T* packed,float* D){','const T* A,const T* B,T* packed,float* D,uint64_t* trace){')
pos=s.index('template<class T> __global__')
s=s[:pos]+'''__device__ __forceinline__ void stamp(uint64_t* trace,int phase){if(threadIdx.x==128){uint64_t t;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t));trace[(((blockIdx.z*8+blockIdx.y)*8+blockIdx.x)*6)+phase]=t;}}
'''+s[pos:]
s=s.replace('    tensor_chain(s,tid,m0,n0,partial+blockIdx.z*1024*1024);','    stamp(trace,0);\n    tensor_chain(s,tid,m0,n0,partial+blockIdx.z*1024*1024);\n    stamp(trace,1);')
s=s.replace('    if(tid==128){\n      auto* counter','    stamp(trace,2);\n    if(tid==128){\n      auto* counter')
s=s.replace('    if(s.last){','    stamp(trace,3);\n    if(tid==128)trace[(((blockIdx.z*8+blockIdx.y)*8+blockIdx.x)*6)+5]=s.last;\n    if(s.last){')
s=s.replace('  // CTA shared storage survives','  if(tid==128)stamp(trace,4);\n  // CTA shared storage survives')
s=s.replace('void* workspace,cudaStream_t stream){','void* workspace,cudaStream_t stream,uint64_t* trace){').replace('&workspace,&D};','&workspace,&D,&trace};')
(p/'trace.cuh').write_text(s)
(p/'driver.cu').write_text('''#include <vector>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "base.cuh"
#include "trace.cuh"
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>void run(megafmm::DType dt,const char* name){
 constexpr int N=1024*1024;void* A;void* B;void* W;float* D;float* E;uint64_t* tr;
 ck(cudaMalloc(&A,N*2));ck(cudaMalloc(&B,N*2));ck(cudaMalloc(&W,12583168));ck(cudaMalloc(&D,N*4));ck(cudaMalloc(&E,N*4));ck(cudaMalloc(&tr,128*6*8));
 megafmm::Problem p;p.m=p.n=p.k=p.lda=p.ldb=p.ldc=p.ldd=1024;p.dtype=dt;
 auto bs=megafmm::base235::make_sw_pipeline(p);auto ts=megafmm::trace235::make_sw_pipeline(p);
 std::vector<T>a(N),b(N);std::vector<float>x(N),y(N);std::vector<uint64_t>traces(128*6);
 for(int trial=0;trial<3;++trial){for(int i=0;i<N;++i){a[i]=T(float((i*17+trial*43)%127-63)/64);b[i]=T(float((i*13+trial*29)%131-65)/64);}
 ck(cudaMemcpy(A,a.data(),N*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(B,b.data(),N*2,cudaMemcpyHostToDevice));
 megafmm::base235::run_sw_pipeline(*bs,A,B,D,W,0);ck(cudaDeviceSynchronize());
 for(int rep=0;rep<5;++rep){megafmm::trace235::run_sw_pipeline(*ts,A,B,E,W,0,tr);ck(cudaDeviceSynchronize());
 ck(cudaMemcpy(x.data(),D,N*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(y.data(),E,N*4,cudaMemcpyDeviceToHost));if(memcmp(x.data(),y.data(),N*4))throw std::runtime_error("trace mismatch");
 ck(cudaMemcpy(traces.data(),tr,traces.size()*8,cudaMemcpyDeviceToHost));for(int c=0;c<128;++c)printf("%s,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu\\n",name,trial,rep,c,(unsigned long long)traces[c*6],(unsigned long long)traces[c*6+1],(unsigned long long)traces[c*6+2],(unsigned long long)traces[c*6+3],(unsigned long long)traces[c*6+4],(unsigned long long)traces[c*6+5]);
 }}
 cudaFree(A);cudaFree(B);cudaFree(W);cudaFree(D);cudaFree(E);cudaFree(tr);
}
int main(){puts("dtype,trial,rep,cta,start,compute,fence,counter,finish,last");run<cute::bfloat16_t>(megafmm::DType::BF16,"bf16");run<cute::half_t>(megafmm::DType::FP16,"fp16");}
''')

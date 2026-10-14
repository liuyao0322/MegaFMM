#include <vector>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "base.cuh"
#include "trace.cuh"
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>void run(megafmm::DType dt,const char* name){
 constexpr int N=1024*1024;void* A;void* B;void* W;float* D;float* E;uint64_t* tr;
 ck(cudaMalloc(&A,N*2));ck(cudaMalloc(&B,N*2));ck(cudaMalloc(&W,4194560));ck(cudaMalloc(&D,N*4));ck(cudaMalloc(&E,N*4));ck(cudaMalloc(&tr,128*12*8));
 megafmm::Problem p;p.m=p.n=p.k=p.lda=p.ldb=p.ldc=p.ldd=1024;p.dtype=dt;
 auto bs=megafmm::base249::make_sw_pipeline(p);auto ts=megafmm::trace249::make_sw_pipeline(p);
 std::vector<T>a(N),b(N);std::vector<float>x(N),y(N);std::vector<uint64_t>traces(128*12);
 for(int trial=0;trial<3;++trial){for(int i=0;i<N;++i){a[i]=T(float((i*17+trial*43)%127-63)/64);b[i]=T(float((i*13+trial*29)%131-65)/64);}
 ck(cudaMemcpy(A,a.data(),N*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(B,b.data(),N*2,cudaMemcpyHostToDevice));
 megafmm::base249::run_sw_pipeline(*bs,A,B,D,W,0);ck(cudaDeviceSynchronize());
 for(int rep=0;rep<5;++rep){megafmm::trace249::run_sw_pipeline(*ts,A,B,E,W,0,tr);ck(cudaDeviceSynchronize());
 ck(cudaMemcpy(x.data(),D,N*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(y.data(),E,N*4,cudaMemcpyDeviceToHost));if(memcmp(x.data(),y.data(),N*4))throw std::runtime_error("trace mismatch");
 ck(cudaMemcpy(traces.data(),tr,traces.size()*8,cudaMemcpyDeviceToHost));for(int c=0;c<128;++c){printf("%s,%d,%d,%d",name,trial,rep,c);for(int j=0;j<11;++j)printf(",%llu",(unsigned long long)traces[c*12+j]);puts("");}
 }}
 cudaFree(A);cudaFree(B);cudaFree(W);cudaFree(D);cudaFree(E);cudaFree(tr);
}
int main(){puts("dtype,trial,rep,cta,tma_start,raw_end,init_acquired,pack_acquired,published,q0_acquired,q1_acquired,coord_start,coord_end,tc_start,tc_end");run<cute::bfloat16_t>(megafmm::DType::BF16,"bf16");run<cute::half_t>(megafmm::DType::FP16,"fp16");}

from pathlib import Path
import subprocess
p=Path('results/opt_goal/reassess228')
for name,commit in [('candidate221','fbf9caa'),('candidate224','62142fc')]:
 s=subprocess.check_output(['git','show',commit+':solution/sw_pipeline.cu'],text=True).replace('namespace megafmm::detail','namespace megafmm::'+name)
 (p/(name+'.cuh')).write_text(s)
(p/'profile.cu').write_text('''#include <cuda_profiler_api.h>
#include <vector>
#include "candidate221.cuh"
#include "candidate224.cuh"
template<class T>void run(megafmm::DType dt){
 constexpr int N=1024*1024;void* A;void* B;void* W;float* D;
 cudaMalloc(&A,N*2);cudaMalloc(&B,N*2);cudaMalloc(&W,N*4);cudaMalloc(&D,N*4);
 std::vector<T>a(N),b(N);for(int i=0;i<N;++i){a[i]=T(float(i%127-63)/64);b[i]=T(float(i%131-65)/64);}
 cudaMemcpy(A,a.data(),N*2,cudaMemcpyHostToDevice);cudaMemcpy(B,b.data(),N*2,cudaMemcpyHostToDevice);
 megafmm::Problem p;p.m=p.n=p.k=p.lda=p.ldb=p.ldc=p.ldd=1024;p.dtype=dt;
 auto a221=megafmm::candidate221::make_sw_pipeline(p);auto a224=megafmm::candidate224::make_sw_pipeline(p);
 for(int i=0;i<20;++i){megafmm::candidate221::run_sw_pipeline(*a221,A,B,D,W,0);megafmm::candidate224::run_sw_pipeline(*a224,A,B,D,W,0);}cudaDeviceSynchronize();
 cudaProfilerStart();megafmm::candidate221::run_sw_pipeline(*a221,A,B,D,W,0);cudaDeviceSynchronize();megafmm::candidate224::run_sw_pipeline(*a224,A,B,D,W,0);cudaDeviceSynchronize();cudaProfilerStop();
 cudaFree(A);cudaFree(B);cudaFree(W);cudaFree(D);
}
int main(){run<cute::bfloat16_t>(megafmm::DType::BF16);run<cute::half_t>(megafmm::DType::FP16);}
''')

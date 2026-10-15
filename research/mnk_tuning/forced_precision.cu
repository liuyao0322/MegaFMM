// Diagnose fixed candidates against the benchmark's original-input FP64 oracle.
// This is a diagnostic, not independent performance certification.
#define main megafmm_benchmark_main
#include "../../bench/benchmark.cu"
#undef main
#include "internal.h"

template<class T>
int forced_precision(int n,megafmm::DType dtype,bool timing) {
  using namespace megafmm;
  Stream stream;
  Problem p;p.m=p.n=p.k=n;p.lda=p.ldb=p.ldc=p.ldd=n;p.dtype=dtype;
  Policy policy;policy.algorithm=Algorithm::Dense;
  auto dense=create_plan(p,policy);
  int device;cudaDeviceProp prop{};
  check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
  const int64_t count=int64_t(n)*n;
  size_t required=0;
  for(int id=94;id<=97;++id) {
    required=std::max(required,detail::hopper_workspace(p,id,prop.multiProcessorCount));
    detail::hopper_dense(id,p,nullptr,nullptr,nullptr,nullptr,nullptr,prop.multiProcessorCount,stream.value,2);
  }
  Buffer a(count*2),b(count*2),out(count*4),base(count*4);
  Buffer a64(count*8),b64(count*8),ref(count*8),base64(count*8);
  Buffer workspace(dense->workspace_bytes()),candidate_workspace(required),scratch(reduction_blocks*sizeof(Partial));
  Blas blas(stream.value);
  auto fill=[&](uint64_t seed) {
    fill_input<<<1024,256,0,stream.value>>>(a.as<T>(),count,seed);
    fill_input<<<1024,256,0,stream.value>>>(b.as<T>(),count,seed^0x632be59bd9b4e019ULL);
    check(cudaGetLastError());
  };
  fill(1234567);
  tune(*dense,a.data,b.data,nullptr,base.as<float>(),workspace.data,stream.value);
  const double floor=dtype==DType::BF16?1e-3:1e-4;
  int failures=0;
  for(int trial=-1;trial<(timing?0:3);++trial) {
    fill(trial<0?1234567ULL:1234567ULL+uint64_t(trial+1)*0x9e3779b97f4a7c15ULL);
    to_double<<<1024,256,0,stream.value>>>(a.as<T>(),a64.as<double>(),count);
    to_double<<<1024,256,0,stream.value>>>(b.as<T>(),b64.as<double>(),count);
    check(cudaGetLastError());
    double one=1,zero=0;
    check(cublasDgemm(blas.value,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&one,b64.as<double>(),n,a64.as<double>(),n,&zero,ref.as<double>(),n));
    run(*dense,a.data,b.data,nullptr,base.as<float>(),workspace.data,stream.value);
    to_double<<<1024,256,0,stream.value>>>(base.as<float>(),base64.as<double>(),count);
    check(cudaGetLastError());
    const auto de=error(base.as<float>(),ref.as<double>(),count,floor,1e-3,1,stream.value,scratch);
    for(int id=94;id<=97;++id) {
      if(!detail::hopper_dense(id,p,a.data,b.data,nullptr,out.as<float>(),candidate_workspace.data,prop.multiProcessorCount,stream.value,1))
        throw std::runtime_error("fixed candidate rejected shape");
      const auto ce=error(out.as<float>(),ref.as<double>(),count,floor,1e-3,1,stream.value,scratch);
      const auto difference=error(out.as<float>(),base64.as<double>(),count,floor,1e-3,1,stream.value,scratch);
      const bool agreement=detail::dense_agreement(p,out.as<float>(),base.as<float>(),stream.value);
      const double gate=3*de.frobenius+floor;
      const bool pass=de.nonfinite==0 && ce.nonfinite==0 && std::isfinite(ce.frobenius) && ce.frobenius<=gate;
      failures+=!pass;
      printf("DIAGNOSTIC n=%d dtype=%s trial=%d candidate=%d dense_ef=%.12g candidate_ef=%.12g gate=%.12g dense_difference_ef=%.12g agreement=%d original_gate_pass=%d nonfinite=%llu\n",n,dtype_name(dtype),trial,id,de.frobenius,ce.frobenius,gate,difference.frobenius,int(agreement),int(pass),static_cast<unsigned long long>(ce.nonfinite));
      fflush(stdout);
      if(timing) {
        auto launch=[&](int route) {
          if(route==0)run(*dense,a.data,b.data,nullptr,base.as<float>(),workspace.data,stream.value);
          else if(!detail::hopper_dense(id,p,a.data,b.data,nullptr,out.as<float>(),candidate_workspace.data,prop.multiProcessorCount,stream.value,1))
            throw std::runtime_error("fixed candidate timing rejected");
        };
        for(int warm=0;warm<20;++warm){launch(0);launch(1);}
        Event start,stop;
        std::vector<float> gains;
        for(int group=0;group<5;++group) {
          float ms[2]{};
          for(int order=0;order<2;++order) {
            int route=order^(group&1);
            check(cudaEventRecord(start.value,stream.value));
            for(int call=0;call<20;++call)launch(route);
            check(cudaEventRecord(stop.value,stream.value));check(cudaEventSynchronize(stop.value));
            check(cudaEventElapsedTime(&ms[route],start.value,stop.value));ms[route]/=20;
          }
          gains.push_back(ms[0]/ms[1]);
          printf("DIAGNOSTIC_TIME n=%d dtype=%s candidate=%d group=%d dense_ms=%.9g custom_ms=%.9g speedup=%.9g calls=20 input_sets=1 certification=unchecked\n",n,dtype_name(dtype),id,group,ms[0],ms[1],gains.back());
          fflush(stdout);
        }
        std::sort(gains.begin(),gains.end());
        printf("DIAGNOSTIC_TIMING_SUMMARY candidate=%d median_speedup=%.9g min_speedup=%.9g certification=unchecked\n",id,gains[2],gains[0]);
        fflush(stdout);
      }
    }
  }
  return failures?2:0;
}
int main(int argc,char**argv) {try {
  const int n=argc>1?std::stoi(argv[1]):16384;
  const bool timing=argc>3 && std::string(argv[3])=="--timing";
  if(argc>2 && std::string(argv[2])=="bf16")return forced_precision<__nv_bfloat16>(n,megafmm::DType::BF16,timing);
  return forced_precision<__half>(n,megafmm::DType::FP16,timing);
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

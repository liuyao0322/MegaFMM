#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#include "traced_f9.cuh"
#include <fstream>

int main(int argc,char** argv){try{
 if(argc!=2)throw std::invalid_argument("trace requires output CSV path");
 fused_int8::init();Stream stream;
 Buffer a(Count*2),b(Count*2),ad(Count*8),bd(Count*8),d(Count*4),trace(128*fused_int8::TraceCTA*8);
 fill<<<512,256,0,stream.s>>>(a.as<__nv_bfloat16>(),b.as<__nv_bfloat16>(),ad.as<double>(),bd.as<double>(),0,1901901);
 CUtensorMap ma,mb;fused_int8::encode(ma,a.p,megafmm::DType::BF16,true);fused_int8::encode(mb,b.p,megafmm::DType::BF16,false);
 auto launch=[&]{fused_int8::kernel<cute::bfloat16_t><<<dim3(16,8),256,sizeof(fused_int8::Shared),stream.s>>>(ma,mb,d.as<float>(),trace.as<uint64_t>());};
 check(cudaMemsetAsync(trace.p,0,128*fused_int8::TraceCTA*8,stream.s));
 for(int i=0;i<10;++i)launch();
 cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));check(cudaEventRecord(begin,stream.s));
 for(int i=0;i<20;++i)launch();check(cudaEventRecord(end,stream.s));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,begin,end));
 printf("TRACE diagnostic_kernel_us=%.9f dtype=bf16 warmup=10 calls=20 timing_acceptance=0\n",ms*50);
 Blas blas(stream.s);Buffer ref(Count*8),base(Count*4),scratch(128*sizeof(Metric));double one=1,zero=0;
 check(cublasDgemm(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&one,bd.as<double>(),N,ad.as<double>(),N,&zero,ref.as<double>(),N));
 float fone=1,fzero=0;check(cublasGemmEx(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&fone,b.p,CUDA_R_16BF,N,a.p,CUDA_R_16BF,N,&fzero,base.p,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
 auto dm=metric(base.as<float>(),ref.as<double>(),scratch,stream.s),cm=metric(d.as<float>(),ref.as<double>(),scratch,stream.s);double gate=3*ef(dm)+1e-3;
 printf("TRACE_CHECK ef=%.12g gate=%.12g nonfinite=%llu pass=%d\n",ef(cm),gate,cm.nonfinite,cm.nonfinite==0&&ef(cm)<=gate);
 if(cm.nonfinite||ef(cm)>gate)return 2;
 std::vector<uint64_t> host(128*fused_int8::TraceCTA);check(cudaMemcpy(host.data(),trace.p,host.size()*8,cudaMemcpyDeviceToHost));
 std::ofstream out(argv[1]);out<<"cta,role,kind,seq,t0,t1,t2,t3,t4,t5\n";
 for(int cta=0;cta<128;++cta)for(int role=0;role<2;++role){
  auto ptr=host.data()+cta*fused_int8::TraceCTA+role*fused_int8::TraceRole;
  for(int seq=0;seq<20;++seq){out<<cta<<','<<role<<",leaf,"<<seq;for(int j=0;j<6;++j)out<<','<<ptr[seq*6+j];out<<'\n';}
  for(int k=0;k<4;++k){out<<cta<<','<<role<<",raw,"<<k;for(int j=0;j<4;++j)out<<','<<ptr[120+k*4+j];out<<",0,0\n";}
 }
 check(cudaEventDestroy(begin));check(cudaEventDestroy(end));return 0;
}catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

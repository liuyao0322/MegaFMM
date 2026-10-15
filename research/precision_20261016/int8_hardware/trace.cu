#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#include "traced_corrected.cuh"
#include <fstream>

int main(int argc,char** argv){try{
 if(argc!=2)throw std::invalid_argument("trace requires output CSV path");
 int8_research::init();Stream stream;
 Buffer a(Count*2),b(Count*2),ad(Count*8),bd(Count*8),d(Count*4),w(int8_research::WorkBytes),trace(128*60*6*8);
 fill<<<512,256,0,stream.s>>>(a.as<__nv_bfloat16>(),b.as<__nv_bfloat16>(),ad.as<double>(),bd.as<double>(),0,1901901);
 auto work=int8_research::Work(w.p);
 int8_research::preprocess<cute::bfloat16_t,true>(a.as<cute::bfloat16_t>(),b.as<cute::bfloat16_t>(),work,stream.s);
 CUtensorMap ha,hb,qa,qb;
 int8_research::encode(ha,work.ha,megafmm::DType::BF16,true,false);int8_research::encode(hb,work.hb,megafmm::DType::BF16,false,false);
 int8_research::encode(qa,work.qa,megafmm::DType::BF16,true,true);int8_research::encode(qb,work.qb,megafmm::DType::BF16,false,true);
 auto launch=[&]{int8_research::corrected<cute::bfloat16_t><<<dim3(16,8),256,sizeof(int8_research::Shared),stream.s>>>(ha,hb,qa,qb,work.ea,work.eb,d.as<float>(),trace.as<uint64_t>());};
 for(int j=0;j<20;++j)launch();
 cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));check(cudaEventRecord(begin,stream.s));
 for(int j=0;j<100;++j)launch();check(cudaEventRecord(end,stream.s));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,begin,end));
 printf("TRACE diagnostic_core_only_us=%.9f excludes_preprocess=1\n",ms*10);
 Blas blas(stream.s);Buffer ref(Count*8),base(Count*4),scratch(128*sizeof(Metric));
 double one=1,zero=0;check(cublasDgemm(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&one,bd.as<double>(),N,ad.as<double>(),N,&zero,ref.as<double>(),N));
 float fone=1,fzero=0;check(cublasGemmEx(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&fone,b.p,CUDA_R_16BF,N,a.p,CUDA_R_16BF,N,&fzero,base.p,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
 auto dm=metric(base.as<float>(),ref.as<double>(),scratch,stream.s),cm=metric(d.as<float>(),ref.as<double>(),scratch,stream.s);
 double gate=3*ef(dm)+1e-3;printf("TRACE_CHECK ef=%.12g gate=%.12g nonfinite=%llu pass=%d\n",ef(cm),gate,cm.nonfinite,cm.nonfinite==0&&ef(cm)<=gate);
 if(cm.nonfinite||ef(cm)>gate)return 2;
 std::vector<uint64_t> host(128*60*6);check(cudaMemcpy(host.data(),trace.p,host.size()*8,cudaMemcpyDeviceToHost));
 std::ofstream out(argv[1]);out<<"cta,step,k,t0,t1,t2,t3,t4,t5\n";
 for(int cta=0;cta<128;++cta)for(int seq=0;seq<60;++seq){
  out<<cta<<','<<seq/4<<','<<seq%4;for(int j=0;j<6;++j)out<<','<<host[(cta*60+seq)*6+j];out<<'\n';
 }
 check(cudaEventDestroy(begin));check(cudaEventDestroy(end));return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

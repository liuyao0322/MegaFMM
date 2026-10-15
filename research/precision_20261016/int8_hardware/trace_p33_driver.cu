#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#include "p33/corrected.cuh"
#define int8_research phase_research
#include "trace_p33/corrected.cuh"
#undef int8_research
#include <cstring>
#include <functional>

static double timing(const std::function<void()>& fn,cudaStream_t stream){
 cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
 check(cudaEventRecord(begin,stream));for(int i=0;i<100;++i)fn();check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
 float ms=0;check(cudaEventElapsedTime(&ms,begin,end));check(cudaEventDestroy(begin));check(cudaEventDestroy(end));return double(ms)*10;
}
template<class T>int investigate(megafmm::DType dtype){
 Stream stream;Blas blas(stream.s);
 Buffer a(Count*2),b(Count*2),ad(Count*8),bd(Count*8),ref(Count*8),base(Count*4),trace(Count*4),scratch(128*sizeof(Metric));
 Buffer wb(int8_research::WorkBytes),wt(phase_research::WorkBytes);
 const char* dt=dtype==megafmm::DType::BF16?"bf16":"fp16";
 std::function<void()> run[2]={
  [&]{int8_research::run(dtype,a.p,b.p,base.as<float>(),wb.p,stream.s);},
  [&]{phase_research::run(dtype,a.p,b.p,trace.as<float>(),wt.p,stream.s);}};
 const int patterns[5]={0,0,0,2,3};const uint64_t seeds[5]={1908191,39007,39011,39013,39019};
 int failures=0;
 for(int sample=0;sample<5;++sample){
  fill<<<512,256,0,stream.s>>>(a.as<T>(),b.as<T>(),ad.as<double>(),bd.as<double>(),patterns[sample],seeds[sample]);
  double one=1,zero=0;
  check(cublasDgemm(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&one,bd.as<double>(),N,ad.as<double>(),N,&zero,ref.as<double>(),N));
  for(int i=0;i<20;++i){run[0]();run[1]();}check(cudaStreamSynchronize(stream.s));
  std::vector<uint64_t> timestamps(288*8*6);
  check(cudaMemcpy(timestamps.data(),phase_research::Work(wt.p).partial,timestamps.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost));
  std::vector<uint32_t> first(Count),second(Count);
  check(cudaMemcpy(first.data(),base.p,Count*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(second.data(),trace.p,Count*4,cudaMemcpyDeviceToHost));
  int different=0;for(int i=0;i<Count;++i)different+=first[i]!=second[i];
  auto bm=metric(base.as<float>(),ref.as<double>(),scratch,stream.s),tm=metric(trace.as<float>(),ref.as<double>(),scratch,stream.s);
  double floor=dtype==megafmm::DType::BF16?1e-3:1e-4;
  bool pass=different==0&&bm.nonfinite==0&&tm.nonfinite==0&&ef(bm)<=floor&&ef(tm)<=floor;failures+=!pass;
  printf("TRACE_CHECK dtype=%s sample=%d pattern=%d seed=%llu different=%d base_ef=%.12g trace_ef=%.12g floor=%.12g pass=%d\n",dt,sample,patterns[sample],(unsigned long long)seeds[sample],different,ef(bm),ef(tm),floor,pass);
  for(int block=0;block<288;++block)for(int warp=0;warp<8;++warp){
   auto p=timestamps.data()+(block*8+warp)*6;
   printf("PHASE dtype=%s sample=%d block=%d warp=%d t0=%llu t1=%llu t2=%llu t3=%llu t4=%llu t5=%llu\n",dt,sample,block,warp,(unsigned long long)p[0],(unsigned long long)p[1],(unsigned long long)p[2],(unsigned long long)p[3],(unsigned long long)p[4],(unsigned long long)p[5]);
  }
  // Calibrate complete-call perturbation on the exact ordinary timing input.
  if(sample==0)for(int group=0;group<5;++group)for(int arm=0;arm<4;++arm){
   int method=(arm==1||arm==2)?1:0;
   printf("TRACE_TIME dtype=%s group=%d arm=%d method=%d calls=100 us=%.9f\n",dt,group,arm,method,timing(run[method],stream.s));
  }
  fflush(stdout);
 }
 return failures;
}
int main(){try{
 int8_research::init();phase_research::init();
 return investigate<__nv_bfloat16>(megafmm::DType::BF16)+investigate<__half>(megafmm::DType::FP16)?2:0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}

// Use the repository's unchanged FP64 reference, inputs and metric utilities.
#define main original_sw_test_main
#include "../../tests/test_sw1024.cu"
#undef main
#ifndef DIRECT_HEADER
#define DIRECT_HEADER "direct.cuh"
#endif
#include DIRECT_HEADER
#include <fstream>
#include <functional>

double timing(const std::function<void()>& launch,cudaStream_t stream){
 cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
 check(cudaEventRecord(start,stream));for(int i=0;i<100;++i)launch();
 check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));float ms;
 check(cudaEventElapsedTime(&ms,start,end));check(cudaEventDestroy(start));check(cudaEventDestroy(end));
 return ms*10; // microseconds / invocation
}
template<class T> int compare(megafmm::DType dtype,bool graph_mode){
 Stream stream;Blas blas(stream.s);Buffer a(Count*2),b(Count*2),a64(Count*8),b64(Count*8),ref(Count*8);
 Buffer direct(Count*4),base(Count*4),swout(Count*4),scratch(128*sizeof(Metric));
 megafmm::Problem p;p.m=p.n=p.k=N;p.dtype=dtype;
 megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Dense;
 auto dense=megafmm::create_plan(p,policy);policy.algorithm=megafmm::Algorithm::SWPipeline;
 auto sw=megafmm::create_plan(p,policy);Buffer workspace(std::max(dense->workspace_bytes(),sw->workspace_bytes()));
 fill<<<512,256,0,stream.s>>>(a.as<T>(),b.as<T>(),a64.as<double>(),b64.as<double>(),0,1234);
 megafmm::tune(*dense,a.p,b.p,nullptr,base.as<float>(),workspace.p,stream.s);
 megafmm::tune(*sw,a.p,b.p,nullptr,swout.as<float>(),workspace.p,stream.s);
 std::function<void()> runs[3]={
  [&]{megafmm::run(*dense,a.p,b.p,nullptr,base.as<float>(),workspace.p,stream.s);},
  [&]{megafmm::run(*sw,a.p,b.p,nullptr,swout.as<float>(),workspace.p,stream.s);},
  [&]{precision_research::run(dtype,a.p,b.p,direct.as<float>(),stream.s);}};
 Graph graphs[3];
 if(graph_mode)for(int j=0;j<3;++j){
  check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));runs[j]();
  check(cudaStreamEndCapture(stream.s,&graphs[j].g));check(cudaGraphInstantiate(&graphs[j].x,graphs[j].g,nullptr,nullptr,0));
  runs[j]=[&,j]{check(cudaGraphLaunch(graphs[j].x,stream.s));};
 }
 const char* name=dtype==megafmm::DType::BF16?"bf16":"fp16";
 int fails=0;
 // Original six distributions, three fresh trials, including same Graph reuse.
 for(int pat=0;pat<6;++pat)for(int trial=0;trial<3;++trial){
  fill<<<512,256,0,stream.s>>>(a.as<T>(),b.as<T>(),a64.as<double>(),b64.as<double>(),pat,1234567ULL+trial*0x9e3779b97f4a7c15ULL+pat*101);
  double one=1,zero=0;check(cublasDgemm(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&one,b64.as<double>(),N,a64.as<double>(),N,&zero,ref.as<double>(),N));
  runs[0]();runs[1]();runs[2]();
  auto dm=metric(base.as<float>(),ref.as<double>(),scratch,stream.s);
  auto sm=metric(swout.as<float>(),ref.as<double>(),scratch,stream.s);
  auto cm=metric(direct.as<float>(),ref.as<double>(),scratch,stream.s);
  double gate=3*ef(dm)+(dtype==megafmm::DType::BF16?1e-3:1e-4);
  bool pass=cm.nonfinite==0 && ef(cm)<=gate;fails+=!pass;
  printf("CHECK %s graph=%d pattern=%d trial=%d dense_ef=%.12g sw_ef=%.12g direct_ef=%.12g gate=%.12g pass=%d\n",name,graph_mode,pat,trial,ef(dm),ef(sm),ef(cm),gate,pass);fflush(stdout);
 }
 // Time fresh bounded inputs. All complete-call packing/descriptor work is inside run.
 fill<<<512,256,0,stream.s>>>(a.as<T>(),b.as<T>(),a64.as<double>(),b64.as<double>(),0,8191);
 for(int i=0;i<20;++i)for(auto& run:runs)run();check(cudaStreamSynchronize(stream.s));
 for(int group=0;group<5;++group){
  int order[4][3]={{0,1,2},{2,1,0},{2,1,0},{0,1,2}};
  for(int arm=0;arm<4;++arm)for(int j:order[arm]){
   double us=timing(runs[j],stream.s);
   printf("TIME %s graph=%d group=%d arm=%d method=%d us=%.9f\n",name,graph_mode,group,arm,j,us);fflush(stdout);
  }
 }
 return fails;
}
int main(int argc,char**argv){try{
 precision_research::init();bool graph=argc>1 && std::string(argv[1])=="--graph";
 int fails=compare<__nv_bfloat16>(megafmm::DType::BF16,graph)+compare<__half>(megafmm::DType::FP16,graph);
 return fails?2:0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}

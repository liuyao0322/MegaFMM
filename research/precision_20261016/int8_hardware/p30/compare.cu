// Independent research acceptance against unchanged original-input FP64 helpers.
#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#include "corrected.cuh"
#include "../direct_b128_w1.cuh"
#include <functional>

// Same extended input definitions as the already accepted dense comparator.
template<class T> __global__ void extended_fill(T* a,T* b,double* ad,double* bd,int pattern,int seed){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Count;i+=gridDim.x*blockDim.x){
  int r=i/N,c=i%N,ai=pattern==2?r*N+c%512:i,bi=pattern==2?(r%512)*N+c:i;
  float x=float(hash(uint64_t(ai)^seed)>>40)*(2.f/16777216.f)-1;
  float y=float(hash(uint64_t(bi)^seed^0x632be59bd9b4e019ULL)>>40)*(2.f/16777216.f)-1;
  if(pattern==0){constexpr bool bf=std::is_same_v<T,__nv_bfloat16>;x=ldexpf(x,bf?60:10);y=ldexpf(y,bf?-60:-10);}
  if(pattern==1){x=ldexpf(x,r<512?6:-6);y=ldexpf(y,c<512?-6:6);}
  if(pattern==2&&c>=512)x=-x+1.f/256;
  if(pattern==3){x=ldexpf(x,c<512?0:-12);y=ldexpf(y,r<512?-12:0);}
  if(pattern==4){x=ldexpf(x,int(hash(i+seed)%17)-8);y=ldexpf(y,int(hash(i+seed+1)%17)-8);}
  if(pattern==5){x=r==c?1:0;y*=2;}
  if(pattern==6){bool af=((c%2)==1)^(c>=512),bf=((r%2)==1)^(r>=512);x=ldexpf(x,af?-12:0);y=ldexpf(y,bf?0:-12);}
  T xx=rounded<T>(x),yy=rounded<T>(y);a[i]=xx;b[i]=yy;ad[i]=float(xx);bd[i]=float(yy);
 }
}
double time_calls(const std::function<void()>& fn,cudaStream_t stream){
 cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));
 check(cudaEventRecord(a,stream));for(int j=0;j<100;++j)fn();check(cudaEventRecord(b,stream));check(cudaEventSynchronize(b));
 float ms;check(cudaEventElapsedTime(&ms,a,b));check(cudaEventDestroy(a));check(cudaEventDestroy(b));return ms*10;
}

template<class T> int compare(megafmm::DType dtype,bool balance,bool quick,bool graph_mode){
 Stream stream;Blas blas(stream.s);Buffer a(Count*2),b(Count*2),ad(Count*8),bd(Count*8),ref(Count*8);
 Buffer denseout(Count*4),swout(Count*4),directout(Count*4),out(Count*4),scratch(128*sizeof(Metric));
 Buffer workspace(int8_research::WorkBytes),reference_inside(1024*sizeof(int));
 megafmm::Problem p;p.m=p.n=p.k=N;p.dtype=dtype;megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Dense;
 auto dense=megafmm::create_plan(p,policy);policy.algorithm=megafmm::Algorithm::SWPipeline;auto sw=megafmm::create_plan(p,policy);
 Buffer basework(std::max(dense->workspace_bytes(),sw->workspace_bytes()));
 fill<<<512,256,0,stream.s>>>(a.as<T>(),b.as<T>(),ad.as<double>(),bd.as<double>(),0,1901);
 megafmm::tune(*dense,a.p,b.p,nullptr,denseout.as<float>(),basework.p,stream.s);
 megafmm::tune(*sw,a.p,b.p,nullptr,swout.as<float>(),basework.p,stream.s);
 std::function<void()> runs[4]={
  [&]{megafmm::run(*dense,a.p,b.p,nullptr,denseout.as<float>(),basework.p,stream.s);},
  [&]{megafmm::run(*sw,a.p,b.p,nullptr,swout.as<float>(),basework.p,stream.s);},
  [&]{precision_research::run(dtype,a.p,b.p,directout.as<float>(),stream.s);},
  [&]{int8_research::run(dtype,a.p,b.p,out.as<float>(),workspace.p,stream.s,balance);}};
 Graph graphs[4];if(graph_mode)for(int j=0;j<4;++j){
  check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));runs[j]();check(cudaStreamEndCapture(stream.s,&graphs[j].g));
  check(cudaGraphInstantiate(&graphs[j].x,graphs[j].g,nullptr,nullptr,0));runs[j]=[&,j]{check(cudaGraphLaunch(graphs[j].x,stream.s));};
 }
 const char* dt=dtype==megafmm::DType::BF16?"bf16":"fp16";int failures=0;
 for(int pat=0;pat<(quick?1:13);++pat)for(int trial=0;trial<(quick?1:3);++trial){
  if(pat<6)fill<<<512,256,0,stream.s>>>(a.as<T>(),b.as<T>(),ad.as<double>(),bd.as<double>(),pat,1901901ULL+trial*271+pat*29);
  else extended_fill<<<512,256,0,stream.s>>>(a.as<T>(),b.as<T>(),ad.as<double>(),bd.as<double>(),pat-6,1901901+trial*271+pat*29);
  double one=1,zero=0;check(cublasDgemm(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&one,bd.as<double>(),N,ad.as<double>(),N,&zero,ref.as<double>(),N));
  runs[0]();runs[3]();
  int inside_errors=0;
  if(balance){using CT=std::conditional_t<std::is_same_v<T,__half>,cute::half_t,cute::bfloat16_t>;
   int8_research::inside_scale<CT><<<1024,256,0,stream.s>>>(static_cast<const CT*>(a.p),static_cast<const CT*>(b.p),reference_inside.as<int>());
   std::vector<int> got(1024),expected(1024);auto iw=int8_research::Work(workspace.p);
   check(cudaMemcpyAsync(got.data(),iw.inside,1024*sizeof(int),cudaMemcpyDeviceToHost,stream.s));
   check(cudaMemcpyAsync(expected.data(),reference_inside.p,1024*sizeof(int),cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
   for(int k=0;k<1024;++k)inside_errors+=got[k]!=expected[k];
   printf("INSIDE_EQUAL dtype=%s pattern=%d trial=%d elements=1024 errors=%d\n",dt,pat,trial,inside_errors);
  }
  Metric dm=metric(denseout.as<float>(),ref.as<double>(),scratch,stream.s),cm=metric(out.as<float>(),ref.as<double>(),scratch,stream.s);
  double gate=3*ef(dm)+(dtype==megafmm::DType::BF16?1e-3:1e-4);
  bool pass=cm.nonfinite==0&&dm.nonfinite==0&&ef(cm)<=gate&&inside_errors==0;failures+=!pass;
  printf("CHECK dtype=%s balance=%d graph=%d pattern=%d trial=%d dense_ef=%.12g corrected_ef=%.12g gate=%.12g nonfinite=%llu pass=%d\n",dt,balance,graph_mode,pat,trial,ef(dm),ef(cm),gate,cm.nonfinite,pass);fflush(stdout);
 }
 fill<<<512,256,0,stream.s>>>(a.as<T>(),b.as<T>(),ad.as<double>(),bd.as<double>(),0,1908191);
 for(int j=0;j<20;++j)for(auto& fn:runs)fn();check(cudaStreamSynchronize(stream.s));
 for(int group=0;group<5;++group){
  int order[4][4]={{0,1,2,3},{3,2,1,0},{3,2,1,0},{0,1,2,3}};
  for(int arm=0;arm<4;++arm)for(int j:order[arm]){
   double us=time_calls(runs[j],stream.s);
   printf("TIME dtype=%s balance=%d graph=%d group=%d arm=%d method=%d us=%.9f\n",dt,balance,graph_mode,group,arm,j,us);fflush(stdout);
  }
 }
 // Diagnostic component timing: never substitute for the complete-call rows.
 auto w=int8_research::Work(workspace.p);
 std::function<void()> prep=[&]{
  using CT=std::conditional_t<std::is_same_v<T,__half>,cute::half_t,cute::bfloat16_t>;
  if(balance)int8_research::preprocess<CT,true>(static_cast<const CT*>(a.p),static_cast<const CT*>(b.p),w,stream.s);
  else int8_research::preprocess<CT,false>(static_cast<const CT*>(a.p),static_cast<const CT*>(b.p),w,stream.s);
 };
 for(int j=0;j<20;++j)prep();
 printf("COMPONENT dtype=%s balance=%d preprocess_us=%.9f diagnostic_only=1\n",dt,balance,time_calls(prep,stream.s));
 using CT=std::conditional_t<std::is_same_v<T,__half>,cute::half_t,cute::bfloat16_t>;
 std::function<void()> components[4]={
  [&]{int8_research::inside_partial<CT><<<dim3(8,32),256,0,stream.s>>>(static_cast<const CT*>(a.p),static_cast<const CT*>(b.p),w.partial);},
  [&]{int8_research::inside_finalize<<<128,256,0,stream.s>>>(w.partial,w.inside);},
  [&]{if(balance)int8_research::pack<CT,true,true><<<3*32,256,sizeof(int8_research::PairPackShared<true>),stream.s>>>(static_cast<const CT*>(a.p),static_cast<CT*>(w.ha),w.qa,w.ea,w.inside);else int8_research::pack<CT,true,false><<<3*32,256,sizeof(int8_research::PairPackShared<true>),stream.s>>>(static_cast<const CT*>(a.p),static_cast<CT*>(w.ha),w.qa,w.ea,w.inside);},
  [&]{if(balance)int8_research::pack<CT,false,true><<<3*64,256,sizeof(int8_research::PairPackShared<false>),stream.s>>>(static_cast<const CT*>(b.p),static_cast<CT*>(w.hb),w.qb,w.eb,w.inside);else int8_research::pack<CT,false,false><<<3*64,256,sizeof(int8_research::PairPackShared<false>),stream.s>>>(static_cast<const CT*>(b.p),static_cast<CT*>(w.hb),w.qb,w.eb,w.inside);}
 };
 for(int j=0;j<4;++j){for(int warm=0;warm<20;++warm)components[j]();printf("PREP_STAGE dtype=%s balance=%d stage=%d us=%.9f diagnostic_only=1\n",dt,balance,j,time_calls(components[j],stream.s));}
 printf("SUMMARY dtype=%s balance=%d checks=%d failures=%d\n",dt,balance,quick?1:39,failures);fflush(stdout);
 return failures;
}
int main(int argc,char** argv){try{
 bool balance=true,quick=false,graph=false;
 for(int i=1;i<argc;++i){std::string arg=argv[i];if(arg=="--no-balance")balance=false;else if(arg=="--quick")quick=true;else if(arg=="--graph")graph=true;else throw std::invalid_argument("unknown argument "+arg);}
 int8_research::init();precision_research::init();
 return compare<__nv_bfloat16>(megafmm::DType::BF16,balance,quick,graph)+compare<__half>(megafmm::DType::FP16,balance,quick,graph)?2:0;
}catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

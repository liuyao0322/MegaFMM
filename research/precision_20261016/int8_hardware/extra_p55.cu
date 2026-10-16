// Own fresh matrix gates, one stream/workspace; not a dual-stream certificate.
#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#include "p55/corrected.cuh"
#include <cfenv>
#include <cstring>
using namespace int8_research;
uint64_t host_hash55(uint64_t v){v+=0x9e3779b97f4a7c15ULL;v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}
float add55(float a,float b){volatile float x=a+b;return x;}
float sub55(float a,float b){volatile float x=a-b;return x;}
uint32_t bits55(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
template<class T> void host_inputs(std::vector<T>& a,std::vector<T>& b,int pattern,int seed){
 for(int i=0;i<Count;++i){
  int r=i/1024,c=i%1024;
  bool cancel=pattern==3||pattern==9||pattern==11||pattern==13;
  int ai=cancel?r*1024+c%512:i,bi=cancel?(r%512)*1024+c:i;
  float x=float(host_hash55(uint64_t(ai)^uint64_t(seed))>>40)*(2.f/16777216.f)-1.f;
  float y=float(host_hash55(uint64_t(bi)^uint64_t(seed)^0x632be59bd9b4e019ULL)>>40)*(2.f/16777216.f)-1.f;
  if(cancel&&c>=512)x=add55(-x,1.f/256);
  if(pattern==1){x=fabsf(x);y=fabsf(y);}
  if(pattern==2){x=std::ldexp(x,int(host_hash55(i+seed)%9)-4);y=std::ldexp(y,int(host_hash55(i+seed+1)%9)-4);}
  if(pattern==4){x=std::ldexp(x,r<512?6:-6);y=std::ldexp(y,c<512?-6:6);}
  if(pattern==5){constexpr bool bf=std::is_same_v<T,cute::bfloat16_t>;x=std::ldexp(x,bf?60:10);y=std::ldexp(y,bf?-60:-10);}
  if(pattern==6){x=std::ldexp(x,c<512?0:-12);y=std::ldexp(y,r<512?-12:0);}
  if(pattern==7){bool af=((c%2)==1)^(c>=512),bf=((r%2)==1)^(r>=512);x=std::ldexp(x,af?-12:0);y=std::ldexp(y,bf?0:-12);}
  if(pattern==8||pattern==9){int p[4]={0,4,-4,2};x=std::ldexp(x,p[(c%512)/128]);y=std::ldexp(y,p[(r%512)/128]);}
  if(pattern==10||pattern==11){constexpr bool bf=std::is_same_v<T,cute::bfloat16_t>;x=std::ldexp(x,bf?-60:-10);y=std::ldexp(y,bf?-60:-10);}
  if(pattern==12||pattern==13){int p[4]={-4,0,4,-2};x=std::ldexp(x,p[(r/8)%4]);y=std::ldexp(y,p[(c/8)%4]);}
  // Edge inputs keep transforms finite, exercise high/subnormal native bins.
  if(pattern==14){constexpr bool bf=std::is_same_v<T,cute::bfloat16_t>;int e=bf?-125:-14;x=std::ldexp(x,e);y=std::ldexp(y,e);}
  if(pattern==15){constexpr bool bf=std::is_same_v<T,cute::bfloat16_t>;x=std::ldexp(x,bf?119:11);y=std::ldexp(y,bf?-119:-11);}
  a[i]=T(x);b[i]=T(y);
 }
}
template<class T> std::vector<int> inside_oracle55(const std::vector<T>& a,const std::vector<T>& b){
 std::vector<int> e(1024);
 for(int k=0;k<1024;++k){
  float am=0,bm=0;for(int r=0;r<1024;++r){am=std::max(am,fabsf(float(a[r*1024+k])));bm=std::max(bm,fabsf(float(b[k*1024+r])));}
  // Independent host traversal and double logarithms, RN ties-even.
  e[k]=(am>0&&bm>0)?int(std::nearbyint(.5*(std::log2(double(am))-std::log2(double(bm))))):0;
 }
 return e;
}

template<class T> int extra55(megafmm::DType dt,const char* dtype){
 Stream stream;Blas blas(stream.s);Buffer a(Count*2),b(Count*2),ad(Count*8),bd(Count*8),ref(Count*8),denseout(Count*4),out((Count+256)*4),workspace(WorkBytes+512),scratch(128*sizeof(Metric));
 std::vector<T> ah(Count),bh(Count);std::vector<double> da(Count),db(Count);Work w((char*)workspace.p+256);
 megafmm::Problem problem;problem.m=problem.n=problem.k=1024;problem.dtype=dt;megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Dense;
 auto plan=megafmm::create_plan(problem,policy);Buffer densework(plan->workspace_bytes());
 host_inputs(ah,bh,0,127163);check(cudaMemcpyAsync(a.p,ah.data(),Count*2,cudaMemcpyHostToDevice,stream.s));check(cudaMemcpyAsync(b.p,bh.data(),Count*2,cudaMemcpyHostToDevice,stream.s));
 megafmm::tune(*plan,a.p,b.p,nullptr,denseout.as<float>(),densework.p,stream.s);
 int failures=0;
 for(int pattern=0;pattern<14;++pattern)for(int trial=0;trial<3;++trial){
  int seed=127163+487*trial+127*pattern+(dt==megafmm::DType::FP16?263:0);
  host_inputs(ah,bh,pattern,seed);auto expected_inside=inside_oracle55(ah,bh);
  for(int i=0;i<Count;++i){da[i]=float(ah[i]);db[i]=float(bh[i]);}
  check(cudaMemcpyAsync(a.p,ah.data(),Count*2,cudaMemcpyHostToDevice,stream.s));check(cudaMemcpyAsync(b.p,bh.data(),Count*2,cudaMemcpyHostToDevice,stream.s));
  check(cudaMemcpyAsync(ad.p,da.data(),Count*8,cudaMemcpyHostToDevice,stream.s));check(cudaMemcpyAsync(bd.p,db.data(),Count*8,cudaMemcpyHostToDevice,stream.s));
  double one=1,zero=0;check(cublasDgemm(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,1024,1024,1024,&one,bd.as<double>(),1024,ad.as<double>(),1024,&zero,ref.as<double>(),1024));
  check(cudaMemsetAsync(workspace.p,0xa5,WorkBytes+512,stream.s));
  std::vector<uint32_t> poison(Count+256,0x4e123456u);check(cudaMemcpyAsync(out.p,poison.data(),poison.size()*4,cudaMemcpyHostToDevice,stream.s));
  check(cudaMemsetAsync(out.as<float>()+128,0xff,Count*4,stream.s));
  megafmm::run(*plan,a.p,b.p,nullptr,denseout.as<float>(),densework.p,stream.s);
  int8_research::run(dt,a.p,b.p,out.as<float>()+128,(char*)workspace.p+256,stream.s,true);
  Metric dm=metric(denseout.as<float>(),ref.as<double>(),scratch,stream.s),cm=metric(out.as<float>()+128,ref.as<double>(),scratch,stream.s);
  uint8_t guards[512];uint32_t og[256];std::vector<int> actual_inside(1024);
  check(cudaMemcpyAsync(guards,workspace.p,256,cudaMemcpyDeviceToHost,stream.s));check(cudaMemcpyAsync(guards+256,(char*)workspace.p+256+WorkBytes,256,cudaMemcpyDeviceToHost,stream.s));
  check(cudaMemcpyAsync(og,out.p,128*4,cudaMemcpyDeviceToHost,stream.s));check(cudaMemcpyAsync(og+128,out.as<float>()+128+Count,128*4,cudaMemcpyDeviceToHost,stream.s));
  check(cudaMemcpyAsync(actual_inside.data(),w.inside,4096,cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
  unsigned wg=0,outputg=0,inside_errors=0;for(auto x:guards)wg+=x!=0xa5;for(auto x:og)outputg+=x!=0x4e123456u;
  for(int i=0;i<1024;++i)inside_errors+=actual_inside[i]!=expected_inside[i];
  double floor=dt==megafmm::DType::BF16?1e-3:1e-4,gate=3*ef(dm)+floor;
  bool valid=cm.nonfinite==0&&dm.nonfinite==0&&wg==0&&outputg==0&&inside_errors==0;
  bool pass=valid&&ef(cm)<=gate,fixed=valid&&ef(cm)<=floor;failures+=!pass;
  printf("EXTRA_CHECK dtype=%s pattern=%d trial=%d seed=%d dense_ef=%.12g corrected_ef=%.12g gate=%.12g fixed_floor=%.12g nonfinite=%llu inside_errors=%u workspace_guard_errors=%u output_guard_errors=%u fixed_pass=%d pass=%d\n",dtype,pattern,trial,seed,ef(dm),ef(cm),gate,floor,cm.nonfinite,inside_errors,wg,outputg,fixed,pass);fflush(stdout);
 }
 return failures;
}
int main(){try{require(std::fesetround(FE_TONEAREST)==0,"host RN");int8_research::init();return extra55<cute::bfloat16_t>(megafmm::DType::BF16,"bf16")+extra55<cute::half_t>(megafmm::DType::FP16,"fp16")?2:0;}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}

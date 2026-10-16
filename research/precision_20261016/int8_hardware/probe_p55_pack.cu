// Own pre-MMA original-rounded-input CPU packing oracle and scalar checks.
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
struct HostPack55 {std::vector<uint16_t> high;std::vector<int8_t> quant;std::vector<int> exponent;uint64_t saturation=0;};
template<class T,bool SideA> HostPack55 host_pack55(const std::vector<T>& x,const std::vector<int>& inside){
 HostPack55 o;o.high.resize(7*Plane);o.quant.resize(9*Plane);o.exponent.resize(9*64);
 std::vector<float> values(4*Plane),raw(Plane);float peaks[5][64]={};
 for(int r=0;r<512;++r)for(int c=0;c<512;++c){
  int i=r*512+c,at=r*1024+c,e0=SideA?-inside[c]:inside[r],e1=SideA?-inside[c+512]:inside[r+512];
  float a=std::ldexp(float(x[at]),e0),b=std::ldexp(float(x[at+512]),SideA?e1:e0);
  float cc=std::ldexp(float(x[at+512*1024]),SideA?e0:e1),d=std::ldexp(float(x[at+512*1024+512]),e1);
  float t[4];
  if constexpr(SideA){t[0]=add55(cc,d);t[1]=sub55(t[0],a);t[2]=sub55(a,cc);t[3]=sub55(b,t[1]);}
  else{t[0]=sub55(b,a);t[1]=sub55(d,t[0]);t[2]=sub55(d,b);t[3]=sub55(t[1],cc);}
  float native[7]={a,SideA?b:cc,SideA?t[3]:d,SideA?d:t[3],t[0],t[1],t[2]};
  for(int p=0;p<7;++p){T h=T(native[p]);std::memcpy(&o.high[p*Plane+i],&h,2);}
  int g=SideA?r/8:c/8;
  for(int p=0;p<4;++p){values[p*Plane+i]=t[p];peaks[p][g]=std::max(peaks[p][g],fabsf(float(T(t[p]))));}
  raw[i]=d;peaks[4][g]=std::max(peaks[4][g],fabsf(float(T(d))));
 }
 constexpr int Bits=std::is_same_v<T,cute::half_t> ? 11:8;
 for(int p=0;p<5;++p)for(int g=0;g<64;++g){
  // Independent mathematical exponent; native high maxima are exactly represented.
  int e=peaks[p][g]>0?int(std::ceil(std::log2(double(peaks[p][g])/127.))):0;
  if(p<4){o.exponent[p*64+g]=e;o.exponent[(p+4)*64+g]=e-Bits;}else o.exponent[8*64+g]=e;
 }
 for(int r=0;r<512;++r)for(int c=0;c<512;++c){
  int i=r*512+c,g=SideA?r/8:c/8,at=SideA?i:c*512+r;
  for(int p=0;p<5;++p){
   int qp=p<4?p:8;float v=p<4?values[p*Plane+i]:raw[i],h=float(T(v));
   int qh=int(std::nearbyint(std::ldexp(h,-o.exponent[qp*64+g])));
   o.saturation+=std::abs(qh)>127;o.quant[qp*Plane+at]=int8_t(std::max(-127,std::min(127,qh)));
   if(p<4){float low=sub55(v,h);int ql=int(std::nearbyint(std::ldexp(low,-o.exponent[(p+4)*64+g])));
    o.saturation+=std::abs(ql)>127;o.quant[(p+4)*Plane+at]=int8_t(std::max(-127,std::min(127,ql)));}
  }
 }
 return o;
}
template<class T> int packing55(const char* dtype){
 Stream stream;Buffer a(Count*2),b(Count*2),workspace(WorkBytes+512);Work w(static_cast<char*>(workspace.p)+256);
 std::vector<T> ah(Count),bh(Count);std::vector<uint16_t> ha(7*Plane),hb(7*Plane);
 std::vector<int8_t> qa(9*Plane),qb(9*Plane);std::vector<int> ea(9*64),eb(9*64),inside(1024);
 int failures=0;
 for(int pattern=0;pattern<16;++pattern)for(int trial=0;trial<3;++trial){
  int seed=125153+467*trial+113*pattern+(std::is_same_v<T,cute::half_t> ? 251:0);
  host_inputs(ah,bh,pattern,seed);auto ex=inside_oracle55(ah,bh);auto ar=host_pack55<T,true>(ah,ex),br=host_pack55<T,false>(bh,ex);
  check(cudaMemcpyAsync(a.p,ah.data(),Count*2,cudaMemcpyHostToDevice,stream.s));check(cudaMemcpyAsync(b.p,bh.data(),Count*2,cudaMemcpyHostToDevice,stream.s));
  check(cudaMemsetAsync(workspace.p,0xa5,WorkBytes+512,stream.s));
  preprocess<T,true>(a.as<T>(),b.as<T>(),w,stream.s);check(cudaGetLastError());
  check(cudaMemcpyAsync(ha.data(),w.ha,HighBytes,cudaMemcpyDeviceToHost,stream.s));check(cudaMemcpyAsync(hb.data(),w.hb,HighBytes,cudaMemcpyDeviceToHost,stream.s));
  check(cudaMemcpyAsync(qa.data(),w.qa,QuantBytes,cudaMemcpyDeviceToHost,stream.s));check(cudaMemcpyAsync(qb.data(),w.qb,QuantBytes,cudaMemcpyDeviceToHost,stream.s));
  check(cudaMemcpyAsync(ea.data(),w.ea,ea.size()*4,cudaMemcpyDeviceToHost,stream.s));check(cudaMemcpyAsync(eb.data(),w.eb,eb.size()*4,cudaMemcpyDeviceToHost,stream.s));
  check(cudaMemcpyAsync(inside.data(),w.inside,4096,cudaMemcpyDeviceToHost,stream.s));uint8_t guards[512];
  check(cudaMemcpyAsync(guards,workspace.p,256,cudaMemcpyDeviceToHost,stream.s));check(cudaMemcpyAsync(guards+256,(char*)workspace.p+256+WorkBytes,256,cudaMemcpyDeviceToHost,stream.s));
  check(cudaStreamSynchronize(stream.s));
  uint64_t he=0,qe=0,ee=0,ie=0,ge=0;
  for(int i=0;i<7*Plane;++i){he+=ha[i]!=ar.high[i];he+=hb[i]!=br.high[i];}
  for(int i=0;i<9*Plane;++i){qe+=qa[i]!=ar.quant[i];qe+=qb[i]!=br.quant[i];}
  for(int i=0;i<9*64;++i){ee+=ea[i]!=ar.exponent[i];ee+=eb[i]!=br.exponent[i];}
  for(int i=0;i<1024;++i)ie+=inside[i]!=ex[i];for(auto x:guards)ge+=x!=0xa5;
  uint64_t sat=ar.saturation+br.saturation;bool pass=he==0&&qe==0&&ee==0&&ie==0&&ge==0&&sat==0;failures+=!pass;
  printf("FINE_PACK dtype=%s pattern=%d trial=%d seed=%d native_values=3670016 quant_bytes=4718592 metadata=1152 inside=1024 native_errors=%llu quant_errors=%llu exponent_errors=%llu inside_errors=%llu guard_errors=%llu reference_saturation=%llu pass=%d\n",dtype,pattern,trial,seed,(unsigned long long)he,(unsigned long long)qe,(unsigned long long)ee,(unsigned long long)ie,(unsigned long long)ge,(unsigned long long)sat,pass);fflush(stdout);
 }
 return failures;
}
__global__ void scalar_scale55(const float* x,const int* e,float* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=scale_pow2(x[i],e[i]);}
__global__ void scalar_exponent55(float* peaks,int* result,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)result[i]=native_high_exponent(peaks[i]);}
int scalar55(){
 Stream stream;std::vector<float> x;std::vector<int> e;
 int ints[11]={0,1,-1,127,-127,65535,-65535,16516096,-16516096,1048576,-1048576};
 for(int power=-300;power<=300;++power)for(int v:ints){x.push_back(float(v));e.push_back(power);}
 Buffer dx(x.size()*4),de(e.size()*4),dy(x.size()*4);std::vector<float> y(x.size());
 check(cudaMemcpyAsync(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice,stream.s));check(cudaMemcpyAsync(de.p,e.data(),e.size()*4,cudaMemcpyHostToDevice,stream.s));
 scalar_scale55<<<(x.size()+255)/256,256,0,stream.s>>>(dx.as<float>(),de.as<int>(),dy.as<float>(),x.size());
 check(cudaMemcpyAsync(y.data(),dy.p,y.size()*4,cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
 unsigned errors=0;for(size_t i=0;i<x.size();++i)errors+=bits55(y[i])!=bits55(std::ldexp(x[i],e[i]));
 std::vector<float> peaks;std::vector<int> expected;
 for(int dt=0;dt<2;++dt)for(int code=0;code<(dt?32640:31744);++code){
  float p;if(dt){uint32_t u=uint32_t(code)<<16;std::memcpy(&p,&u,4);}else{uint16_t u=code;cute::half_t h;std::memcpy(&h,&u,2);p=float(h);}
  peaks.push_back(p);expected.push_back(p>0?int(std::ceil(std::log2(double(p)/127.))):0);
 }
 Buffer dp(peaks.size()*4),dr(peaks.size()*4);std::vector<int> actual(peaks.size());
 check(cudaMemcpyAsync(dp.p,peaks.data(),peaks.size()*4,cudaMemcpyHostToDevice,stream.s));
 scalar_exponent55<<<(peaks.size()+255)/256,256,0,stream.s>>>(dp.as<float>(),dr.as<int>(),peaks.size());
 check(cudaMemcpyAsync(actual.data(),dr.p,actual.size()*4,cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
 unsigned exp_errors=0;for(size_t i=0;i<peaks.size();++i)exp_errors+=expected[i]!=actual[i];
 printf("FINE_SCALAR scaling_cases=%zu scaling_bit_errors=%u native_exponent_cases=%zu exponent_errors=%u negative_expected=%.12g pass=%d\n",x.size(),errors,peaks.size(),exp_errors,std::ldexp(1048576.f,-152),errors==0&&exp_errors==0);fflush(stdout);
 return errors||exp_errors;
}
int main(){try{
 require(std::fesetround(FE_TONEAREST)==0,"host RN");int8_research::init();
 int failures=scalar55();failures+=packing55<cute::bfloat16_t>("bf16");failures+=packing55<cute::half_t>("fp16");return failures?2:0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}

// N1 pre-MMA diagnostic: accepted P47 native planes versus fused raw transforms.
#define main original_sw_test_main
#include "../../../../tests/test_sw1024.cu"
#undef main
#include "../../int8_hardware/p47/pack.cuh"
#include "prototype.cuh"
template<class T> __global__ void varied_fill(T* a,T* b,double* ad,double* bd,int pattern,int seed){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Count;i+=gridDim.x*blockDim.x){
  int r=i/N,c=i%N;
  int ai=pattern==2?r*N+c%512:i,bi=pattern==2?(r%512)*N+c:i;
  float x=float(hash(uint64_t(ai)^seed)>>40)*(2.f/16777216.f)-1;
  float y=float(hash(uint64_t(bi)^seed^0x632be59bd9b4e019ULL)>>40)*(2.f/16777216.f)-1;
  if(pattern==0){
   constexpr bool bf=std::is_same_v<T,__nv_bfloat16>;
   x=ldexpf(x,bf?60:10);y=ldexpf(y,bf?-60:-10);
  }
  if(pattern==1){x=ldexpf(x,r<512?6:-6);y=ldexpf(y,c<512?-6:6);}
  if(pattern==2){if(c>=512)x=-x+1.f/256;}
  if(pattern==3){x=ldexpf(x,c<512?0:-12);y=ldexpf(y,r<512?-12:0);}
  if(pattern==4){x=ldexpf(x,int(hash(i+seed)%17)-8);y=ldexpf(y,int(hash(i+seed+1)%17)-8);}
  if(pattern==5){x=(r==c)?1:0;y=y*2;}
  if(pattern==6){bool af=((c%2)==1)^(c>=512),bf=((r%2)==1)^(r>=512);x=ldexpf(x,af?-12:0);y=ldexpf(y,bf?0:-12);}
  // Different K128 restoration exponents force the original fallback.
  if(pattern==7){x=ldexpf(x,4*((c%512)/128)-6);y=ldexpf(y,4*((r%512)/128)-6);}
  // BF16 restoration exponents below the common-path guard; FP16 stays representable.
  if(pattern==8){constexpr bool bf=std::is_same_v<T,__nv_bfloat16>;x=ldexpf(x,bf?-60:-10);y=ldexpf(y,bf?-60:-10);}
  T xx=rounded<T>(x),yy=rounded<T>(y);
  a[i]=xx;b[i]=yy;ad[i]=float(xx);bd[i]=float(yy);
 }
}


template<class Input,class Native> int probe(const char* dtype,megafmm::DType dt){
 Stream stream;Buffer a(Count*2),b(Count*2),ad(Count*8),bd(Count*8);
 Buffer old_work(int8_research::WorkBytes),new_work(native_fusion::WorkBytes),error(2*sizeof(unsigned));
 int8_research::Work old(old_work.p);native_fusion::Work fresh(new_work.p);
 int8_research::init_cooperative_preprocess<Native>();native_fusion::init_one<Native,true>();
 int failures=0;
 for(int pattern=0;pattern<9;++pattern)for(int trial=0;trial<3;++trial){
  int seed=104003+trial*367+pattern*71;
  varied_fill<<<512,256,0,stream.s>>>(a.as<Input>(),b.as<Input>(),ad.as<double>(),bd.as<double>(),pattern,seed);
  int8_research::preprocess<Native,true>(a.as<Native>(),b.as<Native>(),old,stream.s);
  check(cudaMemsetAsync(error.p,0,2*sizeof(unsigned),stream.s));
  CUtensorMap ma,mb;native_fusion::encode(ma,a.p,dt,true);native_fusion::encode(mb,b.p,dt,false);
  native_fusion::launch<Native,true>(a.p,b.p,nullptr,new_work.p,stream.s,ma,mb,static_cast<const Native*>(old.ha),static_cast<const Native*>(old.hb),error.as<unsigned>());
  unsigned result[2];std::vector<int> got(1024),expected(1024);
  check(cudaMemcpyAsync(result,error.p,sizeof(result),cudaMemcpyDeviceToHost,stream.s));
  check(cudaMemcpyAsync(got.data(),fresh.inside,4096,cudaMemcpyDeviceToHost,stream.s));
  check(cudaMemcpyAsync(expected.data(),old.inside,4096,cudaMemcpyDeviceToHost,stream.s));
  check(cudaStreamSynchronize(stream.s));int exponent_errors=0;
  for(int k=0;k<1024;++k)exponent_errors+=got[k]!=expected[k];
  bool pass=result[0]==0&&result[1]==0&&exponent_errors==0;failures+=!pass;
  printf("NATIVE_LAYOUT dtype=%s pattern=%d trial=%d seed=%d native_comparisons=69206016 inside_exponents=1024 a_errors=%u b_errors=%u exponent_errors=%d pass=%d\n",dtype,pattern,trial,seed,result[0],result[1],exponent_errors,pass);fflush(stdout);
 }
 return failures;
}
int main(){try{
 return (probe<__nv_bfloat16,cute::bfloat16_t>("bf16",megafmm::DType::BF16)+probe<__half,cute::half_t>("fp16",megafmm::DType::FP16))?2:0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}

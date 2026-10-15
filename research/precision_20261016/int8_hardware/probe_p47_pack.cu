// Generated from prior extended validation; original reference unchanged.
#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#define int8_research p46_reference
#include "p46/pack.cuh"
#undef int8_research
#include "p47/corrected.cuh"
#include <cuda_profiler_api.h>

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

// Diagnostic only: compare every native high/low bit, before either MMA path.
template<class T> __global__ void compare_values(const T* x,const T* y,int n,unsigned* errors){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)
  if(x[i]!=y[i])atomicAdd(errors,1u);
}
template<class Input,class Native> int probe(const char* dtype){
 Stream stream;Buffer a(Count*2),b(Count*2),ad(Count*8),bd(Count*8);
 Buffer old_work(p46_reference::WorkBytes),new_work(int8_research::WorkBytes),error(3*sizeof(unsigned));
 p46_reference::Work old(old_work.p);int8_research::Work fresh(new_work.p);
 p46_reference::init_cooperative_preprocess<Native>();
 int8_research::init_cooperative_preprocess<Native>();
 int failures=0;
 for(int pattern=0;pattern<9;++pattern)for(int trial=0;trial<3;++trial){
  int seed=47999+trial*353+pattern*61;
  varied_fill<<<512,256,0,stream.s>>>(a.as<Input>(),b.as<Input>(),ad.as<double>(),bd.as<double>(),pattern,seed);
  p46_reference::preprocess<Native,true>(a.as<Native>(),b.as<Native>(),old,stream.s);
  int8_research::preprocess<Native,true>(a.as<Native>(),b.as<Native>(),fresh,stream.s);
  check(cudaMemsetAsync(error.p,0,3*sizeof(unsigned),stream.s));
  compare_values<<<512,256,0,stream.s>>>((const uint16_t*)old.ha,(const uint16_t*)fresh.ha,11*512*512,error.as<unsigned>());
  compare_values<<<512,256,0,stream.s>>>((const uint16_t*)old.hb,(const uint16_t*)fresh.hb,11*512*512,error.as<unsigned>()+1);
  compare_values<<<4,256,0,stream.s>>>(old.inside,fresh.inside,1024,error.as<unsigned>()+2);
  unsigned result[3];check(cudaMemcpyAsync(result,error.p,sizeof(result),cudaMemcpyDeviceToHost,stream.s));
  check(cudaStreamSynchronize(stream.s));
  bool pass=result[0]==0&&result[1]==0&&result[2]==0;failures+=!pass;
  printf("PACK_BITS dtype=%s pattern=%d trial=%d seed=%d native_values=5767168 inside_exponents=1024 a_errors=%u b_errors=%u exponent_errors=%u pass=%d\n",dtype,pattern,trial,seed,result[0],result[1],result[2],pass);fflush(stdout);
 }
 return failures;
}
int main(){try{
 return (probe<__nv_bfloat16,cute::bfloat16_t>("bf16")+probe<__half,cute::half_t>("fp16"))?2:0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}

// Generated from prior extended validation; original reference unchanged.
#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#define int8_research p56_reference
#include "p56/pack.cuh"
#undef int8_research
#include "p60/pack.cuh"
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
  if(pattern==9||pattern==10){
   constexpr bool bf=std::is_same_v<T,__nv_bfloat16>;
   int hi=bf?127:15,lo=bf?-132:-24;
   x=ldexpf(x,pattern==9?hi:lo);y=ldexpf(y,pattern==9?lo:hi);
  }
  if(pattern==11){
   constexpr bool bf=std::is_same_v<T,__nv_bfloat16>;
   x=ldexpf(x,(c%64)<32?(bf?127:15):0);
   y=ldexpf(y,(r%64)<32?(bf?-132:-24):0);
  }
  T xx=rounded<T>(x),yy=rounded<T>(y);
  a[i]=xx;b[i]=yy;ad[i]=float(xx);bd[i]=float(yy);
 }
}

// Diagnostic only: compare every native high/low bit, before either MMA path.
template<class T> __global__ void compare_values(const T* x,const T* y,int n,unsigned* errors){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)
  if(x[i]!=y[i])atomicAdd(errors,1u);
}

template<class T> __global__ void conversion_oracle(unsigned* err){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=524288)return;
 constexpr bool bf=std::is_same_v<T,cute::bfloat16_t>;
 cutlass::Array<float,2> v;
 #pragma unroll
 for(int k=0;k<2;++k){
  unsigned u=unsigned(hash(uint64_t(i)*2+k+136247)>>32);
  float m=1.f+float(u&0x7fffff)/8388608.f,x=0;
  switch(i%8){
   case 0:x=(k==0?0.f:-0.f);break;
   case 1:x=ldexpf(float(int(u%33)-16),bf?-134:-25);break;
   case 2:x=1.f+float(int(u%129)-64)*ldexpf(1.f,bf?-8:-11);x=nextafterf(x,k?INFINITY:-INFINITY);break;
   case 3:x=bf?ldexpf(m,120):65504.f-float(u%4096)*.5f;break;
   case 4:x=ldexpf(m,bf?int(u%259)-140:int(u%45)-30);break;
   case 5:x=nextafterf(ldexpf(1.f,bf?-133:-24),k?INFINITY:0.f);break;
   case 6:x=__fsub_rn(1.f,ldexpf(m,bf?-24:-15));break;
   default:x=__fadd_rn(ldexpf(m,bf?-100:-14),ldexpf(1.f,bf?-108:-24));break;
  }
  v[k]=(i%8==0)?x:((u>>31)?-x:x);
 }
 auto high=int8_research::paired_native_convert<T>(v);
 cutlass::Array<float,2> residual;
 constexpr float scale=bf?256.f:2048.f;
 #pragma unroll
 for(int k=0;k<2;++k)residual[k]=__fmul_rn(__fsub_rn(v[k],float(high[k])),scale);
 auto low=int8_research::paired_native_convert<T>(residual);
 #pragma unroll
 for(int k=0;k<2;++k){
  T expected(v[k]);T expected_low(__fmul_rn(__fsub_rn(v[k],float(expected)),scale));
  if(high[k].raw()!=expected.raw())atomicAdd(err,1u);
  if(low[k].raw()!=expected_low.raw())atomicAdd(err+1,1u);
  if(high[k].raw()!=T(v[1-k]).raw())atomicAdd(err+2,1u);
 }
}
__global__ void byte_guards(const uint8_t* p,int total,int offset,int length,unsigned* errors){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=gridDim.x*blockDim.x)
  if((i<offset||i>=offset+length)&&p[i]!=0xa5)atomicAdd(errors,1u);
}
template<class Input,class Native> int probe(const char* dtype){
 Stream stream;Buffer a(Count*2+1024),b(Count*2+1024),ad(Count*8),bd(Count*8);
 Buffer old_work(p56_reference::WorkBytes),new_work(int8_research::WorkBytes+512),error(6*sizeof(unsigned));
 p56_reference::Work old(old_work.p);int8_research::Work fresh(new_work.as<uint8_t>()+256);
 check(cudaMemsetAsync(error.p,0,6*sizeof(unsigned),stream.s));
 conversion_oracle<Native><<<2048,256,0,stream.s>>>(error.as<unsigned>());
 unsigned scalar[3];check(cudaMemcpyAsync(scalar,error.p,sizeof(scalar),cudaMemcpyDeviceToHost,stream.s));
 check(cudaStreamSynchronize(stream.s));
 bool scalar_pass=scalar[0]==0&&scalar[1]==0&&scalar[2]>0;
 printf("PAIR_CONVERSION dtype=%s pairs=524288 native_comparisons=2097152 high_errors=%u low_errors=%u swapped_lane_negative=%u pass=%d\n",dtype,scalar[0],scalar[1],scalar[2],scalar_pass);fflush(stdout);
 if(!scalar_pass)return 1;
 p56_reference::init_cooperative_preprocess<Native>();int8_research::init_cooperative_preprocess<Native>();
 cudaFuncAttributes attr{};check(cudaFuncGetAttributes(&attr,int8_research::preprocess_cooperative_columns<Native>));
 printf("PAIR_RESOURCE dtype=%s regs=%d local_bytes=%zu shared_bytes=%zu grid=384\n",dtype,attr.numRegs,attr.localSizeBytes,sizeof(int8_research::PairPackShared<true>));fflush(stdout);
 int failures=0;
 for(int misaligned=0;misaligned<2;++misaligned){
  for(int pattern=0;pattern<12;++pattern)for(int trial=0;trial<(misaligned?2:3);++trial){
   if(misaligned&&pattern!=0&&pattern!=6&&pattern!=11)continue;
   int seed=137263+trial*547+pattern*151+misaligned*10007+(std::is_same_v<Native,cute::half_t>?311:0);
   int offset=256+2*misaligned;auto ap=reinterpret_cast<Input*>(a.as<uint8_t>()+offset),bp=reinterpret_cast<Input*>(b.as<uint8_t>()+offset);
   check(cudaMemsetAsync(a.p,0xa5,Count*2+1024,stream.s));check(cudaMemsetAsync(b.p,0xa5,Count*2+1024,stream.s));
   varied_fill<<<512,256,0,stream.s>>>(ap,bp,ad.as<double>(),bd.as<double>(),pattern,seed);
   p56_reference::preprocess<Native,true>(reinterpret_cast<Native*>(ap),reinterpret_cast<Native*>(bp),old,stream.s);
   check(cudaMemsetAsync(new_work.p,0xa5,int8_research::WorkBytes+512,stream.s));
   int8_research::preprocess<Native,true>(reinterpret_cast<Native*>(ap),reinterpret_cast<Native*>(bp),fresh,stream.s);
   check(cudaMemsetAsync(error.p,0,6*sizeof(unsigned),stream.s));
   compare_values<<<512,256,0,stream.s>>>((const uint16_t*)old.ha,(const uint16_t*)fresh.ha,11*512*512,error.as<unsigned>());
   compare_values<<<512,256,0,stream.s>>>((const uint16_t*)old.hb,(const uint16_t*)fresh.hb,11*512*512,error.as<unsigned>()+1);
   compare_values<<<4,256,0,stream.s>>>(old.inside,fresh.inside,1024,error.as<unsigned>()+2);
   byte_guards<<<32,256,0,stream.s>>>(a.as<uint8_t>(),Count*2+1024,offset,Count*2,error.as<unsigned>()+3);
   byte_guards<<<32,256,0,stream.s>>>(b.as<uint8_t>(),Count*2+1024,offset,Count*2,error.as<unsigned>()+4);
   byte_guards<<<32,256,0,stream.s>>>(new_work.as<uint8_t>(),int8_research::WorkBytes+512,256,int8_research::WorkBytes,error.as<unsigned>()+5);
   std::vector<int> exponents(1024);check(cudaMemcpyAsync(exponents.data(),old.inside,4096,cudaMemcpyDeviceToHost,stream.s));
   unsigned result[6];check(cudaMemcpyAsync(result,error.p,sizeof(result),cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
   int outside=0,maxabs=0;for(int e:exponents){outside+=e< -126||e>127;maxabs=std::max(maxabs,abs(e));}
   bool pass=true;for(unsigned x:result)pass=pass&&x==0;
   if(std::is_same_v<Native,cute::bfloat16_t>&&pattern>=9)pass=pass&&outside>0;
   failures+=!pass;
   printf("PAIR_PACK dtype=%s pattern=%d trial=%d seed=%d input_offset_bytes=%d native_values=5767168 inside_exponents=1024 a_errors=%u b_errors=%u exponent_errors=%u a_guard_errors=%u b_guard_errors=%u workspace_guard_errors=%u outside_normal=%d max_abs_exponent=%d pass=%d\n",dtype,pattern,trial,seed,2*misaligned,result[0],result[1],result[2],result[3],result[4],result[5],outside,maxabs,pass);fflush(stdout);
   if(!pass)return failures;
  }
 }
 return failures;
}
int main(){try{
 int failures=probe<__nv_bfloat16,cute::bfloat16_t>("bf16");if(failures)return 2;
 failures+=probe<__half,cute::half_t>("fp16");return failures?2:0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}

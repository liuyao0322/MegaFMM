"""Own pre-MMA conversion oracle and P56-plane controls, including 2B inputs."""
from pathlib import Path
import hashlib,json
h=Path(__file__).resolve().parent;repo=h.parents[2]
base=(h/'probe_p58_edges.cu').read_text()
prefix=base[:base.index('template<class Input,class Native> int probe')].replace('"p58/pack.cuh"','"p60/pack.cuh"')
body=r'''
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
  if(T(high[k]).raw()!=expected.raw())atomicAdd(err,1u);
  if(T(low[k]).raw()!=expected_low.raw())atomicAdd(err+1,1u);
  if(T(high[k]).raw()!=T(v[1-k]).raw())atomicAdd(err+2,1u);
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
   int seed=137263+trial*547+pattern*151+misaligned*10007+(std::is_same_v<Native,cute::half_t> ? 311 : 0);
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
'''
(h/'probe_p60_pack.cu').write_text(prefix+body)
script='''#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
root=research/precision_20261016/int8_hardware
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \\
 -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -Iinclude -Isolution -Ithird_party/cutlass/include \\
 -Xptxas=-v "$root/probe_p60_pack.cu" build/libmegafmm.a -lcublas -lcublasLt -lcuda -o "$root/bin/probe_p60_pack"
'''
(h/'build_probe_p60_pack.sh').write_text(script)
files=[Path(__file__),h/'probe_p60_pack.cu',h/'build_probe_p60_pack.sh',h/'p60_source_manifest.json',h/'p56/pack.cuh']
(h/'p60_probe_source_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print('P60 own scalar oracle / 72 aligned / 12 misaligned native-plane controls prepared')

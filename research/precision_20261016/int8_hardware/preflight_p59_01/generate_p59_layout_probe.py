"""Own raw-input preparation/publication/TMA generic-reader probe, before MMA."""
from pathlib import Path
import hashlib,json,re
h=Path(__file__).resolve().parent;repo=h.parents[2]
s=(h/'probe_p56_layout.cu').read_text().replace('P56','P59').replace('p56','p59')
s=re.sub(r'(?<=[A-Za-z])56\b','59',s)
s=s.replace('#include "p59/corrected.cuh"','#define int8_research p56_reference\n#include "p56/pack.cuh"\n#undef int8_research\n#include "p59/corrected.cuh"')
a=s.index('__global__ void synthetic59');b=s.index('__global__ void coverage_init59',a)
fill=(h/'probe_p58_edges.cu').read_text();fill=fill[fill.index('template<class T> __global__ void varied_fill'):fill.index('// Diagnostic only:')]
s=s[:a]+fill+s[b:]
s=s.replace(' const uint16_t* expected_a,const uint16_t* expected_b,unsigned* error,unsigned* coverage){',' const T* input_a,const T* input_b,Work w,const uint16_t* expected_a,const uint16_t* expected_b,unsigned* error,unsigned* coverage){')
s=s.replace('m0=blockIdx.y*64,n0=blockIdx.x*32','m0=(blockIdx.x/16)*64,n0=(blockIdx.x%16)*32')
needle=' if(tid==0){for(int j=0;j<Stages;++j){Barrier::init'
i=s.index(needle)
s=s[:i]+''' prepare_resident<T>(input_a,input_b,static_cast<T*>(w.ha),static_cast<T*>(w.hb),
  static_cast<T*>(w.qa),static_cast<T*>(w.qb),w.ea,w.eb,w.inside,w.partial,reinterpret_cast<unsigned char*>(&s));
 cooperative_groups::this_grid().sync();
'''+s[i:]
s=s.replace('template<class T> int probe59','template<class Input,class T> int probe59')
s=s.replace('Stream stream;Buffer work(WorkBytes+512),error(2*sizeof(unsigned)),coverage((Plane+256)*4);Work w((char*)work.p+256);','''Stream stream;Buffer work(WorkBytes+512),error(3*sizeof(unsigned)),coverage((Plane+256)*4);
 Buffer a(Count*2),b(Count*2),ad(Count*8),bd(Count*8),reference_work(p56_reference::WorkBytes);
 Work w((char*)work.p+256);p56_reference::Work expected(reference_work.p);
 p56_reference::init_cooperative_preprocess<T>();''')
s=s.replace('corrected<T>','fused_global<T>')
s=s.replace('"candidate consumer capacity"','"candidate fused capacity"')
s=s.replace('consumer_regs=%d','fused_regs=%d')
needle=' int failures=0;\n for(int trial=0;trial<3;++trial){'
assert s.count(needle)==1
s=s.replace(needle,''' int probe_active=0;
 check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&probe_active,layout59<T>,256,sizeof(ProbeShared59)));
 require(probe_active*prop.multiProcessorCount>=128,"layout probe cooperative capacity");
 printf("FUSED_PROBE_CAPACITY dtype=%s active_blocks_per_sm=%d sm_count=%d grid=128 shared_bytes=%zu\\n",dtype,probe_active,prop.multiProcessorCount,sizeof(ProbeShared59));
 int failures=0;
 for(int trial=0;trial<3;++trial){''')
s=s.replace('int seed=128171+491*trial+(dt==megafmm::DType::BF16?0:269);','int pattern=trial==0?0:trial==1?2:11;int seed=134227+523*trial+(dt==megafmm::DType::BF16?0:283);')
s=s.replace('cudaMemsetAsync(error.p,0,8,stream.s)','cudaMemsetAsync(error.p,0,12,stream.s)')
s=s.replace('  synthetic59<<<512,256,0,stream.s>>>((uint16_t*)w.ha,(uint16_t*)w.hb,seed,trial);','''  varied_fill<<<512,256,0,stream.s>>>(a.as<Input>(),b.as<Input>(),ad.as<double>(),bd.as<double>(),pattern,seed);
  p56_reference::preprocess<T,true>(a.as<T>(),b.as<T>(),expected,stream.s);''')
old='  layout59<T><<<dim3(16,8),256,sizeof(ProbeShared59),stream.s>>>(ha,hb,(uint16_t*)w.ha,(uint16_t*)w.hb,error.as<unsigned>(),coverage.as<unsigned>());check(cudaGetLastError());'
assert old in s
s=s.replace(old,'''  const T* input_a=a.as<T>();const T* input_b=b.as<T>();
  auto expected_a=(const uint16_t*)expected.ha;auto expected_b=(const uint16_t*)expected.hb;
  auto error_ptr=error.as<unsigned>();auto coverage_ptr=coverage.as<unsigned>();
  void* args[]={&ha,&hb,&input_a,&input_b,&w,&expected_a,&expected_b,&error_ptr,&coverage_ptr};
  check(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(layout59<T>),dim3(128),dim3(256),args,sizeof(ProbeShared59),stream.s));
  compare_inside59<<<4,256,0,stream.s>>>(expected.inside,w.inside,error.as<unsigned>()+2);check(cudaGetLastError());''')
s=s.replace('unsigned result[2]','unsigned result[3]').replace('cudaMemcpyAsync(result,error.p,8,','cudaMemcpyAsync(result,error.p,12,')
s=s.replace('bool pass=result[0]==0&&result[1]==0&&ce==0&&ge==0','bool pass=result[0]==0&&result[1]==0&&result[2]==0&&ce==0&&ge==0')
s=s.replace('trial=%d seed=%d ctas=','trial=%d seed=%d pattern=%d ctas=')
s=s.replace('workspace_guard_errors=%u pass=%d','workspace_guard_errors=%u inside_exponents=1024 inside_errors=%u pass=%d')
s=s.replace('dtype,trial,seed,result[0],result[1],ce,ge,pass','dtype,trial,seed,pattern,result[0],result[1],ce,ge,result[2],pass')
s=s.replace('probe59<cute::bfloat16_t>','probe59<__nv_bfloat16,cute::bfloat16_t>').replace('probe59<cute::half_t>','probe59<__half,cute::half_t>')
needle='template<class Input,class T> int probe59'
i=s.index(needle)
s=s[:i]+'''__global__ void compare_inside59(const int* expected,const int* actual,unsigned* errors){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<1024&&expected[i]!=actual[i])atomicAdd(errors,1u);
}
'''+s[i:]
(h/'probe_p59_layout.cu').write_text(s)
(h/'build_probe_p59_layout.sh').write_text((h/'build_probe_p56_layout.sh').read_text().replace('p56','p59'))
(h/'run_probe_p59_layout.sh').write_text((h/'run_probe_p56_layout.sh').read_text().replace('p56','p59'))
files=[Path(__file__),h/'probe_p59_layout.cu',h/'build_probe_p59_layout.sh',h/'run_probe_p59_layout.sh',h/'p59_source_manifest.json',h/'p56_manifest.json',h/'probe_p58_edges.cu',Path(repo/'tests/test_sw1024.cu')]
(h/'p59_probe_source_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print('P59 own six raw-input->global-plane->TMA probe source frozen before compile')

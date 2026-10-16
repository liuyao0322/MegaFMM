"""Fresh P61 full-plane/exponent tests before first MMA, independent scalar inside."""
from pathlib import Path
import hashlib,json
h=Path(__file__).resolve().parent;repo=h.parents[2]
s=(h/'probe_p60_pack.cu').read_text().replace('"p60/pack.cuh"','"p61/pack.cuh"')
a=s.index('template<class T> __global__ void conversion_oracle');b=s.index('__global__ void byte_guards',a)
s=s[:a]+s[b:]
a=s.index(' check(cudaMemsetAsync(error.p,0,6*sizeof(unsigned),stream.s));');b=s.index(' p56_reference::init_cooperative_preprocess',a)
s=s[:a]+s[b:]
s=s.replace('new_work(int8_research::WorkBytes+512),error(6*sizeof(unsigned))','new_work(int8_research::WorkBytes+512),error(7*sizeof(unsigned)),reference_inside(4096)')
s=s.replace('6*sizeof(unsigned)','7*sizeof(unsigned)').replace('unsigned result[6]','unsigned result[7]')
key='   check(cudaMemsetAsync(error.p,0,7*sizeof(unsigned),stream.s));'
assert s.count(key)==1
s=s.replace(key,key+'''
   p56_reference::inside_scale<Native><<<1024,256,0,stream.s>>>(reinterpret_cast<Native*>(ap),reinterpret_cast<Native*>(bp),reference_inside.as<int>());
   compare_values<<<4,256,0,stream.s>>>(reference_inside.as<int>(),fresh.inside,1024,error.as<unsigned>()+6);''')
s=s.replace('137263+trial*547+pattern*151+misaligned*10007+(std::is_same_v<Native,cute::half_t> ? 311 : 0)','139301+trial*563+pattern*163+misaligned*10009+(std::is_same_v<Native,cute::half_t> ? 331 : 0)')
s=s.replace('PAIR_RESOURCE','INSIDE_RESOURCE').replace('PAIR_PACK','INSIDE_PACK')
s=s.replace('workspace_guard_errors=%u outside_normal=','workspace_guard_errors=%u scalar_inside_errors=%u outside_normal=')
s=s.replace('result[3],result[4],result[5],outside','result[3],result[4],result[5],result[6],outside')
(h/'probe_p61_pack.cu').write_text(s)
s=(h/'build_probe_p60_pack.sh').read_text().replace('p60','p61')
(h/'build_probe_p61_pack.sh').write_text(s)
files=[Path(__file__),h/'probe_p61_pack.cu',h/'build_probe_p61_pack.sh',h/'p61_source_manifest.json',h/'p56/pack.cuh',h/'probe_p60_pack.cu']
(h/'p61_probe_source_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print('P61 own84fresh cases/native planes/inside/independent scalar inside/guards prepared')

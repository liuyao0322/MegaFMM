"""Own P59 raw preparation/publication/TMA guards before candidate MMA."""
from pathlib import Path
import hashlib,json,re
h=Path(__file__).resolve().parent;repo=h.parents[2]
old=(h/'p56/corrected.cuh').read_text();c=(h/'p59/corrected.cuh').read_text();p=(h/'probe_p59_layout.cu').read_text()
def consumer(text):
 a=text.index('template<class T> __device__ __forceinline__ void consume')
 return text[a:text.index('template<class T> __global__ __launch_bounds__(256,1)',a)]
assert consumer(old)==consumer(c)
f=c[c.index('void fused_global('):c.index('\nvoid encode(')]
a=f[f.index(' if(tid==0){\n  for(int step=0;step<7;'):f.index('}else if(tid>=128)consume')+1]
b=p[p.index(' // PRODUCER_BEGIN')+len(' // PRODUCER_BEGIN'):p.index(' // PRODUCER_END')]
assert ''.join(a.split())==''.join(b.split())
pack58=(h/'p58/pack.cuh').read_text();pack59=(h/'p59/pack.cuh').read_text()
body58=pack58[pack58.index(' constexpr bool Balance=true;',pack58.index('void preprocess_cooperative_columns')):pack58.index('\ntemplate<class T> void init_cooperative_preprocess')]
body58=body58.replace(' extern __shared__ __align__(128) unsigned char scratch_buffer[];\n','')
body59=pack59[pack59.index(' constexpr bool Balance=true;',pack59.index('void prepare_resident')):pack59.index('// All128CTAs participate')]
assert body58.strip()==body59.strip()
for text in [f,p[p.index('void layout59('):p.index('// PRODUCER_BEGIN')]]:
 assert text.index('prepare_resident<T>')<text.index('cooperative_groups::this_grid().sync();')<text.index('Barrier::init')
 assert 'm0=(blockIdx.x/16)*64,n0=(blockIdx.x%16)*32' in text
text=(h/'probe_p59_layout.log').read_text()
def records(prefix):return [dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in text.splitlines() if s.startswith(prefix+' ')]
rows=records('STATIC_LAYOUT');assert len(rows)==6
for x in rows:
 t=int(x['trial']);assert int(x['seed'])==134227+523*t+(283 if x['dtype']=='fp16' else 0)
 assert int(x['pattern'])==[0,2,11][t]
 assert x['pass']=='1' and all(x[k]=='0' for k in ['native_errors','shared_guard_errors','coverage_errors','workspace_guard_errors','inside_errors'])
 assert (x['ctas'],x['sequences'],x['slots'],x['native_comparisons'],x['inside_exponents'])==('128','28','4','69206016','1024')
resources=records('STATIC_RESOURCE');fused=records('FUSED_INIT');probe=records('FUSED_PROBE_CAPACITY')
assert len(resources)==len(fused)==len(probe)==2
for x in resources:assert x['shared_bytes']=='196736' and x['active_blocks_per_sm']=='1' and x['sm_count']=='132'
for x in fused:assert x['supported']=='1' and x['grid']=='128' and x['active_blocks_per_sm']=='1' and x['sm_count']=='132' and x['shared_bytes']=='196736'
for x in probe:assert x['grid']=='128' and x['active_blocks_per_sm']=='1' and x['sm_count']=='132' and x['shared_bytes']=='197888'
assert all(next(x for x in resources if x['dtype']==y['dtype'])['fused_regs']==y['regs'] for y in fused)
out=dict(scope='Own six newly seeded original raw-input -> P58 exact preparation helper -> global11planes -> second grid barrier -> candidate verbatim producer/TMA -> generic shared reads vs frozen P56 independent expected planes. All28batches/fourslots/full128grid, inside/shared/workspace/coordinate guards; no MMA execution here. Does not certify actual WGMMA wait-order, full matrix gates, sanitizer or performance. Actual fused and generic-probe resources/capacity separately queried.',configurations=6,native_uint16_comparisons_including_cta_reuse=415236096,output_coordinates=1572864,inside_exponent_comparisons=6144,errors=0,resources=resources,fused_cooperative_capacity=fused,probe_cooperative_capacity=probe,rows=rows)
(h/'probe_p59_layout.json').write_text(json.dumps(out,indent=2)+'\n')
files=[Path(__file__)]+[h/n for n in ['probe_p59_layout.cu','build_probe_p59_layout.sh','run_probe_p59_layout.sh','probe_p59_layout_build.log','probe_p59_layout.log','probe_p59_layout.json','p59_source_manifest.json','p59_probe_source_manifest.json','bin/probe_p59_layout','p59_probe_generation_preflight.log','p59_probe_visibility_preflight.log']]
(h/'p59_layout_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['scope','rows']},indent=2))

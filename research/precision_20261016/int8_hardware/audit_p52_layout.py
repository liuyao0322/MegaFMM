"""Freeze P52's own TMA/layout diagnostic before any new MMA execution."""
import json,hashlib,re
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[2]
source=(h/'p52/corrected.cuh').read_text();probe=(h/'probe_p52_layout.cu').read_text()
a=source[source.index(' if(tid==0){\n  for(int step'):source.index('}else if(tid>=128)consume')+1]
b=probe[probe.index(' if(tid==0){\n  for(int step'):probe.index(' // PRODUCER_END')]
assert ''.join(a.split())==''.join(b.split())
for n in ['pack.cuh','compare.cu','pow2.cuh']:assert (h/'p52'/n).read_bytes()==(h/'p51'/n).read_bytes()
text=(h/'probe_p52_layout.log').read_text()
rows=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('CONSUMER_LAYOUT ')]
assert len(rows)==6 and {(x['dtype'],int(x['trial'])) for x in rows}=={(d,t) for d in ['bf16','fp16'] for t in range(3)}
for x in rows:
 assert int(x['seed'])==114067+433*int(x['trial'])+(0 if x['dtype']=='bf16' else 227)
 assert x['pass']=='1' and all(x[k]=='0' for k in ['a_errors','b_errors','shared_guard_errors','coverage_errors','workspace_guard_errors'])
 assert x['ctas']=='256' and x['sequences']=='28' and x['slots']=='2' and x['native_comparisons']=='115343360'
init=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('CONSUMER_INIT ')]
assert len(init)==2 and {x['dtype'] for x in init}=={'bf16','fp16'}
assert all(int(x['active_blocks_per_sm'])>=1 and x['shared_bytes']=='82048' and x['grid']=='256' for x in init)
out=dict(scope='Pre-MMA only, six own synthetic plane-distinguishing configurations: low index16bits, upper indexbits, fresh hasheduint16. Both native dtypes, all256CTAs/28sequences/two-slot reuse; every loaded high/required-low value plus output-coordinate bijection and shared/workspace/output guards. Candidate TMA producer copied verbatim; generic reads retire current slot without executing any MMA, so this is not a WGMMA wait-order, matrix, performance or sanitizer certificate.',configurations=6,native_uint16_comparisons_including_cta_reuse=692060160,unique_workspace_native_values_per_configuration=5767168,bit_errors=0,output_coordinate_checks=6291456,consumer_runtime_capacity=init,rows=rows)
(h/'probe_p52_layout.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p52_layout.py','probe_p52_layout.cu','build_probe_p52_layout.sh','probe_p52_layout_build.log','probe_p52_layout.log','probe_p52_layout.json','bin/probe_p52_layout','p52_source_manifest.json','../../../tests/test_sw1024.cu']
(h/'p52_layout_manifest.json').write_text(json.dumps({str((h/f).resolve().relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

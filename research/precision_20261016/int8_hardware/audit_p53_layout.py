"""Own P53 pre-MMA TMA/layout and GPU scalar recombination evidence only."""
import hashlib
import json
import re
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[2]
source=(h/'p53/corrected.cuh').read_text();probe=(h/'probe_p53_layout.cu').read_text()
a=source[source.index(' if(tid==0){\n  for(int k=0;k<4;++k)'):source.index('}else if(tid>=128)consume')+1]
b=probe[probe.index(' if(tid==0){\n  for(int k=0;k<4;++k)'):probe.index(' // PRODUCER_END')]
assert ''.join(a.split())==''.join(b.split())
for n in ['compare.cu','pow2.cuh']:assert (h/'p53'/n).read_bytes()==(h/'p51'/n).read_bytes()
text=(h/'probe_p53_layout.log').read_text()
rows=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('CONSUMER_LAYOUT ')]
assert len(rows)==6 and {(x['dtype'],int(x['trial'])) for x in rows}=={(d,t) for d in ['bf16','fp16'] for t in range(3)}
for x in rows:
 assert int(x['seed'])==116081+443*int(x['trial'])+(0 if x['dtype']=='bf16' else 233)
 assert x['pass']=='1' and all(x[k]=='0' for k in ['a_errors','b_errors','shared_guard_errors','coverage_errors','workspace_guard_errors'])
 assert x['ctas']=='896' and x['sequences']=='4' and x['slots']=='2' and x['native_comparisons']=='69206016'
init=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('LEAF_CONSUMER_INIT ')]
assert len(init)==2 and {x['dtype'] for x in init}=={'bf16','fp16'}
assert all(int(x['active_blocks_per_sm'])>=1 and x['shared_bytes']=='98432' and x['grid']=='896' for x in init)
recombine=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('LEAF_RECOMBINE ')]
assert len(recombine)==3 and {int(x['trial']) for x in recombine}==set(range(3))
for x in recombine:
 assert int(x['seed'])==118097+449*int(x['trial']) and x['output_bit_comparisons']=='1048576'
 assert x['pass']=='1' and all(x[k]=='0' for k in ['bit_errors','guard_errors','input_errors','nonfinite'])
 assert x['input_values']=='1835264' and int(x['changed_order_negative_differences'])>0
out=dict(scope='Pre-MMA only: own six synthetic native plane/TMA layouts, all896CTAs/four sequences/two-slot reuse, seven-plane output-coordinate bijection/shared+workspace+output guards. Generic reads release current slot without any MMA; no WGMMA wait-order, original-input matrix, performance or sanitizer certificate. Separate GPU recombination kernel compared against independent P51-order host oracle on three newly seeded signed/scaled/canceling leaf arrays and signed zeros, guarded output and unchanged leaf input.',configurations=6,native_uint16_comparisons_including_cta_reuse=415236096,unique_workspace_native_values_per_configuration=5767168,bit_errors=0,leaf_coordinate_checks=11010048,consumer_runtime_capacity=init,rows=rows,recombine_trials=3,recombine_output_bit_comparisons=3145728,recombine_bit_errors=0,recombine_rows=recombine)
(h/'probe_p53_layout.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p53_layout.py','probe_p53_layout.cu','build_probe_p53_layout.sh','run_probe_p53_layout.sh','probe_p53_layout_build.log','probe_p53_layout.log','probe_p53_layout.json','bin/probe_p53_layout','p53_source_manifest.json','../../../tests/test_sw1024.cu']
(h/'p53_layout_manifest.json').write_text(json.dumps({str((h/f).resolve().relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['rows','recombine_rows']},indent=2))

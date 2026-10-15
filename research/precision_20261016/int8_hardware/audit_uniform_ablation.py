"""Audit P31–P34 arithmetic-preserving hardware controls without scope transfer."""
import hashlib
import json
import re
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[2]
def read(name):return json.loads((h/name).read_text())
def rows(name,prefix):return [dict(re.findall(r'(\w+)=([^ ]+)',x)) for x in (h/name).read_text().splitlines() if x.startswith(prefix)]
reference=read('p30_balanced.json')['checks'];candidates={}
for label,regs in [('p31',198),('p32',198),('p33',198),('p34',196)]:
    j=read(label+'_balanced.json');rr=j['checks']
    assert len(rr)==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in rr)
    assert all(all(a[k]==b[k] for k in ['dtype','pattern','trial','corrected_ef']) for a,b in zip(rr,reference))
    inside=rows(label+'_balanced.log','INSIDE_EQUAL ')
    assert len(inside)==78 and all(x['elements']=='1024' and x['errors']=='0' for x in inside)
    build=(h/(label+'_build.log')).read_text();assert 'C7515' not in build and 'C7510' not in build
    consumer=re.findall(r"Compiling entry function '([^']*corrected[^']*)'.*?(?=ptxas info    : Compiling entry|\Z)",build,re.S)
    assert len(consumer)==2
    # Inspect the full function records, not only their symbol names.
    records=[x.group(0) for x in re.finditer(r"Compiling entry function '([^']*corrected[^']*)'.*?(?=ptxas info    : Compiling entry|\Z)",build,re.S)]
    assert all(f'Used {regs} registers' in x and '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in x for x in records)
    d=read(label+'_scale_diagnostics.json')
    assert all(x['groups']==24960 for x in d['ordinary_correctness_summary'])
    if label=='p34':assert all(x['paired_eligible']==0 for x in d['ordinary_correctness_summary'])
    static=read(label+'_static_stats.json')
    candidates[label]=dict(ordinary_checks=78,equal_per_column_exponents=79872,all_printed_ef_equal_p30=True,
       full_call_us={s['dtype']:s['corrected_us'] for s in j['summaries']},consumer_registers=regs,
       static_bf16_instructions=static['instructions'],static_shfl=sum(v for k,v in static['opcodes'].items() if k.startswith('SHFL')),
       metadata_diagnostics=d['ordinary_correctness_summary'],performance_accepted=False,
       extra_validation='own full Graph78, fresh9-pattern stress108/864calls, scoped tools36/72calls each' if label=='p33' else ('fresh9-pattern stress108/864calls only; no full Graph78 or sanitizer claim' if label=='p31' else 'ordinary78 only; no Graph/stress/sanitizer claim'))
for label in ['p31','p33']:
    d=read(label+'_scale_diagnostics.json');assert d['stress_configurations']==108 and d['stress_interleaved_calls']==864
v=read('p33_validation.json');assert v['independent_stress_configurations']==108 and v['sanitizer_cases_per_tool']==36
assert (h/'p33_graph.log').read_bytes()==(h/'p33_graph_abba_b0.log').read_bytes()
for name in ['p33_abba.json','p33_graph_abba.json','p34_abba.json']:
    j=read(name);assert j['correctness_checks']==312 and j['independent_numerical_configurations']==78
assert all(0<x['relative_reduction']<.03 for x in read('p33_abba.json')['summary'])
assert all(x['relative_reduction']<0 for x in read('p33_graph_abba.json')['summary'])
assert all(x['relative_reduction']<0 for x in read('p34_abba.json')['summary'])
out=dict(status='accuracy_validated_performance_rejected',goal_achieved=False,retained_ordinary='p33',retained_graph='p27',
 candidates=candidates,ordinary_p30_vs_p33=read('p33_abba.json')['summary'],graph_p27_vs_p33=read('p33_graph_abba.json')['summary'],
 ordinary_p33_vs_p34=read('p34_abba.json')['summary'],
 scope='Full original-input FP64 matrix gates; printed eF equality is not a bitwise-output claim. Host eligibility diagnostics outside timing. No production integration. Unlocked clocks, no confidence interval.')
(h/'uniform_ablation.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_uniform_ablation.py','uniform_ablation.json','generate_uniform_scale.py','generate_precomputed_uniform.py','generate_uniform_wait0.py','generate_single_uniform.py',
 'export_uniform_static.py','analyze_uniform_scale.py','prepare_uniform_validation.py','build_stress_p31.sh','validate_p31.sh','p31_scale_diagnostics_manifest.json','p33_validation_manifest.json']
for label in ['p30','p31','p32','p33','p34']:
    files+=[label+'_static_manifest.json']
for label in ['p31','p32','p33','p34']:
    files += [label+'_manifest.json',label+'_scale_diagnostics_manifest.json']
files+=['p33_abba_manifest.json','p33_graph_abba_manifest.json','p34_abba_manifest.json','profile_p31_manifest.json','profile_p33_manifest.json']
(h/'uniform_ablation_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='candidates'},indent=2))

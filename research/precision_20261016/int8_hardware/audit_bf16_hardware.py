"""Keep P43/P44 representation accuracy separate from rejected performance."""
import hashlib
import json
import math
import re
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[2]
def rows(file,prefix):
 return [dict(re.findall(r'(\w+)=([^ ]+)',x)) for x in (h/file).read_text().splitlines() if x.startswith(prefix)]
results=[]
for label in ('p43','p44'):
 j=json.loads((h/f'{label}_balanced.json').read_text());c=j['checks']
 assert len(c)==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in c)
 inside=rows(f'{label}_balanced.log','INSIDE_EQUAL ')
 assert len(inside)==78 and all(x['elements']=='1024' and x['errors']=='0' for x in inside)
 init=rows(f'{label}_balanced.log','COOPERATIVE_INIT ')
 assert len(init)==2 and all(x['supported']=='1' and int(x['active_blocks_per_sm'])*int(x['sm_count'])>=288 for x in init)
 build=(h/f'{label}_build.log').read_text()
 assert 'C7515' not in build and 'C7510' not in build
 abba=json.loads((h/f'{label}_abba.json').read_text())
 assert abba['correctness_checks']==312 and abba['independent_numerical_configurations']==78
 assert all(x['relative_reduction']<0 for x in abba['summary'])
 r=dict(label=label,ordinary_checks=78,inside_exponent_checks=79872,
  max_ef={s['dtype']:s['max_ef'] for s in j['summaries']},
  initial_full_call_us={s['dtype']:s['corrected_us'] for s in j['summaries']},
  ordinary_abba=abba['summary'],cooperative_init=init,
  performance_accepted=False,retained=False,independent_graph_timing=False,
  scoped_sanitizers=False)
 if label=='p43':
  st=rows('stress_p43.log','EXTRA ')
  assert len(st)==108 and all(x['pass']=='1' and x['guards']=='1' and math.isfinite(float(x['ef'])) for x in st)
  assert {int(x['pattern']) for x in st}==set(range(9))
  assert '43871+trial*347+s*197+pat*59' in (h/'stress_p43.cu').read_text()
  r.update(fresh_stress_configurations=108,fresh_stress_calls=864,
   stress_scope='two independent streams/workspaces, ordinary/Graph interleaving, 128float guards each side; not independent Graph timing or sanitizer acceptance',
   stress_seed_formula='43871+trial*347+stream*197+pattern*59',
   stress_max_ef={dt:max(float(x['ef']) for x in st if x['dtype']==dt) for dt in ('bf16','fp16')},
   stress_tiny_max_ef={dt:max(float(x['ef']) for x in st if x['dtype']==dt and x['pattern']=='8') for dt in ('bf16','fp16')})
 else:r['fresh_stress_configurations']=0
 results.append(r)
c43=json.loads((h/'p43_balanced.json').read_text())['checks'];c44=json.loads((h/'p44_balanced.json').read_text())['checks']
assert all(all(a[k]==b[k] for k in ('dtype','pattern','trial','corrected_ef')) for a,b in zip(c43,c44))
out=dict(status='native_main_bf16_correction_accuracy_passed_performance_rejected',goal_achieved=False,
 scope='original contiguous1024cubed inputs, bothdtypes, original FP64/gates, seven main and all eight correction dots; each candidate has only its own listed evidence',
 candidates=results,printed_standard_errors_equal_p43_p44=True,
 compile_descriptor_preflight='separately frozen initialP43 rank3 256Bfast box failed host encoding; no candidate matrix execution or latency',
 retained='P39 remains fully validated ordinary/Graph research reference; production unchanged',
 limitations='No P43/P44 independent Graph timing or sanitizer acceptance; P44 has no fresh stress. FP32bmm M11 model is not hardware proof or performance.')
(h/'bf16_hardware_ablation.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_bf16_hardware.py','bf16_hardware_ablation.json','BF16_CORRECTION_HARDWARE.md',
 'generate_bf16_correction.py','generate_bf16_no_transpose.py','p43_descriptor_preflight_manifest.json',
 'stress_p43.cu','build_stress_p43.sh','stress_p43_build.log','stress_p43.log','bin/stress_p43',
 'm11_manifest.json','m11_summary.json','analyze.py','analyze_abba.py','prepare_uniform_validation.py']
for label in ('p43','p44'):
 files += [f'{label}_manifest.json',f'bin/compare_{label}',f'{label}_balanced.log',f'{label}_balanced.json',f'{label}_build.log',f'{label}_abba.json',f'{label}_abba_manifest.json']
(h/'bf16_hardware_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

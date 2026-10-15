"""P35/P36/P37: numerical changes and diagnostic exclusions kept explicit."""
import hashlib
import json
import re
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[2]
def read(name):return json.loads((h/name).read_text())
def rows(label,prefix):return [dict(re.findall(r'(\w+)=([^ ]+)',x)) for x in (h/(label+'_balanced.log')).read_text().splitlines() if x.startswith(prefix)]
candidates={}
for label in ['p35','p36','p37']:
    j=read(label+'_balanced.json');rr=j['checks']
    assert len(rr)==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in rr)
    inside=rows(label,'INSIDE_EQUAL ');assert len(inside)==78 and all(x['errors']=='0' and x['elements']=='1024' for x in inside)
    meta=rows(label,'SCALE_ELIGIBILITY ');correct=[x for x in meta if int(x['pattern'])>=0]
    assert len(correct)==78 and all(x['eligible']=='640' and x['paired_eligible']=='384' for x in correct)
    init=rows(label,'COOPERATIVE_INIT ');assert len(init)==2
    assert all(x['supported']=='1' and int(x['active_blocks_per_sm'])*int(x['sm_count'])>=288 for x in init)
    if label in ['p36','p37']:
        timing=[x for x in meta if int(x['pattern'])<0];assert len(timing)==2 and all(x['eligible']=='640' for x in timing)
    build=(h/(label+'_build.log')).read_text();assert 'C7515' not in build and 'C7510' not in build
    assert (h/label/'corrected.cuh').read_bytes()==(h/'p33/corrected.cuh').read_bytes()
    candidates[label]=dict(ordinary_checks=78,equal_per_column_exponents=79872,
        full_call_us={s['dtype']:s['corrected_us'] for s in j['summaries']},max_ef={s['dtype']:s['max_ef'] for s in j['summaries']},
        complete_preprocess_diagnostic_us={x['dtype']:float(x['preprocess_us']) for x in j['components']},
        correctness_metadata_groups=78*640,all_correctness_groups_eligible=True,
        timing_metadata='two final pattern=-1 rows explicitly excluded: older standalone probes overwrite workspace' if label=='p35' else 'actual complete prep rerun after all timings; both rows640eligible',
        performance_accepted=False,validation_scope='ordinary78 only; no ABBA/full Graph/fresh stress/scoped tools claim')
a=read('p35_balanced.json')['checks']
for label in ['p36','p37']:
    b=read(label+'_balanced.json')['checks']
    assert all(all(x[k]==y[k] for k in ['dtype','pattern','trial','corrected_ef']) for x,y in zip(a,b))
arithmetic=read('requantized_integer_arithmetic.json');assert arithmetic['checks']==77056 and arithmetic['errors']==0
out=dict(status='ordinary_accuracy_passed_performance_rejected',goal_achieved=False,candidates=candidates,
    p36_p37_all_printed_ef_equal_p35=True,bitwise_output_equality_claimed=False,
    cpu_cap8_scalar_checks=77056,cpu_cap8_errors=0,
    retained='P33 ordinary and P27 Graph, each with their own previously recorded validation; no transfer to P35/P36')
(h/'requantized_hardware.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_requantized_hardware.py','requantized_hardware.json','generate_requantized_hardware.py','generate_vector_requantized.py',
       'p35_manifest.json','p36_manifest.json','p37_manifest.json','p37_scale_diagnostics_manifest.json','generate_skip_requantized.py','profile_p36_manifest.json','p36_scale_diagnostics_manifest.json','m10_manifest.json','requantized_integer_arithmetic_manifest.json']
(h/'requantized_hardware_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

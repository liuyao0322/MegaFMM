"""Freeze M10's numerical screen, keeping failed arms explicit if any."""
import hashlib
import json
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[2]
j=json.loads((h/'requantized_model.json').read_text());rows=j['rows']
assert len(rows)==180 and len({(x['dtype'],x['pattern'],x['seed']) for x in rows})==60
assert {x['seed'] for x in rows}=={5201,5202,5203}
assert (h/'requantized_model.log').read_text().count('CONTROL ')==3 # two CONTROL + one NEGATIVE_CONTROL
assert (h/'requantized_model.log').read_text().count('NEGATIVE_CONTROL ')==1
summary=[]
for dt in ['float16','bfloat16']:
    for arm in ['baseline_k128','exactwide_k512','requantized_k512']:
        r=[x for x in rows if x['dtype']==dt and x['arm']==arm];assert len(r)==30
        assert all(x['inside']=='per_column' and x['max_absolute_integer']<2**24 for x in r)
        summary.append(dict(dtype=dt,arm=arm,configurations=30,passes=sum(x['pass_fixed_floor'] for x in r),
            nonfinite=sum(x['nonfinite'] for x in r),saturated_low_values=sum(x['saturated_low_values'] for x in r),
            max_ef=max(x['ef'] for x in r),max_absolute_integer=max(x['max_absolute_integer'] for x in r)))
        if arm!='requantized_k512':assert all(x['pass_fixed_floor'] for x in r)
out=dict(scope='60fresh matrix configurations,3paired numerical arms; not hardware/synchronization/performance acceptance',
    configurations=60,paired_arm_rows=180,summary=summary,
    double_rounding_negative_control='RN(RN(1.49)/2)=0 vs RN(1.49/2)=1',
    same_input_controls='first uniform seed5201 per dtype: baselineK128 and exactwideK512 exactly match M9 eF and max integer',
    hardware_accepted=False,performance_accepted=False)
(h/'m10_summary.json').write_text(json.dumps(out,indent=2)+'\n')
files=['generate_requantized_model.py','requantized_model.py','requantized_model.json','requantized_model.log','summarize_requantized_model.py','m10_summary.json','REQUANTIZATION_DESIGN.md',
 'wide_quant_model.py','paired_column_model.py','alternating_inside_model.py','global_k128_model.py','../ablate.py']
(h/'m10_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

"""Audit and freeze M13's isolated numerical recombination-order screen."""
import hashlib
import json
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[2]
d=json.loads((h/'leaf_reuse_model.json').read_text());rows=d['rows']
assert len(rows)==144 and len({(x['dtype'],x['pattern'],x['seed']) for x in rows})==72
assert {x['seed'] for x in rows}=={8301,8302,8303}
assert len(d['controls'])==2 and all(x['native_arm_matches_m12'] for x in d['controls'])
assert all(x['pass_fixed_floor'] and x['pass_original_gate'] for x in rows)
summary=[]
for dt in ('float16','bfloat16'):
    byarm={arm:[x for x in rows if x['dtype']==dt and x['arm']==arm] for arm in ('native_low_scaled','native_leaf_scaled')}
    for arm,r in byarm.items():
        assert len(r)==36 and len({(x['pattern'],x['seed']) for x in r})==36
        assert all(x['floor']==(1e-4 if dt=='float16' else 1e-3) for x in r)
        summary.append(dict(dtype=dt,arm=arm,configurations=36,passes=36,
            max_ef=max(x['ef'] for x in r),nonfinite=sum(x['nonfinite'] for x in r),
            max_paired_output_delta_ef=max(x['paired_output_delta_ef'] for x in r)))
out=dict(scope=d['scope'],configurations=72,rows=144,summary=summary,
    same_input_controls=d['controls'],hardware_accepted=False,performance_accepted=False)
(h/'m13_summary.json').write_text(json.dumps(out,indent=2)+'\n')
files=['leaf_reuse_model.py','leaf_reuse_model.json','leaf_reuse_model.log','summarize_leaf_reuse_model.py',
       'm13_summary.json','LEAF_REUSE_DESIGN.md','native_correction_model.py','m12_manifest.json',
       'bf16_correction_model.py','wide_quant_model.py','paired_column_model.py',
       'alternating_inside_model.py','global_k128_model.py','../ablate.py']
(h/'m13_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

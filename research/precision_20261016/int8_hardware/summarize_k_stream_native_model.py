"""Audit/freeze M14; numerical models never confer hardware acceptance."""
import hashlib
import json
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[2]
d=json.loads((h/'k_stream_native_model.json').read_text());r=d['rows']
assert len(r)==288 and len({(x['dtype'],x['pattern'],x['seed']) for x in r})==72
assert {x['seed'] for x in r}=={9301,9302,9303}
assert len(d['controls'])==2 and all(x['full_k128_matches_m13_leaf'] for x in d['controls'])
assert all(x['pass_fixed_floor'] and x['pass_original_gate'] and x['nonfinite']==0 and x['nonfinite_correction_operands']==0 for x in r)
summary=[];paired=[]
for dt in ['float16','bfloat16']:
    for arm in ['full_k128','full_k64','chunk_k64','stream_k64']:
        a=[x for x in r if x['dtype']==dt and x['arm']==arm]
        assert len(a)==36 and len({(x['pattern'],x['seed']) for x in a})==36
        assert all(x['floor']==(1e-4 if dt=='float16' else 1e-3) for x in a)
        worst=max(a,key=lambda x:x['ef'])
        summary.append(dict(dtype=dt,arm=arm,configurations=36,passes=36,nonfinite=0,
            max_ef=worst['ef'],worst_pattern=worst['pattern'],worst_seed=worst['seed'],
            max_output_delta_to_full_k128_ef=max(x['delta_to_full_k128_ef'] for x in a)))
    for arm in ['full_k64','chunk_k64','stream_k64']:
        keys=sorted({(x['pattern'],x['seed']) for x in r if x['dtype']==dt})
        delta=[]
        for pattern,seed in keys:
            by={x['arm']:x for x in r if (x['dtype'],x['pattern'],x['seed'])==(dt,pattern,seed)}
            delta.append(by[arm]['ef']-by['full_k128']['ef'])
        paired.append(dict(dtype=dt,arm=arm,lower_error_cases=sum(x<0 for x in delta),
            equal_error_cases=sum(x==0 for x in delta),higher_error_cases=sum(x>0 for x in delta),
            max_ef_increase=max(delta),min_ef_change=min(delta)))
out=dict(scope=d['scope'],configurations=72,rows=288,summary=summary,paired=paired,
    same_input_controls=d['controls'],hardware_accepted=False,performance_accepted=False)
(h/'m14_summary.json').write_text(json.dumps(out,indent=2)+'\n')
files=['k_stream_native_model.py','k_stream_native_model.json','k_stream_native_model.log',
       'summarize_k_stream_native_model.py','m14_summary.json','FUSED_NATIVE_DESIGN.md',
       'leaf_reuse_model.py','m13_manifest.json','native_correction_model.py','bf16_correction_model.py',
       'wide_quant_model.py','paired_column_model.py','alternating_inside_model.py',
       'global_k128_model.py','../ablate.py']
(h/'m14_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

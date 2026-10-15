"""Summarize frozen F9 diagnostic role timing, preserving same-CTA overlap."""
import csv
import json
import statistics
from pathlib import Path
r=Path(__file__).resolve().parent
rows=[]
for row in csv.DictReader((r/'trace_f9.csv').open()):
    rows.append({k:(v if k=='kind' else int(v)) for k,v in row.items()})
assert len(rows)==128*2*24
per_cta=[]
for cta in range(128):
    roles=[];work=[]
    for role in range(2):
        rr=[x for x in rows if x['cta']==cta and x['role']==role]
        leaves=sorted([x for x in rr if x['kind']=='leaf'],key=lambda x:x['seq'])
        raw=sorted([x for x in rr if x['kind']=='raw'],key=lambda x:x['seq'])
        assert len(leaves)==20 and len(raw)==4
        sums={};ordered=[];intervals=[]
        def collect(row,names):
            for i,name in enumerate(names):
                a,b=row[f't{i}'],row[f't{i+1}'];assert b>=a>0,(cta,role,row,i)
                sums[name]=sums.get(name,0)+(b-a)/1000
            ordered.append((row['t0'],row[f't{len(names)}']))
        for k in range(4):
            x=raw[k]
            collect(x,['raw_free_wait','raw_issue_ready','inside'] if role==0 else ['raw_ready_wait','raw_native','raw_release'])
            intervals.append((x['t2'],x['t3']) if role==0 else (x['t1'],x['t2']))
            for x in leaves[k*5:(k+1)*5]:
                collect(x,['slot_free_wait','pack','publish'] if role==0 else ['slot_ready_wait','native','correction0','correction1','slot_release'])
                intervals.append((x['t1'],x['t2']) if role==0 else (x['t1'],x['t4']))
        gaps=sum(b[0]-a[1] for a,b in zip(ordered,ordered[1:]));assert gaps>=0
        assert all(b[0]>=a[1] for a,b in zip(ordered,ordered[1:]))
        sums['control_gaps']=gaps/1000;sums['total']=(ordered[-1][1]-ordered[0][0])/1000
        assert abs(sum(v for k,v in sums.items() if k!='total')-sums['total'])<1e-6
        roles.append(sums);work.append(intervals)
    overlap=sum(max(0,min(a1,b1)-max(a0,b0)) for a0,a1 in work[0] for b0,b1 in work[1])/1000
    per_cta.append(dict(cta=cta,producer=roles[0],consumer=roles[1],useful_role_overlap_us=overlap))
out=dict(scope='Frozen F9, one uniform BF16 input. Globaltimer stores and extra registers perturb scheduling. Sum intervals within each CTA, then report medians. Role durations overlap and must not be added. Instrumented arithmetic checked against original-input FP64; not a full precision or complete-call performance acceptance.',
    median_us={role:{k:statistics.median(x[role][k] for x in per_cta) for k in per_cta[0][role]} for role in ['producer','consumer']},
    median_overlap_us=statistics.median(x['useful_role_overlap_us'] for x in per_cta),per_cta=per_cta)
(r/'trace_f9_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='per_cta'},indent=2))

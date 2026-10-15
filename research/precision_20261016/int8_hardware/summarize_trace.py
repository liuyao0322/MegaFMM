"""Sum phases within each CTA first; do not add overlapping role durations."""
import csv
import json
import statistics
from pathlib import Path

r=Path(__file__).resolve().parent
rows=[{k:int(v) for k,v in row.items()} for row in csv.DictReader((r/'trace.csv').open())]
assert len(rows)==128*60
names=['operand_ready','prepare_issue','explicit_wait','scale','free']
per_cta=[]
for cta in range(128):
    rr=[x for x in rows if x['cta']==cta]
    assert len(rr)==60
    assert all(x[f't{i+1}']>=x[f't{i}'] for x in rr for i in range(5))
    values={name:sum(x[f't{i+1}']-x[f't{i}'] for x in rr)/1000 for i,name in enumerate(names)}
    values['between_stages']=sum(b['t0']-a['t5'] for a,b in zip(rr,rr[1:]))/1000
    values['total']=(rr[-1]['t5']-rr[0]['t0'])/1000
    assert abs(sum(v for k,v in values.items() if k!='total')-values['total'])<1e-6
    per_cta.append(dict(cta=cta,**values))
out=dict(scope='Instrumented BF16 uniform consumer. Median of same-CTA sums. Prepare/issue includes overlapped asynchronous execution; explicit wait is only its exposed tail. Diagnostic stores and extra registers perturb scheduling. Not complete-call timing.',
         median_us={k:statistics.median(x[k] for x in per_cta) for k in per_cta[0] if k!='cta'},per_cta=per_cta)
(r/'trace_summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out['median_us'],indent=2))

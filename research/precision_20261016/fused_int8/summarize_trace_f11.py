"""First sum subphase costs within each CTA; medians are descriptive, not additive."""
import csv,json,statistics
from pathlib import Path
r=Path(__file__).resolve().parent
rows=[]
for row in csv.DictReader((r/'trace_f11.csv').open()):rows.append({k:(v if k=='kind' else int(v)) for k,v in row.items()})
assert len(rows)==128*(48+20)
names=['native_a_pack','native_b_pack','maximum_reduce','exponent_reduce','a_quantize','b_quantize','publish_sync']
per_cta=[]
for cta in range(128):
    pp=sorted([x for x in rows if x['cta']==cta and x['kind']=='pack'],key=lambda x:x['seq']);assert len(pp)==20
    assert all(x[f't{i+1}']>=x[f't{i}']>0 for x in pp for i in range(7))
    phases={name:sum(x[f't{i+1}']-x[f't{i}'] for x in pp)/1000 for i,name in enumerate(names)}
    phases['pack_total']=sum(x['t7']-x['t0'] for x in pp)/1000
    assert abs(sum(phases[n] for n in names)-phases['pack_total'])<1e-6
    producer=[x for x in rows if x['cta']==cta and x['kind']=='leaf' and x['role']==0]
    phases['producer_pack_envelope']=sum(x['t2']-x['t1'] for x in producer)/1000
    per_cta.append(dict(cta=cta,**phases))
out=dict(scope='FrozenF11, one uniformBF16 input. Eight stamps per pack; stores/register allocation perturb scheduling. Role traces are overlapping. No full-call timing or general numerical verdict.',median_us={k:statistics.median(x[k] for x in per_cta) for k in per_cta[0] if k!='cta'},per_cta=per_cta)
(r/'trace_f11_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['median_us'],indent=2))

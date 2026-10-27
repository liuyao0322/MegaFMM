from pathlib import Path
import csv,statistics,json,collections,sys
root=Path(sys.argv[1] if len(sys.argv)>1 else 'results/production_chain')
rows=list(csv.DictReader((root/'records.csv').open()))
groups=collections.defaultdict(list)
for r in rows:
 k=(r['dtype'],int(r['role']),int(r['sample']))
 ts=[int(r[f't{i}']) for i in range(4)]
 groups[k].append([ts[i+1]-ts[i] for i in range(3)])
summary={}
for dtype in ['bf16','fp16']:
 print('\nDTYPE',dtype);summary[dtype]={}
 for role in range(5):
  ss=sorted(s for d,ro,s in groups if d==dtype and ro==role)
  per=[]
  for s in ss:
   vs=groups[dtype,role,s]
   per.append([statistics.median(v[j] for v in vs)/1000 for j in range(3)])
  summary[dtype][str(role)]={'per_stage_median_us':per,'sum_stage_medians_us':[sum(x[j] for x in per) for j in range(3)]}
  print('ROLE',role,'SUM',summary[dtype][str(role)]['sum_stage_medians_us'])
  if role<2:
   for s,p in zip(ss,per):print(s,[round(x,3) for x in p])
 timing=[r for r in csv.DictReader((root/'timing.csv').open()) if r['dtype']==dtype]
 baseline=statistics.median(float(r['event_us']) for r in timing if r['role']=='-1')
 print('baseline',baseline)
 for role in range(5):
  ts=[float(r['event_us']) for r in timing if int(r['role'])==role]
  print('time role',role,'median',statistics.median(ts),'min',min(ts),'max',max(ts),'ratio',statistics.median(ts)/baseline)
 summary[dtype]['baseline_us']=baseline
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

aggregate_path=root/'aggregate'/'records.csv'
if not aggregate_path.exists():aggregate_path=root/'records.csv'
aggregate_rows=[r for r in csv.DictReader(aggregate_path.open()) if int(r['role'])==5]
aggregate_summary={}
for dtype in ['bf16','fp16']:
 data=[[int(r['t0']),int(r['t1']),int(r['t2']),int(r['t3'])] for r in aggregate_rows if r['dtype']==dtype]
 if not data:continue
 result={}
 for j,key in enumerate(['operand_wait','prepare_issue','explicit_wait0','producer_elapsed']):
  vs=sorted(r[j]/1000 for r in data)
  result[key]={'p10_us':vs[int(.1*(len(vs)-1))],'median_us':statistics.median(vs),'p90_us':vs[int(.9*(len(vs)-1))]}
  if j<3:result[key]['median_fraction_of_producer']=statistics.median(r[j]/r[3] for r in data)
 result['residual_median_us']=statistics.median((r[3]-sum(r[:3]))/1000 for r in data)
 result['samples']=len(data)
 aggregate_summary[dtype]=result
(root/'aggregate_summary.json').write_text(json.dumps(aggregate_summary,indent=2)+'\n')
print('Aggregate:',json.dumps(aggregate_summary,indent=2))

"""Full-call alternating comparison; unchanged original FP64 gates remain failing."""
from pathlib import Path
import csv,json,subprocess,statistics,sys
base=Path('research/falcon_layout_fusion')
names=sys.argv[1:] or ['iter331','iter332','iter333','iter335']
out=base/('comparison_'+'_'.join(names) if sys.argv[1:] else 'comparison');out.mkdir(exist_ok=True)
records=[]
for mode in ['launch','graph']:
 for i,name in enumerate(names+names[::-1]):
  stem=f'{mode}_{name}_{i}';path=out/(stem+'.csv')
  cmd=[str(base/(name+'_bench')),'--algorithm','sw','--shapes','1024,1024,1024','--dtype','both','--rotate-inputs','1','--trials','100','--warmup','20','--groups','5','--correct-trials','3','--output',str(path)]
  if mode=='graph':cmd.append('--graph')
  with (out/(stem+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  assert r.returncode==2,(stem,r.returncode)
  rows=list(csv.DictReader(path.open()));assert len(rows)==2
  for row in rows:
   assert row['nonfinite']=='0' and row['numerical_pass']=='0'
   assert .005< float(row['solution_ef']) < .007 if row['dtype']=='bf16' else .0007<float(row['solution_ef'])<.0008
   records.append(dict(mode=mode,version=name,dtype=row['dtype'],sw_us=float(row['solution_event_ms'])*1000,dense_us=float(row['dense_event_ms'])*1000,ef=float(row['solution_ef']),limit=float(row['gate_limit'])))
  print(stem,'done',flush=True)
summary=[]
for mode in ['launch','graph']:
 for dt in ['bf16','fp16']:
  values={n:statistics.median(x['sw_us'] for x in records if x['mode']==mode and x['dtype']==dt and x['version']==n) for n in names}
  gains={n:100*(values['iter331']-values[n])/values['iter331'] for n in names[1:]}
  row=dict(mode=mode,dtype=dt,latencies_us=values,latency_reduction_percent=gains)
  summary.append(row);print(row,flush=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'records.json').write_text(json.dumps(records,indent=2)+'\n')

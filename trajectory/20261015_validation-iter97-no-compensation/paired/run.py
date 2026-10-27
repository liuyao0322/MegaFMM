import subprocess,csv,json,statistics
from pathlib import Path
base=Path('results/compare_iter97')
versions={'iter96':str(base/'iter96_bench'),'iter97':'./build/megafmm_bench'}
summary=[]
for mode in ('launch','graph'):
 for version,repeat in [('iter96',1),('iter97',1),('iter97',2),('iter96',2)]:
  stem=f'{mode}_{version}_{repeat}'
  args=[versions[version],'--algorithm','sw','--shapes','1024,1024,1024','--dtype','both','--trials','100','--warmup','20','--groups','5','--correct-trials','3','--output',str(base/(stem+'.csv'))]
  if mode=='graph':args.append('--graph')
  with (base/(stem+'.log')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
  assert r.returncode==(0 if version=='iter96' else 2),(stem,r.returncode)
  rows=list(csv.DictReader((base/(stem+'.csv')).open()));assert len(rows)==2
  for row in rows:
   assert row['nonfinite']=='0'
   assert row['numerical_pass']==('1' if version=='iter96' else '0')
   entry=dict(mode=mode,version=version,repeat=repeat,dtype=row['dtype'],sw_us=float(row['solution_event_ms'])*1000,dense_us=float(row['dense_event_ms'])*1000,ef=float(row['solution_ef']),limit=float(row['gate_limit']),pass_gate=row['numerical_pass'])
   summary.append(entry);print(entry,flush=True)
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for mode in ('launch','graph'):
 for dtype in ('bf16','fp16'):
  a,b=[statistics.median(x['sw_us'] for x in summary if x['mode']==mode and x['dtype']==dtype and x['version']==v) for v in ('iter96','iter97')]
  print(f'COMPARISON {mode} {dtype} compensated={a:.5f} seven={b:.5f} reduction={100*(1-b/a):.3f}% speedup={a/b:.4f}x',flush=True)

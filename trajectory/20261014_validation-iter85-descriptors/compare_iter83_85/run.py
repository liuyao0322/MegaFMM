import subprocess,csv,json
from pathlib import Path
base=Path('results/compare_iter83_85')
versions={'iter83':'results/compare_iter83_85/build/megafmm_bench','iter85':'./build/megafmm_bench'}
cases=[('launch','iter83',1),('launch','iter85',1),('launch','iter85',2),('launch','iter83',2),('graph','iter83',1),('graph','iter85',1),('graph','iter85',2),('graph','iter83',2)]
summary=[]
for mode,version,repeat in cases:
 stem=f'{mode}_{version}_{repeat}'
 args=[versions[version],'--algorithm','sw','--shapes','1024,1024,1024','--dtype','both','--trials','100','--warmup','20','--groups','5','--correct-trials','3','--output',str(base/(stem+'.csv'))]
 if mode=='graph':args.append('--graph')
 with (base/(stem+'.log')).open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
 for row in csv.DictReader((base/(stem+'.csv')).open()):
  assert row['numerical_pass']=='1'
  result=dict(mode=mode,version=version,repeat=repeat,dtype=row['dtype'],sw_us=float(row['solution_event_ms'])*1000,dense_us=float(row['dense_event_ms'])*1000)
  summary.append(result);print(result,flush=True)
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

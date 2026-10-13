from pathlib import Path
import subprocess,json,sys
runs=[(f'psplit_bn16_cpu_{k}_model_iter39',['python3','results/psplit_bn16_cpu_model_runner_iter39.py',k]) for k in ['raw','ss','tail']]
for old,new in [('results/psplit_hybrid_s_cpu_dag_iter38.txt','psplit_bn16_cpu_dag_iter39'),('results/psplit_hybrid_s_cpu_stage_iter38.txt','psplit_bn16_cpu_stage_iter39')]:
 first=Path(old).read_text().splitlines()[0];assert first.startswith('COMMAND ');runs.append((new,json.loads(first[8:])))
for name,cmd in runs:
 p=Path('results',name+'.txt');assert not p.exists();print('START',name,flush=True)
 with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',name,code,flush=True)
 if code:sys.exit(code)

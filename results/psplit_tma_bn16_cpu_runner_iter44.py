from pathlib import Path
import subprocess,json,sys
runs=[('psplit_tma_bn16_cpu_dag_iter44',['python3','tests/test_dag.py']),('psplit_tma_bn16_cpu_stage_iter44',['python3','tests/test_stage_m2.py']),('psplit_tma_bn16_cpu_state_model_iter44',['python3','results/psplit_tma_bn16_cpu_state_model_iter44.py'])]
for name,cmd in runs:
 p=Path('results')/(name+'.txt');assert not p.exists();print('START',name,flush=True)
 with p.open('w') as f: f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',name,code,flush=True)
 if code:sys.exit(code)

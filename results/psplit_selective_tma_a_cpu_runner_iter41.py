from pathlib import Path
import subprocess,json,sys
runs=[('psplit_selective_tma_a_cpu_dag_iter41',['python3','tests/test_dag.py']),('psplit_selective_tma_a_cpu_stage_iter41',['python3','tests/test_stage_m2.py']),('psplit_selective_tma_a_cpu_state_model_iter41',['python3','results/psplit_selective_tma_a_cpu_state_model_iter41.py'])]
for name,cmd in runs:
 p=Path('results')/(name+'.txt');assert not p.exists();print('START',name,flush=True)
 with p.open('w') as f: f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',name,code,flush=True)
 if code:sys.exit(code)

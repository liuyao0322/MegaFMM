from pathlib import Path
import subprocess,json,sys
runs=[('psplit_sw128_simt_control_cpu_dag_iter42',['python3','tests/test_dag.py']),('psplit_sw128_simt_control_cpu_stage_iter42',['python3','tests/test_stage_m2.py']),('psplit_sw128_simt_control_cpu_state_model_iter42',['python3','results/psplit_sw128_simt_control_cpu_state_model_iter42.py'])]
for name,cmd in runs:
 p=Path('results')/(name+'.txt');assert not p.exists();print('START',name,flush=True)
 with p.open('w') as f: f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',name,code,flush=True)
 if code:sys.exit(code)

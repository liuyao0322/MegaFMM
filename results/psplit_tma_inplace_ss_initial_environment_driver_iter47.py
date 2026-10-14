from pathlib import Path
import subprocess,json,sys
runs=[('psplit_tma_inplace_ss_initial_environment_iter47', ['bash', 'scripts/on_gpu.sh', 'python3', 'results/psplit_tma_inplace_ss_environment_snapshot_runner_iter47.py'], {0})]
for name,cmd,ok in runs:
 p=Path('results',name+'.txt');assert not p.exists();print('START',name,flush=True)
 with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',name,code,flush=True)
 if code not in ok:sys.exit(code or 98)
print('ALL47_CLOSED',flush=True)

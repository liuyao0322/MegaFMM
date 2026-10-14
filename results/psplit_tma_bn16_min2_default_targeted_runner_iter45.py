from pathlib import Path
import subprocess,json,sys
runs=[('correctness_psplit_tma_bn16_min2_iter45', ['bash', 'scripts/on_gpu.sh', './build/test_correctness']), ('correctness_psplit_tma_bn16_min249_targeted_iter45', ['bash', 'scripts/on_gpu.sh', './build/test_correctness', '--psplit-tma-bn16-min2-variant49-only'])]
for name,cmd in runs:
 p=Path('results',name+'.txt');assert not p.exists();print('START',name,flush=True)
 with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',name,code,flush=True)
 if code:sys.exit(code)

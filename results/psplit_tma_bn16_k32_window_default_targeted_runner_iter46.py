from pathlib import Path
import subprocess,json,sys
runs=[('correctness_psplit_tma_bn16_k32_window_iter46', ['bash', 'scripts/on_gpu.sh', './build/test_correctness']), ('correctness_psplit_tma_bn16_k32_window50_targeted_iter46', ['bash', 'scripts/on_gpu.sh', './build/test_correctness', '--psplit-tma-bn16-k32-window-variant50-only'])]
for name,cmd in runs:
 p=Path('results',name+'.txt');assert not p.exists();print('START',name,flush=True)
 with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',name,code,flush=True)
 if code:sys.exit(code)

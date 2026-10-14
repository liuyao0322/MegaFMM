from pathlib import Path
import subprocess,json,sys
runs=[('correctness_psplit_tma_inplace_ss_iter47', ['bash', 'scripts/on_gpu.sh', './build/test_correctness'], {0}), ('correctness_psplit_tma_inplace_ss51_targeted_iter47', ['bash', 'scripts/on_gpu.sh', './build/test_correctness', '--psplit-tma-inplace-ss-variant51-only'], {0}), ('correctness_psplit_tma_inplace_ss52_targeted_iter47', ['bash', 'scripts/on_gpu.sh', './build/test_correctness', '--psplit-tma-inplace-ss-variant52-only'], {0})]
for name,cmd,ok in runs:
 p=Path('results',name+'.txt');assert not p.exists();print('START',name,flush=True)
 with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',name,code,flush=True)
 if code not in ok:sys.exit(code or 98)
print('ALL47_CLOSED',flush=True)

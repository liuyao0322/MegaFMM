from pathlib import Path
import subprocess,json,sys
runs=[('correctness_psplit_tma_bn64_iter43',['bash','scripts/on_gpu.sh','./build/test_correctness']),('correctness_psplit_tma_bn6447_targeted_iter43',['bash','scripts/on_gpu.sh','./build/test_correctness','--smoke-candidates','--mega-only','--mega-variant','47'])]
for name,cmd in runs:
 p=Path('results',name+'.txt');assert not p.exists();print('START',name,flush=True)
 with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',name,code,flush=True)
 if code:sys.exit(code)

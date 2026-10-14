from pathlib import Path
import subprocess,json,sys
runs=[('psplit_tma_inplace_ss_final_test_build_iter47',['bash','scripts/on_gpu.sh','cmake','--build','build','--target','test_correctness','-j4'])]
for name,cmd in runs:
 p=Path('results',name+'.txt');assert not p.exists();print('START',name,flush=True)
 with p.open('w') as f:
  f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',name,code,flush=True)
 if code:sys.exit(code)

from pathlib import Path
import subprocess,sys,json
runs=[('correctness_psplit_hybrid_s_iter38',[]),('correctness_psplit_hybrid_s41_targeted_iter38',['--smoke-candidates','--mega-only','--mega-variant','41'])]
for name,args in runs:
 p=Path('results',name+'.txt');assert not p.exists(),p
 cmd=['bash','scripts/on_gpu.sh','./build/test_correctness']+args
 print('START',name,json.dumps(cmd),flush=True)
 with p.open('w') as f:
  f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush()
  code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n')
 print('DONE',name,'exit',code,flush=True)
 if code:sys.exit(code)

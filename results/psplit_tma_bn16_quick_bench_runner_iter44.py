from pathlib import Path
import subprocess,os,json
p=Path('results/psplit_tma_bn16_bench_iter44.txt');assert not p.exists()
env=os.environ.copy();env.update(MEGAFMM_ALGORITHM='mega',MEGAFMM_SHAPES='1024,1024,1024',MEGAFMM_DTYPE='both',MEGAFMM_MEGA_VARIANT='48',MEGAFMM_TRIALS='20',MEGAFMM_WARMUP='20',MEGAFMM_GROUPS='5',MEGAFMM_CORRECT_TRIALS='1',MEGAFMM_ROTATE_INPUTS='1',MEGAFMM_GRAPH='0')
cmd=['bash','scripts/bench.sh','iter-44']
with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.write('MEGAFMM_ENV '+json.dumps({k:v for k,v in env.items() if k.startswith('MEGAFMM_')},sort_keys=True)+'\n');f.flush();code=subprocess.call(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
p.with_suffix('.exit').write_text(str(code)+'\n');print('AKO44_CLOSED',code,flush=True);raise SystemExit(code)

from pathlib import Path
import subprocess,os,json
p=Path('results/psplit_tma_bn64_bench_iter43.txt');assert not p.exists()
e=os.environ.copy();e.update({'MEGAFMM_GRAPH':'0','MEGAFMM_DTYPE':'both','MEGAFMM_ALGORITHM':'mega','MEGAFMM_MEGA_VARIANT':'47','MEGAFMM_SHAPES':'1024,1024,1024','MEGAFMM_TRIALS':'20','MEGAFMM_WARMUP':'20','MEGAFMM_GROUPS':'5','MEGAFMM_CORRECT_TRIALS':'1','MEGAFMM_ROTATE_INPUTS':'1'})
cmd=['bash','scripts/bench.sh','iter-43']
with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();c=subprocess.call(cmd,env=e,stdout=f,stderr=subprocess.STDOUT)
p.with_suffix('.exit').write_text(str(c)+'\n');print('AKO_QUICK43_CLOSED',c,flush=True);raise SystemExit(c)

from pathlib import Path
import os,subprocess,json
p=Path('results/psplit_sw128_simt_control_bench_iter42')
env=os.environ.copy();env.update({'MEGAFMM_ALGORITHM':'mega','MEGAFMM_MEGA_VARIANT':'46','MEGAFMM_SHAPES':json.loads(Path('bench/small_shapes.json').read_text())['cli_shapes'],'MEGAFMM_DTYPE':'both','MEGAFMM_TRIALS':'20','MEGAFMM_WARMUP':'20','MEGAFMM_GROUPS':'5','MEGAFMM_CORRECT_TRIALS':'1','MEGAFMM_ROTATE_INPUTS':'1','MEGAFMM_GRAPH':'0'})
with p.with_suffix('.txt').open('w') as out:
 r=subprocess.run(['bash','scripts/bench.sh','iter-42'],env=env,stdout=out,stderr=subprocess.STDOUT)
p.with_suffix('.exit').write_text(str(r.returncode)+'\n');print('AKO_ITER42_COMPLETED',r.returncode)
raise SystemExit(r.returncode)

from pathlib import Path
import subprocess,json
p=Path('results/psplit_tma_bn16_revision1_build_iter44.txt');assert not p.exists()
cmd=['bash','scripts/on_gpu.sh','cmake','--build','build','--target','megafmm_bench','-j4']
with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();c=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
p.with_suffix('.exit').write_text(str(c)+'\n');print('REVISION1_BUILD43_CLOSED',c,flush=True);raise SystemExit(c)

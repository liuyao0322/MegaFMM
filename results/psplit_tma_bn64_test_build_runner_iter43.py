from pathlib import Path
import subprocess,json
p=Path('results/psplit_tma_bn64_test_build_iter43.txt');assert not p.exists()
cmd=['bash','scripts/on_gpu.sh','cmake','--build','build','--target','test_correctness','-j4']
with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();c=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
p.with_suffix('.exit').write_text(str(c)+'\n');print('TEST_BUILD43_CLOSED',c,flush=True);raise SystemExit(c)

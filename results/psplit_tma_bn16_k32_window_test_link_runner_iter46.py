from pathlib import Path
import subprocess,json
p=Path('results/psplit_tma_bn16_k32_window_test_link_iter46.txt');assert not p.exists()
cmd=['bash','scripts/on_gpu.sh','cmake','--build','build','--target','test_correctness','-j4']
with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
p.with_suffix('.exit').write_text(str(code)+'\n');print('TEST_LINK44_CLOSED',code,flush=True);raise SystemExit(code)

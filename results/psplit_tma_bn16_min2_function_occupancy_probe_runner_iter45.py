from pathlib import Path
import json,subprocess
p=Path('results/psplit_tma_bn16_min2_function_occupancy_probe_iter45.txt');assert not p.exists()
cmd=['bash','scripts/on_gpu.sh','python3','results/psplit_tma_bn16_min2_function_occupancy_probe_iter45.py']
print('START_READONLY_NO_KERNEL',flush=True)
with p.open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',code,flush=True);raise SystemExit(code)

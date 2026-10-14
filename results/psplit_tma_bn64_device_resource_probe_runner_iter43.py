from pathlib import Path
import subprocess,json
p=Path('results/psplit_tma_bn64_device_resource_probe_iter43.txt');assert not p.exists()
cmd=['bash','scripts/on_gpu.sh','python3','results/psplit_tma_bn64_device_resource_probe_iter43.py']
with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();c=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
p.with_suffix('.exit').write_text(str(c)+'\n');print('DEVICE_RESOURCE_PROBE_CLOSED',c);raise SystemExit(c)

from pathlib import Path
import subprocess,json
p=Path('results/psplit_sw128_simt_control_validation_driver_iter42.txt');assert not p.exists()
cmd=['python3','results/psplit_sw128_simt_control_validation_runner_iter42.py']
with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
p.with_suffix('.exit').write_text(str(code)+'\n');print('VALIDATION_DRIVER42_CLOSED',code,flush=True);raise SystemExit(code)

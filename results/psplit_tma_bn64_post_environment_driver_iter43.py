from pathlib import Path
import subprocess,json,sys
p=Path('results/small_psplit_tma_bn64_environment_iter43.txt');assert not p.exists()
cmd=['bash','scripts/on_gpu.sh','python3','results/psplit_tma_bn64_environment_runner_iter43.py']
with p.open('w') as f:
 f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
p.with_suffix('.exit').write_text(str(code)+'\n')
print('POST_ENVIRONMENT43_CLOSED',code)
sys.exit(code)

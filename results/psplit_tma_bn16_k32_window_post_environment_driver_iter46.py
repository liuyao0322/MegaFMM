from pathlib import Path
import subprocess,json,sys
p=Path('results/small_psplit_tma_bn16_k32_window_environment_iter46.txt');assert not p.exists()
cmd=['bash','scripts/on_gpu.sh','python3','results/psplit_tma_bn16_k32_window_environment_runner_iter46.py']
with p.open('w') as f:
 f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
p.with_suffix('.exit').write_text(str(code)+'\n')
print('POST_ENVIRONMENT44_CLOSED',code)
sys.exit(code)

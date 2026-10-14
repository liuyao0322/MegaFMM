from pathlib import Path
import subprocess,sys
for stem in ['full','supplement']:
 cmd=['python',f'results/psplit_tma_bn16_k32_window_{stem}_runner_iter46.py']
 code=subprocess.call(cmd)
 if code:sys.exit(code)
print('PRIMARY20_PLUS_SUPPLEMENT8_ALL_CLOSED',flush=True)

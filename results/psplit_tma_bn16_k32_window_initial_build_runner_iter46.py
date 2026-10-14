from pathlib import Path
import subprocess,json,sys
runs=[('psplit_tma_bn16_k32_window_initial_allocation_iter46',['scontrol','show','job','107488']),('psplit_tma_bn16_k32_window_initial_environment_iter46',['bash','scripts/on_gpu.sh','python3','results/psplit_tma_bn16_k32_window_initial_environment_runner_iter46.py']),('psplit_tma_bn16_k32_window_initial_build_iter46',['bash','scripts/on_gpu.sh','bash','-c','cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target megafmm_bench -j4'])]
for name,cmd in runs:
 p=Path('results',name+'.txt');assert not p.exists();print('START',name,flush=True)
 with p.open('w') as f:
  f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',name,code,flush=True)
 if code:sys.exit(code)

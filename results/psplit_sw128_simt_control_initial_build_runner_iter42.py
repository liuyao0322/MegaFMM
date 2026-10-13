from pathlib import Path
import subprocess,json,sys
runs=[('psplit_sw128_simt_control_initial_allocation_iter42',['scontrol','show','job','107488']),('psplit_sw128_simt_control_initial_environment_iter42',['bash','scripts/on_gpu.sh','python3','results/psplit_sw128_simt_control_initial_environment_runner_iter42.py']),('psplit_sw128_simt_control_initial_build_iter42',['bash','scripts/on_gpu.sh','bash','-c','cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target megafmm_bench -j4'])]
for name,cmd in runs:
 p=Path('results',name+'.txt');assert not p.exists();print('START',name,flush=True)
 with p.open('w') as f:
  f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',name,code,flush=True)
 if code:sys.exit(code)

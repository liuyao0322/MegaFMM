from pathlib import Path
import subprocess,json,sys
p=Path('results/psplit_bn16_initial_environment_runner_iter39.py');src=Path('results/psplit_hybrid_s_initial_environment_runner_iter38.py');assert not p.exists();p.write_text(src.read_text().replace('iteration38','iteration39').replace('new41','new42/43'))
runs=[('psplit_bn16_initial_allocation_iter39',['scontrol','show','job','107488']),('psplit_bn16_initial_environment_iter39',['bash','scripts/on_gpu.sh','python3',str(p)]),('psplit_bn16_initial_build_iter39',['bash','scripts/on_gpu.sh','bash','-c','cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target megafmm_bench -j4'])]
for name,cmd in runs:
 p=Path('results',name+'.txt');assert not p.exists();print('START',name,flush=True)
 with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',name,code,flush=True)
 if code:sys.exit(code)

from pathlib import Path
import subprocess,json,sys
root='psplit_tma_bn16_k32_window'
runs=[('strict_'+root+'_iter46',['bash','scripts/on_gpu.sh','./build/test_correctness','--strict-psplit-tma-bn16-k32-window-enabled-only'],{2}),('disabled_'+root+'37_cli_iter46',['bash','scripts/on_gpu.sh','./build/test_correctness','--smoke-candidates','--mega-only','--mega-variant','37'],{1})]
for tool in ['memcheck','racecheck','synccheck']:
 cmd=['bash','scripts/on_gpu.sh','/share/app/cuda/cuda-12.8/compute-sanitizer/compute-sanitizer','--tool',tool,'--error-exitcode','97']
 if tool=='memcheck':cmd+=['--leak-check','full']
 cmd+=['./build/test_correctness','--psplit-tma-bn16-k32-window-variant50-only']
 runs.append(('sanitizer_'+root+'50_' +tool+'_iter46',cmd,{0}))
for name,cmd,expected in runs:
 p=Path('results',name+'.txt');assert not p.exists();print('START',name,flush=True)
 with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE',name,code,flush=True)
 if code not in expected:sys.exit(code or 98)
print('VALIDATION46_ALL_CLOSED',flush=True)

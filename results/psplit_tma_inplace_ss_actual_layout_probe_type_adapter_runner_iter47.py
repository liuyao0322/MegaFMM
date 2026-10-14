from pathlib import Path
import subprocess,json
root='psplit_tma_inplace_ss_actual_layout_probe_type_adapter'
runs=[(root+'_build_iter47',['g++','-x','c++','-std=c++17','-O2','-Iinclude','-Ithird_party/cutlass/include','-I/share/app/cuda/cuda-12.4/include','results/'+root+'_iter47.cu','-o','.scratch/'+root+'_iter47']), (root+'_execution_iter47',['.scratch/'+root+'_iter47'])]
for name,cmd in runs:
 p=Path('results',name+'.txt');assert not p.exists()
 with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 p.with_suffix('.exit').write_text(str(code)+'\n');print(name,code,flush=True)
 if code:raise SystemExit(code)

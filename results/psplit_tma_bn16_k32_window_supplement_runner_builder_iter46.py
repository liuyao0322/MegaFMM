from pathlib import Path
import json,hashlib
pre='psplit_tma_bn16_k32_window';pp=f'results/{pre}_input_placement_supplement_plan_iter46.json';ip=f'results/{pre}_measured_identity_iter46.json';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();plan=json.loads(Path(pp).read_text());ident=json.loads(Path(ip).read_text());assert ident['supplemental_predeclared_plan']=={'path':pp,'sha256':sha(pp)}
runs=[]
for route in plan['routes']:
 for mode in plan['modes']:
  path=f"results/small_{route['label']}_{mode}_iter46_full1024.csv";assert path in plan['csv_paths'];cmd=['bash','scripts/on_gpu.sh','./build/megafmm_bench','--algorithm','hopper','--hopper-variant',str(route['variant']),'--shapes','1024,1024,1024','--dtype','both','--trials','100','--warmup','20','--groups','5','--correct-trials','3','--rotate-inputs','1','--output',path]
  if mode=='graph':cmd.append('--graph')
  runs.append((path,cmd))
code='''from pathlib import Path
import json,hashlib,subprocess,sys
'''+f'PLAN_PATH={pp!r}\nIDENTITY_PATH={ip!r}\nPLAN_SHA={sha(pp)!r}\nIDENTITY_SHA={sha(ip)!r}\nRUNS={runs!r}\n'+'''sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert sha(PLAN_PATH)==PLAN_SHA and sha(IDENTITY_PATH)==IDENTITY_SHA
 plan=json.loads(Path(PLAN_PATH).read_text());ident=json.loads(Path(IDENTITY_PATH).read_text())
 assert ident['supplemental_predeclared_plan']=={'path':PLAN_PATH,'sha256':PLAN_SHA}
 for p,w in {**plan['frozen_sources'],**plan['frozen_binaries']}.items():assert sha(p)==w,p
 for path in plan['csv_paths']:
  for suffix in ('.csv','.log','.exit'):assert not Path(path).with_suffix(suffix).exists()
 inner=[]
 for path,cmd in RUNS:
  p=Path(path);print('START_SUPPLEMENT',path,flush=True)
  with p.with_suffix('.log').open('x') as f:
   f.write('COMMAND '+json.dumps(cmd)+'\\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
  p.with_suffix('.exit').write_text(str(code)+'\\n');inner.append(code);print('DONE_SUPPLEMENT',path,'exit',code,flush=True)
  if code not in (0,2):return code
 print('SUPPLEMENT8_ALL_TERMINAL',inner,flush=True);return 0
if __name__=='__main__':sys.exit(main())
'''
p=Path(f'results/{pre}_supplement_runner_iter46.py');assert not p.exists();p.write_text(code)
p=Path(f'results/{pre}_all_full_driver_iter46.py');assert not p.exists();p.write_text("from pathlib import Path\nimport subprocess,sys\nfor stem in ['full','supplement']:\n cmd=['python',f'results/psplit_tma_bn16_k32_window_{stem}_runner_iter46.py']\n code=subprocess.call(cmd)\n if code:sys.exit(code)\nprint('PRIMARY20_PLUS_SUPPLEMENT8_ALL_CLOSED',flush=True)\n")
print('SUPPLEMENT_RUNNER_PREDECLARED',sha(pp),sha(ip))

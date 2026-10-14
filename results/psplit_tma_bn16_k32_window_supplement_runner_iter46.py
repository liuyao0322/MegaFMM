from pathlib import Path
import json,hashlib,subprocess,sys
PLAN_PATH='results/psplit_tma_bn16_k32_window_input_placement_supplement_plan_iter46.json'
IDENTITY_PATH='results/psplit_tma_bn16_k32_window_measured_identity_iter46.json'
PLAN_SHA='0e5f1eb3a48f70c699294613141cadb6f2650e909a8809816dbb465035a37f88'
IDENTITY_SHA='a4afcdd2f93c7248174bb895a5ac1588ace047e2f7496f3989d93e5a1cd6a4f8'
RUNS=[('results/small_hopper_fused10_launch_iter46_full1024.csv', ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'hopper', '--hopper-variant', '10', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_hopper_fused10_launch_iter46_full1024.csv']), ('results/small_hopper_fused10_graph_iter46_full1024.csv', ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'hopper', '--hopper-variant', '10', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_hopper_fused10_graph_iter46_full1024.csv', '--graph']), ('results/small_hopper_st_once12_launch_iter46_full1024.csv', ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'hopper', '--hopper-variant', '12', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_hopper_st_once12_launch_iter46_full1024.csv']), ('results/small_hopper_st_once12_graph_iter46_full1024.csv', ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'hopper', '--hopper-variant', '12', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_hopper_st_once12_graph_iter46_full1024.csv', '--graph'])]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
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
   f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
  p.with_suffix('.exit').write_text(str(code)+'\n');inner.append(code);print('DONE_SUPPLEMENT',path,'exit',code,flush=True)
  if code not in (0,2):return code
 print('SUPPLEMENT8_ALL_TERMINAL',inner,flush=True);return 0
if __name__=='__main__':sys.exit(main())

from pathlib import Path
import csv,json,subprocess,hashlib,shutil,gzip,re
root=Path('trajectory/20261015_optimization-goal-checkpoint-248');root.mkdir(exist_ok=True)
manifest={}
def copy(p,d=None):
 p=Path(p);d=root/(d or str(p).removeprefix('results/opt_goal/'))
 d.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.csv' and p.stat().st_size>600000:
  d=d.with_suffix(d.suffix+'.gz');d.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
 else:shutil.copyfile(p,d)
 manifest[str(d.relative_to(root))]={'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sha256':hashlib.sha256(d.read_bytes()).hexdigest()}
for name in ['check224','profile218','reassess228','trace235','profile230','profile234','profile238','profile242']:
 for p in Path('results/opt_goal',name).rglob('*'):
  if p.is_file() and p.suffix in ['.cu','.cuh','.py','.sh','.log','.csv'] and 'tmp' not in p.parts:copy(p)
# Existing paired measurement logs and drivers.
for name in ['compare_iter162_iter215_iter218','compare_iter218_iter224']:
 for p in Path('results/opt_seven_round',name).rglob('*'):
  if p.is_file() and p.suffix in ['.json','.csv','.log','.py','.sh','.cu']:
   copy(p,'paired/'+name+'/'+str(p.relative_to(Path('results/opt_seven_round',name))))
# Keep experimental generators; prior checkpoint owns older sources.
for p in Path('results/opt_goal').rglob('generate.py'):
 if 'checkpoint' not in str(p) and p.parent.name not in ['check224','profile218','reassess228','trace235']:
  if re.search(r'(209|210|211|212|213|214|215|216|217|218|219|220|221|222|223|224|225|226|227|228|229|230|231|232|233|234|235|236|237|238|239|240|241|242|243|244|245|246)',str(p)):
   copy(p,'generators/'+str(p.relative_to('results/opt_goal')))
rows=[]
for n in range(209,249):
 for p in sorted(Path('trajectory').glob(f'*_iter-{n}')):
  f=p/'results.csv'; rr=list(csv.DictReader(f.open())) if f.exists() and f.stat().st_size else []
  rows.append({'iteration':n,'trajectory':str(p),'rows':rr,'output_sha256':hashlib.sha256((p/'output.txt').read_bytes()).hexdigest()})
(root/'iterations209_248.json').write_text(json.dumps(rows,indent=2)+'\n')
expected={p.name:hashlib.sha256(subprocess.check_output(['git','show','62142fc:solution/'+p.name])).hexdigest() for p in Path('solution').iterdir() if p.is_file()}
actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('solution').iterdir() if p.is_file()}
assert expected==actual,(expected,actual)
candidate=Path('results/opt_goal/check224/candidate.cuh').read_text().replace('namespace megafmm::diagnostic_candidate','namespace megafmm::detail')
assert candidate==Path('solution/sw_pipeline.cu').read_text()
(root/'state.json').write_text(json.dumps({'goal_status':'active','target_met':False,'selected_iteration':224,'selected_commit':'62142fc','launch_refresh_iteration':247,'graph_refresh_iteration':248,'original_numerical_gate':'FAIL both','auto':'Dense','solution_sha256':actual,'verbatim_selected_source':True,'validation_candidate_source_matches':True},indent=2)+'\n')
(root/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('archived',len(manifest),'evidence files; source and validation hashes match224')

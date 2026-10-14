from pathlib import Path
import csv,json,subprocess,hashlib,shutil,gzip,re
root=Path('trajectory/20261015_optimization-goal-checkpoint-313');root.mkdir(exist_ok=True)
manifest={}
def copy(p,d=None):
 p=Path(p);d=root/(d or str(p).removeprefix('results/opt_goal/'));d.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.csv' and p.stat().st_size>600000:d=d.with_suffix(d.suffix+'.gz');d.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
 else:shutil.copyfile(p,d)
 manifest[str(d.relative_to(root))]={'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sha256':hashlib.sha256(d.read_bytes()).hexdigest()}
for name in ['reassess292','reassess295','reassess298','reassess303','profile299','profile306','profile308','profile311','check303']:
 for p in Path('results/opt_goal',name).rglob('*'):
  if p.is_file() and p.suffix in ['.cu','.cuh','.py','.sh','.log','.csv','.json','.md'] and 'tmp' not in p.parts:copy(p)
name='compare_iter282_iter303'
for p in Path('results/opt_seven_round',name).rglob('*'):
 if p.is_file() and p.suffix in ['.json','.csv','.log','.py','.sh','.cu']:copy(p,'paired/'+name+'/'+str(p.relative_to(Path('results/opt_seven_round',name))))
for p in Path('results/opt_goal').rglob('*'):
 if p.is_file() and p.name in ['generate.py','make_generator.py','PROTOCOL.md'] and re.search(r'(29[2-9]|30[0-9]|31[0-2])',str(p)) and 'checkpoint' not in str(p):copy(p,'generators/'+str(p.relative_to('results/opt_goal')))
for name in ['dsm_split_proposal_next.md','local_transform_proposal_next.md','global_split_proposal_next.md']:copy(Path('results/opt_goal',name),'proposals/'+name)
rows=[]
for n in range(292,314):
 for p in sorted(Path('trajectory').glob(f'*_iter-{n}')):
  f=p/'results.csv';rr=list(csv.DictReader(f.open())) if f.exists() and f.stat().st_size else []
  rows.append({'iteration':n,'trajectory':str(p),'rows':rr,'output_sha256':hashlib.sha256((p/'output.txt').read_bytes()).hexdigest(),'invalid_for_ranking':n in [298,305,306,307,311]})
assert len(rows)==22,len(rows)
(root/'iterations292_313.json').write_text(json.dumps(rows,indent=2)+'\n')
expected={p.name:hashlib.sha256(subprocess.check_output(['git','show','1bdfb29:solution/'+p.name])).hexdigest() for p in Path('solution').iterdir() if p.is_file()}
actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('solution').iterdir() if p.is_file()};assert expected==actual
candidate=Path('results/opt_goal/check282/candidate.cuh').read_text().replace('namespace megafmm::diagnostic_candidate','namespace megafmm::detail');assert candidate==Path('solution/sw_pipeline.cu').read_text()
(root/'state.json').write_text(json.dumps({'goal_status':'active','target_met':False,'selected_iteration':282,'selected_commit':'1bdfb29','launch_refresh_iteration':313,'graph_refresh_iteration':291,'original_numerical_gate':'FAIL both','auto':'Dense','solution_sha256':actual,'verbatim_selected_source':True,'validation_candidate_source_matches':True,'303_rejected_despite_independent_validation':True,'next_iteration':314,'next_direction':'PDL global split4plus3; rank1 computes all three leaves before acquiring per-warp global U flags'},indent=2)+'\n')
copy('results/opt_goal/checkpoint313/archive.py','archive.py')
copy('results/opt_goal/checkpoint313/REPORT.md','REPORT.md')
(root/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n');print('archived',len(manifest),'files; source and validation hashes match282')

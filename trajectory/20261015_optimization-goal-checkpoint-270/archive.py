from pathlib import Path
import csv,json,subprocess,hashlib,shutil,gzip,re
root=Path('trajectory/20261015_optimization-goal-checkpoint-270');root.mkdir(exist_ok=True)
manifest={}
def copy(p,d=None):
 p=Path(p);d=root/(d or str(p).removeprefix('results/opt_goal/'))
 d.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.csv' and p.stat().st_size>600000:
  d=d.with_suffix(d.suffix+'.gz');d.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
 else:shutil.copyfile(p,d)
 manifest[str(d.relative_to(root))]={'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sha256':hashlib.sha256(d.read_bytes()).hexdigest()}
for name in ['check254','check266','profile254','profile266','reassess251','trace249','trace254','batchtrace254','phasetrace254','nsys252','nsys254']:
 for p in Path('results/opt_goal',name).rglob('*'):
  if p.is_file() and p.suffix in ['.cu','.cuh','.py','.sh','.log','.csv','.json'] and 'tmp' not in p.parts:copy(p)
for name in ['compare_iter224_iter254','compare_iter254_iter266']:
 for p in Path('results/opt_seven_round',name).rglob('*'):
  if p.is_file() and p.suffix in ['.json','.csv','.log','.py','.sh','.cu']:copy(p,'paired/'+name+'/'+str(p.relative_to(Path('results/opt_seven_round',name))))
for p in Path('results/opt_goal').rglob('generate.py'):
 if 'checkpoint' not in str(p) and re.search(r'(249|250|251|252|253|254|255|256|257|258|259|260|261|262|263|264|265|266|267|268)',str(p)):
  copy(p,'generators/'+str(p.relative_to('results/opt_goal')))
rows=[]
for n in range(249,271):
 for p in sorted(Path('trajectory').glob(f'*_iter-{n}')):
  f=p/'results.csv';rr=list(csv.DictReader(f.open())) if f.exists() and f.stat().st_size else []
  rows.append({'iteration':n,'trajectory':str(p),'rows':rr,'output_sha256':hashlib.sha256((p/'output.txt').read_bytes()).hexdigest()})
(root/'iterations249_270.json').write_text(json.dumps(rows,indent=2)+'\n')
expected={p.name:hashlib.sha256(subprocess.check_output(['git','show','18f9a6f:solution/'+p.name])).hexdigest() for p in Path('solution').iterdir() if p.is_file()}
actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('solution').iterdir() if p.is_file()}
assert expected==actual,(expected,actual)
candidate=Path('results/opt_goal/check266/candidate.cuh').read_text().replace('namespace megafmm::diagnostic_candidate','namespace megafmm::detail')
assert candidate==Path('solution/sw_pipeline.cu').read_text()
(root/'state.json').write_text(json.dumps({'goal_status':'active','target_met':False,'selected_iteration':266,'selected_commit':'18f9a6f','launch_refresh_iteration':269,'graph_refresh_iteration':270,'original_numerical_gate':'FAIL both','auto':'Dense','solution_sha256':actual,'verbatim_selected_source':True,'validation_candidate_source_matches':True},indent=2)+'\n')
copy('results/opt_goal/checkpoint270/archive.py','archive.py')
(root/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('archived',len(manifest),'files; source and validation hashes match266')

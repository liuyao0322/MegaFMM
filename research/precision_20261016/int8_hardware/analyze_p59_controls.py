"""Independent P58/P59 ABBA, using actual frozen binary identities."""
import hashlib,json,re,statistics
from pathlib import Path
from analyze import analyze
h=Path(__file__).resolve().parent;repo=h.parents[2]
for mode,suffix in [('ordinary','abba'),('graph','graph_abba')]:
 prefix='p59_control_'+suffix;runs=[];files=[]
 for arm in ['a0','b0','b1','a1']:
  f=h/f'{prefix}_{arm}.log';data=analyze(f)
  assert len(data['checks'])==78 and all(x['pass']=='1' and x['nonfinite']=='0' and x['graph']==str(int(mode=='graph')) for x in data['checks'])
  inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in f.read_text().splitlines() if l.startswith('INSIDE_EQUAL ')]
  assert len(inside)==78 and all(x['errors']=='0' and x['elements']=='1024' for x in inside)
  j=f.with_suffix('.json');j.write_text(json.dumps(data,indent=2)+'\n');files.extend([f,j])
  runs.append(dict(arm=arm,candidate='p58' if arm[0]=='a' else 'p59',summaries=data['summaries']))
 summary=[]
 for dtype in ['bf16','fp16']:
  groups={label:[g['corrected_us'] for run in runs if run['candidate']==label for row in run['summaries'] if row['dtype']==dtype for g in row['groups']] for label in ['p58','p59']}
  old,new=(statistics.median(groups[label]) for label in ['p58','p59'])
  summary.append(dict(dtype=dtype,old_candidate='p58',new_candidate='p59',old_group_median_us=old,new_group_median_us=new,relative_reduction=1-new/old))
 report=h/f'{prefix}.json';report.write_text(json.dumps(dict(scope='Four independent fresh-process ABBA controls: resident128 two-kernel P58 vs resident128 single-kernel P59; full invocation/all arms, original78 deterministic configurations repeated, unlocked clocks/noCI. Graph descriptor encoding occurs only at capture.',correctness_checks=312,independent_numerical_configurations=78,inside_exponents=319488,runs=runs,summary=summary),indent=2)+'\n')
 files += [report,Path(__file__),h/f'run_{prefix}.sh',h/f'{prefix}_driver.log',h/'analyze.py']
 for label in ['p58','p59']:files.extend([h/f'{label}_manifest.json',h/'bin'/f'compare_{label}'])
 (h/f'{prefix}_manifest.json').write_text(json.dumps({str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
 print(mode,json.dumps(summary))

"""Independent N3 complete-call evidence; profiler processes never enter timing."""
import csv,hashlib,json,re,statistics,sys
from pathlib import Path
r=Path(__file__).resolve().parent;repo=r.parents[2]
sys.path.insert(0,str(r.parent/'int8_hardware'))
from analyze import analyze
rows=[];runs=[];files=[r/'analyze_n3.py',r/'run_n3_abba.sh',r/'n3_abba_driver.log',r/'n3/build.sh',r/'n3/compare.cu',r/'n3_build.log',r/'n3/bin/compare',r/'n3_layout_manifest.json']
for mode in ['ordinary','graph']:
 local=[]
 for arm in ['a0','b0','b1','a1']:
  f=r/f'n3_{mode}_abba_{arm}.log';d=analyze(f)
  assert len(d['checks'])==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in d['checks'])
  inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in f.read_text().splitlines() if l.startswith('INSIDE_EQUAL ')]
  assert len(inside)==78 and all(x['errors']=='0' for x in inside)
  out=f.with_suffix('.json');out.write_text(json.dumps(d,indent=2)+'\n');files.extend([f,out]);local.append(dict(arm=arm,candidate='p47' if arm[0]=='a' else 'n3',summaries=d['summaries']))
 for dt in ['bf16','fp16']:
  groups={label:[g['corrected_us'] for x in local if x['candidate']==label for s in x['summaries'] if s['dtype']==dt for g in s['groups']] for label in ['p47','n3']}
  old,new=(statistics.median(groups[x]) for x in ['p47','n3'])
  rows.append(dict(mode=mode,dtype=dt,p47_us=old,n3_us=new,n3_over_p47=new/old,relative_reduction=1-new/old))
 runs.extend(dict(mode=mode,**x) for x in local)
f=r/'n3_ordinary.log';d=analyze(f);o=f.with_suffix('.json');o.write_text(json.dumps(d,indent=2)+'\n');files.extend([f,o])
report=dict(scope='Independent four-process ABBA per ordinary/Graph mode, deterministic78configurations repeated; ten timing groups per dtype/candidate, unlocked clocks, no confidence interval',correctness_checks_per_mode=312,exact_inside_exponents_per_mode=319488,performance_accepted=False,fresh_stream_guard_acceptance=False,sanitizer_acceptance=False,goal_achieved=False,summary=rows,runs=runs)
(r/'n3_summary.json').write_text(json.dumps(report,indent=2)+'\n');files.append(r/'n3_summary.json')
files.extend([r.parent/'int8_hardware/bin/compare_p47',r.parent/'int8_hardware/p47_manifest.json',r.parent/'int8_hardware/analyze.py'])
(r/'n3_manifest.json').write_text(json.dumps({str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
raw=list(csv.reader((r/'n3_profile_raw.csv').open()));assert len(raw)==3
names,units,values=raw;profile={k:dict(value=v,unit=u) for k,u,v in zip(names,units,values)}
(r/'n3_profile_stats.json').write_text(json.dumps(dict(scope='One first balancedBF16 native fusion kernel;17replay passes, no cache/clock control. Every profiler-process timing arm excluded from ordinary/Graph ABBA.',metrics=profile),indent=2)+'\n')
files=[r/x for x in ['profile_n3.sh','n3_profile.log','n3_profile_raw.csv','n3_profile_stats.json','ncu/profile_n3.ncu-rep','n3_manifest.json','analyze_n3.py']]
(r/'n3_profile_manifest.json').write_text(json.dumps({str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(rows,indent=2))

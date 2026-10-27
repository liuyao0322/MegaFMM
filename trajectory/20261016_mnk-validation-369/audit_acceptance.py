#!/usr/bin/env python3
"""Independently audit frozen acceptance against raw logs, CSVs and current files."""
import csv, hashlib, json, math, re, statistics
from pathlib import Path
root=Path.cwd(); dest=root/'results/mnk369_acceptance'
r=json.loads((dest/'registry.json').read_text())
assert r['repeats_required']==3 and r['groups']==5 and r['trials']==100
assert r['correctness_trials']==3 and r['rotate_inputs']==2 and r['mode']=='launch'
assert r['margin']==.03 and r['expected_combinations']==28
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'build/megafmm_bench')==r['binary_sha256']
assert r['binary_sha256']=='d945f5def5f90f6571bdd31841fbe1210797e805f49740fef1e763612df17d24'
for p,digest in r['source_sha256'].items():assert sha(root/p)==digest,p
spec=json.loads((root/'bench/shapes.json').read_text())
expected={tuple(map(str,shape))+(dtype,) for group in ('squares','rectangles','boundaries') for shape in spec[group] for dtype in spec['dtypes']}
key=lambda row:tuple(str(row[x]) for x in ('m','n','k','dtype'))
entries={key(e):e for e in r['entries']}
assert len(entries)==len(r['entries'])==28 and set(entries)==expected
assert len(r['processes'])==3
raw={k:[] for k in expected};artifact_hashes={}
for repeat,process in enumerate(r['processes']):
 assert process['repeat']==repeat and process['exit_code'] in (0,2)
 log=dest/f'run{repeat}.log'; cp=dest/f'run{repeat}.csv'
 artifact_hashes[log.name]=sha(log);artifact_hashes[cp.name]=sha(cp)
 text=log.read_text(); rows=list(csv.DictReader(cp.open()))
 assert len(rows)==28 and {key(x) for x in rows}==expected
 groups={};checks={}
 for m,n,k,d,g,c,t in re.findall(r'TIME shape=(\d+),(\d+),(\d+) dtype=(\w+) group=(\d+) case=(dense|solution) event_ms=([\deE.+-]+)',text):
  slot=groups.setdefault((m,n,k,d),{}).setdefault(int(g),{})
  assert c not in slot,'duplicate timing';slot[c]=float(t)
 for m,n,k,d,t,de,se,gate,passed in re.findall(r'CHECK (\d+),(\d+),(\d+) (\w+) trial=(\d+) dense_eF=(\S+) solution_eF=(\S+) gate=(\S+) pass=(\d+)',text):
  slot=checks.setdefault((m,n,k,d),{});assert int(t) not in slot
  assert passed=='1' and all(math.isfinite(float(v)) for v in (de,se,gate))
  # Console reports six significant figures, so trust printed pass and verify
  # original full-precision aggregate errors from CSV below.
  slot[int(t)]=float(se)
 for row in rows:
  k=key(row);assert set(groups[k])==set(range(5)) and set(checks[k])==set(range(3))
  assert row['numerical_pass']=='1' and row['nonfinite']=='0'
  assert int(row['correct_trials'])==3 and int(row['trials'])==100 and int(row['groups'])==5
  assert row['rotate_inputs']=='2' and row['mode']=='launch'
  assert math.isfinite(float(row['solution_ef'])) and float(row['solution_ef'])<=float(row['gate_limit'])
  config=dict(field.split('=',1) for field in row['solution_lt_config'].split(';') if '=' in field)
  selection=config['selection']; custom=selection!='dense_cublaslt' and row['actual']!='tuned_dense_cublaslt'
  pairs=[]
  for g in range(5):
   a=groups[k][g];assert set(a)=={'dense','solution'}
   assert all(math.isfinite(v) and v>0 for v in a.values());pairs.append(a['dense']/a['solution'])
  obs=entries[k]['observations'][repeat]
  assert obs['repeat']==repeat and obs['actual']==row['actual'] and obs['selection']==selection
  assert obs['numerical_pass'] and obs['correct_trials']==3 and obs['paired_speedups']==pairs
  assert obs['solution_ms']==float(row['solution_event_ms']) and obs['dense_ms']==float(row['dense_event_ms'])
  raw[k].append({'custom':custom,'pairs':pairs,'ef':float(row['solution_ef']),'selection':selection,'actual':row['actual']})
summary=[]
for k,observations in sorted(raw.items()):
 e=entries[k];pairs=[p for o in observations for p in o['pairs']]
 assert len(observations)==3 and len(pairs)==15
 stable=all(o['custom'] for o in observations) and min(pairs)>=1.03
 assert e['complete'] and e['precision_pass']
 assert e['custom_every_repeat']==all(o['custom'] for o in observations)
 assert e['min_paired_speedup']==min(pairs) and e['median_paired_speedup']==statistics.median(pairs)
 assert (e['status']=='stable_win')==stable
 summary.append(dict(zip(('m','n','k','dtype'),k),stable=stable,min_paired_speedup=min(pairs),median_paired_speedup=statistics.median(pairs),max_ef=max(o['ef'] for o in observations),selections=[o['selection'] for o in observations]))
wins=sum(x['stable'] for x in summary)
assert wins==r['stable_wins'] and r['objective_achieved']==(wins==28)
report={'audit_pass':True,'stable_wins':wins,'combinations':28,'precision_rows_passed':84,'raw_timing_pairs':420,'fresh_precision_checks':252,'binary_sha256':r['binary_sha256'],'source_hashes_checked':len(r['source_sha256']),'artifact_sha256':artifact_hashes,'entries':summary}
(dest/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'PASS independent audit: {wins}/28 stable; 84/84 precision; 420 paired groups; 252 fresh checks; binary and {len(r["source_sha256"])} source hashes unchanged')
for x in summary:print(x['m'],x['n'],x['k'],x['dtype'],'stable' if x['stable'] else 'unresolved',f'min={x["min_paired_speedup"]:.6f}',','.join(x['selections']))

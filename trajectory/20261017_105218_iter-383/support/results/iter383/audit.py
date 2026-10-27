from pathlib import Path
import csv,hashlib,json,re,statistics
p=Path('results/iter383');log=(p/'driver.log').read_text()
rows=list(csv.DictReader((p/'current.csv').open()));candidates=list(csv.DictReader((p/'current.csv.sw_tuning.csv').open()))
shapes=[(512,512,512),(1024,1024,1024),(2048,2048,2048),(4096,4096,4096),(8192,8192,8192),(16384,16384,16384),(8192,4096,8192),(4096,8192,8192),(8192,8192,4096),(8192,1024,512),(1024,8192,512),(1025,1537,769),(514,770,258),(1,1,1)]
expected={(str(m),str(n),str(k),d) for m,n,k in shapes for d in ['bf16','fp16']};key=lambda r:tuple(r[x] for x in ['m','n','k','dtype'])
assert len(rows)==28 and set(map(key,rows))==expected
checks={q:[] for q in expected};groups={}
for m in re.finditer(r'CHECK (\d+),(\d+),(\d+) (bf16|fp16) trial=(\d+) dense_eF=([\d.e+-]+) solution_eF=([\d.e+-]+) gate=([\d.e+-]+) pass=(\d+)',log):
 m,n,k,d,t,de,se,g,passed=m.groups();checks[(m,n,k,d)].append((int(t),float(de),float(se),float(g),int(passed)))
for m in re.finditer(r'TIME shape=(\d+),(\d+),(\d+) dtype=(bf16|fp16) group=(\d+) case=(dense|solution) event_ms=([\d.e+-]+)',log):
 m,n,k,d,g,c,t=m.groups();assert (m,n,k,d,int(g),c) not in groups;groups[(m,n,k,d,int(g),c)]=float(t)
assert len(groups)==280 and all(len(x)==3 for x in checks.values())
assert len(candidates)==2316
prior=list(csv.DictReader(Path('trajectory/20261017_100233_iter-382/results.csv.sw_tuning.csv').open()))
for q in expected:
 old={c['id']:c['config'] for c in prior if key(c)==q}
 current={c['id']:c['config'] for c in candidates if key(c)==q}
 assert all(current.get(i)==v for i,v in old.items()),q
 added=set(current)-set(old)
 wanted=set()
 m,n,k=map(int,q[:3])
 if m%64==0 and n%128==0 and k%256==0:
  wanted={str(500000+i) for i in range(6 if k==512 else 3)}
 assert added==wanted,q

summary=[]
for r in rows:
 q=key(r);assert r['requested']=='sw-tuned' and r['actual']=='sw_tuned' and r['precision_required']=='0' and r['accepted']=='1' and r['nonfinite']=='0'
 assert r['trials']=='100' and r['groups']=='5' and r['correct_trials']=='3' and r['rotate_inputs']=='2'
 assert sorted(c[0] for c in checks[q])==[0,1,2] and int(r['numerical_pass'])==int(all(c[4] for c in checks[q]))
 pool=[c for c in candidates if key(c)==q];assert all(c['ranking_ms'] and len(c['ranking_ms'].split('|'))==5 for c in pool)
 assert all('leaf_terms=7' in c['config'] for c in pool)
 confirmed=[c for c in pool if c['confirmation_ms']];assert len(confirmed)==min(4,len(pool))
 median=lambda c:statistics.median(map(float,c['confirmation_ms'].split('|')))
 selected=int(re.search(r';selected_id=(-?\d+);',r['solution_lt_config'])[1]);winner=next(c for c in confirmed if int(c['id'])==selected)
 assert median(winner)==min(map(median,confirmed))
 paired=[groups[(*q,g,'dense')]/groups[(*q,g,'solution')] for g in range(5)]
 summary.append({'shape_dtype':q,'sw_us':1000*float(r['solution_event_ms']),'cublas_us':1000*float(r['dense_event_ms']),'speedup':float(r['speedup']),'min_group_speedup':min(paired),'all5_above_1p03':min(paired)>=1.03,'ef':float(r['solution_ef']),'historical_precision_pass':int(r['numerical_pass']),'selected':selected,'config':winner['config']})
sourcefiles=list(Path('solution').glob('*'))+[Path('include/megafmm/gemm.h'),Path('bench/benchmark.cu'),Path('tests/test_sw_tuner.cu'),Path('CMakeLists.txt')]
manifest={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sourcefiles if f.is_file()}
for f,h in manifest.items():
 if f.startswith('solution/') or f in ['include/megafmm/gemm.h','bench/benchmark.cu']:assert h+'  '+f in log
assert manifest['solution/sw1024.cu']=='ebc37e9dd171dd5855cdaed91fbb030d250c052a4c459746dd33b82cc634264b'
for f in ['libmegafmm.a','megafmm_bench','test_sw_tuner']:
 file=Path('build-sw-small')/f;manifest[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
archive=Path(re.findall(r'Trajectory saved to: (\S+)',log)[-1]);(archive/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
result={'rows':28,'fresh_fp64_checks':84,'candidate_records':len(candidates),'finite_accepted':28,'historical_precision_pass':sum(int(r['numerical_pass']) for r in rows),'single_sweep_5group_wins_1p03':sum(s['all5_above_1p03'] for s in summary),'scope':'full28 launch single-sweep; independent repeats/Graph/sanitizers not certified','summary':summary}
(archive/'policy_audit.json').write_text(json.dumps(result,indent=2)+'\n');(p/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

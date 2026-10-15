from pathlib import Path
import csv,hashlib,json,re,statistics,shutil
root=Path.cwd();formal=sorted((root/'trajectory').glob('*_iter-370'))[-1]
rows=list(csv.DictReader((formal/'results.csv').open()));text=(formal/'output.txt').read_text()
dest=root/'trajectory/20261016_mnk-validation-370';dest.mkdir(exist_ok=True)
groups={}
for m,n,k,d,g,c,t in re.findall(r'TIME shape=(\d+),(\d+),(\d+) dtype=(\w+) group=(\d+) case=(dense|solution) event_ms=([\deE.+-]+)',text):groups.setdefault((m,n,k,d),{}).setdefault(g,{})[c]=float(t)
perf=[]
for row in rows:
 cfg=dict(field.split('=',1) for field in row['solution_lt_config'].split(';') if '=' in field)
 key=tuple(row[k] for k in ('m','n','k','dtype'));pairs=[g['dense']/g['solution'] for g in groups[key].values()]
 assert len(pairs)==5 and row['numerical_pass']=='1'
 attempts=[]
 for raw in cfg.get('confirmation_trace','').split('|'):
  if not raw:continue
  fields=raw.split(':');candidate,n,status=int(fields[0]),int(fields[1]),fields[2]
  timings=[tuple(map(float,p.split(','))) for p in fields[3:]]
  assert n==len(timings) and 1<=n<=5
  assert status!='pass' or n==5
  attempts.append({'candidate_id':candidate,'groups':n,'status':status,'dense_custom_ms':timings})
 assert len(attempts)==int(cfg.get('confirmation_attempts',0))
 if attempts:assert int(cfg['confirmation_groups'])==attempts[-1]['groups']
 if cfg['selection']!='dense_cublaslt':assert attempts and attempts[-1]['status']=='pass'
 item={k:row[k] for k in ('m','n','k','dtype','actual','dense_event_ms','solution_event_ms','speedup','solution_ef','numerical_pass','tune_ms')}
 item.update(selection=cfg['selection'],confirmation_status=cfg['confirmation_status'],attempts=attempts,paired_speedups=pairs,min_paired_speedup=min(pairs),median_paired_speedup=statistics.median(pairs),confirmation_after_two=len(attempts)>2 and attempts[-1]['status']=='pass')
 perf.append(item)
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'solution').glob('*') if p.is_file()}
# Source copied by the formal harness must match the files under test.
for name,digest in manifest.items():assert hashlib.sha256((formal/Path(name).name).read_bytes()).hexdigest()==digest
(dest/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(dest/'performance.json').write_text(json.dumps({'formal_archive':str(formal.relative_to(root)),'scope':'one formal sweep only, not independent certification','rows':len(perf),'precision_pass_rows':sum(x['numerical_pass']=='1' for x in perf),'single_sweep_custom_wins':sum(x['selection']!='dense_cublaslt' and x['min_paired_speedup']>=1.03 for x in perf),'confirmed_after_two_nominees':sum(x['confirmation_after_two'] for x in perf),'entries':perf},indent=2)+'\n')
shutil.copy2('results/archive370.py',dest/'archive370.py')
for x in perf:print(x['m'],x['n'],x['k'],x['dtype'],x['selection'],f'median={float(x["speedup"]):.5f}',f'min5={x["min_paired_speedup"]:.5f}',f'attempts={len(x["attempts"])}',','.join(str(a['candidate_id'])+':'+a['status'] for a in x['attempts']))

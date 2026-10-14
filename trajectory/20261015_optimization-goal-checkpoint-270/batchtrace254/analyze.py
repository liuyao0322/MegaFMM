import csv,json,statistics,collections
p='results/opt_goal/batchtrace254/'
g=collections.defaultdict(list)
with open(p+'timing.csv') as f:
 for r in csv.DictReader(f):g[tuple(r[k] for k in ('dtype','trial','graph','rep','invocation'))].append({k:int(v) for k,v in r.items() if k not in ['dtype']})
rows=[]
for (dtype,trial,graph,rep,inv),rs in g.items():
 ps=min(r['pack_start'] for r in rs); pe=max(r['pack_lane0_finish'] for r in rs)
 rows.append(dict(dtype=dtype,trial=int(trial),graph=int(graph),rep=int(rep),invocation=int(inv),pack_span=(pe-ps)/1000,entry_minus_pack_finish=(min(r['compute_entry'] for r in rs)-pe)/1000,tc_start_minus_pack_finish=(min(r['tc_start'] for r in rs)-pe)/1000,span=(max(r['tc_end'] for r in rs)-ps)/1000,dep_wait=statistics.median(r['dep_after']-r['dep_before'] for r in rs)/1000,tc_chain=statistics.median(r['tc_end']-r['tc_start'] for r in rs)/1000))
summary={}
for dtype in ['bf16','fp16']:
 for graph in [0,1]:
  for region in ['all','steady']:
   rs=[r for r in rows if r['dtype']==dtype and r['graph']==graph and (region=='all' or r['invocation']>0)]
   summary[f'{dtype}_graph{graph}_{region}']={k:statistics.median(r[k] for r in rs) for k in ('pack_span','entry_minus_pack_finish','tc_start_minus_pack_finish','span','dep_wait','tc_chain')}
print(json.dumps(summary,indent=2));open(p+'summary.json','w').write(json.dumps(summary,indent=2)+'\n')

from pathlib import Path
import sqlite3,json,statistics
root=Path(__file__).resolve().parent
c=sqlite3.connect(root/'timeline.sqlite');c.row_factory=sqlite3.Row
rows=[dict(r) for r in c.execute('select k.*,s.value as name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id where s.value like "%megafmm::detail%" order by k.start')]
assert len(rows)==7*524*2,len(rows)
shapes={(8,4):(512,512,512),(8,16):(2048,2048,2048),(16,32):(4096,4096,4096),(64,128):(16384,16384,16384),(32,8):(8192,1024,512),(16,64):(1024,8192,512),(17,7):(514,770,258)}
samples={}
for a,b in zip(rows[::2],rows[1::2]):
 assert ('::transform<' in a['name'] or '::pack_padded<' in a['name']) and '::compute<' in b['name'],(a['name'],b['name'])
 key=shapes[(b['gridX'],b['gridY'])]
 overlap=max(0,min(a['end'],b['end'])-max(a['start'],b['start']))
 samples.setdefault(key,[]).append(dict(pack_us=(a['end']-a['start'])/1000,compute_us=(b['end']-b['start'])/1000,interval_union_us=(max(a['end'],b['end'])-a['start'])/1000,interval_overlap_us=overlap/1000,start_gap_us=(b['start']-a['end'])/1000))
out=[]
for key,x in samples.items():
 assert len(x)==524
 # The final 500 invocations are the five timed groups. Exclude setup/correctness/warmup.
 x=x[-500:]
 item=dict(shape=key,samples=500,median={k:statistics.median(r[k] for r in x) for k in x[0]},p10={k:sorted(r[k] for r in x)[49] for k in x[0]},p90={k:sorted(r[k] for r in x)[449] for k in x[0]})
 out.append(item)
(root/'timeline_summary.json').write_text(json.dumps(out,indent=2))
for r in out:print(r['shape'],r['median'])

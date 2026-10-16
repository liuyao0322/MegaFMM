"""Own 84 fresh original-input gates; one stream, no stress/tool certificate."""
from pathlib import Path
import hashlib,json,re
h=Path(__file__).resolve().parent;repo=h.parents[2]
rows=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in (h/'extra_p55.log').read_text().splitlines() if s.startswith('EXTRA_CHECK ')]
assert len(rows)==84
assert len({(r['dtype'],r['pattern'],r['trial']) for r in rows})==84
for r in rows:
 assert int(r['seed'])==127163+487*int(r['trial'])+127*int(r['pattern'])+(263 if r['dtype']=='fp16' else 0)
 assert r['pass']=='1' and all(r[k]=='0' for k in ['nonfinite','inside_errors','workspace_guard_errors','output_guard_errors'])
summary={dt:dict(configurations=42,max_ef=max(float(r['corrected_ef']) for r in rows if r['dtype']==dt),fixed_floor_failures=sum(r['fixed_pass']!='1' for r in rows if r['dtype']==dt)) for dt in ['bf16','fp16']}
out=dict(scope='84 newly seeded original rounded input configurations,14patterns/3trials/bothdtypes; independent host inputs/inside, original FP64 AB/tuned Dense gate, one stream/workspace, output and workspace guards. Full two-kernel P55. Extra fixed floors recorded separately. Not Graph/two-stream/reuse stress or sanitizer certificate.',original_gates=84,inside_exponents=86016,original_gate_failures=0,summary=summary,rows=rows)
(h/'extra_p55.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p55_extra.py','extra_p55.cu','build_extra_p55.sh','extra_p55_build.log','extra_p55.log','extra_p55.json','bin/extra_p55','p55_extra_source_manifest.json','p55_manifest.json']
(h/'p55_extra_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(summary,indent=2))

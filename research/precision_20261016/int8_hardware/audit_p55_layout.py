"""Own P55 native+INT8 pre-MMA TMA probe only, not a wait-order certificate."""
from pathlib import Path
import hashlib,json,re
h=Path(__file__).resolve().parent;repo=h.parents[2]
a=(h/'p55/corrected.cuh').read_text();b=(h/'probe_p55_layout.cu').read_text()
a=a[a.index(' if(tid==0){\n  for(int step=0;step<12;++step)'):a.index('}else if(tid>=128)consume')+1]
b=b[b.index(' if(tid==0){\n  for(int step=0;step<12;++step)'):b.index(' // PRODUCER_END')]
assert ''.join(a.split())==''.join(b.split())
rows=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in (h/'probe_p55_layout.log').read_text().splitlines() if s.startswith('FINE_LAYOUT ')]
assert len(rows)==6
for r in rows:
 assert int(r['seed'])==126157+479*int(r['trial'])+(257 if r['dtype']=='fp16' else 0)
 assert r['pass']=='1' and all(r[k]=='0' for k in ['native_errors','integer_errors','metadata_errors','shared_guard_errors','coverage_errors','workspace_guard_errors'])
 assert (r['ctas'],r['sequences'],r['slots'])==('128','48','8')
 assert (r['native_comparisons'],r['integer_comparisons'],r['coordinates'])==('44040192','50331648','262144')
out=dict(scope='Own six synthetic native+INT8 TMA/shared swizzle configurations, all128CTAs/48batches/eightslot reuse, fine metadata/local-global coordinate and paired exponents, output CLayout bijection, shared/workspace/output guards. Verbatim frozen producer; generic reads retire current slot without MMA. Not WGMMA wait-order, original-input matrix, sanitizer or performance evidence.',configurations=6,native_uint16_comparisons_including_cta_reuse=6*44040192,int8_comparisons_including_cta_reuse=6*50331648,output_coordinates=6*262144,errors=0,rows=rows)
(h/'probe_p55_layout.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p55_layout.py','probe_p55_layout.cu','build_probe_p55_layout.sh','run_probe_p55_layout.sh','probe_p55_layout_build.log','probe_p55_layout.log','probe_p55_layout.json','bin/probe_p55_layout','p55_source_manifest.json','p55_probe_source_manifest.json']
(h/'p55_layout_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['rows','scope']},indent=2))

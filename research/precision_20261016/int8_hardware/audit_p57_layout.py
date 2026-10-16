"""Own P57 TMA values/guards and compiled resources before first candidate MMA."""
from pathlib import Path
import hashlib,json,re
h=Path(__file__).resolve().parent;repo=h.parents[2]
c=(h/'p57/corrected.cuh').read_text();old=(h/'p56/corrected.cuh').read_text()
assert c.replace(' if(tid==0){\n  #pragma unroll 7',' if(tid==0){')==old
p=(h/'probe_p57_layout.cu').read_text();a=c[c.index(' if(tid==0){\n  #pragma unroll 7'):c.index('}else if(tid>=128)consume')+1];b=p[p.index(' // PRODUCER_BEGIN')+len(' // PRODUCER_BEGIN'):p.index(' // PRODUCER_END')];assert ''.join(a.split())==''.join(b.split())
text=(h/'probe_p57_layout.log').read_text()
rows=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in text.splitlines() if s.startswith('STATIC_LAYOUT ')]
assert len(rows)==6
for r in rows:
 assert int(r['seed'])==131179+503*int(r['trial'])+(281 if r['dtype']=='fp16' else 0)
 assert r['pass']=='1' and all(r[k]=='0' for k in ['native_errors','shared_guard_errors','coverage_errors','workspace_guard_errors'])
 assert (r['ctas'],r['sequences'],r['slots'],r['native_comparisons'])==('128','28','4','69206016')
resources=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in text.splitlines() if s.startswith('STATIC_RESOURCE ')]
assert len(resources)==2 and all(0<int(r['consumer_regs'])<=255 and r['active_blocks_per_sm']=='1' and r['shared_bytes']=='196736' for r in resources)
assert all(r['prep_regs']==('52' if r['dtype']=='bf16' else '48') for r in resources)
coop=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in text.splitlines() if s.startswith('COOPERATIVE_INIT ')]
assert len(coop)==2 and all(r['active_blocks_per_sm']=='3' and r['grid']=='384' and r['sm_count']=='132' for r in coop)
out=dict(scope='Own six newly seeded full-grid pre-MMA TMA/shared/ring28batch/fourslot tests, all native high/low values and guards/coordinate coverage. Producer verbatim candidate. Generic reads retire current slots without MMA: not actual WGMMA wait-order, numerical matrix, input packing, sanitizer or performance acceptance. Candidate MMA resource queries are compiled but no corrected launch here; layout generic reader itself has32B stack,0spill.',configurations=6,native_uint16_comparisons_including_cta_reuse=415236096,output_coordinates=1572864,errors=0,resources=resources,cooperative_capacity=coop,rows=rows)
(h/'probe_p57_layout.json').write_text(json.dumps(out,indent=2)+'\n')
f=[h/n for n in ['audit_p57_layout.py','probe_p57_layout.cu','build_probe_p57_layout.sh','run_probe_p57_layout.sh','probe_p57_layout_build.log','probe_p57_layout.log','probe_p57_layout.json','p57_source_manifest.json','p57_probe_source_manifest.json','bin/probe_p57_layout']]
(h/'p57_layout_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in f},indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['rows','scope']},indent=2))

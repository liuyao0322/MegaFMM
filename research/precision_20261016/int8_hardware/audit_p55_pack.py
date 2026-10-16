"""P55 own pre-MMA rounded-original packing and scalar checks, not matrix gates."""
from pathlib import Path
import re
import hashlib
import json
h=Path(__file__).resolve().parent;repo=h.parents[2]
log=(h/'probe_p55_pack.log').read_text()
rows=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in log.splitlines() if s.startswith('FINE_PACK ')]
assert len(rows)==96 and {(r['dtype'],int(r['pattern']),int(r['trial'])) for r in rows}=={(d,p,t) for d in ['fp16','bf16'] for p in range(16) for t in range(3)}
for r in rows:
    assert int(r['seed'])==125153+467*int(r['trial'])+113*int(r['pattern'])+(251 if r['dtype']=='fp16' else 0)
    assert r['pass']=='1' and all(r[k]=='0' for k in ['native_errors','quant_errors','exponent_errors','inside_errors','guard_errors','reference_saturation'])
    assert (r['native_values'],r['quant_bytes'],r['metadata'],r['inside'])==('3670016','4718592','1152','1024')
scalar=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in log.splitlines() if s.startswith('FINE_SCALAR ')]
assert len(scalar)==1 and scalar[0]['scaling_cases']=='6611' and scalar[0]['native_exponent_cases']=='64384'
assert scalar[0]['scaling_bit_errors']==scalar[0]['exponent_errors']=='0'
capacity=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in log.splitlines() if s.startswith(('COOPERATIVE_INIT ','FINE_CONSUMER_INIT '))]
assert len(capacity)==4
assert all(r['active_blocks_per_sm']==('2' if 'threads' in r else '1') for r in capacity)
build=(h/'probe_p55_pack_build.log').read_text()
assert all(x not in build for x in ['C7515','C7510','warning:'])
assert all(int(a)==int(b)==int(c)==0 for a,b,c in re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',build))
out=dict(scope='Own pre-MMA CPU oracle over original rounded16patterns/3trials/bothdtypes. Native high bits, direct INT8 bytes and integer exponents, original inside, guards and reference preclamp saturation; scalar safe pow2 and exhaustive native high exponent domain. Not matrix/WGMMA/performance/sanitizer acceptance.',configurations=96,native_uint16_comparisons=96*3670016,int8_byte_comparisons=96*4718592,quant_exponents=96*1152,inside_exponents=96*1024,errors=0,scalar=scalar[0],capacity=capacity,rows=rows)
(h/'probe_p55_pack.json').write_text(json.dumps(out,indent=2)+'\n')
files=['probe_p55_pack.cu','probe_p55_pack_build.log','probe_p55_pack.log','probe_p55_pack.json','build_probe_p55_pack.sh','run_probe_p55_pack.sh','audit_p55_pack.py','bin/probe_p55_pack','p55_source_manifest.json','p55_probe_source_manifest.json','p55_preflight_manifest.json']
(h/'p55_pack_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['rows','scope','capacity']},indent=2))

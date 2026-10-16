"""Audit and freeze P58's own pre-MMA finite-range edge diagnostic."""
from pathlib import Path
import json,re,hashlib
h=Path(__file__).resolve().parent;repo=h.parents[2]
rows=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in (h/'probe_p58_edges.log').read_text().splitlines() if l.startswith('PACK_EDGE ')]
assert len(rows)==18 and len({(x['dtype'],x['pattern'],x['trial']) for x in rows})==18
assert {(x['dtype'],int(x['pattern']),int(x['trial'])) for x in rows}=={(d,p,t) for d in ['bf16','fp16'] for p in range(9,12) for t in range(3)}
for x in rows:
 assert int(x['seed'])==133201+521*int(x['trial'])+139*int(x['pattern'])
 assert x['pass']=='1' and all(x[k]=='0' for k in ['a_errors','b_errors','exponent_errors'])
 assert int(x['native_values'])==5767168 and int(x['inside_exponents'])==1024
 assert int(x['outside_normal'])==(0 if x['dtype']=='fp16' else 512 if x['pattern']=='11' else 1024)
 assert int(x['max_abs_exponent'])==(20 if x['dtype']=='fp16' else 130)
source=(h/'probe_p58_edges.cu').read_text()
assert '(c%64)<32' in source and '(r%64)<32' in source and '#include "p58/pack.cuh"' in source
out=dict(scope='Packing only before any P58 MMA. Same fresh input, frozen P56 vsP58, all eleven planes bothsides; freshly poisoned new workspace. BF16 high/subnormal, reversed, alternating32Kslabs extreme/unit, FP16 representable controls. No performance, full matrix or sanitizer acceptance.',configurations=18,native_value_comparisons=103809024,exponent_comparisons=18432,bitwise_errors=0,seed_formula='133201+521trial+139pattern',rows=rows)
(h/'probe_p58_edges.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p58_edges.py','probe_p58_edges.cu','probe_p58_edges.log','probe_p58_edges.json','build_probe_p58_edges.sh','probe_p58_edges_build.log','bin/probe_p58_edges','p58_source_manifest.json','p56_manifest.json','p58_edges_probe_source_manifest.json']
(h/'p58_edges_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

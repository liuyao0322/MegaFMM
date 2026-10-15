"""Audit the separate finite-range P50 packing edge diagnostic."""
from pathlib import Path
import json,re,hashlib
h=Path(__file__).resolve().parent;repo=h.parents[2]
rows=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in (h/'probe_p50_edges.log').read_text().splitlines() if l.startswith('PACK_EDGE ')]
assert len(rows)==18 and len({(x['dtype'],x['pattern'],x['trial']) for x in rows})==18
for x in rows:
 assert int(x['seed'])==111031+397*int(x['trial'])+97*int(x['pattern'])
 assert x['pass']=='1' and all(x[k]=='0' for k in ['a_errors','b_errors','exponent_errors'])
 assert int(x['native_values'])==5767168 and int(x['inside_exponents'])==1024
 assert int(x['outside_normal'])==(0 if x['dtype']=='fp16' else 512 if x['pattern']=='11' else 1024)
 assert int(x['max_abs_exponent'])==(19 if x['dtype']=='fp16' else 130)
assert json.loads((h/'probe_p50_edges.json').read_text())['rows']==rows
files=['audit_p50_edges.py','p50_edges_manifest.json']
(h/'p50_edge_audit_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print('P50 pre-MMA edge cases:',len(rows),'pass; realBF16outside-fast-range coverage verified')

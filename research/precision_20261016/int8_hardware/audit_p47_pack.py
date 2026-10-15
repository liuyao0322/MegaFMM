"""Freeze the P47-vs-frozen-P46 full native plane bit comparison."""
import hashlib
import json
import re
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[2]
rows=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in (h/'probe_p47_pack.log').read_text().splitlines() if line.startswith('PACK_BITS ')]
assert len(rows)==54 and len({(x['dtype'],x['pattern'],x['trial']) for x in rows})==54
assert {x['dtype'] for x in rows}=={'bf16','fp16'}
assert {int(x['pattern']) for x in rows}==set(range(9))
assert {int(x['trial']) for x in rows}==set(range(3))
assert all(int(x['seed'])==47999+int(x['trial'])*353+int(x['pattern'])*61 for x in rows)
assert all(x['pass']=='1' and all(x[k]=='0' for k in ['a_errors','b_errors','exponent_errors']) for x in rows)
assert all(int(x['native_values'])==5767168 and int(x['inside_exponents'])==1024 for x in rows)
out=dict(scope='same fresh inputs for frozen P46 and P47 preprocessors; all eleven native high/low planes on both sides and exact inside exponents, before MMA. Diagnostic only, no timing or original-input FP64 matrix acceptance; not separately sanitizer-instrumented.',
    configurations=54,native_value_comparisons=311427072,exponent_comparisons=55296,
    bitwise_errors=0,seed_formula='47999+trial*353+pattern*61',rows=rows)
(h/'probe_p47_pack.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p47_pack.py','probe_p47_pack.cu','build_probe_p47_pack.sh',
       'probe_p47_pack_build.log','probe_p47_pack.log','probe_p47_pack.json',
       'bin/probe_p47_pack','p46_manifest.json','p47_manifest.json','../../../tests/test_sw1024.cu']
(h/'p47_pack_bits_manifest.json').write_text(json.dumps({str((h/f).resolve().relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

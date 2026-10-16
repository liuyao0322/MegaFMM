"""Freeze independently regenerated P55 sources before first GPU compilation."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

h=Path(__file__).resolve().parent
repo=h.parents[2]
with tempfile.TemporaryDirectory(prefix='p55_cpu_') as d:
    subprocess.run(['python3',str(h/'generate_direct_fine_quant.py'),'--output',d],check=True,capture_output=True)
    for f in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:
        assert (h/'p55'/f).read_bytes()==(Path(d)/f).read_bytes()
        assert (h/f).read_bytes()==(h/'p51'/f).read_bytes()
p=(h/'p55/pack.cuh').read_text();c=(h/'p55/corrected.cuh').read_text()
for needle in ['sizeof(DirectPackShared)==55808','WorkBytes==12329472','__launch_bounds__(768,1)','cooperative_groups::this_grid().sync();']:
    assert needle in p,needle
for needle in ['sizeof(Shared)==197248','s.ae[ap][(mn/64)/8]+s.be[bp][(mn%64)/8]','warpgroup_wait<1>();','uint32_t integer[16]={};']:
    assert needle in c,needle
assert 'meta_common' not in c and 'meta_e0' not in c
files=['generate_direct_fine_quant.py','check_p55_sources.py','p55_proposal_manifest.json','quant_exponent_proposal.cuh']+['p55/'+f for f in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']]
(h/'p55_source_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print('P55 isolated regeneration/source identity pass; working P51 unchanged; no GPU execution')

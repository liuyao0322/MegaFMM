"""P39 metadata: standalone probes preserve its quantizer, not a fresh full-prep replay."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('label');ap.add_argument('--stress',action='store_true');args=ap.parse_args()
h=Path(__file__).resolve().parent;repo=h.parents[2]
def parse(path,prefix):
    return [dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in path.read_text().splitlines() if l.startswith(prefix)]
def eligibility(path):
    rows=parse(path,'SCALE_ELIGIBILITY ')
    assert rows
    for x in rows:
        assert int(x['groups'])==640 and int(x['eligible'])+int(x['fallback'])==640
        assert 0<=int(x['paired_eligible'])<=min(384,int(x['eligible']))
        assert 0<=int(x['exceptional'])<=int(x['fallback'])
    return rows
log=h/(args.label+'_balanced.log');ordinary=eligibility(log)
correctness=[x for x in ordinary if int(x['pattern'])>=0]
timing_input=[x for x in ordinary if int(x['pattern'])<0]
assert len(correctness)==78 and len(timing_input) in [0,2]
summary=[]
for dtype in ['bf16','fp16']:
    r=[x for x in correctness if x['dtype']==dtype];assert len(r)==39
    summary.append(dict(dtype=dtype,configurations=39,**{k:sum(int(x[k]) for x in r) for k in ['groups','eligible','paired_eligible','fallback','exceptional']}))
files=[log,h/'analyze_p39_scale.py',h/(args.label+'_manifest.json')]
out=dict(scope='Host copies of current quantizer metadata outside accepted timing; counts eligible CTA correction groups, not dynamic instruction counters',
    ordinary_correctness_summary=summary,timing_input_after_quantization_equivalent_standalone_probes=timing_input)
if args.stress:
    path=h/('stress_'+args.label+'.log');r=eligibility(path);extra=parse(path,'EXTRA ')
    assert len(r)==108 and len(extra)==108 and all(x['pass']=='1' and x['guards']=='1' for x in extra)
    assert {int(x['pattern']) for x in r}==set(range(9))
    assert any(int(x['eligible'])>0 for x in r)
    assert all(x['eligible']=='0' and x['fallback']=='640' for x in r if x['pattern']=='7')
    assert all(x['eligible']=='0' and x['exceptional']=='640' for x in r if x['pattern']=='8' and x['dtype']=='bf16')
    out.update(stress_configurations=108,stress_interleaved_calls=864,new_cases='pattern7: changing K128 scales, all640groups fallback; pattern8 BF16: tiny input, all640groups exceptional fallback',stress_metadata=r)
    files+=[path,h/('stress_'+args.label+'.cu'),h/('stress_'+args.label+'_build.log'),h/'bin'/('stress_'+args.label)]
target=h/(args.label+'_scale_diagnostics.json');target.write_text(json.dumps(out,indent=2)+'\n');files.append(target)
(h/(args.label+'_scale_diagnostics_manifest.json')).write_text(json.dumps({str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='stress_metadata'},indent=2))

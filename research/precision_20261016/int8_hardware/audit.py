"""Audit frozen evidence, including the explicit performance rejection."""
import hashlib
import json
import re
from pathlib import Path

r=Path(__file__).resolve().parent
repo=r.parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
baseline=json.loads((r.parent/'source_manifest.json').read_text())['production_unchanged_sha256']
for name,sha in baseline.items():assert digest(repo/name)==sha,name
for candidate in range(1,8):
    manifest=json.loads((r/f'p{candidate}_manifest.json').read_text())
    for name,sha in manifest.items():assert digest(repo/name)==sha,name
# Audit frozen P7, independently of later working candidates.
if (r/'validation_manifest.json').exists():
    for name,sha in json.loads((r/'validation_manifest.json').read_text()).items():
        assert digest(repo/name)==sha,name
def check_rows(name,prefix,count):
    text=(r/name).read_text()
    rows=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith(prefix)]
    assert len(rows)==count,(name,len(rows))
    assert all(x['pass']=='1' for x in rows),name
    if prefix=='EXTRA ':assert all(x['guards']=='1' for x in rows),name
    return rows
ordinary=check_rows('p7_balanced.log','CHECK ',72)
graph=check_rows('p7_graph.log','CHECK ',72)
stress=check_rows('stress.log','EXTRA ',72)
for tool in ['memcheck','racecheck','synccheck']:
    check_rows(f'{tool}.log','EXTRA ',24)
    text=(r/f'{tool}.log').read_text()
    assert ('0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors') in text,tool
primitive=[l for l in (r/'primitive.log').read_text().splitlines() if l.startswith('PRIMITIVE ')]
assert len(primitive)==18 and all('errors=0' in x for x in primitive)
assert 'POW2 cases=1048576 exponents=-300..300 errors=0' in (r/'p2_build.log').read_text()
timings=json.loads((r/'p7_balanced.json').read_text())
groups=[g for s in timings['summaries'] for g in s['groups']]
performance_pass=all(g['speedup_vs_sw']>=1 for g in groups)
assert not performance_pass,'P7 is a rejected performance candidate; re-audit if this changes'
for x in ordinary+graph:
    assert int(x['nonfinite'])==0
    assert float(x['corrected_ef'])<=(1e-3 if x['dtype']=='bf16' else 1e-4)
out=dict(status='accuracy_validated_performance_rejected',goal_achieved=False,production_unchanged=True,
         checks=dict(primitive_exact=18,pow2_equivalence=1048576,ordinary=72,graph=72,dual_stream=72,dual_stream_calls=576,
                     sanitizer_cases_per_tool=24,sanitizer_calls_per_tool=48),
         scope='1024^3 contiguous no-transpose alpha1 beta0, H100. Seven native main products plus eight INT8 first-order correction products. Repeated deterministic seeds across variants are not independent new accuracy samples.',
         max_ef={dt:max(float(x['corrected_ef']) for x in ordinary if x['dtype']==dt) for dt in ('bf16','fp16')},
         performance_pass=performance_pass,required_remaining='A precision-accepted full FMM invocation with no measured regression versus current SW; production integration and final acceptance remain outstanding.')
(r/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))

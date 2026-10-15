"""Audit P24's own Graph, independent-stream guards and sanitizer evidence."""
import hashlib
import json
import math
import re
from pathlib import Path
from analyze import analyze
h=Path(__file__).resolve().parent;repo=h.parents[2]
def rows(name,prefix,count,guards=False):
    text=(h/name).read_text();r=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith(prefix)]
    assert len(r)==count,(name,len(r))
    assert all(x['pass']=='1' for x in r),name
    if guards:assert all(x['guards']=='1' and math.isfinite(float(x['ef'])) for x in r),name
    return r
for name in ['p24_balanced.log','p24_graph.log']:
    rows(name,'CHECK ',78)
    text=(h/name).read_text();inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('INSIDE_EQUAL ')]
    assert len(inside)==78 and all(x['errors']=='0' for x in inside)
graph=analyze(h/'p24_graph.log');assert all(x['graph']=='1' for x in graph['checks'])
(h/'p24_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
stress=rows('stress_p24.log','EXTRA ',84,True)
assert set(int(x['pattern']) for x in stress)==set(range(7))
for tool in ['memcheck','racecheck','synccheck']:
    name=f'p24_{tool}.log';rows(name,'EXTRA ',28,True)
    text=(h/name).read_text();assert ('0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors') in text,tool
out=dict(status='p24_accuracy_and_scoped_synchronization_validated_performance_rejected',goal_achieved=False,
    ordinary_original_input_fp64_checks=78,graph_original_input_fp64_checks=78,graph_same_deterministic_configurations=True,
    equal_per_column_exponents_per_mode=79872,independent_stress_configurations=84,independent_stress_calls=672,
    sanitizer_cases_per_tool=28,sanitizer_calls_per_tool=56,sanitizers=['memcheck','racecheck','synccheck'],sanitizer_kernel_scope='kns=int8_research',
    stress_dense_control='historical cuBLAS GemmEx; ordinary and Graph compare use production tuned Dense gate',
    stress_output_guards='128 floats before and after each output; two independent streams/workspaces',
    graph_timing_us={s['dtype']:s['corrected_us'] for s in graph['summaries']},
    limitations='Only contiguous non-transposed1024^3, alpha1 beta0, FP16/BF16; no arbitrary-range proof, no production integration, no transfer to later modified candidates.')
(h/'p24_validation.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p24_validation.py','analyze.py','stress_p24.cu','build_stress_p24.sh','stress_p24_build.log','validate_p24.sh','p24_validation_driver.log','p24_graph.log','p24_graph.json','stress_p24.log','p24_memcheck.log','p24_racecheck.log','p24_synccheck.log','p24_validation.json','bin/stress_p24','bin/compare_p24','p24_manifest.json']
(h/'p24_validation_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

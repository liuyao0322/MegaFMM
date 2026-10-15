"""Audit P27's own Graph, independent-stream guards and sanitizer evidence."""
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
for name in ['p27_balanced.log','p27_graph.log']:
    rows(name,'CHECK ',78)
    text=(h/name).read_text();inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('INSIDE_EQUAL ')]
    assert len(inside)==78 and all(x['errors']=='0' for x in inside)
graph=analyze(h/'p27_graph.log');assert all(x['graph']=='1' for x in graph['checks'])
(h/'p27_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
stress=rows('stress_p27.log','EXTRA ',84,True)
assert set(int(x['pattern']) for x in stress)==set(range(7))
for tool in ['memcheck','racecheck','synccheck']:
    name=f'p27_{tool}.log';rows(name,'EXTRA ',28,True)
    text=(h/name).read_text();assert ('0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors') in text,tool
out=dict(status='p27_accuracy_and_scoped_synchronization_validated_performance_rejected',goal_achieved=False,
    ordinary_original_input_fp64_checks=78,graph_original_input_fp64_checks=78,graph_same_deterministic_configurations=True,
    equal_per_column_exponents_per_mode=79872,independent_stress_configurations=84,independent_stress_calls=672,
    sanitizer_cases_per_tool=28,sanitizer_calls_per_tool=56,sanitizers=['memcheck','racecheck','synccheck'],sanitizer_kernel_scope='kns=int8_research',
    stress_dense_control='historical cuBLAS GemmEx; ordinary and Graph compare use production tuned Dense gate',
    stress_output_guards='128 floats before and after each output; two independent streams/workspaces',
    graph_timing_us={s['dtype']:s['corrected_us'] for s in graph['summaries']},
    limitations='Only contiguous non-transposed1024^3, alpha1 beta0, FP16/BF16; no arbitrary-range proof, no production integration, no transfer to later modified candidates.')
(h/'p27_validation.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p27_validation.py','analyze.py','stress_p27.cu','build_stress_p27.sh','stress_p27_build.log','validate_p27.sh','p27_validation_driver.log','p27_graph.log','p27_graph.json','stress_p27.log','p27_memcheck.log','p27_racecheck.log','p27_synccheck.log','p27_validation.json','bin/stress_p27','bin/compare_p27','p27_manifest.json']
(h/'p27_validation_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

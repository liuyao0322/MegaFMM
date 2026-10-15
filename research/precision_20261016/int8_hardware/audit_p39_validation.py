"""Audit P39's own Graph, independent-stream guards and sanitizer evidence."""
import hashlib
import json
import math
import re
from pathlib import Path
from analyze import analyze
h=Path(__file__).resolve().parent;repo=h.parents[2]
assert (h/'p39_graph.log').read_bytes()==(h/'p39_graph_abba_b0.log').read_bytes()
def rows(name,prefix,count,guards=False):
    text=(h/name).read_text();r=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith(prefix)]
    assert len(r)==count,(name,len(r))
    assert all(x['pass']=='1' for x in r),name
    if guards:assert all(x['guards']=='1' and math.isfinite(float(x['ef'])) for x in r),name
    return r
for name in ['p39_balanced.log','p39_graph.log']:
    rows(name,'CHECK ',78)
    text=(h/name).read_text();inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('INSIDE_EQUAL ')]
    assert len(inside)==78 and all(x['errors']=='0' for x in inside)
graph=analyze(h/'p39_graph.log');assert all(x['graph']=='1' for x in graph['checks'])
(h/'p39_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
stress=rows('stress_p39.log','EXTRA ',108,True)
assert set(int(x['pattern']) for x in stress)==set(range(9))
for tool in ['memcheck','racecheck','synccheck']:
    name=f'p39_{tool}.log';rows(name,'EXTRA ',36,True)
    text=(h/name).read_text();assert ('0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors') in text,tool
for name in ['p39_balanced.log','p39_graph.log','stress_p39.log','p39_memcheck.log','p39_racecheck.log','p39_synccheck.log']:
    init=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in (h/name).read_text().splitlines() if line.startswith('COOPERATIVE_INIT ')]
    assert len(init)==2 and {x['dtype'] for x in init}=={'bf16','fp16'},name
    assert all(x['supported']=='1' and int(x['active_blocks_per_sm'])*int(x['sm_count'])>=int(x['grid'])==288 for x in init),name
assert '39727+trial*347+s*197+pat*59' in (h/'stress_p39.cu').read_text()
out=dict(status='p39_accuracy_and_scoped_synchronization_validated_performance_rejected',goal_achieved=False,
    ordinary_original_input_fp64_checks=78,graph_original_input_fp64_checks=78,graph_same_deterministic_configurations=True,
    graph_process_provenance='p39_graph.log is an exact copy of existing own p39_graph_abba_b0.log; no extra independent Graph sample',
    equal_per_column_exponents_per_mode=79872,independent_stress_configurations=108,independent_stress_calls=864,
    sanitizer_cases_per_tool=36,sanitizer_calls_per_tool=72,sanitizers=['memcheck','racecheck','synccheck'],sanitizer_kernel_scope='kns=int8_research',
    stress_dense_control='historical cuBLAS GemmEx; ordinary and Graph compare use production tuned Dense gate',
    stress_output_guards='128 floats before and after each output; two independent streams/workspaces',
    graph_timing_us={s['dtype']:s['corrected_us'] for s in graph['summaries']},
    cooperative_preprocessing='288CTAs, one unconditional grid.sync barrier; actual supported/resident capacity checked outside run in every validation process',
    stress_seed_formula='39727+trial*347+stream*197+pattern*59',
    research_retention='P39 ordinary and Graph research reference; production performance target remains unachieved',
    independent_graph_comparison=json.loads((h/'p39_graph_abba.json').read_text())['summary'],
    limitations='Only contiguous non-transposed1024^3, alpha1 beta0, FP16/BF16; no arbitrary-range proof, no production integration, no transfer to later modified candidates.')
(h/'p39_validation.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p39_validation.py','analyze.py','stress_p39.cu','build_stress_p39.sh','stress_p39_build.log','validate_p39.sh','p39_validation_driver.log','p39_graph.log','p39_graph.json','stress_p39.log','p39_memcheck.log','p39_racecheck.log','p39_synccheck.log','p39_validation.json','bin/stress_p39','bin/compare_p39','p39_manifest.json']
files+=['p39_graph_abba.json','p39_graph_abba_manifest.json','p39_graph_abba_b0.log','prepare_uniform_validation.py']
(h/'p39_validation_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

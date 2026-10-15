"""Audit P45's own Graph, independent-stream guards and sanitizer evidence."""
import hashlib
import json
import math
import re
from pathlib import Path
from analyze import analyze
h=Path(__file__).resolve().parent;repo=h.parents[2]
assert (h/'p45_graph.log').read_bytes()==(h/'p45_graph_abba_b0.log').read_bytes()
def rows(name,prefix,count,guards=False):
    text=(h/name).read_text();r=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith(prefix)]
    assert len(r)==count,(name,len(r))
    assert all(x['pass']=='1' for x in r),name
    if guards:assert all(x['guards']=='1' and math.isfinite(float(x['ef'])) for x in r),name
    return r
for name in ['p45_balanced.log','p45_graph.log']:
    rows(name,'CHECK ',78)
    text=(h/name).read_text();inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('INSIDE_EQUAL ')]
    assert len(inside)==78 and all(x['errors']=='0' for x in inside)
graph=analyze(h/'p45_graph.log');assert all(x['graph']=='1' for x in graph['checks'])
(h/'p45_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
stress=rows('stress_p45.log','EXTRA ',108,True)
assert set(int(x['pattern']) for x in stress)==set(range(9))
for tool in ['memcheck','racecheck','synccheck']:
    name=f'p45_{tool}.log';rows(name,'EXTRA ',36,True)
    text=(h/name).read_text();assert ('0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors') in text,tool
for name in ['p45_balanced.log','p45_graph.log','stress_p45.log','p45_memcheck.log','p45_racecheck.log','p45_synccheck.log']:
    init=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in (h/name).read_text().splitlines() if line.startswith('COOPERATIVE_INIT ')]
    assert len(init)==2 and {x['dtype'] for x in init}=={'bf16','fp16'},name
    assert all(x['supported']=='1' and int(x['active_blocks_per_sm'])*int(x['sm_count'])>=int(x['grid'])==288 for x in init),name
assert '45937+trial*347+s*197+pat*59' in (h/'stress_p45.cu').read_text()
ordinary=json.loads((h/'p45_abba.json').read_text())
graph_abba=json.loads((h/'p45_graph_abba.json').read_text())
assert all(x['relative_reduction']>0 for result in [ordinary,graph_abba] for x in result['summary'])
out=dict(status='p45_accuracy_scoped_tools_validated_production_performance_target_unachieved',goal_achieved=False,
    ordinary_original_input_fp64_checks=78,graph_original_input_fp64_checks=78,graph_same_deterministic_configurations=True,
    graph_process_provenance='p45_graph.log is an exact copy of existing own p45_graph_abba_b0.log; no extra independent Graph sample',
    equal_per_column_exponents_per_mode=79872,independent_stress_configurations=108,independent_stress_calls=864,
    sanitizer_cases_per_tool=36,sanitizer_calls_per_tool=72,sanitizers=['memcheck','racecheck','synccheck'],sanitizer_kernel_scope='kns=int8_research',
    stress_dense_control='historical cuBLAS GemmEx; ordinary and Graph compare use production tuned Dense gate',
    stress_output_guards='128 floats before and after each output; two independent streams/workspaces',
    graph_timing_us={s['dtype']:s['corrected_us'] for s in graph['summaries']},
    cooperative_preprocessing='288CTAs, one unconditional grid.sync barrier; actual supported/resident capacity checked outside run in every validation process',
    stress_seed_formula='45937+trial*347+stream*197+pattern*59',
    research_retention='P45 ordinary and Graph research reference with its own complete evidence; production performance target remains unachieved',
    independent_ordinary_comparison=ordinary['summary'],independent_graph_comparison=graph_abba['summary'],
    limitations='Only contiguous non-transposed1024^3, alpha1 beta0, FP16/BF16; no arbitrary-range proof, no production integration, no transfer to later modified candidates.')
(h/'p45_validation.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p45_validation.py','analyze.py','stress_p45.cu','build_stress_p45.sh','stress_p45_build.log','validate_p45.sh','p45_validation_driver.log','p45_graph.log','p45_graph.json','stress_p45.log','p45_memcheck.log','p45_racecheck.log','p45_synccheck.log','p45_validation.json','bin/stress_p45','bin/compare_p45','p45_manifest.json']
files+=['p45_abba.json','p45_abba_manifest.json','p45_graph_abba.json','p45_graph_abba_manifest.json','p45_graph_abba_b0.log','prepare_uniform_validation.py']
(h/'p45_validation_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

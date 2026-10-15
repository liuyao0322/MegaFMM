"""Audit P50's own Graph, independent-stream guards and sanitizer evidence."""
import hashlib
import json
import math
import re
from pathlib import Path
from analyze import analyze
h=Path(__file__).resolve().parent;repo=h.parents[2]
assert (h/'p50_graph.log').read_bytes()==(h/'p50_graph_abba_b0.log').read_bytes()
def rows(name,prefix,count,guards=False):
    text=(h/name).read_text();r=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith(prefix)]
    assert len(r)==count,(name,len(r))
    assert all(x['pass']=='1' for x in r),name
    if guards:
        assert all(x['guards']=='1' and x['fixed_floor']=='1' and x['nonfinite']=='0' for x in r),name
        for x in r:
            v,d,g=map(float,[x['ef'],x['dense'],x['gate']]);floor=1e-3 if x['dtype']=='bf16' else 1e-4
            assert all(math.isfinite(y) for y in [v,d,g]) and v<=floor and v<=g,name
            assert math.isclose(g,3*d+floor,rel_tol=1e-10),name
        trials=3 if count==144 else 1
        assert {(x['dtype'],int(x['pattern']),int(x['trial']),int(x['stream'])) for x in r}=={(d,p,t,s) for d in ['bf16','fp16'] for p in range(12) for t in range(trials) for s in range(2)},name
    return r
for name in ['p50_balanced.log','p50_graph.log']:
    assert all(x['nonfinite']=='0' for x in rows(name,'CHECK ',78))
    text=(h/name).read_text();inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('INSIDE_EQUAL ')]
    assert len(inside)==78 and all(x['errors']=='0' for x in inside)
graph=analyze(h/'p50_graph.log');assert all(x['graph']=='1' for x in graph['checks'])
(h/'p50_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
stress=rows('stress_p50.log','EXTRA ',144,True)
assert set(int(x['pattern']) for x in stress)==set(range(12))
for tool in ['memcheck','racecheck','synccheck']:
    name=f'p50_{tool}.log';rows(name,'EXTRA ',48,True)
    text=(h/name).read_text();assert ('0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors') in text,tool
for name in ['p50_balanced.log','p50_graph.log','stress_p50.log','p50_memcheck.log','p50_racecheck.log','p50_synccheck.log']:
    init=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in (h/name).read_text().splitlines() if line.startswith('COOPERATIVE_INIT ')]
    assert len(init)==2 and {x['dtype'] for x in init}=={'bf16','fp16'},name
    assert all(x['supported']=='1' and int(x['active_blocks_per_sm'])*int(x['sm_count'])>=int(x['grid'])==288 for x in init),name
source=(h/'stress_p50.cu').read_text()
for required in ['50957+trial*401+s*211+pat*101','pat<12','quick?2:8','quick?1:3','dm.nonfinite==0&&cm.nonfinite==0&&ef(cm)<=limit&&fixed','pattern==9||pattern==10','pattern==11','(c%128)<64','(r%128)<64']:
    assert required in source,required

ordinary=json.loads((h/'p50_abba.json').read_text())
graph_abba=json.loads((h/'p50_graph_abba.json').read_text())
assert all(x['relative_reduction']>0 for result in [ordinary,graph_abba] for x in result['summary'])
out=dict(status='p50_accuracy_scoped_tools_validated_production_performance_target_unachieved',goal_achieved=False,
    ordinary_original_input_fp64_checks=78,graph_original_input_fp64_checks=78,graph_same_deterministic_configurations=True,
    graph_process_provenance='p50_graph.log is an exact copy of existing own p50_graph_abba_b0.log; no extra independent Graph sample',
    equal_per_column_exponents_per_mode=79872,independent_stress_configurations=144,independent_stress_calls=1152,
    sanitizer_cases_per_tool=48,sanitizer_calls_per_tool=96,sanitizers=['memcheck','racecheck','synccheck'],sanitizer_kernel_scope='kns=int8_research',
    stress_dense_control='historical cuBLAS GemmEx; ordinary and Graph compare use production tuned Dense gate',
    stress_additional_strict_fixed_floor='BF16 eF<=1e-3 and FP16 eF<=1e-4; original Dense-dependent gate also enforced',
    stress_patterns='0..11: original nine plus extreme/subnormal complementary, reversed, and alternating64K slabs; original rounded inputs and FP64AB',
    stress_output_guards='128 floats before and after each output; two independent streams/workspaces',
    graph_timing_us={s['dtype']:s['corrected_us'] for s in graph['summaries']},
    cooperative_preprocessing='288CTAs, one unconditional grid.sync barrier; actual supported/resident capacity checked outside run in every validation process',
    stress_seed_formula='50957+trial*401+stream*211+pattern*101',
    research_retention='P50 ordinary and Graph research reference with its own complete evidence; production performance target remains unachieved',
    independent_ordinary_comparison=ordinary['summary'],independent_graph_comparison=graph_abba['summary'],
    limitations='Only contiguous non-transposed1024^3, alpha1 beta0, FP16/BF16; no arbitrary-range proof, no production integration, no transfer to later modified candidates.')
(h/'p50_validation.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p50_validation.py','analyze.py','stress_p50.cu','build_stress_p50.sh','stress_p50_build.log','validate_p50.sh','p50_validation_driver.log','p50_graph.log','p50_graph.json','stress_p50.log','p50_memcheck.log','p50_racecheck.log','p50_synccheck.log','p50_validation.json','bin/stress_p50','bin/compare_p50','p50_manifest.json']
files+=['p50_abba.json','p50_abba_manifest.json','p50_graph_abba.json','p50_graph_abba_manifest.json','p50_graph_abba_b0.log','prepare_uniform_validation.py','p50_pack_bits_manifest.json','p50_edges_manifest.json','p50_edge_audit_manifest.json']
(h/'p50_validation_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

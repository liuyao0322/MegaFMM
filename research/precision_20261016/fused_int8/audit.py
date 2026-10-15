"""Audit actual numerical successes and failed hardware candidates separately."""
import hashlib
import json
import re
from pathlib import Path
r=Path(__file__).resolve().parent
repo=r.parents[2]
h=r.parent/'int8_hardware'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for path in list(h.glob('p*_manifest.json'))+list(r.glob('f*_manifest.json'))+[h/'tile_inside_adaptive_manifest.json',h/'f2_f3_manifest.json',h/'m4_manifest.json',r/'probe_manifest.json',r/'profile_manifest.json',r/'trace_f9_manifest.json',r/'trace_f11_manifest.json',r/'compile_stopped_f12_manifest.json',r/'layout_access_manifest.json',h/'m5_manifest.json',h/'m6_manifest.json',h/'m7_manifest.json',h/'m8_manifest.json',h/'m9_manifest.json',h/'m10_manifest.json',h/'alternating_hardware_manifest.json',h/'uniform_scale_arithmetic_manifest.json',h/'uniform_ablation_manifest.json',h/'requantized_hardware_manifest.json',h/'requantized_integer_arithmetic_manifest.json',h/'trace_p33_manifest.json']:
    for name,sha in json.loads(path.read_text()).items():assert digest(repo/name)==sha,(path,name)
for name,sha in json.loads((r.parent/'source_manifest.json').read_text())['production_unchanged_sha256'].items():
    assert digest(repo/name)==sha,name
frozen=sorted((p for p in r.glob('f[0-9]*') if p.is_dir() and re.fullmatch(r'f[0-9]+',p.name)),key=lambda p:int(p.name[1:]))
matching=[p.name for p in frozen if all((p/name).exists() and digest(r/name)==digest(p/name) for name in ['prototype.cuh','compare.cu','build.sh'])]
assert matching,'Working source must match a measured, frozen candidate'
current=matching[-1]
candidates={}
for label,failures in [('f6',66),('f7',60),('f8',0),('f9',0),('f10',0),('f11',0),('f13',0)]+([('f12',0)] if (r/'f12_manifest.json').exists() else []):
    j=json.loads((r/(label+'_balanced.json')).read_text())
    assert len(j['checks'])==72
    assert all(g['speedup_vs_sw']<1 for s in j['summaries'] for g in s['groups'])
    assert sum(x['pass']!='1' for x in j['checks'])==failures
    if not failures:assert all(int(x['nonfinite'])==0 for x in j['checks'])
    candidates[label]=dict(checks=72,failures=failures,performance_accepted=False,
        full_call_us={s['dtype']:s['corrected_us'] for s in j['summaries']},
        max_ef={s['dtype']:s['max_ef'] for s in j['summaries']})
lines=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in (r/'probe_fixed.log').read_text().splitlines() if l.startswith('PROBE ')]
assert len(lines)==2
assert all(all(int(x[k])==0 for k in ['raw_errors','high_errors','exponent_errors','quant_errors','integer_errors']) for x in lines)
models={}
for name,passes in [('tile_inside_adaptive',42),('tile_inside_interleave',42),('tile_inside_paired',39),('tile_inside_paired_extra',42),('bound_scale_model',36),('global_k128_model',42)]:
    rows=json.loads((h/(name+'.json')).read_text())['rows'];assert len(rows)==42
    assert sum(bool(x['pass_fixed_floor']) for x in rows)==passes
    models[name]=dict(cases=42,pass_fixed_floor=passes)
m8=json.loads((h/'paired_column_model.json').read_text())['rows']
assert len(m8)==144
for arm in ['adaptive_unpaired','relative_unpaired','relative_paired']:
    rows=[x for x in m8 if x['arm']==arm]
    assert len(rows)==48 and all(x['pass_fixed_floor'] and x['nonfinite']==0 and x['saturated_low_values']==0 for x in rows)
    models['m8_'+arm]=dict(fresh_configurations=48,passes=48,hardware_accepted=False)
m9=json.loads((h/'wide_quant_model.json').read_text())['rows']
assert len(m9)==180 and len({(x['dtype'],x['pattern'],x['seed']) for x in m9})==60
assert all(x['pass_fixed_floor'] and x['nonfinite']==0 and x['saturated_low_values']==0 for x in m9)
assert (h/'wide_quant_model.log').read_text().count('CONTROL ')==2
for width in [128,256,512]:
    rows=[x for x in m9 if x['quant_k']==width];assert len(rows)==60
    models['m9_quant_k'+str(width)]=dict(fresh_configurations=60,passes=60,hardware_accepted=False)
m10=json.loads((h/'requantized_model.json').read_text())['rows']
assert len(m10)==180 and len({(x['dtype'],x['pattern'],x['seed']) for x in m10})==60
assert all(x['pass_fixed_floor'] and x['nonfinite']==0 and x['saturated_low_values']==0 for x in m10)
assert (h/'requantized_model.log').read_text().count('NEGATIVE_CONTROL ')==1
for arm in ['baseline_k128','exactwide_k512','requantized_k512']:
    rows=[x for x in m10 if x['arm']==arm];assert len(rows)==60
    models['m10_'+arm]=dict(fresh_configurations=60,passes=60,hardware_accepted=False)
arithmetic=json.loads((h/'uniform_scale_arithmetic.json').read_text())
assert len(arithmetic['rows'])==460 and arithmetic['bitwise_errors']==0
assert arithmetic['scaled_arithmetic_checks']==9203680
assert all(x['bitwise_errors']==0 for x in arithmetic['rows'])
assert arithmetic['differing_scale_negative_control']['original']!=arithmetic['differing_scale_negative_control']['invalid_merged']
models['uniform_scale_arithmetic']=dict(scope='CPU restoration arithmetic only; not matrix or hardware acceptance',generated_tuples_per_dot_count=20008,exponents_per_dot_count=230,bitwise_errors=0,hardware_accepted=False)
abba=json.loads((h/'p17_abba.json').read_text())
assert abba['correctness_checks']==312 and abba['independent_numerical_configurations']==78
for label in ['p8','p9','p10','p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30','p31','p32','p33','p34','p38','p39']:
    j=json.loads((h/(label+'_balanced.json')).read_text());count=78 if label in ['p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30','p31','p32','p33','p34','p38','p39'] else 72
    assert len(j['checks'])==count and all(x['pass']=='1' and int(x['nonfinite'])==0 for x in j['checks'])
    assert all(g['speedup_vs_sw']<1 for z in j['summaries'] for g in z['groups'])
    candidates[label]=dict(checks=count,failures=0,performance_accepted=False,full_call_us={s['dtype']:s['corrected_us'] for s in j['summaries']})
fresh=json.loads((h/'alternating_hardware.json').read_text())
assert len(fresh['checks'])==12
for label,passes in [('p9',6),('p10',0)]:
    rows=[x for x in fresh['checks'] if x['candidate']==label]
    assert len(rows)==6 and sum(x['pass']=='1' for x in rows)==passes
    assert all(int(x['nonfinite'])==0 for x in rows)
    candidates[label]['fresh_alternating_checks']=6
    candidates[label]['fresh_alternating_passes']=passes
    candidates[label]['retained']=False
m7=json.loads((h/'alternating_inside_model.json').read_text())['rows']
assert len(m7)==12
for granularity,passes in [('K128',0),('column',6)]:
    rows=[x for x in m7 if x['inside_granularity']==granularity]
    assert len(rows)==6 and sum(x['pass_fixed_floor'] for x in rows)==passes
models['alternating_inside']=dict(fresh_configurations=6,paired_variants=2,coarse_passes=0,column_passes=6)
for label in ['p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30','p31','p32','p33','p34','p38','p39']:
    rows=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in (h/(label+'_balanced.log')).read_text().splitlines() if line.startswith('INSIDE_EQUAL ')]
    assert len(rows)==78 and all(x['errors']=='0' and x['elements']=='1024' for x in rows)
    j=json.loads((h/(label+'_balanced.json')).read_text())
    added=[x for x in j['checks'] if x['pattern']=='12'];assert len(added)==6 and all(x['pass']=='1' for x in added)
    candidates[label]['equal_per_column_exponents']=79872
    candidates[label]['additional_alternating_checks_passed']=6
    candidates[label]['retained']=label=='p39'
for label in ['p28','p29','p30','p31','p32','p33','p34','p38','p39']:
    reference=json.loads((h/'p27_balanced.json').read_text())['checks']
    rows=json.loads((h/(label+'_balanced.json')).read_text())['checks']
    assert all(all(a[k]==b[k] for k in ['dtype','pattern','trial','corrected_ef']) for a,b in zip(rows,reference))
assert json.loads((h/'p28_preflight/manifest.json').read_text())['device_executed'] is False
ordinary29=json.loads((h/'p29_abba.json').read_text())
graph29=json.loads((h/'p29_graph_abba.json').read_text())
for result in [ordinary29,graph29]:
    assert result['correctness_checks']==312 and result['independent_numerical_configurations']==78
assert all(x['relative_reduction']>0.03 for x in ordinary29['summary'])
assert all(x['relative_reduction']<0 for x in graph29['summary'])
ordinary30=json.loads((h/'p30_abba.json').read_text())
graph30=json.loads((h/'p30_graph_abba.json').read_text())
for result in [ordinary30,graph30]:
    assert result['correctness_checks']==312 and result['independent_numerical_configurations']==78
assert all(x['relative_reduction']>0.03 for x in ordinary30['summary'])
assert all(x['relative_reduction']<0 for x in graph30['summary'])
candidates['p27']['retained_for']='historical fully validated Graph fallback'
candidates['p30']['retained_for']='historical fully validated ordinary fallback'
candidates['p33']['retained_for']='historical fully validated ordinary fallback'
for label in ['p35','p36','p37']:
    j=json.loads((h/(label+'_balanced.json')).read_text());rr=j['checks']
    assert len(rr)==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in rr)
    inside=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in (h/(label+'_balanced.log')).read_text().splitlines() if line.startswith('INSIDE_EQUAL ')]
    assert len(inside)==78 and all(x['errors']=='0' and x['elements']=='1024' for x in inside)
    assert all(g['speedup_vs_sw']<1 for z in j['summaries'] for g in z['groups'])
    candidates[label]=dict(checks=78,failures=0,equal_per_column_exponents=79872,performance_accepted=False,retained=False,
        full_call_us={x['dtype']:x['corrected_us'] for x in j['summaries']},
        numerical_algorithm_changed='doubleRN existingK128 INT8 into commonK512; seven main/eight correction dots retained',
        validation_scope='ordinary78 only; no ABBA/Graph/stress/sanitizer transfer')
candidates['p39']['retained_for']='ordinary and Graph research reference; own fresh validation only'
candidates['p38']['validation_scope']='ordinary78 plus independent ordinaryABBA312; no Graph/stress/sanitizer transfer'
ordinary39=json.loads((h/'p39_abba.json').read_text())
graph39=json.loads((h/'p39_graph_abba.json').read_text())
for result in [ordinary39,graph39]:
    assert result['correctness_checks']==312 and result['independent_numerical_configurations']==78
    assert all(x['relative_reduction']>0 for x in result['summary'])
validation39=json.loads((h/'p39_validation.json').read_text())
assert validation39['independent_stress_configurations']==108 and validation39['sanitizer_cases_per_tool']==36
ordinary38=json.loads((h/'p38_abba.json').read_text())
assert ordinary38['correctness_checks']==312 and ordinary38['independent_numerical_configurations']==78
assert ordinary38['summary'][0]['relative_reduction']>0 and ordinary38['summary'][1]['relative_reduction']<0
trace33=json.loads((h/'trace_p33_summary.json').read_text())
assert trace33['trace_cases']==10 and trace33['new_seed_cases']==8
assert trace33['bitwise_output_comparisons']==10485760 and trace33['bitwise_differences']==0
assert trace33['warp_records']==23040 and trace33['interpreted_warp0_records']==2880
assert all(x['excluded_nonwarp0_early_grid_return_stamps']==2016 for x in trace33['cases'])
assert all(0<x['median_paired_relative_change']<.02 for x in trace33['perturbation'])
global_frozen=sorted((p for p in h.glob('p[0-9]*') if p.is_dir() and re.fullmatch(r'p[0-9]+',p.name)),key=lambda p:int(p.name[1:]))
global_matching=[p.name for p in global_frozen if all((p/name).exists() and digest(h/name)==digest(p/name) for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh'])]
assert global_matching,'Global working source must match a measured frozen candidate'
global_current=global_matching[-1]
assert digest(h/'bin/compare')==digest(h/'bin'/('compare_'+global_current))
profile13=json.loads((h/'profile_b_p13_stats.json').read_text())['metrics']
profile17=json.loads((h/'profile_b_p17_stats.json').read_text())['metrics']
bank_key='l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'
assert int(profile13[bank_key]['value'])>500000
assert int(profile17[bank_key]['value'])<2000
assert 'cases=1048576 exponents=-300..300 errors=0' in (h/'p14_pow2.log').read_text()
out=dict(status='progress_accuracy_validated_performance_rejected',goal_achieved=False,production_unchanged=True,
    working_candidate=current,working_global_candidate=global_current,compact_pow2_equivalence_cases=1048576,retained_global_candidate='p39 ordinary and Graph research reference, own ordinary/Graph78, fresh9-pattern stress108/864calls, scoped tools36/72calls each. p33/p27 remain historical validated fallbacks. No production performance winner.',compile_stopped_candidates=['F12: no numerical or latency measurement','P28 initial compile-only preflight: missing read-to-reuse hardware fence, superseded before device execution'],candidates=candidates,numerical_models=models,
    diagnostic_traces=['F9 role-level, one BF16 uniform case','F11 pack subphases, one BF16 uniform case','P33 warp0 paths,10 cases, zero bitwise differences; other warp records excluded from interpretation'],p38_independent_performance=ordinary38['summary'],p33_phase_trace=trace33,probe_scope='one P5 K128 tile, fresh uniform CPU inputs, two dtypes; not full-call acceptance',
    p20_validation=json.loads((h/'p20_validation.json').read_text()),
    p20_independent_performance=json.loads((h/'p20_abba.json').read_text())['summary'],
    p24_validation=json.loads((h/'p24_validation.json').read_text()),
    p24_independent_performance=json.loads((h/'p24_abba.json').read_text())['summary'],
    p26_validation=json.loads((h/'p26_validation.json').read_text()),
    p26_independent_performance=json.loads((h/'p26_abba.json').read_text())['summary'],
    p27_validation=json.loads((h/'p27_validation.json').read_text()),
    p27_independent_performance=json.loads((h/'p27_abba.json').read_text())['summary'],
    p29_validation=json.loads((h/'p29_validation.json').read_text()),
    p29_independent_performance=ordinary29['summary'],p29_graph_regression=graph29['summary'],
    p30_validation=json.loads((h/'p30_validation.json').read_text()),
    p30_independent_performance=ordinary30['summary'],p30_graph_regression=graph30['summary'],
    p33_validation=json.loads((h/'p33_validation.json').read_text()),
    p39_validation=validation39,p39_independent_performance=ordinary39['summary'],p39_graph_performance=graph39['summary'],
    requantized_hardware=json.loads((h/'requantized_hardware.json').read_text()),
    uniform_scale_ablation=json.loads((h/'uniform_ablation.json').read_text()),
    validations_not_transferred='P20, P24, P26, P27, P29 and P30 each have their own Graph78, independent stress84/672calls and scoped sanitizers28/56calls per tool. P7 remains historical; fusion candidates retain only their separately recorded scope. P33 has its own9-pattern108/864stress and36/72scoped tools; rejectedP31 has only9-pattern stress, P32/P34/P35/P36/P37 only ordinary, P38 ordinary plus ABBA. P39 owns its Graph78, fresh9-pattern108/864stress and36/72tools. No transfer to later changes.',
    remaining='Resolve the end-to-end latency gap, then independent performance repeats, Graph/dual-stream/sanitizer and production integration.')
(r/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))

"""Audit actual numerical successes and failed hardware candidates separately."""
import hashlib
import json
import re
from pathlib import Path
r=Path(__file__).resolve().parent
repo=r.parents[2]
h=r.parent/'int8_hardware'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for path in list(h.glob('p*_manifest.json'))+list(r.glob('f*_manifest.json'))+list((r.parent/'fused_native').glob('n[0-9]*_manifest.json'))+[h/'tile_inside_adaptive_manifest.json',h/'f2_f3_manifest.json',h/'m4_manifest.json',r/'probe_manifest.json',r/'profile_manifest.json',r/'trace_f9_manifest.json',r/'trace_f11_manifest.json',r/'compile_stopped_f12_manifest.json',r/'layout_access_manifest.json',h/'m5_manifest.json',h/'m6_manifest.json',h/'m7_manifest.json',h/'m8_manifest.json',h/'m9_manifest.json',h/'m10_manifest.json',h/'alternating_hardware_manifest.json',h/'uniform_scale_arithmetic_manifest.json',h/'uniform_ablation_manifest.json',h/'requantized_hardware_manifest.json',h/'requantized_integer_arithmetic_manifest.json',h/'trace_p33_manifest.json',h/'nonnegative_max_arithmetic_manifest.json',h/'native_quant_exponent_arithmetic_manifest.json',h/'m11_manifest.json',h/'m12_manifest.json',h/'m13_manifest.json',h/'m14_manifest.json',h/'m15_premeasurement_manifest.json',h/'m15_manifest.json',h/'m16_premeasurement_manifest.json',h/'m16_manifest.json',h/'bf16_hardware_manifest.json']:
    for name,sha in json.loads(path.read_text()).items():
        target=repo/name
        # Frozen restoration records describe the then-working P51 snapshot.
        # Keep those manifests untouched when a later candidate becomes current.
        if path.name in ['p54_restore_manifest.json','p55_restore_manifest.json']:
            rel=target.relative_to(h) if target.is_relative_to(h) else None
            if rel is not None and str(rel) in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh','bin/compare']:
                target=h/'bin/compare_p51' if str(rel)=='bin/compare' else h/'p51'/str(rel)
        assert digest(target)==sha,(path,name,target)
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
m11=json.loads((h/'bf16_correction_model.json').read_text())['rows']
assert len(m11)==216 and len({(x['dtype'],x['pattern'],x['seed']) for x in m11})==72
assert all(x['pass_fixed_floor'] and x['nonfinite']==0 and x['nonfinite_correction_operands']==0 for x in m11)
assert sum(line.startswith('CONTROL ') for line in (h/'bf16_correction_model.log').read_text().splitlines())==2
m11_summary=json.loads((h/'m11_summary.json').read_text())
assert all(m11_summary['fixed_scale16_overflow_negative_control'].values())
for arm in ['int8_k128','bf16_low_unscaled','bf16_low_scale16']:
    rows=[x for x in m11 if x['arm']==arm];assert len(rows)==72
    models['m11_'+arm]=dict(fresh_configurations=72,passes=72,hardware_accepted=False)
m12=json.loads((h/'native_correction_model.json').read_text())['rows']
assert len(m12)==216 and len({(x['dtype'],x['pattern'],x['seed']) for x in m12})==72
assert all(x['pass_fixed_floor'] and x['nonfinite']==0 and x['nonfinite_correction_operands']==0 for x in m12)
m12_summary=json.loads((h/'m12_summary.json').read_text())
assert sum(x['sampled_transforms'] for x in m12_summary['scalar_range_controls'])==515072
assert all(x['scaled_low_nonfinite']==0 for x in m12_summary['scalar_range_controls'])
for arm in ['int8_k128','bf16_low_unscaled','native_low_scaled']:
    rows=[x for x in m12 if x['arm']==arm];assert len(rows)==72
    models['m12_'+arm]=dict(fresh_configurations=72,passes=72,hardware_accepted=False)
m13_summary=json.loads((h/'m13_summary.json').read_text())
m13=json.loads((h/'leaf_reuse_model.json').read_text())
assert len(m13['rows'])==144 and len({(x['dtype'],x['pattern'],x['seed']) for x in m13['rows']})==72
assert all(x['pass_fixed_floor'] and x['pass_original_gate'] and x['nonfinite']==0 and x['nonfinite_correction_operands']==0 for x in m13['rows'])
assert len(m13['controls'])==2 and all(x['native_arm_matches_m12'] for x in m13['controls'])
for arm in ['native_low_scaled','native_leaf_scaled']:
    rows=[x for x in m13['rows'] if x['arm']==arm];assert len(rows)==72
    models['m13_'+arm]=dict(fresh_configurations=72,passes=72,hardware_accepted=False)
m14_summary=json.loads((h/'m14_summary.json').read_text())
m14=json.loads((h/'k_stream_native_model.json').read_text())
assert len(m14['rows'])==288 and len({(x['dtype'],x['pattern'],x['seed']) for x in m14['rows']})==72
assert all(x['pass_fixed_floor'] and x['pass_original_gate'] and x['nonfinite']==0 and x['nonfinite_correction_operands']==0 for x in m14['rows'])
assert len(m14['controls'])==2 and all(x['full_k128_matches_m13_leaf'] for x in m14['controls'])
for arm in ['full_k128','full_k64','chunk_k64','stream_k64']:
    rows=[x for x in m14['rows'] if x['arm']==arm];assert len(rows)==72
    models['m14_'+arm]=dict(fresh_configurations=72,passes=72,hardware_accepted=False)
m15_summary=json.loads((h/'m15_summary.json').read_text())
m15=json.loads((h/'fine_quant_model.json').read_text())
assert len(m15['rows'])==420 and m15['configurations']==84
assert all(x['pass_original_gate'] and x['inside_cpu_errors']==0 for x in m15['rows'])
assert sum(not x['pass_fixed_floor'] for x in m15['rows'])==6
assert m15_summary['fine_original_and_fixed_passes']==84 and not m15_summary['hardware_accepted']
m16_summary=json.loads((h/'m16_summary.json').read_text())
m16=json.loads((h/'exponent_restore_model.json').read_text())
assert len(m16['rows'])==850 and m16['scalar']['cases']==5409 and m16['scalar']['bit_errors']==0
assert all(x['pass_original_gate'] and x['inside_cpu_errors']==0 for x in m16['rows'])
assert all(x['pass_fixed_floor'] for x in m16['rows'] if not x['arm'].startswith('coarse'))
assert m16_summary['paired_arm_ef_equalities']==425 and m16_summary['correction_bit_differences']==0
assert not m16_summary['hardware_accepted'] and not m16_summary['goal_achieved']
for label,model in [('m15',m15),('m16',m16)]:
    models[label+'_fine_quantization']=dict(rows=len(model['rows']),hardware_accepted=False,performance_measured=False)
nonnegative=json.loads((h/'nonnegative_max_arithmetic.json').read_text())
assert nonnegative['generated_values']==1048576 and nonnegative['bitwise_errors']==0
assert nonnegative['signed_negative_control']['invalid_unsigned_bit_max']!=nonnegative['signed_negative_control']['correct_float_max']
models['nonnegative_max_arithmetic']=nonnegative
native_exponent=json.loads((h/'native_quant_exponent_arithmetic.json').read_text())
assert native_exponent['cases']==64384 and native_exponent['errors']==0
models['native_quant_exponent_arithmetic']=native_exponent
arithmetic=json.loads((h/'uniform_scale_arithmetic.json').read_text())
assert len(arithmetic['rows'])==460 and arithmetic['bitwise_errors']==0
assert arithmetic['scaled_arithmetic_checks']==9203680
assert all(x['bitwise_errors']==0 for x in arithmetic['rows'])
assert arithmetic['differing_scale_negative_control']['original']!=arithmetic['differing_scale_negative_control']['invalid_merged']
models['uniform_scale_arithmetic']=dict(scope='CPU restoration arithmetic only; not matrix or hardware acceptance',generated_tuples_per_dot_count=20008,exponents_per_dot_count=230,bitwise_errors=0,hardware_accepted=False)
abba=json.loads((h/'p17_abba.json').read_text())
assert abba['correctness_checks']==312 and abba['independent_numerical_configurations']==78
for label in ['p8','p9','p10','p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30','p31','p32','p33','p34','p38','p39','p40','p41','p42','p43','p44','p45','p46','p47','p48','p49','p50','p51','p52','p53','p54','p55','p56','p57','p58','p59','p60']:
    j=json.loads((h/(label+'_balanced.json')).read_text());count=78 if label in ['p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30','p31','p32','p33','p34','p38','p39','p40','p41','p42','p43','p44','p45','p46','p47','p48','p49','p50','p51','p52','p53','p54','p55','p56','p57','p58','p59','p60'] else 72
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
for label in ['p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30','p31','p32','p33','p34','p38','p39','p40','p41','p42','p43','p44','p45','p46','p47','p48','p49','p50','p51','p52','p53','p54','p55','p56','p57','p58','p59','p60']:
    rows=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in (h/(label+'_balanced.log')).read_text().splitlines() if line.startswith('INSIDE_EQUAL ')]
    assert len(rows)==78 and all(x['errors']=='0' and x['elements']=='1024' for x in rows)
    j=json.loads((h/(label+'_balanced.json')).read_text())
    added=[x for x in j['checks'] if x['pattern']=='12'];assert len(added)==6 and all(x['pass']=='1' for x in added)
    candidates[label]['equal_per_column_exponents']=79872
    candidates[label]['additional_alternating_checks_passed']=6
    candidates[label]['retained']=label=='p56'
for label in ['p28','p29','p30','p31','p32','p33','p34','p38','p39','p40','p41','p42']:
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
candidates['p39']['retained_for']='historical fully validated ordinary and Graph fallback'
candidates['p45']['retained_for']='historical fully validated ordinary and Graph fallback'
candidates['p46']['retained_for']='historical fully validated ordinary and Graph fallback'
candidates['p47']['retained_for']='historical fully validated ordinary and Graph fallback'
candidates['p50']['retained_for']='historical fully validated ordinary and Graph fallback'
candidates['p51']['retained_for']='historical fully validated ordinary and Graph fallback'
candidates['p56']['retained_for']='ordinary and Graph research reference; own fresh validation only'
candidates['p38']['validation_scope']='ordinary78 plus independent ordinaryABBA312; no Graph/stress/sanitizer transfer'
ordinary39=json.loads((h/'p39_abba.json').read_text())
graph39=json.loads((h/'p39_graph_abba.json').read_text())
for result in [ordinary39,graph39]:
    assert result['correctness_checks']==312 and result['independent_numerical_configurations']==78
    assert all(x['relative_reduction']>0 for x in result['summary'])
validation39=json.loads((h/'p39_validation.json').read_text())
assert validation39['independent_stress_configurations']==108 and validation39['sanitizer_cases_per_tool']==36
local_controls={}
for label in ['p40','p41','p42']:
    result=json.loads((h/(label+'_abba.json')).read_text())
    assert result['correctness_checks']==312 and result['independent_numerical_configurations']==78
    candidates[label]['retained']=False
    candidates[label]['validation_scope']='own ordinary78 plus ordinaryABBA312; no fresh stress/sanitizer acceptance or certificate transfer'
    local_controls[label]=dict(ordinary=result['summary'])
    assert all(x['relative_reduction']<.03 for x in result['summary'])
graph40=json.loads((h/'p40_graph_abba.json').read_text())
assert graph40['correctness_checks']==312 and graph40['independent_numerical_configurations']==78
local_controls['p40']['graph']=graph40['summary']
candidates['p40']['validation_scope']+='; additionally own GraphABBA312, no independent own stress/tools'
bf16_hardware=json.loads((h/'bf16_hardware_ablation.json').read_text())
for x in bf16_hardware['candidates']:
    candidates[x['label']].update(x)
ordinary45=json.loads((h/'p45_abba.json').read_text())
graph45=json.loads((h/'p45_graph_abba.json').read_text())
for result in [ordinary45,graph45]:
    assert result['correctness_checks']==312 and result['independent_numerical_configurations']==78
    assert all(x['relative_reduction']>0 for x in result['summary'])
validation45=json.loads((h/'p45_validation.json').read_text())
assert validation45['independent_stress_configurations']==108 and validation45['sanitizer_cases_per_tool']==36
candidates['p45']['validation_scope']='own ordinary/Graph78, each independentABBA312; fresh9-pattern108configurations/864calls, scoped tools36/72calls each'
candidates['p45']['numerical_algorithm_changed']='native T correction high aliases + p11FP16/p8BF16 scaled low, all eight correction dots; two maps/eleven planes per side'
ordinary46=json.loads((h/'p46_abba.json').read_text())
graph46=json.loads((h/'p46_graph_abba.json').read_text())
for result in [ordinary46,graph46]:
    assert result['correctness_checks']==312 and result['independent_numerical_configurations']==78
    assert all(x['relative_reduction']>.03 for x in result['summary'])
validation46=json.loads((h/'p46_validation.json').read_text())
assert validation46['independent_stress_configurations']==108 and validation46['sanitizer_cases_per_tool']==36
for name in ['pack.cuh','compare.cu','pow2.cuh']:assert digest(h/'p46'/name)==digest(h/'p45'/name)
candidates['p46']['validation_scope']='own ordinary/Graph78, each independentABBA312; fresh9-pattern108configurations/864calls, scoped tools36/72calls each'
candidates['p46']['numerical_algorithm_changed']='early corrected-leaf recombination; reuse loaded native highs, all seven main/eight cross dots, native p11/8 lows unchanged'
ordinary47=json.loads((h/'p47_abba.json').read_text())
graph47=json.loads((h/'p47_graph_abba.json').read_text())
for result in [ordinary47,graph47]:
    assert result['correctness_checks']==312 and result['independent_numerical_configurations']==78
    assert all(x['relative_reduction']>.03 for x in result['summary'])
validation47=json.loads((h/'p47_validation.json').read_text())
assert validation47['independent_stress_configurations']==108 and validation47['sanitizer_cases_per_tool']==36
for name in ['corrected.cuh','compare.cu','pow2.cuh']:assert digest(h/'p47'/name)==digest(h/'p46'/name)
old=json.loads((h/'p46_balanced.json').read_text())['checks']
new=json.loads((h/'p47_balanced.json').read_text())['checks']
assert all(x['corrected_ef']==y['corrected_ef'] for x,y in zip(old,new))
candidates['p47']['validation_scope']='own ordinary/Graph78, each independentABBA312; fresh9-pattern108configurations/864calls, scoped tools36/72calls each; separate54configuration full-plane bit comparison'
candidates['p47']['numerical_algorithm_changed']='storage-only direct native high/low writes; unchanged explicit RN transforms and P46 consumer; shared requests retained'
pack47=json.loads((h/'probe_p47_pack.json').read_text())
assert pack47['configurations']==54 and pack47['native_value_comparisons']==311427072 and pack47['exponent_comparisons']==55296
assert pack47['bitwise_errors']==0
ordinary48=json.loads((h/'p48_abba.json').read_text())
graph48=json.loads((h/'p48_graph_abba.json').read_text())
for result in [ordinary48,graph48]:
    assert result['correctness_checks']==312 and result['independent_numerical_configurations']==78
assert all(x['relative_reduction']<.03 for x in ordinary48['summary'])
assert all(x['relative_reduction']<0 for x in graph48['summary'])
candidates['p48']['validation_scope']='own ordinary78 plus independent ordinary/GraphABBA312 each; no own fresh stress or sanitizer certificate, rejected performance'
static48=json.loads((h/'p47_p48_static.json').read_text())
assert len(static48['device_normalized_instruction_text_identical'])==4 and all(static48['device_normalized_instruction_text_identical'].values())
native_pack_controls={}
for label in ['p49','p50','p51']:
    modes={mode:json.loads((h/(label+suffix)).read_text()) for mode,suffix in [('ordinary','_abba.json'),('graph','_graph_abba.json')]}
    for result in modes.values():
        assert result['correctness_checks']==312 and result['independent_numerical_configurations']==78
        assert all((x['relative_reduction']<0 if label=='p49' else 0<x['relative_reduction']<.03) for x in result['summary'])
    bits=json.loads((h/f'probe_{label}_pack.json').read_text())
    assert bits['configurations']==54 and bits['native_value_comparisons']==311427072 and bits['exponent_comparisons']==55296 and bits['bitwise_errors']==0
    for name in ['corrected.cuh','pow2.cuh']+([] if label=='p51' else ['compare.cu']):assert digest(h/label/name)==digest(h/'p47'/name)
    native_pack_controls[label]=dict(ordinary=modes['ordinary']['summary'],graph=modes['graph']['summary'],packing_bits=bits)
    candidates[label]['numerical_algorithm_changed']='storage/geometry only: all-native seven high/four low planes; unchanged P47 consumer, RN transforms and exact inside'
    candidates[label]['validation_scope']=('own ordinary78 and independent ordinary/Graph ABBA312 each,54fresh full-plane bit configurations; rejected performance, no own stress/tools' if label=='p49' else 'own ordinary/Graph78, independent ABBA312 each,54fresh full-plane bits,18additional packing edges, fresh12-pattern144configurations/1152calls, scoped tools48/96calls each')
validation50=json.loads((h/'p50_validation.json').read_text())
assert validation50['independent_stress_configurations']==144 and validation50['independent_stress_calls']==1152
assert validation50['sanitizer_cases_per_tool']==48 and validation50['sanitizer_calls_per_tool']==96
edges50=json.loads((h/'probe_p50_edges.json').read_text())
assert len(edges50['rows'])==18 and all(x['pass']=='1' for x in edges50['rows'])
native_pack_controls['p50'].update(validation=validation50,packing_edges=edges50)
validation51=json.loads((h/'p51_validation.json').read_text())
assert validation51['independent_stress_configurations']==144 and validation51['independent_stress_calls']==1152
assert validation51['sanitizer_cases_per_tool']==48 and validation51['sanitizer_calls_per_tool']==96
edges51=json.loads((h/'probe_p51_edges.json').read_text())
assert len(edges51['rows'])==18 and all(x['pass']=='1' for x in edges51['rows'])
assert '(c%64)<32' in (h/'stress_p51.cu').read_text()
assert all(json.loads((h/'p51_reproduction.json').read_text())['reproduced'].values())
native_pack_controls['p51'].update(validation=validation51,packing_edges=edges51)
candidates['p51']['numerical_algorithm_changed']='geometry only A16x128/B32x32 packing,384CTA cooperative grid, diagnostics128A/256B; unchanged P47 consumer/RN/planes/inside'

consumer52=json.loads((h/'p52_hardware.json').read_text())
assert not consumer52['performance_accepted'] and consumer52['retained']=='P51' and not consumer52['goal_achieved']
assert consumer52['ordinary_original_fp64_checks']==78 and consumer52['equal_inside_exponents']==79872
assert consumer52['layout']['bit_errors']==0 and consumer52['layout']['configurations']==6
assert not consumer52['fresh_stream_guard_acceptance'] and not consumer52['sanitizer_acceptance']
for mode in ['ordinary','graph']:
    data=json.loads((h/('p52_abba.json' if mode=='ordinary' else 'p52_graph_abba.json')).read_text())
    assert data['correctness_checks']==312 and data['independent_numerical_configurations']==78
    assert all(x['relative_reduction']<-.4 for x in data['summary'])
for name in ['pack.cuh','compare.cu','pow2.cuh']:assert digest(h/'p52'/name)==digest(h/'p51'/name)
candidates['p52']['validation_scope']='own six synthetic pre-MMA TMA layouts, standard78, independent ordinary/GraphABBA312 each and own two-kernel profile; rejected performance, no own fresh dual-stream or sanitizer certificate'
candidates['p52']['numerical_algorithm_changed']='native m64n16k16/eight-value fragments, two actual40960Bslots/shared82048B/grid256; fullK512/all7+8dots/RN/planes/inside preserved'
proposal53=json.loads((h/'p53_proposal_cpu.json').read_text())
assert proposal53['gpu_executed'] is False and proposal53['run_kernel_count']==3
assert all(proposal53['unchanged'].values())

leaf53=json.loads((h/'p53_hardware.json').read_text())
assert not leaf53['performance_accepted'] and leaf53['retained']=='P51' and not leaf53['goal_achieved']
assert leaf53['ordinary_original_fp64_checks']==78 and leaf53['equal_inside_exponents']==79872
assert leaf53['layout']['bit_errors']==0 and leaf53['layout']['configurations']==6
assert leaf53['layout']['recombine_output_bit_comparisons']==3145728 and leaf53['layout']['recombine_bit_errors']==0
assert not leaf53['fresh_stream_guard_acceptance'] and not leaf53['sanitizer_acceptance']
assert len(leaf53['own_profile']['kernels'])==3
for mode in ['ordinary','graph']:
    data=json.loads((h/('p53_abba.json' if mode=='ordinary' else 'p53_graph_abba.json')).read_text())
    assert data['correctness_checks']==312 and data['independent_numerical_configurations']==78
    assert all(x['relative_reduction']<-.3 for x in data['summary'])
for name in ['compare.cu','pow2.cuh']:assert digest(h/'p53'/name)==digest(h/'p51'/name)
candidates['p53']['validation_scope']='own six synthetic pre-MMA TMA layouts and three scalar GPU recombination trials, standard78, independent ordinary/GraphABBA312 each and own three-kernel profile; rejected performance, no own fresh dual-stream or sanitizer certificate'
candidates['p53']['numerical_algorithm_changed']='native N32/fullK512/all7+8dots/RN/planes/inside preserved, one leaf per CTA/grid896/two49152Bslots/Shared98432B, extra7MiB FP32leaf workspace and same-stream scalar recombine kernel; three full kernels'
proposal54=json.loads((h/'p54_proposal_cpu.json').read_text())
assert proposal54['gpu_executed'] is False and proposal54['run_kernel_count']==3
assert all(proposal54['unchanged'].values()) and proposal54['recombine_kernel_byte_identical_to_p53']

leaf54=json.loads((h/'p54_hardware.json').read_text())
assert not leaf54['performance_accepted'] and leaf54['retained']=='P51' and not leaf54['goal_achieved']
assert leaf54['ordinary_original_fp64_checks']==78 and leaf54['equal_inside_exponents']==79872
assert leaf54['layout']['bit_errors']==0 and leaf54['layout']['native_uint16_comparisons_including_cta_reuse']==276824064
assert leaf54['layout']['recombine_output_bit_comparisons']==3145728 and leaf54['layout']['recombine_bit_errors']==0
assert not leaf54['fresh_stream_guard_acceptance'] and not leaf54['sanitizer_acceptance']
assert len(leaf54['own_profile']['kernels'])==3
assert leaf54['own_profile']['kernels'][1]['launch__registers_per_thread']['value']=='106'
for suffix in ['abba','graph_abba','shape_abba','shape_graph_abba']:
    data=json.loads((h/f'p54_{suffix}.json').read_text())
    assert data['correctness_checks']==312 and all(x['relative_reduction']<0 for x in data['summary'])
for name in ['compare.cu','pow2.cuh']:assert digest(h/'p54'/name)==digest(h/'p51'/name)
candidates['p54']['validation_scope']='own six synthetic pre-MMA layouts, three scalar GPU recombination trials, standard78, independent P53/P54 and P51/P54 ordinary/Graph312 each, own three-kernel profile; rejected, no own fresh dual-stream matrix stress/tools'
candidates['p54']['numerical_algorithm_changed']='native N64/fullK512/all7+8dots/RN preserved, grid448/two65536Bslots/Shared131200B, same three-kernel leaf workspace/recombination as P53'

proposal55=json.loads((h/'p55_proposal_cpu.json').read_text())
assert proposal55['proposal_only'] and not proposal55['gpu_executed'] and not proposal55['cpp_generated']
assert proposal55['retained']=='P51' and proposal55['run_kernels']==2
assert proposal55['workspace_bytes']==12329472 and proposal55['source_mma_instructions_per_cta']==352
assert proposal55['source_tma_bytes_whole_grid']==138412032 and proposal55['source_tma_commands_whole_grid']==15360
assert not proposal55['actual_capacity_checked'] and not proposal55['hardware_accepted']


fine55=json.loads((h/'p55_hardware.json').read_text())
assert not fine55['performance_accepted'] and fine55['retained']=='P51' and not fine55['goal_achieved']
assert fine55['ordinary_original_fp64_checks']==78 and fine55['equal_inside_exponents']==79872
assert fine55['pre_mma_pack']['configurations']==96 and fine55['pre_mma_pack']['errors']==0
assert fine55['pre_mma_layout']['configurations']==6 and fine55['pre_mma_layout']['errors']==0
assert fine55['extra_original_input']['original_gates']==84 and fine55['extra_original_input']['original_gate_failures']==0
assert all(x['fixed_floor_failures']==0 for x in fine55['extra_original_input']['summary'].values())
assert not fine55['fresh_dual_stream_reuse_acceptance'] and not fine55['sanitizer_acceptance']
assert [x['launch__registers_per_thread']['value'] for x in fine55['own_profile']['kernels']]==['40','182']
for suffix in ['abba','graph_abba']:
    data=json.loads((h/f'p55_{suffix}.json').read_text())
    assert data['correctness_checks']==312 and all(x['relative_reduction']<-.6 for x in data['summary'])
candidates['p55']['validation_scope']='own96 CPU-oracle packing/scalar/exponent configurations and six mixed TMA layouts before first MMA, standard78, own fresh84 single-stream matrix gates/guards, independent ordinary/GraphABBA312 each, own two-kernel profile; rejected, no own dual-stream/reuse stress or tools'
candidates['p55']['numerical_algorithm_changed']='A8/B8 direct fullK512 INT8 quantization,7native9integer planes,128x768prep,48batches/8slots/fine exponent metadata/post corrections; all7main8cross/exactinside/originalgates preserved'


static56=json.loads((h/'p56_hardware.json').read_text())
assert static56['research_retained'] and static56['retained']=='P56' and not static56['goal_achieved']
assert not static56['production_performance_accepted'] and static56['printed_per_case_ef_matches_p51']
assert static56['standard_original_fp64_gates']==78 and static56['exact_inside_exponents_per_mode']==79872
assert static56['own_pre_mma_layout']['errors']==0 and static56['own_pre_mma_layout']['configurations']==6
assert static56['own_validation']['independent_stress_configurations']==168 and static56['own_validation']['sanitizer_cases_per_tool']==56
assert static56['own_profile']['kernels'][1]['launch__registers_per_thread']['value']=='128'
for suffix in ['abba','graph_abba']:
    data=json.loads((h/f'p56_{suffix}.json').read_text())
    assert data['correctness_checks']==312 and all(x['relative_reduction']>.03 for x in data['summary'])
for name in ['pack.cuh','compare.cu','pow2.cuh']:assert digest(h/'p56'/name)==digest(h/'p51'/name)
c56=(h/'p56/corrected.cuh').read_text()
assert c56.replace(' #pragma unroll 7\n for(int step=0;step<7;++step)',' #pragma unroll 1\n for(int step=0;step<7;++step)')==(h/'p51/corrected.cuh').read_text()
candidates['p56']['validation_scope']='own six newly seeded pre-MMA TMA layouts, original78, independent ordinary/Graph312 each, fresh14pattern168configs/1344interleaved calls and output/workspace guards, scoped56configs/112calls per tool, own static/two-kernel profile; research retained, production target unachieved'
candidates['p56']['numerical_algorithm_changed']='Exactly one consumer pragma unroll1 to7; all remaining source bytes identical P51, same RN graph/fullK512/exactinside/all7+8dots/planes/producer'

producer57=json.loads((h/'p57_hardware.json').read_text())
assert not producer57['performance_accepted'] and producer57['retained']=='P56' and not producer57['goal_achieved']
assert producer57['standard_original_fp64_gates']==78 and producer57['exact_inside_exponents']==79872
assert producer57['own_pre_mma_layout']['configurations']==6 and producer57['own_pre_mma_layout']['errors']==0
assert producer57['printed_per_case_ef_matches_p56'] and not producer57['fresh_dual_stream_reuse_acceptance'] and not producer57['sanitizer_acceptance']
for suffix in ['abba','graph_abba']:
    data=json.loads((h/f'p57_{suffix}.json').read_text())
    assert data['correctness_checks']==312 and all(x['relative_reduction']<.01 for x in data['summary'])
candidates['p57']['validation_scope']='Own six pre-MMA TMA cases, standard78, independent ordinary/Graph312each, own static/two-kernel profile; rejected, no fresh dual-stream reuse stress or scoped tools, byte-exact P56 restoration plus new-process78'
candidates['p57']['numerical_algorithm_changed']='One added producer pragma unroll7; all other source bytes identical P56, same numerical graph and all7+8dots/fullK512/inside/fences'

resident58=json.loads((h/'p58_hardware.json').read_text())
assert not resident58['performance_accepted'] and resident58['retained']=='P56' and not resident58['goal_achieved']
assert resident58['standard_original_fp64_gates']==78 and resident58['exact_inside_exponents']==79872
assert resident58['own_pack_bits']['configurations']==54 and resident58['own_edge_bits']['configurations']==18
assert resident58['own_pack_bits']['bitwise_errors']==resident58['own_edge_bits']['bitwise_errors']==0
assert not resident58['fresh_dual_stream_reuse_acceptance'] and not resident58['sanitizer_acceptance']
for suffix in ['abba','graph_abba']:
    data=json.loads((h/f'p58_{suffix}.json').read_text())
    assert data['correctness_checks']==312 and all(0<x['relative_reduction']<.03 for x in data['summary'])
candidates['p58']['validation_scope']='Own54packing+18edge pre-MMA bit controls, standard78, independent ordinary/Graph312each, own static/two-kernel profile; sub3percent point gains only, not upgraded, no freshstream/tools; byte-exact P56restore/new-process78'
candidates['p58']['numerical_algorithm_changed']='Prep grid384to128/3logical tasks perCTA and explicit helper tile argument; same math/input-output coordinates, consumer/pow2/compare byte-exact P56 and consumer SASSidentical'

phase59=json.loads((h/'p59_hardware.json').read_text())
assert phase59['retained']=='P56' and not phase59['goal_achieved']
assert not phase59['performance_accepted'] and not phase59['research_retained']
assert phase59['standard_original_fp64_gates']==78 and phase59['exact_inside_exponents']==79872
assert phase59['own_pre_mma_layout']['configurations']==6 and phase59['own_pre_mma_layout']['errors']==0
assert phase59['own_pre_mma_layout']['inside_exponent_comparisons']==6144
assert all(x['fused_regs']=='114' for x in phase59['own_pre_mma_layout']['resources'])
assert phase59['printed_per_case_ef_matches_p56'] and not phase59['full_output_bit_equality_proven']
assert not phase59['fresh_dual_stream_reuse_acceptance'] and not phase59['sanitizer_acceptance']
assert phase59['stress_and_tool_sources_prepared_but_not_executed']
for control in ['', 'control_']:
    for suffix in ['abba','graph_abba']:
        data=json.loads((h/f'p59_{control}{suffix}.json').read_text())
        assert data['correctness_checks']==312 and data['independent_numerical_configurations']==78
        assert all(x['relative_reduction']<.03 for x in data['summary'])
        if control:assert all(x['relative_reduction']<0 for x in data['summary'])
candidates['p59']['validation_scope']='Own six raw-input to pack to TMA pre-MMA controls, standard78, independent P56/P59 and P58/P59 ordinary/Graph312each, own static/single-kernel profile; rejected, no own fresh streams/tools; byte-exact P56restore/new-process78'
candidates['p59']['numerical_algorithm_changed']='P58 exact preparation then second unconditional grid barrier and verbatim P56 producer/consumer in one128CTA cooperative kernel; global11planes/fullK512/all7+8dots/RN/inside preserved'

proposal60=json.loads((h/'p60_proposal_cpu.json').read_text())
assert proposal60['proposal_only'] and not proposal60['cpp_generated'] and not proposal60['gpu_executed']
assert proposal60['retained']=='P56' and not proposal60['actual_capacity_checked'] and not proposal60['hardware_accepted']
assert proposal60['cooperative_grid']==384 and proposal60['paired_coordinates']==262144
assert proposal60['covered_elements']=={'A':262144,'B':262144}
assert proposal60['original_input_alignment_preserved'] and proposal60['input_offset2_requires_scalar_fallback']==1048576

pair60=json.loads((h/'p60_hardware.json').read_text())
assert pair60['retained']=='P56' and not pair60['goal_achieved'] and not pair60['performance_accepted']
assert pair60['standard_original_fp64_gates']==78 and pair60['exact_inside_exponents']==79872
assert pair60['own_pre_mma_pack']['packing_configurations']==84 and pair60['own_pre_mma_pack']['misaligned_input']==12
assert pair60['own_pre_mma_pack']['native_uint16_comparisons']==484442112
assert pair60['own_pre_mma_pack']['conversion_native_comparisons']==4194304
assert pair60['own_pre_mma_pack']['bitwise_errors']==pair60['own_pre_mma_pack']['guard_errors']==0
assert not pair60['fresh_dual_stream_reuse_acceptance'] and not pair60['sanitizer_acceptance']
assert pair60['stress_and_tool_sources_prepared_but_not_executed']
for suffix in ['abba','graph_abba']:
    data=json.loads((h/f'p60_{suffix}.json').read_text())
    assert data['correctness_checks']==312 and all(x['relative_reduction']<.03 for x in data['summary'])
candidates['p60']['validation_scope']='Own scalar pair conversion/residual oracle,54aligned+18edge+12misaligned full-plane controls, standard78, independent ordinary/Graph312each, own static/two-kernel profile; rejected, no own fresh streams/tools; byte-exact P56restore/new-process78'
candidates['p60']['numerical_algorithm_changed']='Adjacent-column paired raw loads/RN native casts/32bit plane writes; componentwise FP32 RN, 2B scalar fallback, all7+8dots/inside/consumer and original cooperative reduction unchanged P56'
proposal61=json.loads((h/'p61_proposal_cpu.json').read_text())
assert proposal61['proposal_only'] and not proposal61['cpp_generated'] and not proposal61['gpu_executed']
assert proposal61['retained']=='P56' and not proposal61['actual_capacity_checked'] and not proposal61['hardware_accepted']
assert proposal61['input_positions_each_side']==1048576 and proposal61['coverage_min']==proposal61['coverage_max']==1
assert proposal61['reduction_ctas']==64 and proposal61['columns_per_reduction_cta']==16 and proposal61['launch_ctas']==384

pack61=json.loads((h/'probe_p61_pack.json').read_text())
assert pack61['retained']=='P56' and pack61['packing_configurations']==84
assert pack61['native_uint16_comparisons']==484442112 and pack61['bitwise_errors']==pack61['guard_errors']==0
assert pack61['inside_comparisons_against_p56']==pack61['inside_comparisons_against_scalar_reference']==86016
assert not pack61['matrix_acceptance'] and not pack61['performance_acceptance'] and not pack61['sanitizer_acceptance']

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
    working_candidate=current,working_global_candidate=global_current,compact_pow2_equivalence_cases=1048576,retained_global_candidate='p56 ordinary and Graph research reference, own ordinary/Graph78, fresh14-pattern stress168/1344calls, scoped tools56/112calls each and output/workspace guards. p51/p50/p47/p46/p45/p39/p33/p27 remain historical validated fallbacks. No production performance winner.',compile_stopped_candidates=['P60 initial probe host ternary and CUTLASS mutable Array proxy preflights: no binary/device execution, archived separately before corrected probe subsequently passed','P59 initial host ternary compile-only preflight: no binary/device execution, archived before whitespace-only retry subsequently measured','F12: no numerical or latency measurement','P28 initial compile-only preflight: missing read-to-reuse hardware fence, superseded before device execution','P43 initial host descriptor preflight: invalid256Bfast box, no matrix invocation','P45 initial syntax compile-only preflight: no candidate binary/device execution','P52 initial host ternary compile-only preflight: no binary/device execution, whitespace-only retry subsequently measured','P55 initial variable-template ternary compile-only preflight: no binary/device execution, whitespace-only retry subsequently measured'],candidates=candidates,numerical_models=models,
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
    p40_p42_local_controls=local_controls,m11_numerical_screen=m11_summary,m12_numerical_screen=m12_summary,m13_numerical_screen=m13_summary,m14_numerical_screen=m14_summary,m15_numerical_screen=m15_summary,m16_numerical_screen=m16_summary,
    native_pack_controls=native_pack_controls,p52_consumer_control=consumer52,p53_historical_premeasurement_proposal=proposal53,p53_leaf_parallel_control=leaf53,p54_historical_premeasurement_proposal=proposal54,p54_leaf_n64_control=leaf54,p55_historical_premeasurement_proposal=proposal55,p55_direct_fine_quant_control=fine55,p56_static_leaf_control=static56,p57_static_producer_control=producer57,p58_resident_pack_control=resident58,p59_global_phase_control=phase59,p60_historical_premeasurement_proposal=proposal60,p60_pair_native_pack_control=pair60,p61_historical_premeasurement_proposal=proposal61,p61_pre_mma_preparation_only=pack61,
    p48_independent_performance=ordinary48['summary'],p48_graph_performance=graph48['summary'],p48_device_identity=static48,
    p47_pack_bits=pack47,p47_validation=validation47,p47_independent_performance=ordinary47['summary'],p47_graph_performance=graph47['summary'],
    p46_validation=validation46,p46_independent_performance=ordinary46['summary'],p46_graph_performance=graph46['summary'],
    bf16_correction_hardware=bf16_hardware,p45_validation=validation45,p45_independent_performance=ordinary45['summary'],p45_graph_performance=graph45['summary'],
    p39_validation=validation39,p39_independent_performance=ordinary39['summary'],p39_graph_performance=graph39['summary'],
    requantized_hardware=json.loads((h/'requantized_hardware.json').read_text()),
    uniform_scale_ablation=json.loads((h/'uniform_ablation.json').read_text()),
    validations_not_transferred='P20, P24, P26, P27, P29 and P30 each have their own Graph78, independent stress84/672calls and scoped sanitizers28/56calls per tool. P7 remains historical; fusion candidates retain only their separately recorded scope. P33 has its own9-pattern108/864stress and36/72scoped tools; rejectedP31 has only9-pattern stress, P32/P34/P35/P36/P37 only ordinary, P38 ordinary plus ABBA. P39 owns its Graph78, fresh9-pattern108/864stress and36/72tools. P40 owns ordinary/GraphABBA312 each, P41/P42 ordinaryABBA312 only; no fresh stress/tools acceptance. M11/M12/M13/M14 are numerical only. P43 owns fresh9-pattern108/864stress without tools/independent Graph timing; P44 ordinaryABBA only. P45 owns ordinary/Graph78, fresh108/864stress and36/72tools; P46 owns its independent ordinary/GraphABBA, fresh108/864stress and36/72tools; P47 owns independent ordinary/GraphABBA, fresh108/864stress,36/72tools and separate54configurations of full native plane/exponent bit comparisons; P48 has only ordinary/GraphABBA and matching device code, no fresh stress/tools and is rejected; P49 has own ordinary/Graph ABBA and54fresh packing configurations but is rejected, no fresh stress/tools; P50 owns ordinary/Graph ABBA,54fresh packing configurations plus18edge packing configurations, fresh12-pattern144/1152stress and48/96tools per tool; P51 owns ordinary/Graph ABBA,54fresh packing configurations plus18edge32K packing configurations, fresh12-pattern144/1152stress and48/96tools per tool; P52 owns six synthetic pre-MMA layouts, ordinary78 and independent ordinary/GraphABBA312 each plus ownprofile; rejected, no own fresh stress/tools. P53 owns six synthetic pre-MMA layouts, three scalar GPU recombination trials, ordinary78, independent ordinary/GraphABBA312 each and own three-kernel profile; rejected, no own fresh dual-stream matrix stress/tools. P54 owns six synthetic pre-MMA layouts, three scalar GPU recombination trials, standard78, independent P53/P54 and P51/P54 ordinary/Graph312 each and own three-kernel profile; rejected, no own fresh dual-stream matrix stress/tools. P55 owns96pre-MMA packing/scalar/exponent configurations, six mixed TMA layouts, standard78, fresh84single-stream matrices/guards, independent ordinary/GraphABBA312each and own two-kernel profile; rejected, no dual-stream/reuse stress or tools. P56 has own six pre-MMA TMA cases, standard78, independent ordinary/Graph312each, fresh14-pattern168/1344streams/guards,56/112scoped tools and static/two-kernel profile; research retained, production target incomplete. M15/M16 are separate numerical screens only; no hardware/performance certificates transfer to later changes.',
    remaining='Resolve the end-to-end latency gap, then independent performance repeats, Graph/dual-stream/sanitizer and production integration.')
native=json.loads((r.parent/'fused_native/audit.json').read_text())
assert native['goal_achieved']==False and native['production_unchanged'] and native['retained_candidate']=='P56'
out['historical_restoration_manifest_scope']='P54/P55 restore manifests retain original mutable-path hashes, now verified against frozen P51 sources/binary; current working identity is checked separately against its measured frozen candidate.'
out['native_fusion_hardware']=dict(audit='research/precision_20261016/fused_native/audit.json',retained=False,candidates={label:dict(summary=data['summary'],precision_checks_per_mode=data['precision_checks_per_mode'],exact_exponents_per_mode=data['exact_exponents_per_mode'],fresh_stream_guard_acceptance=False,sanitizer_acceptance=False) for label,data in native['candidates'].items()})
(r/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))

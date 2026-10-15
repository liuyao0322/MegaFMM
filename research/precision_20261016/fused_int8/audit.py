"""Audit actual numerical successes and failed hardware candidates separately."""
import hashlib
import json
import re
from pathlib import Path
r=Path(__file__).resolve().parent
repo=r.parents[2]
h=r.parent/'int8_hardware'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for path in list(h.glob('p*_manifest.json'))+list(r.glob('f*_manifest.json'))+[h/'tile_inside_adaptive_manifest.json',h/'f2_f3_manifest.json',h/'m4_manifest.json',r/'probe_manifest.json',r/'profile_manifest.json',r/'trace_f9_manifest.json',r/'trace_f11_manifest.json',r/'compile_stopped_f12_manifest.json',r/'layout_access_manifest.json',h/'m5_manifest.json',h/'m6_manifest.json',h/'m7_manifest.json',h/'m8_manifest.json',h/'alternating_hardware_manifest.json']:
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
abba=json.loads((h/'p17_abba.json').read_text())
assert abba['correctness_checks']==312 and abba['independent_numerical_configurations']==78
for label in ['p8','p9','p10','p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24']:
    j=json.loads((h/(label+'_balanced.json')).read_text());count=78 if label in ['p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24'] else 72
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
for label in ['p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24']:
    rows=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in (h/(label+'_balanced.log')).read_text().splitlines() if line.startswith('INSIDE_EQUAL ')]
    assert len(rows)==78 and all(x['errors']=='0' and x['elements']=='1024' for x in rows)
    j=json.loads((h/(label+'_balanced.json')).read_text())
    added=[x for x in j['checks'] if x['pattern']=='12'];assert len(added)==6 and all(x['pass']=='1' for x in added)
    candidates[label]['equal_per_column_exponents']=79872
    candidates[label]['additional_alternating_checks_passed']=6
    candidates[label]['retained']=label=='p24'
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
    working_candidate=current,working_global_candidate=global_current,compact_pow2_equivalence_cases=1048576,retained_global_candidate='p24, ordinary/Graph78, own fresh stress84/672calls and scoped sanitizer28/56calls per tool; p10 rejected by fresh hardware',compile_stopped_candidates=['F12: no numerical or latency measurement'],candidates=candidates,numerical_models=models,
    diagnostic_traces=['F9 role-level, one BF16 uniform case','F11 pack subphases, one BF16 uniform case'],probe_scope='one P5 K128 tile, fresh uniform CPU inputs, two dtypes; not full-call acceptance',
    p20_validation=json.loads((h/'p20_validation.json').read_text()),
    p20_independent_performance=json.loads((h/'p20_abba.json').read_text())['summary'],
    p24_validation=json.loads((h/'p24_validation.json').read_text()),
    p24_independent_performance=json.loads((h/'p24_abba.json').read_text())['summary'],
    validations_not_transferred='P20 and P24 each have their own Graph78, independent stress84/672calls and scoped sanitizers28/56calls per tool. P7 remains historical; fusion candidates retain only their separately recorded scope. No transfer to later changes.',
    remaining='Resolve the end-to-end latency gap, then independent performance repeats, Graph/dual-stream/sanitizer and production integration.')
(r/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))

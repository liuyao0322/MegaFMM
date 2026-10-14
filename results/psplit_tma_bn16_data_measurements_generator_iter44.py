#!/usr/bin/env python3
"""Owner-run dynamic archive generation; declared routes/pairs are the authority.

No GPU/build/Git/model execution. Default stdout; only owner selects an output.
Original CSV strings and original compiled certificate fields remain unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True


def common():
    path = Path(__file__).with_name('psplit_tma_bn16_data_common_iter44.py')
    spec = importlib.util.spec_from_file_location('bn16_tma_archive_math', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def key(row): return row['requested'], str(row['variant']), row['dtype'], row['mode']
def key_object(value): return dict(zip(('algorithm','variant','dtype','mode'),value))
def us(row, field='solution_event_ms'): return float(row[field]) * 1000


def route_identity(route): return route['algorithm'], str(route['variant'])


def pair_contract(plan, declared):
    result = []
    for pair in plan['comparison_pairs']:
        if 'candidate' in pair:
            candidate = pair['candidate']['algorithm'], str(pair['candidate']['variant'])
            baseline = pair['baseline']['algorithm'], str(pair['baseline']['variant'])
        else:
            candidate = pair.get('candidate_algorithm','mega'), str(pair['candidate_variant'])
            baseline = pair.get('baseline_algorithm','mega'), str(pair['baseline_variant'])
        require(candidate in declared and baseline in declared and candidate != baseline and pair.get('axis'), 'Undeclared comparison endpoint/axis')
        require(pair.get('controlled_axis_claimed',False) is False, 'Own latency does not establish an isolated causal axis')
        result.append({'candidate_algorithm':candidate[0], 'candidate_variant':int(candidate[1]),
            'baseline_algorithm':baseline[0], 'baseline_variant':int(baseline[1]), 'axis':pair['axis'], 'controlled_axis_claimed':False})
    require(len({(x['candidate_algorithm'],x['candidate_variant'],x['baseline_algorithm'],x['baseline_variant']) for x in result}) == len(result)
            and len({x['axis'] for x in result}) == len(result), 'Duplicate comparison endpoint/axis')
    return result


def compiled_binding(plan, identity):
    match = re.search(r'iter(\d+)',plan['schema']); require(match, 'Plan schema must identify its iteration')
    iteration = int(match[1]); descriptor = identity['independent_compiled_gate']
    require(plan['independent_compiled_gate']==descriptor, 'Plan/identity original gate differs')
    require(descriptor.get('path') and descriptor.get('sha256') and descriptor.get('status') and
            sha(descriptor['path']) == descriptor['sha256'], 'Original fresh compiled gate binding differs')
    original = read(descriptor['path'])
    require(f'iter{iteration}' in original.get('schema','') and 'compiled' in original.get('schema','') and
            original.get('status') == descriptor['status'] and not any(x in descriptor['status'].upper() for x in ('PENDING','UNKNOWN')) and
            original.get(f'compiled_gate_inherited_from_iteration{iteration-1}') is False, 'Require fresh original terminal gate; no inherited certification')
    require(original.get(f'compiled_gate_inherited_from_iteration{iteration-2}') is False or
            original.get('compiled_gate_inherited_from_older_iterations') is False, 'Fresh gate cannot inherit an older route certificate')
    for path,wanted in {**plan['frozen_sources'],**plan['frozen_binaries']}.items():
        require(original.get('frozen_bindings',{}).get(path)==wanted, 'Original gate source/bin binding differs: '+path)
    require(original.get('frozen_bindings',{}).get(identity['linked_identity_path'])==identity['linked_identity_sha256'],
            'Original gate linked identity binding differs')
    for variant in plan['new_variants']:
        config = plan['route_configurations'][str(variant)]
        entries = [x for x in original.get('instances' if config['enabled'] else 'disabled_instances',[]) if x.get('variant')==variant]
        require(len(entries)==len(plan['dtypes']) and {str(x.get('dtype','')).lower() for x in entries}==set(plan['dtypes']), 'Fresh new route must cover the exact declared dtype domain')
        require(not any(x.get('variant')==variant for x in original.get('disabled_instances' if config['enabled'] else 'instances',[])), 'Original gate conflicts with enabled scope')
        for entry in entries:
            require(str(entry.get('status','')).startswith('PASS' if config['enabled'] else 'FAIL'), 'Original dtype status conflicts with enabled scope')
            if config['enabled']:
                require(entry.get('producer_quota')==config['producer_quota'] and entry.get('consumer_quotas')==config['consumer_quotas'] and
                        entry.get('dynamic_shared_bytes')==config['dynamic_shared_bytes'] and entry.get('static_shared_bytes')==config['static_shared_bytes'] and
                        entry.get('bm')==config['BM'] and entry.get('bn')==config['BN'] and entry.get('bk')==config['BK'] and
                        entry.get('threads')==config['threads'], 'Actual gate quota/driver shared/geometry differs from plan')
    return descriptor, original


def comparison(mode,dtype,baseline,candidate,pair,configurations):
    old,new = baseline['csv_row'],candidate['csv_row']; a,b = us(old),us(new)
    record = {**pair, 'mode':mode, 'dtype':dtype, 'baseline_path':baseline['path'],'baseline_line':baseline['line'],
        'candidate_path':candidate['path'],'candidate_line':candidate['line'],'baseline_us':a,'candidate_us':b,
        'baseline_dense_us':us(old,'dense_event_ms'),'candidate_dense_us':us(new,'dense_event_ms'),
        'baseline_batch_wall_us':us(old,'solution_wall_ms'),'candidate_batch_wall_us':us(new,'solution_wall_ms'),
        'baseline_dense_batch_wall_us':us(old,'dense_wall_ms'),'candidate_dense_batch_wall_us':us(new,'dense_wall_ms'),
        'baseline_over_candidate_own_latency':a/b,'own_latency_decrease_percent':(1-b/a)*100,
        'baseline_same_row_dense_over_sw':float(old['speedup']),'candidate_same_row_dense_over_sw':float(new['speedup']),
        'baseline_numerical_pass':old['numerical_pass']=='1','candidate_numerical_pass':new['numerical_pass']=='1','same_binary':True,
        'controlled_axis_scope':'Observed declared whole-call comparison; no isolated cause, dynamic residency, transaction, stall or traffic claim.'}
    axis = dict(record)
    for prefix, row in (('baseline',old),('candidate',new)):
        config = configurations.get(str(row['variant']),{}) if row['requested']=='mega' else {}
        axis[prefix+'_configuration'] = config
        for field in ('BM','BN','BK','producer_quota','consumer_quotas','launch_bounds_min_ctas'):
            if field in config: axis[prefix+'_'+field] = config[field]
    return record,axis


def generate(plan_path,identity_path,exclusions_path=None):
    math_common = common(); plan,identity = read(plan_path),read(identity_path)
    require(plan['frozen_sources']==identity['frozen_sources'] and plan['frozen_binaries']==identity['frozen_binaries'] and
            plan['frozen_sources'] and len(plan['frozen_binaries'])==2, 'Plan/identity must share actual nonempty source map and two binaries')
    require(plan['new_source_paths'] and set(plan['new_source_paths'])<=set(plan['frozen_sources']), 'New TU paths must come from frozen plan')
    for path,wanted in {**plan['frozen_sources'],**plan['frozen_binaries']}.items(): require(sha(path)==wanted,'Frozen source/bin changed: '+path)
    plan_binding = {'path':str(plan_path),'sha256':sha(plan_path)}
    require(identity['predeclared_plan']==plan_binding and identity['predeclared_plan_sha256']==plan_binding['sha256'], 'Complete identity must bind this once-declared plan')
    require(identity['route_configurations']==plan['route_configurations'], 'Identity/plan actual configurations differ')
    require(all(identity[field]==plan[field] for field in ('new_variants','new_source_paths','comparison_pairs')),
            'Complete identity must preserve declared new-source/variant/comparison domain')
    require(sha(identity['linked_identity_path'])==identity['linked_identity_sha256'], 'Complete identity link SHA differs')
    require(len(plan['shapes'])==1 and len(plan['shapes'][0])==3, 'Each archive plan describes one matrix shape')
    require(len(set(plan['dtypes']))==len(plan['dtypes'])>0 and len(set(plan['modes']))==len(plan['modes'])>0 and
            set(plan['modes'])<={'launch','graph'}, 'Unique declared dtypes/modes required')
    routes = {route_identity(x):x for x in plan['routes']}
    require(len(routes)==len(plan['routes']) and routes and len(set(x['label'] for x in routes.values()))==len(routes), 'Unique declared routes and labels required')
    require(len(set(plan['new_variants']))==len(plan['new_variants']) and plan['new_variants'] and
            all(('mega',str(value)) in routes for value in plan['new_variants']), 'Unique new variants must be declared routes')
    for value,route in routes.items():
        require(route.get('actual_name') and isinstance(route.get('enabled'),bool) and isinstance(route.get('native'),bool), 'Final route actual name/enabled/native must be explicitly frozen')
        if route['algorithm']=='mega':
            config=plan['route_configurations'][value[1]]
            require(config['actual_name']==route['actual_name'] and config['enabled'] is route['enabled'], 'Route/actual configuration differs')
            require(not route['enabled'] or all(config.get(x) is not None for x in ('BM','BN','BK','producer_quota','consumer_quotas','launch_bounds_min_ctas','dynamic_shared_bytes','static_shared_bytes')), 'Enabled actual resources incomplete')
        require(route['enabled'] or (route['algorithm']=='mega' and int(route['variant']) in plan['new_variants']), 'Only new compiled-blocked routes may be missing')
    expected = {(*route,dt,mode) for route in routes for dt in plan['dtypes'] for mode in plan['modes']}
    require(plan['expected_row_count']==len(expected), 'Expected row count must match declared routes/dtypes/modes')
    file_domains=[(route_identity(route),mode) for route in plan['routes'] for mode in plan['modes']]
    require(len(plan['csv_paths'])==len(file_domains) and len(set(plan['csv_paths']))==len(file_domains), 'Exact unique declared route/mode CSV paths required')
    descriptor,original=compiled_binding(plan,identity)
    blocked={value for value in expected if not routes[value[:2]]['enabled']}
    require(plan.get('enabled_row_count',len(expected)-len(blocked))==len(expected)-len(blocked), 'Enabled row count differs from declared route states')
    excluded={}; exclusion_binding=None
    if exclusions_path:
        obj=read(exclusions_path)
        iteration=re.search(r'iter(\d+)',plan['schema'])[1]
        require(obj['schema']==f'megafmm.iter{iteration}.measurement-exclusions.v1' and obj['predeclared_plan']==plan_binding and
                all(obj[k]==identity[k] for k in ('measurement_source_commit','frozen_sources','frozen_binaries')), 'Exclusions identity/schema differs')
        for item in obj['unmeasured_rows']:
            value=item['algorithm'],str(item['variant']),item['dtype'],item['mode']
            require(value in blocked and value not in excluded and item.get('classification')=='compiled_safety_blocked' and item.get('reason') and
                    (item.get('evidence_path'),item.get('evidence_sha256'))==(descriptor['path'],descriptor['sha256']), 'Excluded key must bind current exact compiled block')
            excluded[value]=item
        exclusion_binding={'path':str(exclusions_path),'sha256':sha(exclusions_path),'rows':obj['unmeasured_rows'],
            **{k:obj[k] for k in ('measurement_source_commit','frozen_sources','frozen_binaries','predeclared_plan')}}
    require(set(excluded)==blocked, 'All and only blocked declared keys require exclusions; no old-data substitution')
    shape=','.join(map(str,plan['shapes'][0])); groups,trials,fresh=int(plan['groups']),int(plan['trials']),int(plan['correct_trials'])
    require(groups>0 and trials>0 and fresh>0 and int(plan['rotate_inputs'])>0, 'Positive declared group/call/fresh/input counts required')
    runs=[]; rows=[];seen=set();missing_files=[]
    for filename,(route_id,mode) in zip(plan['csv_paths'],file_domains):
        path=Path(filename)
        if not path.exists(): missing_files.append(filename);continue
        with path.open(newline='') as stream: csv_rows=list(csv.DictReader(stream))
        require(len(csv_rows)==len(plan['dtypes']), 'Each route/mode run must retain exact declared dtype rows')
        log_path,exit_path=path.with_suffix('.log'),path.with_suffix('.exit');text=log_path.read_text()
        times,checks,tunes=math_common.records(text)
        require((len(times),len(checks),len(tunes))==(2*groups*len(csv_rows),fresh*len(csv_rows),2*len(csv_rows)), 'Incomplete raw TIME/CHECK/TUNE archive')
        commands=[json.loads(line[8:]) for line in text.splitlines() if line.startswith('COMMAND ')]
        require(len(commands)==1 and commands[0][:2]==['bash','scripts/on_gpu.sh'], 'Exactly one actual runner COMMAND required')
        command=commands[0]; flags={};at=3
        require(len(command)>=3 and command[2].removeprefix('./') in plan['frozen_binaries'], 'COMMAND benchmark must be a frozen binary')
        devices=[line for line in text.splitlines() if line.startswith('DEVICE ')]
        require(len(devices)==1 and ' precision=FP32-output ' in devices[0] and
                re.search(r' workspace_limit=(\d+)$',devices[0]) and
                int(re.search(r' workspace_limit=(\d+)$',devices[0])[1])==int(plan['workspace_limit']), 'Actual DEVICE precision/workspace binding differs')
        while at<len(command):
            field=command[at]; require(field.startswith('--') and field not in flags,'Duplicate/invalid COMMAND flag')
            if field=='--graph':flags[field]=True;at+=1
            else:require(at+1<len(command),'COMMAND flag value missing');flags[field]=command[at+1];at+=2
        variant_flag={'mega':'--mega-variant','lt':'--lt-variant','hopper':'--hopper-variant'}[route_id[0]]
        require(flags.get('--algorithm')==route_id[0] and flags.get(variant_flag)==route_id[1] and bool(flags.get('--graph'))==(mode=='graph') and
                flags.get('--shapes')==shape and flags.get('--output')==filename and
                flags.get('--dtype')==('both' if set(plan['dtypes'])=={'bf16','fp16'} else plan['dtypes'][0]), 'COMMAND actual route/mode/shape/dtype/output differs')
        for flag,field in (('--warmup','warmup'),('--trials','trials'),('--groups','groups'),('--correct-trials','correct_trials'),('--rotate-inputs','rotate_inputs')):
            require(flags.get(flag)==str(plan[field]), 'COMMAND changed '+field)
        file_pass=True
        for line,row in enumerate(csv_rows,2):
            value=key(row);route=routes.get(value[:2])
            require(value in expected and value not in seen and value not in excluded and value[:2]==route_id and row['mode']==mode, 'Unexpected/duplicate/excluded/wrong-file measured row')
            require(','.join(row[x] for x in ('m','n','k'))==shape and row['actual']==route['actual_name'], 'CSV shape/frozen actual name differs')
            require(all(row[field]==str(plan[field]) for field in ('trials','groups','correct_trials','rotate_inputs','workspace_limit')), 'CSV mandatory protocol differs')
            require(row['cache']==('warm_cache' if int(plan['rotate_inputs'])==1 else 'rotating_inputs'), 'CSV cache policy differs')
            dtype_bytes=plan.get('dtype_input_bytes',{'bf16':2,'fp16':2})
            require(row['dtype'] in dtype_bytes, 'Declared dtype input-byte width absent')
            mm,nn,kk=map(int,plan['shapes'][0])
            require(int(row['input_working_set_bytes'])==(mm*kk+kk*nn)*int(dtype_bytes[row['dtype']])*int(plan['rotate_inputs']), 'CSV input working-set bytes differ')
            if route['native']: require(row['solution_workspace']=='0','Declared native route has matrix workspace')
            own_times,summaries=math_common.verify_timing_row(row,times,tunes,groups)
            math_common.verify_dense_binding(row,int(plan['workspace_limit']))
            bound=re.search(r';bound_candidate=(\d+)/(\d+);',row['dense_lt_config'])
            require(bound and int(bound[2])<=int(plan['heuristic_candidates']), 'Dense bound candidate domain exceeds declared heuristic budget')
            floors=plan.get('gate_floors',{'bf16':.001,'fp16':.0001}); require(row['dtype'] in floors,'Declared dtype original floor absent')
            own_checks,passed=math_common.verify_fresh_row(row,checks,fresh,float(floors[row['dtype']]))
            require(all(math.isfinite(float(row[x])) and float(row[x])>0 for x in ('dense_event_ms','solution_event_ms','dense_wall_ms','solution_wall_ms','speedup','equivalent_tflops')), 'Invalid complete timing/throughput metric')
            require(math.isclose(float(row['speedup']),float(row['dense_event_ms'])/float(row['solution_event_ms']),rel_tol=2e-8,abs_tol=1e-10),'Same-row Dense/SW ratio differs from CSV event medians')
            require(math.isclose(float(row['equivalent_tflops']),2*mm*nn*kk/(float(row['solution_event_ms'])*1e9),rel_tol=2e-8,abs_tol=1e-10), 'Equivalent throughput differs from original complete-call formula')
            if 'gate_limit' in row:
                require(abs(float(row['gate_limit'])-max(float(c['gate_limit']) for c in own_checks))<=
                        math_common.printed_error(row['gate_limit'])+max(math_common.printed_error(c['gate_limit']) for c in own_checks)+1e-12,
                        'CSV maximum original gate differs from current stdout')
            file_pass &= passed;seen.add(value)
            rows.append({'path':filename,'line':line,'csv_row':row,'raw_groups':own_times,'fresh_checks':own_checks,'verified_group_statistics':summaries})
        code=int(exit_path.read_text());require(code==(0 if file_pass else 2),'Actual inner exit differs from original numerical result')
        runs.append({'path':filename,'sha256':sha(path),'log_path':str(log_path),'log_sha256':sha(log_path),
            'exit_path':str(exit_path),'exit_sha256':sha(exit_path),'exit':code,'rows':csv_rows})
    missing=expected-seen;require(missing==set(excluded) and seen,'Missing declared rows require exact fresh compiled exclusions and at least one measured row')
    contract=pair_contract(plan,routes);own=[];unavailable=[];axes={x['axis']:[] for x in contract};winners=[];native_winners=[]
    for mode in plan['modes']:
        for dtype in plan['dtypes']:
            group=[z for z in rows if (z['csv_row']['mode'],z['csv_row']['dtype'])==(mode,dtype)]
            by_route={(z['csv_row']['requested'],z['csv_row']['variant']):z for z in group}
            for pair in contract:
                candidate=pair['candidate_algorithm'],str(pair['candidate_variant']);baseline=pair['baseline_algorithm'],str(pair['baseline_variant'])
                if candidate in by_route and baseline in by_route:
                    record,axis=comparison(mode,dtype,by_route[baseline],by_route[candidate],pair,plan['route_configurations']);own.append(record);axes[pair['axis']].append(axis)
                else:unavailable.append({**pair,'mode':mode,'dtype':dtype,'missing_keys':[key_object((*x,dtype,mode)) for x in (candidate,baseline) if x not in by_route]})
            for candidates,target in ((group,winners),([z for x,z in by_route.items() if routes[x]['native']],native_winners)):
                if candidates:
                    winner=min(candidates,key=lambda z:us(z['csv_row']));row=winner['csv_row']
                    target.append({'mode':mode,'dtype':dtype,'path':winner['path'],'line':winner['line'],'requested':row['requested'],
                        'actual':row['actual'],'variant':int(row['variant']),'sw_us':us(row),'dense_us':us(row,'dense_event_ms'),
                        'same_row_dense_over_sw':float(row['speedup']),'numerical_pass':row['numerical_pass']=='1'})
    checks=[c for z in rows for c in z['fresh_checks']]
    noisy=[{'path':z['path'],'line':z['line'],'row_key':key_object(key(z['csv_row'])),'dense_cv':float(z['csv_row']['dense_cv']),
        'solution_cv':float(z['csv_row']['solution_cv']),'dense_event_group_means_us':[float(t['event_ms'])*1000 for t in sorted(z['raw_groups'],key=lambda t:int(t['group'])) if t['case']=='dense'],
        'all_declared_groups_retained':True} for z in rows if float(z['csv_row']['dense_cv'])>=.02]
    return {'schema':'megafmm.iter44.psplit-tma-bn16-measurements.v1','measurement_source_commit':identity['measurement_source_commit'],
        'frozen_sources':identity['frozen_sources'],'frozen_binaries':identity['frozen_binaries'],'route_configurations':plan['route_configurations'],
        'independent_compiled_gate':descriptor,'compiled_gate_original':original,'compiled_gate_content_independently_reaudited_by_generator':False,
        'predeclared_plan':plan_binding,'measurement_scope':{'expected_row_count':len(expected),'actual_row_count':len(rows),'complete_full_matrix':seen==expected,
            'expected_rows':[key_object(x) for x in sorted(expected)],'observed_rows':[key_object(x) for x in sorted(seen)],'missing_rows':[key_object(x) for x in sorted(missing)],
            'missing_csv_paths':missing_files,'exclusions':exclusion_binding},'row_count':len(rows),'timing_group_case_count':sum(len(z['raw_groups']) for z in rows),
        'fresh_gate_count':len(checks),'fresh_gate_pass':sum(c['pass']=='1' for c in checks),'fresh_gate_fail':sum(c['pass']=='0' for c in checks),
        'nonfinite':sum(int(z['csv_row']['nonfinite']) for z in rows),'runs':runs,'rows':rows,'own_latency_pair_contract':contract,
        'same_binary_own_latency_comparison':own,'each_new_vs_retained_controls':own,'axis_comparisons':axes,'unavailable_comparisons':unavailable,
        'observed_all_sw_route_winners':winners,'observed_native_sw_route_winners':native_winners,
        'actually_measured_native_variants':sorted({int(z['csv_row']['variant']) for z in rows if routes[key(z['csv_row'])[:2]]['native']}),
        'dense_cv_at_least_2_percent':noisy,'graph_fixed_capture_bindings_fresh_contents':True,
        'tuning_protocol':'Independent same-row Dense tuning before capture; same ordinary/Graph timing mode; no Graph-specific tuning.',
        'batchwall_protocol':f"Median{groups} group batchwall/call from{trials} calls with batch sync; not independently synchronized single-call CPU wall.",
        'interpretation':'Own pairs and winner sets derive only from declared routes and actual current CSV rows. Original gates and FAIL remain unchanged; Dense denominator noise is not evidence of code benefit. No precision/Auto policy/goal or isolated causal certification.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    parser.add_argument('--identity',required=True)
    parser.add_argument('--exclusions');parser.add_argument('--output',help='Owner-selected formal output; default stdout')
    args=parser.parse_args();value=json.dumps(generate(args.plan,args.identity,args.exclusions),indent=2)+'\n'
    if args.output:
        with Path(args.output).open('x') as stream:stream.write(value)
    else:print(value,end='')

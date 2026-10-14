#!/usr/bin/env python3
"""Owner-run iteration46 profile builder; archived actual records supply scope.

This generator never certifies the generated profile. Its statistics, resource
bindings, reported tests and Git archive are independently reviewed afterward.
All reference-loop runs remain explicitly distinct from candidate-only runs.
"""
from pathlib import Path
import argparse, copy, hashlib, json, re, subprocess

BASE='71de54d8dabb370ee4fc2b107145049858ff23da'
PRIOR='results/small_psplit_tma_bn16_min2_profile_iter45.json'
PREFIX='psplit_tma_bn16_k32_window'

def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def base_bytes(path):return subprocess.check_output(['git','show',BASE+':'+path])
def stdout_json(path):
    text=Path(path).read_text();at=text.find('{')
    require(at>=0,'Archived actual CPU stdout JSON missing: '+str(path))
    return json.JSONDecoder().raw_decode(text[at:])[0]
def log_record(record):
    if not isinstance(record,dict) or not record.get('path'):return None
    path=Path(record['path']);ep=Path(record.get('exit_path',str(path.with_suffix('.exit'))))
    require(path.is_file() and ep.is_file(),'Actual log/terminal sidecar required: '+str(path))
    lines=path.read_text().splitlines();code=int(ep.read_text())
    require(record['sha256']==sha(path) and record['exit_sha256']==sha(ep) and record['exit_code']==code,'Actual coverage terminal binding: '+str(path))
    return {'path':str(path),'exit':code,'CASE':sum(line.startswith('CASE ') for line in lines),
            'PASS':sum(line.startswith('PASS') for line in lines),'BEGIN':sum(line.startswith('BEGIN') for line in lines),
            'source':'Actual closed stdout and archived process exit; never expected counts.'}

def generate(plan_path,identity_path,measurements_path,coverage_path,compiled_path=None,environment_path=None,report_path=None,supplemental_measurements_path=None):
    plan=json.loads(Path(plan_path).read_text());identity=json.loads(Path(identity_path).read_text())
    metrics=json.loads(Path(measurements_path).read_text());coverage=json.loads(Path(coverage_path).read_text())
    require(identity['base_checkpoint']==BASE and metrics['measurement_source_commit']==identity['measurement_source_commit'],'Current iteration46 measurement identity')
    require(plan['frozen_sources']==identity['frozen_sources']==metrics['frozen_sources'] and plan['frozen_binaries']==identity['frozen_binaries']==metrics['frozen_binaries'],'Plan/identity/statistics source and binary maps')
    require(len(identity['frozen_binaries'])==2 and metrics['predeclared_plan']=={'path':str(plan_path),'sha256':sha(plan_path)},'Current dual binary and exact predeclared plan')
    for path,wanted in {**identity['frozen_sources'],**identity['frozen_binaries']}.items():require(sha(path)==wanted,'Actual frozen source/bin SHA: '+path)
    gate=identity['independent_compiled_gate'];require(sha(gate['path'])==gate['sha256'],'Original fresh gate SHA')
    original=json.loads(Path(gate['path']).read_text())
    require('iter46' in original['schema'] and 'compiled' in original['schema'] and not original['status'].lower().startswith('pending'),'Actual iteration46 compiled review, not inherited old gate')
    flags={k:v for k,v in original.items() if k.startswith('compiled_gate_inherited_from_')}
    require(flags and all(v is False for v in flags.values()) and
            (flags.get('compiled_gate_inherited_from_iteration45') is False or
             any(('any_prior' in k or 'older_iterations' in k or 'prior_iterations' in k) for k in flags)),
            'Fresh current46 compiled proof is not inherited from retained controls')
    require(metrics['independent_compiled_gate']==gate,'Statistics bind original fresh gate')
    for path,wanted in {**identity['frozen_sources'],**identity['frozen_binaries']}.items():require(original['frozen_bindings'].get(path)==wanted,'Fresh gate current source/bin: '+path)
    configs=plan['route_configurations'];new_variants=list(map(int,plan.get('new_variants',[50])))
    require(new_variants==[50] and isinstance(configs['50']['enabled'],bool),'Only registered new50 BN16 K32-window scope;51 remains invalid')
    enabled=[v for v in new_variants if configs[str(v)]['enabled']]
    prior=json.loads(base_bytes(PRIOR));prior_metrics=json.loads(base_bytes('results/small_psplit_tma_bn16_min2_measurements_iter45.json'))
    retained={**prior['retained_old_evidence_sha256'],**prior['raw_sha256']}
    extras=[PRIOR,'results/small_psplit_tma_bn16_min2_measurements_iter45.json','results/1024_MEGAKERNEL_PSPLIT_TMA_BN16_MIN2_REPORT.md',
            'results/psplit_tma_bn16_min2_test_coverage_iter45.json','results/psplit_tma_bn16_min2_data_audit_iter45.json',
            'results/psplit_tma_bn16_min2_static_review_final_iter45.json','results/psplit_tma_bn16_min2_static_review_final_iter45.md']
    retained.update({path:hashlib.sha256(base_bytes(path)).hexdigest() for path in extras})
    tracked=set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],text=True).splitlines())
    retained_tracked=sorted(set(retained)&tracked);retained_onsite=sorted(set(retained)-tracked)
    for path,wanted in retained.items():
        require(sha(path)==wanted,'Retained historical evidence onsite unchanged: '+path)
        if path in tracked:require(hashlib.sha256(base_bytes(path)).hexdigest()==wanted,'Retained old Git object: '+path)
    prior_csv=dict(prior['prior_csv_sha256'])
    prior_csv.update({r['path']:r['sha256'] for r in prior['quick_measurements']})
    prior_csv.update({r['path']:r['sha256'] for r in prior_metrics['runs']})
    old_cuda={path:hashlib.sha256(base_bytes(path)).hexdigest() for path in tracked if path.startswith('solution/') and path.endswith('.cu')}
    for path,wanted in {**prior_csv,**old_cuda}.items():require(sha(path)==wanted,'Historical source/CSV bytes: '+path)
    actual=coverage['actual_validation'];measured=actual['measured_source_identity']
    require(measured['sha256']==sha(identity_path) and measured['measurement_source_commit']==identity['measurement_source_commit'],'Actual coverage current measurement identity')
    require(coverage['source_sha256']==identity['frozen_sources']['tests/test_correctness.cu'],'Actual coverage current test source')
    validation={'default':log_record(actual.get('default')),'selected50':log_record(actual.get('selected50')),
                'disabled37_cli':log_record(actual.get('disabled37_cli')),'test_coverage_audit_path':str(coverage_path),
                'actual_route_scopes':copy.deepcopy(actual.get('default',{}).get('new_routes',{})),
                'new_by_variant':{},'strict':None,'strict_by_variant':{},'CPU_checks':{},'CPU_models':[]}
    for v in new_variants:
        record=validation['actual_route_scopes'].get(str(v));validation['new_by_variant'][str(v)]=None
        if record and record.get('enabled'):
            bit=record['rawU_hostepilogue'];gr=record['Graph'];ab=record['current_A_B_rebind'];ep=record['epilogue_rebind']
            validation['new_by_variant'][str(v)]={
                'enabled':True,'reference_variant':bit['reference_variant'],'reference_relationship':bit['reference_relationship'],
                'bitwise_rawU_host_epilogue_groups':bit['PASS_groups'],
                'rawU_reference_scope':'Actual source-bound relationship and reference variant are recorded from current coverage. Strict bit fixtures and host RN epilogue are distinct from general FP64 precision certification.',
                'rawU_total_bound_fixture_kernel_runs':bit['kernel_runs_completed_by_PASS_bound_fixture_loops'],
                'rawU_bound_reference_kernel_runs':bit['reference_kernel_runs_completed_by_PASS_bound_fixture_loops'],
                'rawU_bound_candidate_kernel_runs':bit['candidate_kernel_runs_completed_by_PASS_bound_fixture_loops'],
                'rawU_run_scope':'Completed fixture loops include the explicitly recorded reference variant and candidate50. BEGIN without PASS is not a completed candidate run.',
                'stride_metadata':copy.deepcopy(record.get('stride_metadata')),
                'current_pointer_binding_checks':ab['bindings'],'epilogue_groups':ep['PASS_groups'],
                'epilogue_kernel_runs':ep['kernel_runs_completed_by_PASS_bound_fixture_loops'],
                'new_graph_configurations':gr['config_groups'],'new_graph_fresh_replays':gr['fresh_AB_replays']}
    default=actual.get('default') or {};validation['all_default_Graph']=copy.deepcopy(default.get('all_Graph'))
    strict_groups=[];strict_totals={k:0 for k in ('gates','pass','fail','nonfinite','native_gates','Dense_range_fallback_gates')};strict_paths=[]
    strict_records=[(label,record) for label,record in actual.items() if label.startswith('strict_') and isinstance(record,dict) and record.get('path')]
    for label,record in strict_records:
        meta=log_record(record);stats={
            'gates':record['candidate_gates'],'pass':record['candidate_PASS'],'fail':record['candidate_FAIL'],
            'nonfinite':record['candidate_nonfinite'],
            'native_gates':sum(row['actual_route'] in {configs[str(v)]['actual_name'] for v in enabled} for row in record.get('candidate_gate_records',[])),
            'Dense_range_fallback_gates':sum(row['actual_route']=='dense_cublaslt' for row in record.get('candidate_gate_records',[]))}
        strict_paths.append({'record_label':label,**meta,**stats,'shared_Dense_reference_CASE':record['Dense_reference_CASE']})
        for key in strict_totals:strict_totals[key]+=stats[key]
        for v in record.get('candidate_variants',[]):
            own=record['per_variant'][str(v)]
            validation['strict_by_variant'][str(v)]={'record_label':label,'path':record['path'],'exit':record['exit_code'],
                'gates':own['candidate_gates'],'pass':own['candidate_PASS'],'fail':own['candidate_FAIL'],'nonfinite':own['candidate_nonfinite'],
                'native_gates':own['native_gates'],'native_PASS':own['native_PASS'],'native_FAIL':own['native_FAIL'],
                'Dense_range_fallback_gates':own['HighHalf_Dense_fallback_gates'],'Dense_fallback_PASS':own['HighHalf_Dense_fallback_PASS']}
        for key,group in record.get('per_variant_dtype_pattern',{}).items():strict_groups.append(copy.deepcopy(group))
    if strict_paths:
        scope=coverage.get('actual_scope_summary',{})
        validation['strict']={**strict_totals,'paths':strict_paths,'BF16_floor':.001,'FP16_floor':.0001,
            'full_code_scope_count':scope.get('full_numerical_code_domain'),
            'full_code_scope_executed':scope.get('full_numerical_code_domain_actually_executed'),
            'scope':'Only actual current50 strict logs. Shared Dense reference runs are separate from candidate gates. Unexecuted current full code domain and historical runs retain distinct identities; Dense fallback PASS does not certify native.'}
    for label in ('dag','stage'):
        path=Path(f'results/{PREFIX}_cpu_{label}_iter46.txt')
        if path.is_file() and path.with_suffix('.exit').is_file():
            counts=re.findall(r'Ran (\d+) tests',path.read_text());require(len(counts)==1,'Actual CPU unittest counts: '+str(path))
            validation['CPU_checks'][label]={'path':str(path),'tests':int(counts[0]),'exit':int(path.with_suffix('.exit').read_text()),'exit_path':str(path.with_suffix('.exit'))}
    for path in sorted(Path('results').glob(PREFIX+'*model*iter46.txt')):
        exit_path=path.with_suffix('.exit')
        if not exit_path.is_file():continue
        obj=stdout_json(path)
        validation['CPU_models'].append({'path':str(path),'exit_path':str(exit_path),'exit':int(exit_path.read_text()),'exit_recorded':True,
            'status':obj.get('status'),'stdout_summary':obj,'scope':'Actual CPU source/abstract model stdout and process exit only. No register-fit, hardware liveness, residency, transaction, GEMM precision or performance certification.'})
    compiled=[];compiler_audit={'status':'pending actual mapped compiler input','gate_claimed_PASS':False}
    if compiled_path:
        mapped=json.loads(Path(compiled_path).read_text());compiled=copy.deepcopy(mapped.get('compiled',mapped.get('instances',[])))
        originals={(r['variant'],r['dtype']):r for r in original['instances']}
        require(len(compiled)==len(originals),'Actual mapped compiler instance count')
        for r in compiled:
            orig=originals[r['variant'],r['dtype']];require(all(r.get(k)==v for k,v in orig.items()),'Preserve all original compiled fields')
        compiler_audit=copy.deepcopy(mapped.get('compiler_audit',{}))
        compiler_audit.update({'original_review_path':gate['path'],'original_review_sha256':gate['sha256'],
            'compiled_input_path':str(compiled_path),'compiled_input_sha256':sha(compiled_path),
            'independent_compiled_gate':gate,'mapped_input_scope':'Original fields preserved; mapped input and original compiled gate have independent SHA bindings.'})
    sanitizers=[]
    for record in (actual.get('sanitizers') or {}).get('reports',[]):
        sanitizers.append({**copy.deepcopy(record),'exit':record['exit_code'],
            'test_source_sha256':identity['frozen_sources']['tests/test_correctness.cu'],
            'test_binary_sha256':identity['frozen_binaries']['build/test_correctness']})
    environment=json.loads(Path(environment_path).read_text()) if environment_path else None
    quick_paths=set(Path(identity['quick_archive']).rglob('*.csv'))
    # The immutable quick trajectory and original quick CSV are separate
    # archives. Exclude predeclared full paths and bind both quick copies.
    full_paths={str(p) for p in plan['csv_paths']}
    quick_paths.update(p for p in Path('results').glob('small_'+PREFIX+'*_iter46.csv') if str(p) not in full_paths)
    quick=[{'path':str(path),'sha256':sha(path)} for path in sorted(quick_paths)]
    omitted={f'small_{PREFIX}_profile_iter46.json',f'{PREFIX}_data_audit_iter46.json',f'{PREFIX}_static_review_final_iter46.json',f'{PREFIX}_static_review_final_iter46.md'}
    raw_paths={str(p) for p in Path('results').iterdir() if p.is_file() and 'iter46' in p.name and p.name not in omitted}
    raw_paths.update(str(p) for p in Path(identity['quick_archive']).rglob('*') if p.is_file())
    if report_path and Path(report_path).is_file():raw_paths.add(str(report_path))
    raw_sha={path:sha(path) for path in sorted(raw_paths)}
    result={'schema':'megafmm.iter46.psplit-tma-bn16-k32-window-profile.v1','base_checkpoint':BASE,
        'measurement_source_commit':identity['measurement_source_commit'],'source_hashes':identity['frozen_sources'],
        'binaries':identity['frozen_binaries'],'old_cuda_sha256':old_cuda,'retained_old_evidence_sha256':retained,
        'retained_git_tracked_paths':retained_tracked,'retained_onsite_only_paths':retained_onsite,
        'prior_csv_sha256':prior_csv,'raw_sha256':raw_sha,'independent_compiled_gate':gate,
        'interface':{'new_variants':new_variants,'final_route_configurations':configs,'maximum_registered_variant':50,
            'maximum_enabled_native_variant':max(int(v) for v,c in configs.items() if c['enabled']),'disabled_native_variants':[37,*[v for v in new_variants if v not in enabled]],
            'geometry_scope':'Source/geometry/quotas/shared/barriers only from current final plan and fresh gate. Retained control metadata is not a new function-attribute query.',
            'large_lda_scope':'The A lda<2^39 predicate controls current tensor-map encode eligibility. Actual metadata fixtures certify only helper predicates and supported Plan setup; no huge-stride device allocation or GPU fallback execution is certified.',
            'precision_scope':'Original FP32 RN DAG, final onecast, wholeK seven FP32 products, fixedU and RNalpha/activebetaFMA remain; original FP64 fresh gates unchanged.'},
        'validation':validation,'strict_groups':strict_groups,'sanitizers':sanitizers,'compiled':compiled,'compiler_audit':compiler_audit,
        'environment':environment,'quick_measurements':quick,
        'quick_protocol':{'screen_only':True,'warmup':20,'calls_per_group':20,'groups':5,'fresh_checks_per_row':1},
        'full1024':{'plan_path':str(plan_path),'measurements_path':str(measurements_path),
            **{k:copy.deepcopy(v) for k,v in metrics.items() if k not in ('rows','runs','frozen_sources','frozen_binaries','measurement_source_commit')}},
        'independent_data_audit':{'agent':'pipeline_data30','raw_status':'pending independent raw audit','final_profile_status':'pending independent final audit'},
        'independent_static_final_audit':{'agent':'pipeline_review30','status':'pending unique-prestatus final review'},
        'precision_certified':False,'goal_complete':False,
        'interpretation':['Current same-binary ownSW latency ranks code; independently tuned same-row Dense/SW ratio may change through its Dense denominator.',
            'All groups, original numerical failures/nonfinite and exits remain preserved. Reference loops, range Dense fallback and actual native candidate counts are distinct.',
            'Only50 BN16 K32-window TMA is new;51 is neither registered nor measured. Current rawA descriptor is consumed by TMA, B remains SIMT, and original RN DAG/onecasts remain. Reference atom relationship comes from current source-bound fixtures; CROSSATOM agreement cannot be called sameatom or general FP64 certification. Fresh current source/bin register-fit, typed shared and lifetime proof cannot inherit45 or any older gates.',
            'Driver static shared, resource tool SHARED accounting and occupancy API prediction have distinct meanings. No measured residency, bank-transaction, stall, physical-floor or causal performance certification. Ordinary50,49 and45 each include one current-A tensor-map encode consumed for TMA. Graph50/49/45 encode at capture only, fixed capture addresses read fresh contents. Retained39 encodes no descriptor; control49 is retained BN16 atom-local min2 TMA and uses TMA and current-A encode. Own comparisons50vs49,50vs45,45vs39,50vs39 use current same-binary SW latency; no pure transport causality is certified.']}
    if supplemental_measurements_path:
        sup=json.loads(Path(supplemental_measurements_path).read_text())
        require(sup['schema']=='megafmm.iter46.supplemental-original-hopper-measurements.v1' and sup['predeclared_plan']==identity.get('supplemental_predeclared_plan') and all(sup[k]==identity[k] for k in ('measurement_source_commit','frozen_sources','frozen_binaries','independent_compiled_gate')),'Separate supplement same onceidentity/source/bin/predeclared plan binding')
        require((sup['row_count'],sup['timing_group_case_count'],sup['fresh_gate_count'],len(sup['same_binary_own_latency_comparison']))==(8,80,24,4) and sup['measurement_scope']['complete_full_matrix'],'Supplement actual eight rows/eighty TIME/twentyfour fresh CHECK/four own pairs must be complete')
        result['supplemental_two_routes']={'measurements_path':str(supplemental_measurements_path),'measurements_sha256':sha(supplemental_measurements_path),'mandatory_rows':metrics['row_count'],'supplemental_rows':sup['row_count'],'total_observed_rows':metrics['row_count']+sup['row_count'],'declared_expected_rows':28,'supplemental_scope':'Original H10 fused input two-launch and H12 ST-once three-launch routes; measured separately from mandatory20/currentnativeCAN50, with original gates and all groups preserved.',**copy.deepcopy(sup)}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--identity',required=True)
    p.add_argument('--measurements',required=True);p.add_argument('--coverage',required=True)
    p.add_argument('--supplemental-measurements');p.add_argument('--compiled-json');p.add_argument('--environment-json');p.add_argument('--report');p.add_argument('--output')
    args=p.parse_args();value=generate(args.plan,args.identity,args.measurements,args.coverage,args.compiled_json,args.environment_json,args.report,args.supplemental_measurements)
    text=json.dumps(value,indent=2)+'\n'
    if args.output:Path(args.output).write_text(text)
    else:print(text,end='')

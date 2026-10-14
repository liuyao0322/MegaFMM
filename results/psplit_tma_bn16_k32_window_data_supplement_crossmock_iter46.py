#!/usr/bin/env python3
"""Actual owner generators, with purely in-memory synthetic current46 archives."""
import contextlib
import csv
import fnmatch
import hashlib
import io
import json
import math
from pathlib import Path as RealPath
import types
import sys
sys.dont_write_bytecode=True
ROOT=RealPath('/share/home/liuyao/workspace/MegaFMM')


def load(stem):
    name='psplit_tma_bn16_k32_window_data_'+stem+'_generator_iter46.py'
    path=ROOT/'.scratch'/name;data=path.read_bytes()
    module=types.ModuleType(stem);module.__file__=str(path)
    exec(compile(data,str(path),'exec'),module.__dict__)
    return module,str(path.relative_to(ROOT)),hashlib.sha256(data).hexdigest()


def main():
    h,hp,hs=load('fullplan');i,ip,ih=load('completeidentity_supplement');runner,rup,ruh=load('fullrunner')
    m,mp,mh=load('measurements');p,pp,ph=load('profile_supplement');r,rp,rh=load('report_supplement')
    cp=ROOT/'.scratch/psplit_tma_bn16_k32_window_data_common_iter46.py';cb=cp.read_bytes()
    cm=types.ModuleType('archive_math46');exec(compile(cb,str(cp),'exec'),cm.__dict__)
    sm,smp,smh=load('supplement_measurements')
    files={}
    class MemoryPath:
        def __init__(self,path):self.path=str(path)
        def __str__(self):return self.path
        def __fspath__(self):return self.path
        def __truediv__(self,other):return MemoryPath(self.path.rstrip('/')+'/'+str(other))
        def __lt__(self,other):return self.path<str(other)
        def __hash__(self):return hash(self.path)
        def __eq__(self,other):return isinstance(other,MemoryPath) and self.path==other.path
        @property
        def name(self):return RealPath(self.path).name
        @property
        def stem(self):return RealPath(self.path).stem
        def resolve(self):return MemoryPath(str(RealPath(self.path).absolute()))
        def read_bytes(self):return files[self.path].encode()
        def read_text(self):return files[self.path]
        def exists(self):return self.path in files
        def is_file(self):return self.exists()
        def with_suffix(self,suffix):return MemoryPath(str(RealPath(self.path).with_suffix(suffix)))
        def open(self,mode='r',newline=None):
            if mode=='r':return io.StringIO(files[self.path],newline=newline)
            assert mode=='x' and not self.exists()
            target=self.path
            class Output(io.StringIO):
                def close(self):
                    files[target]=self.getvalue();super().close()
            return Output()
        def glob(self,pattern):
            prefix=self.path.rstrip('/')+'/'
            return [MemoryPath(k) for k in files if k.startswith(prefix) and '/' not in k[len(prefix):] and fnmatch.fnmatch(k[len(prefix):],pattern)]
        def rglob(self,pattern):
            prefix=self.path.rstrip('/')+'/'
            return [MemoryPath(k) for k in files if k.startswith(prefix) and fnmatch.fnmatch(RealPath(k).name,pattern)]
        def iterdir(self):
            prefix=self.path.rstrip('/')+'/'
            return [MemoryPath(k) for k in files if k.startswith(prefix) and '/' not in k[len(prefix):]]
    def save(path,value):files[path]=json.dumps(value)
    def sha(path):return hashlib.sha256(MemoryPath(path).read_bytes()).hexdigest()
    for module in (h,i,runner,m,p,r,sm):module.Path=MemoryPath
    i.helpers=lambda:h;m.common=lambda:cm;sm.common=lambda:cm;r.module=lambda path:m
    git_module=i.subprocess;original_git=git_module.check_output
    def fake_git(args,text=False):
        if args[1:3]==['rev-parse','HEAD']:return 'SYNTHETIC_CURRENT46_QUICKLOGGER_COMMIT\n'
        if args[1]=='ls-tree':return '\n'.join(sorted(base_files))
        assert args[1]=='show'
        return base_files[args[2].split(':',1)[1]].encode() if args[2].startswith(p.BASE+':') else files[args[2].split(':',1)[1]].encode()
    git_module.check_output=fake_git
    base_files={}
    for path in (p.PRIOR,'results/small_psplit_tma_bn16_min2_measurements_iter45.json','results/1024_MEGAKERNEL_PSPLIT_TMA_BN16_MIN2_REPORT.md',
        'results/psplit_tma_bn16_min2_test_coverage_iter45.json','results/psplit_tma_bn16_min2_data_audit_iter45.json',
        'results/psplit_tma_bn16_min2_static_review_final_iter45.json','results/psplit_tma_bn16_min2_static_review_final_iter45.md'):
        base_files[path]='SYNTHETIC_RETAINED_BASE_BYTES '+path
    base_files[p.PRIOR]=json.dumps({'retained_old_evidence_sha256':{},'raw_sha256':{},'prior_csv_sha256':{},'quick_measurements':[]})
    base_files['results/small_psplit_tma_bn16_min2_measurements_iter45.json']=json.dumps({'runs':[]})
    files.update(base_files)
    try:
        template=json.loads((ROOT/h.TEMPLATE).read_text());save(h.TEMPLATE,template)
        for path in h.SOURCES|h.BINS:files[path]='SYNTHETIC_CURRENT46_BYTES '+path
        sources={path:sha(path) for path in h.SOURCES};bins={path:sha(path) for path in h.BINS}
        link='fixture/current46-linked.json';save(link,bins)
        initial='fixture/initial45.json';save(initial,{'base_checkpoint':p.BASE,'frozen_sources':sources,'frozen_binaries':bins,
            'linked_identity_path':link,'linked_identity_sha256':sha(link),'status':'PENDING_HISTORICAL_ONLY'})
        files['fixture/quick45/trajectory/input.csv']='SYNTHETIC_QUICK_ONLY'
        files['fixture/quick45/results.csv']='SYNTHETIC_CLOSED_QUICK_LOGGER_RESULTS'
        files['results/small_psplit_tma_bn16_k32_window_50_launch_iter46.csv']='SYNTHETIC_QUICK_ORIGINAL'
        baseline=dict(files);summaries=[];negative_names=[]
        for blocked in (False,True):
            files.clear();files.update(baseline)
            cfg={str(v):{'actual_name':('megakernel_wgmma_psplit_tma_bn16_k32_window_64x16' if v==50 else 'SYNTHETIC_CONTROL'+str(v)),
                'enabled':not blocked or v!=50,'BM':64,'BN':16 if v in (50,49) else 32,'BK':64,'threads':384,
                'producer_quota':40 if v in (50,49) else 64,'consumer_quotas':[112,112] if v==50 else ([88,112] if v==49 else [192,192]),'launch_bounds_min_ctas':2 if v==50 else 1,
                'dynamic_shared_bytes':None if v==50 and blocked else {50:95360,49:95232,45:123904,39:123008}[v],
                'static_shared_bytes':None if v==50 and blocked else 0,'predicted_dynamic_shared_bytes':95232 if v==50 else None}
                for v in (50,49,45,39)}
            gate={'schema':'megafmm.iter46.compiled-fixture.v1','status':'TERMINAL_SYNTHETIC_ONLY',
                'compiled_gate_inherited_from_iteration45':False,'compiled_gate_inherited_from_older_iterations':False,'frozen_bindings':{**sources,**bins,link:sha(link)},'instances':[],'disabled_instances':[],'revision1_source_repair':{'scope':'SYNTHETIC_DICTIONARY_RENDER_ONLY_NO_ACTUAL_REVISION_OR_GATE'}}
            if blocked:
                del gate['compiled_gate_inherited_from_older_iterations']
                gate['compiled_gate_inherited_from_older_iterations']=False
            for dtype in ('BF16','FP16'):
                entry={'variant':50,'dtype':dtype,'status':'FAIL_SYNTHETIC_ONLY' if blocked else 'PASS_SYNTHETIC_ONLY',
                    'bm':64,'bn':16,'bk':64,'threads':384,'producer_quota':40,'consumer_quotas':[112,112],
                    'dynamic_shared_bytes':cfg['50']['dynamic_shared_bytes'],'actual_driver_static_shared_bytes':cfg['50']['static_shared_bytes'],
                    'resource_tool_shared_bytes':1024,'occupancy_API_predicted_active_CTAs_per_SM':1,'measured_residency':None,
                    'actual_reported_registers':104,'effective_initial_registers_per_thread':104,
                    'initial_owned_CTA_pool_registers':39936,'requested_role_pool_registers':33792,
                    'A_transport':'SYNTHETIC_TMA_RAW_A_SW128','B_transport':'SIMT','encoded_A_map_consumed_by_kernel':True,
                    'ordinary_current_A_encode_count':1,'graph_replay_current_A_encode_count':0,
                    'stack_bytes':0,'spill_load_bytes':0,'spill_store_bytes':0,
                    'role_CFG':[{'role':role,'gpr_highest_including_implicit':highest} for role,highest in [('producer',37),('WG0',85),('WG1',110)]],
                    'all_P_clear_after_successful_INC':True,'legal_initial_owned_preINC_zero_present':False}
                gate['disabled_instances' if blocked else 'instances'].append(entry)
            gp,cf='fixture/gate45.json','fixture/config45.json';save(gp,gate);save(cf,cfg)
            plan=h.generate(gp,initial,cf);plan_path='fixture/plan45.json';save(plan_path,plan)
            sup={k:plan[k] for k in h.PROTOCOL};sup.update(schema='megafmm.iter46.supplemental-plan.v1',declaration_scope='supplemental_original_hopper_routes',frozen_sources=sources,frozen_binaries=bins,independent_compiled_gate=plan['independent_compiled_gate'],new_source_paths=[],new_variants=[],route_configurations={},expected_row_count=8,enabled_row_count=8)
            sup['routes']=[{'label':label,'algorithm':'hopper','variant':v,'actual_name':'SYNTHETIC_HOPPER'+str(v),'enabled':True,'native':False,'complete_run_kernel_launches':launches} for label,v,launches in [('hopper_fused10',10,2),('hopper_st_once12',12,3)]]
            sup['csv_paths']=[f"results/small_{route['label']}_{mode}_iter46_full1024.csv" for route in sup['routes'] for mode in sup['modes']]
            sup['comparison_pairs']=[{'candidate_algorithm':'hopper','candidate_variant':12,'baseline_algorithm':'hopper','baseline_variant':10,'axis':'hopper12_vs_hopper10_input_placement','controlled_axis_claimed':False}]
            sup_path='fixture/supplement_plan46.json';save(sup_path,sup)
            identity=i.generate(plan_path,gp,initial,cf,'fixture/quick45',sup_path);identity_path='fixture/measured45.json';save(identity_path,identity)
            assert identity['supplemental_predeclared_plan']=={'path':sup_path,'sha256':sha(sup_path)}

            exclusion=i.exclusions(identity);exclusions_path='fixture/exclusions45.json' if blocked else None
            if exclusion:save(exclusions_path,exclusion)
            selected=runner.select_runs(plan)
            assert len(selected)==(8 if blocked else 10)
            generated=runner.generate(plan_path,identity_path);namespace={'__name__':'synthetic_cpu_generated_runner45'}
            exec(compile(generated,'synthetic_generated_runner45','exec'),namespace)
            inner_exits=[]
            class FakePopen:
                def __init__(self,command,**kwargs):
                    assert command in [run['command'] for run in selected]
                    files[command[command.index('--output')+1]]='SYNTHETIC_RUNNER_ONLY'
                    self.stdout=['CHECK SYNTHETIC_RUNNER_ONLY\n']
                    self.code=0 if len(inner_exits)%2==0 else 2;inner_exits.append(self.code)
                def wait(self):return self.code
            namespace['Path']=MemoryPath
            namespace['subprocess']=types.SimpleNamespace(Popen=FakePopen,PIPE=-1,STDOUT=-2)
            capture=io.StringIO()
            with contextlib.redirect_stdout(capture):runner_exit=namespace['main']()
            assert runner_exit==0 and 'TERMINAL_RUNNER_EXIT 0 INNER_EXITS' in capture.getvalue()
            assert [int(files[str(RealPath(run['path']).with_suffix('.exit'))]) for run in selected]==inner_exits
            for run in selected:
                for suffix in ('.csv','.log','.exit'):del files[str(RealPath(run['path']).with_suffix(suffix))]
            if blocked:assert not any('small_psplit_tma_bn16_k32_window_50_' in path and '_full1024' in path for path in files)
            for run in runner.select_runs(plan):
                filename=run['path'];route=next(x for x in plan['routes'] if f"small_{x['label']}_" in filename)
                mode='graph' if '--graph' in run['command'] else 'launch';lines=['COMMAND '+json.dumps(run['command']),
                    'DEVICE SYNTHETIC_CPU_FIXTURE sm=1 cc=0.0 driver=0 runtime=0 precision=FP32-output seed=1234567 workspace_limit='+str(plan['workspace_limit'])]
                rows=[];file_pass=True
                for dtype in plan['dtypes']:
                    variant=route['variant'];dense=[.030,.031,.032,.033,.034]
                    center={50:.019,39:.020,45:.022,49:.024,1:.026}[variant]+(.001 if dtype=='fp16' else 0)
                    sol=[center-.002,center-.001,center,center+.001,center+.002]
                    dense_config='dense:synthetic;tuned=1;tune=4w/3x10median;bound_candidate=0/2;tuned_candidate=0;bound_workspace=0;end=1'
                    solution_config='none' if route['native'] else 'batch7:synthetic'
                    lines.extend(f'TUNE shape=1024,1024,1024 dtype={dtype} case={case} config={config}' for case,config in [('dense',dense_config),('solution',solution_config)])
                    for group in range(5):
                        for case,values in [('dense',dense),('solution',sol)]:
                            lines.append(f'TIME shape=1024,1024,1024 dtype={dtype} group={group} case={case} event_ms={values[group]:.9f} wall_ms={values[group]+.004:.9f}')
                    passed=not (variant==50 and dtype=='bf16');floor=.001 if dtype=='bf16' else .0001;limit=3*.00001+floor
                    solution_ef=.01 if not passed else .00002
                    for fresh in range(3):lines.append(f'CHECK 1024,1024,1024 {dtype} trial={fresh} dense_eF=0.000010000 solution_eF={solution_ef:.9f} gate={limit:.9f} pass={int(passed)}')
                    row={'m':'1024','n':'1024','k':'1024','dtype':dtype,'requested':route['algorithm'],'actual':route['actual_name'],
                        'variant':str(variant),'mode':mode,'cache':'warm_cache','rotate_inputs':'1','input_working_set_bytes':'4194304',
                        'quantile_unit':'group_mean','trials':'100','groups':'5','correct_trials':'3','workspace_limit':'8589934592',
                        'dense_workspace':'0','solution_workspace':'0' if route['native'] else '1024',
                        'dense_event_ms':f'{dense[2]:.12f}','dense_p10_ms':'0.030400000000','dense_p90_ms':'0.033600000000',
                        'dense_cv':format(math.sqrt(2)*.001/.032,'.12f'),'solution_event_ms':f'{center:.12f}',
                        'solution_p10_ms':f'{center-.0016:.12f}','solution_p90_ms':f'{center+.0016:.12f}',
                        'solution_cv':format(math.sqrt(2)*.001/center,'.12f'),'dense_wall_ms':'0.036000000000','solution_wall_ms':f'{center+.004:.12f}',
                        'speedup':format(.032/center,'.12f'),'equivalent_tflops':format(2*1024**3/(center*1e9),'.12f'),
                        'dense_ef':'0.000010000','solution_ef':f'{solution_ef:.9f}','nonfinite':'0','gate_floor':str(floor),'gate_limit':f'{limit:.9f}',
                        'numerical_pass':str(int(passed)),'dense_lt_config':dense_config,'solution_lt_config':solution_config}
                    rows.append(row);file_pass &= passed
                output=io.StringIO();writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                files[filename]=output.getvalue();files[str(RealPath(filename).with_suffix('.log'))]='\n'.join(lines)+'\n'
                files[str(RealPath(filename).with_suffix('.exit'))]=str(0 if file_pass else 2)+'\n'
            metrics=m.generate(plan_path,identity_path,exclusions_path)
            assert metrics['row_count']==(16 if blocked else 20) and metrics['measurement_scope']['complete_full_matrix'] is (not blocked)
            assert metrics['timing_group_case_count']==(160 if blocked else 200) and metrics['fresh_gate_count']==(48 if blocked else 60)
            assert (metrics['fresh_gate_pass'],metrics['fresh_gate_fail'],metrics['nonfinite'])==((48,0,0) if blocked else (54,6,0))
            assert len(metrics['same_binary_own_latency_comparison'])==(4 if blocked else 16)
            assert len(metrics['unavailable_comparisons'])==(12 if blocked else 0)
            assert metrics['compiled_gate_original']==gate and len(metrics['dense_cv_at_least_2_percent'])==metrics['row_count']
            assert all(len(x['dense_event_group_means_us'])==5 and x['all_declared_groups_retained'] for x in metrics['dense_cv_at_least_2_percent'])
            assert {(p['candidate_variant'],p['baseline_variant']) for p in metrics['own_latency_pair_contract']}=={(50,49),(50,45),(45,39),(50,39)}
            assert all(len(metrics['axis_comparisons'][pair['axis']])==(4 if not blocked or pair['candidate_variant']==45 else 0) for pair in metrics['own_latency_pair_contract'])
            assert all(record['controlled_axis_claimed'] is False for record in metrics['same_binary_own_latency_comparison'])
            if blocked:assert all((pair['candidate_variant'],pair['baseline_variant'])==(45,39) for pair in metrics['same_binary_own_latency_comparison'])
            if blocked:
                assert len(metrics['measurement_scope']['missing_rows'])==4
                assert all([key['variant'] for key in record['missing_keys']]==['50'] for record in metrics['unavailable_comparisons'])
            metrics_path='fixture/metrics45.json';save(metrics_path,metrics)
            # Copy complete in-memory Lt control rows to two explicit original Hopper identities.
            for route in sup['routes']:
                for mode in sup['modes']:
                    source_row=next(x for x in metrics['rows'] if x['csv_row']['variant']=='1' and x['csv_row']['mode']==mode)
                    src=source_row['path'];dst=f"results/small_{route['label']}_{mode}_iter46_full1024.csv"
                    data=list(csv.DictReader(io.StringIO(files[src])))
                    for row in data:row.update(requested='hopper',variant=str(route['variant']),actual=route['actual_name'])
                    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data);files[dst]=out.getvalue()
                    log=files[str(RealPath(src).with_suffix('.log'))];command=json.loads(log.splitlines()[0][8:]);command[command.index('--algorithm')+1]='hopper';idx=command.index('--lt-variant');command[idx]='--hopper-variant';command[idx+1]=str(route['variant']);command[command.index('--output')+1]=dst
                    files[str(RealPath(dst).with_suffix('.log'))]='COMMAND '+json.dumps(command)+'\n'+'\n'.join(log.splitlines()[1:])+'\n';files[str(RealPath(dst).with_suffix('.exit'))]=files[str(RealPath(src).with_suffix('.exit'))]
            sup_metrics=sm.generate(sup_path,identity_path);sup_metrics_path='fixture/supplement_measurements46.json';save(sup_metrics_path,sup_metrics)
            assert (sup_metrics['row_count'],sup_metrics['timing_group_case_count'],sup_metrics['fresh_gate_count'],len(sup_metrics['same_binary_own_latency_comparison']))==(8,80,24,4)

            default='results/fixture_tma_bn16_default_iter46.txt';files[default]='CASE SYNTHETIC_CURRENT50\nPASS SYNTHETIC_CURRENT50\n';files[str(RealPath(default).with_suffix('.exit'))]='0\n'
            def log_binding(path):
                ep=str(RealPath(path).with_suffix('.exit'))
                return {'path':path,'sha256':sha(path),'exit_path':ep,'exit_sha256':sha(ep),'exit_code':int(files[ep])}
            new={'50':{'enabled':False}} if blocked else {'50':{'enabled':True,'rawU_hostepilogue':{
                'reference_variant':45,'reference_relationship':'CROSSATOM_N16_N32','PASS_groups':6,
                'kernel_runs_completed_by_PASS_bound_fixture_loops':16,
                'reference_kernel_runs_completed_by_PASS_bound_fixture_loops':6,
                'candidate_kernel_runs_completed_by_PASS_bound_fixture_loops':10},
                'Graph':{'config_groups':3,'fresh_AB_replays':9},'current_A_B_rebind':{'bindings':4},
                'epilogue_rebind':{'PASS_groups':2,'kernel_runs_completed_by_PASS_bound_fixture_loops':6}}}
            default_record={**log_binding(default),'new_routes':new,'all_Graph':{'config_groups':3,'fresh_AB_replays':9}}
            coverage={'source_sha256':sources['tests/test_correctness.cu'],'actual_validation':{
                'measured_source_identity':{'sha256':sha(identity_path),'measurement_source_commit':identity['measurement_source_commit']},
                'default':default_record,'sanitizers':{'reports':[]}},
                'actual_scope_summary':{'full_numerical_code_domain':777,'full_numerical_code_domain_actually_executed':False}}
            if not blocked:
                strict='results/fixture_tma_bn16_strict_iter46.txt'
                files[strict]='CASE SYNTHETIC_CURRENT50_OR_DENSE_REFERENCE\n'*60;files[str(RealPath(strict).with_suffix('.exit'))]='1\n'
                coverage['actual_validation']['strict_tma_bn16_enabled_only']={**log_binding(strict),'candidate_gates':30,'candidate_PASS':18,
                    'candidate_FAIL':12,'candidate_nonfinite':0,'Dense_reference_CASE':30,'candidate_variants':[50],
                    'candidate_gate_records':[{'actual_route':cfg['50']['actual_name']}] * 24 + [{'actual_route':'dense_cublaslt'}]*6,
                    'per_variant':{'50':{'candidate_gates':30,'candidate_PASS':18,'candidate_FAIL':12,'candidate_nonfinite':0,
                        'native_gates':24,'native_PASS':12,'native_FAIL':12,'HighHalf_Dense_fallback_gates':6,'HighHalf_Dense_fallback_PASS':6}},
                    'per_variant_dtype_pattern':{}}
            else:
                selected_path='results/fixture_tma_bn1645_disabled_iter46.txt'
                files[selected_path]='FAIL Mega50 is disabled: SYNTHETIC_COMPILED_BLOCK\n';files[str(RealPath(selected_path).with_suffix('.exit'))]='1\n'
                coverage['actual_validation']['selected50']=log_binding(selected_path)
                strict='results/fixture_tma_bn1645_strict_blocked_iter46.txt'
                files[strict]='FAIL requested selective TMA-A numerical scope has no enabled routes\n';files[str(RealPath(strict).with_suffix('.exit'))]='1\n'
                coverage['actual_validation']['strict_tma_bn16_enabled_only']={**log_binding(strict),'candidate_variants':[],
                    'candidate_gates':0,'candidate_PASS':0,'candidate_FAIL':0,'candidate_nonfinite':0,'Dense_reference_CASE':0,'status':'BLOCKED_NOT_EXECUTED'}
            coverage_path='fixture/coverage45.json';save(coverage_path,coverage)
            mapped_path='fixture/mapped45.json';save(mapped_path,{'compiled':gate['instances'],'compiler_audit':{'status':'SYNTHETIC_PENDING_INDEPENDENT_AUDIT'}})
            model='results/psplit_tma_bn16_k32_window_cpu_fixture_model_iter46.txt';save(model,{'status':'PASS_SYNTHETIC_CPU_JSON_ONLY'});files[str(RealPath(model).with_suffix('.exit'))]='7\n'
            profile=p.generate(plan_path,identity_path,metrics_path,coverage_path,mapped_path,None,None,sup_metrics_path)
            assert profile['interface']['maximum_registered_variant']==50
            assert profile['interface']['maximum_enabled_native_variant']==(49 if blocked else 50)
            assert profile['precision_certified'] is False and profile['goal_complete'] is False
            assert profile['validation']['CPU_models'][0]['exit']==7
            assert len(profile['quick_measurements'])==3
            assert profile['compiler_audit']['original_review_path']==gp and profile['compiler_audit']['compiled_input_path']==mapped_path
            assert profile['compiled']==gate['instances']
            if not blocked:
                current=profile['validation']['new_by_variant']['50']
                assert (current['reference_variant'],current['rawU_total_bound_fixture_kernel_runs'],current['rawU_bound_reference_kernel_runs'],current['rawU_bound_candidate_kernel_runs'])==(45,16,6,10)
                assert profile['validation']['strict']['gates']==30 and profile['validation']['strict']['native_gates']==24
            else:
                assert profile['validation']['new_by_variant']['50'] is None
                assert profile['validation']['strict']['gates']==0 and profile['validation']['strict']['paths'][0]['exit']==1
                assert profile['validation']['selected50']['exit']==1 and profile['sanitizers']==[]
            profile_path='fixture/profile45.json';save(profile_path,profile)
            report=r.generate(plan_path,identity_path,profile_path,metrics_path,mp)
            main_rows=[line for line in report.splitlines() if line.startswith('| Mega') or line.startswith('| Lt')]
            own_rows=[line for line in report.splitlines() if (line.startswith('| launch |') or line.startswith('| graph |')) and len(line.split('|')[1:-1])==9]
            assert len([line for line in report.splitlines() if line.startswith('| H10 /') or line.startswith('| H12 /')])==8
            assert profile['supplemental_two_routes']['total_observed_rows']==(24 if blocked else 28)
            assert len(main_rows)==metrics['row_count'] and all(len(line.split('|')[1:-1])==10 for line in main_rows)
            assert len(own_rows)==(4 if blocked else 16) and all(len(line.split('|')[1:-1])==9 for line in own_rows)
            assert 'CPU模型actualexit=7' in report and 'full code domain=777、actually executed=False' in report if not blocked else 'CPU模型actualexit=7' in report
            if not blocked:
                assert 'Driver static_shared_bytes=0' in report and 'resource_tool_reported_SHARED_bytes=1024' in report
                assert 'occupancy API predicted CTAs/SM=1' in report and 'measured_residency=None' in report
                assert 'rawU groups=6，reference=45' in report and 'reference=6，candidate=10' in report
                resource_rows=[index for index,line in enumerate(report.splitlines()) if line.startswith('| 50 |')]
                assert len(resource_rows)==2 and resource_rows[1]==resource_rows[0]+1
                assert ' | 95360 | ' in report.splitlines()[resource_rows[0]]
                assert '95232' in report and '一个A tensor map' in report and 'map实际被四路rawA TMA消费' in report and 'Graph50/49/45在capture时编码' in report and 'CROSSATOM_N16_N32' in report and '通用FP64精度认证' in report
            else:
                assert '当前strict candidate gates=0' in report and 'selected50：actualexit=1，CASE/PASS/BEGIN=0/0/0' in report
            saved=dict(files)
            def reject(name,mutate,ep=exclusions_path):
                files.clear();files.update(saved);mutate()
                try:m.generate(plan_path,identity_path,ep)
                except ValueError:negative_names.append(name)
                else:raise AssertionError('negative fixture unexpectedly accepted: '+name)
                files.clear();files.update(saved)
            if not blocked:
                def changed_gate(field,value):
                    altered=json.loads(files[gp]);altered[field]=value;save(gp,altered)
                    altered_plan=json.loads(files[plan_path]);altered_plan['independent_compiled_gate']['sha256']=sha(gp);save(plan_path,altered_plan)
                    altered_identity=json.loads(files[identity_path]);altered_identity['independent_compiled_gate']['sha256']=sha(gp)
                    altered_identity['predeclared_plan']['sha256']=sha(plan_path);altered_identity['predeclared_plan_sha256']=sha(plan_path);save(identity_path,altered_identity)
                reject('fresh_gate_inherited45_rejected',lambda:changed_gate('compiled_gate_inherited_from_iteration45',True))
                def swap_driver():
                    altered_plan=json.loads(files[plan_path]);altered_plan['route_configurations']['50']['static_shared_bytes']=1024;save(plan_path,altered_plan)
                    altered_identity=json.loads(files[identity_path]);altered_identity['route_configurations']=altered_plan['route_configurations']
                    altered_identity['predeclared_plan']['sha256']=sha(plan_path);altered_identity['predeclared_plan_sha256']=sha(plan_path);save(identity_path,altered_identity)
                reject('resource_tool_SHARED1024_cannot_replace_driver_static0',swap_driver)
                def swap_dynamic():
                    altered_plan=json.loads(files[plan_path]);altered_plan['route_configurations']['50']['dynamic_shared_bytes']=95232;save(plan_path,altered_plan)
                    altered_identity=json.loads(files[identity_path]);altered_identity['route_configurations']=altered_plan['route_configurations']
                    altered_identity['predeclared_plan']['sha256']=sha(plan_path);altered_identity['predeclared_plan_sha256']=sha(plan_path);save(identity_path,altered_identity)
                reject('predicted_shared95232_cannot_replace_synthetic_actual95360',swap_dynamic)
                filename=plan['csv_paths'][0]
                reject('sampleCV_cannot_replace_populationCV',lambda:files.__setitem__(filename,files[filename].replace('0.044194173824','0.050410588440')))
                lp=str(RealPath(filename).with_suffix('.log'))
                reject('wrong_single_bound_TUNE',lambda:files.__setitem__(lp,files[lp].replace('case=solution config=none','case=solution config=other',1)))
            else:
                def omit_key():
                    obj=json.loads(files[exclusions_path]);obj['unmeasured_rows'].pop();save(exclusions_path,obj)
                reject('compiled_block_exact4_missing_keys_required',omit_key)
                def fake_complete():
                    obj=json.loads(files[plan_path]);obj['enabled_row_count']=20;save(plan_path,obj)
                    ident=json.loads(files[identity_path]);ident['predeclared_plan']['sha256']=sha(plan_path);ident['predeclared_plan_sha256']=sha(plan_path);save(identity_path,ident)
                    exc=json.loads(files[exclusions_path]);exc['predeclared_plan']['sha256']=sha(plan_path);save(exclusions_path,exc)
                reject('partial16_cannot_be_claimed_complete20',fake_complete)
            summaries.append({'blocked50':blocked,'rows':metrics['row_count'],'runs':len(metrics['runs']),'TIME':metrics['timing_group_case_count'],
                'CHECK':metrics['fresh_gate_count'],'fresh_PASS':metrics['fresh_gate_pass'],'fresh_FAIL':metrics['fresh_gate_fail'],
                'supplemental_rows':8,'supplemental_TIME':80,'supplemental_CHECK':24,'supplemental_own_pairs':4,'total_observed_rows':profile['supplemental_two_routes']['total_observed_rows'],'own_pairs':len(metrics['same_binary_own_latency_comparison']),'unavailable_pairs':len(metrics['unavailable_comparisons']),
                'missing_keys':len(metrics['measurement_scope']['missing_rows']),'main_table_cells':10,'own_table_cells':9,
                'original_gate_entirely_preserved':True,'original_numerical_failures_preserved':True,'CPU_JSON_PASS_actual_exit':7,
                'actual_generated_runner_mock_terminal':runner_exit,'mock_inner_exits_retained':inner_exits,
                'current_45_resource_scope':'Synthetic95360 actual field vs predicted95232; Driver0/tool1024/APIpred1/null distinct. No actual46 compiled certificate.',
                'validation_scope':'Synthetic current50/CROSSATOM-ref45 ABI with explicit reference/candidate counters; synthetic full777 unexecuted, SAN and independent audits pending.'})
        print(json.dumps({'status':'PASS_PURE_CPU_PRIMARY20_AND_ORIGINAL_HOPPER_SUPPLEMENT8_CHAIN_ITER46',
            'script_SHA256':{hp:hs,ip:ih,rup:ruh,mp:mh,pp:ph,rp:rh,str(cp.relative_to(ROOT)):hashlib.sha256(cb).hexdigest()},
            'scenarios':summaries,'negative_fixtures_rejected':negative_names,
            'scope':'Actual functions executed with in-memory synthetic current metadata, CSV/stdout/exit, original gate, coverage and historical Git objects. No current formal raw read, no real Git/GPU/build/benchmark/model execution or formal write.'},indent=2))
    finally:git_module.check_output=original_git


if __name__=='__main__':main()

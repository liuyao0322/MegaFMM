from pathlib import Path
import json,hashlib
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
init=json.loads(Path('results/psplit_rsa_bk32_initial_frozen_identity_iter37.json').read_text())
gatep=Path('results/psplit_rsa_bk32_static_review_initial_iter37.json');gate=json.loads(gatep.read_text());assert gate['compiled_gate_inherited_from_iteration36'] is False
for p,h in {**init['frozen_sources'],**init['frozen_binaries']}.items():assert sha(p)==h,p
gb={'path':str(gatep),'sha256':sha(gatep),'status':gate['status']}
base=json.loads(Path('results/psplit_rsa1024_expanded_plan_iter36.json').read_text())
p={k:base[k] for k in ['shapes','dtypes','modes','warmup','trials','groups','correct_trials','rotate_inputs','heuristic_candidates','workspace_limit','reference']}
p.update(schema='megafmm.iter37.full1024-plan.v1',declared_before_measurement=True,reason='Mandatory new40 RS-A BK32 vs retained39 RS-A BK64 prior measured best, matched BK32 SS36, multilaunch Lt1; full16 irrespective quick3percent. H10/H12 full24 iteration36 preserved as historical evidence, not remeasured in37.')
p['routes']=[{'label':a,'algorithm':b,'variant':c} for a,b,c in [('psplit_rsa_bk3240','mega',40),('psplit_rsa39','mega',39),('psplit36','mega',36),('packed_lt1','lt',1)]]
p['timing_contract']='Each new40 native call currentAB rawA4/finalB7 to register finalA to whole7P fixedU/alphaRNbetaFMA in one kernel with zero global matrix workspace. Each stage two ascendingK16/P and complete A24/32 retained until wait0. All retained controls complete same-binary invocation. Graph fixed captured addresses fresh contents. Complete GPU event and host 100-call batch末同步wall/100 median; tune/setup/capture excluded. Same-row Dense is independently tuned before capture; no Graph-specific tuning. Original FP64 gates/floors unchanged.'
p['frozen_sources']=init['frozen_sources'];p['frozen_binaries']=init['frozen_binaries']
p['csv_paths']=[f"results/small_{r['label']}_{m}_iter37_full1024.csv" for r in p['routes'] for m in p['modes']]
p['freeze_status']='Fresh CAN_LAUNCH40 PASS for final source7/bin2 before firsttiming; all3role implicit widths/A+P stage lifetime/init/pool/barrier/ABI/objectbenchtest verified independently. Numerical/performance separate,37 disabled.'
p['route_configurations']={k:base['route_configurations'][k] for k in ('39','36')}
p['route_configurations']['40']={'actual_name':'megakernel_wgmma_psplit_rsa_bk32_64x32','producer_quota':64,'consumer_quotas':[192,192],'launch_bounds_min_ctas':1,'enabled':True,'BM':64,'BN':32,'BK':32,'threads':384,'dynamic_shared_bytes':61568,'predicted_dynamic_shared_bytes':61568,'static_shared_bytes':1024,'drain_barrier':'nonaligned barrier.sync6,384; collective epochs0/6/0','compiled_gate':gb,'initial_registers_per_thread':168,'initial_owned_CTA_pool_registers':64512,'requested_role_pool_registers':57344,'producer_gpr_highest_including_implicit':39,'wg0_gpr_highest_including_implicit_by_dtype':{'fp16':124,'bf16':150},'wg1_gpr_highest_including_implicit':141}
p['expected_row_count']=16;p['enabled_row_count']=16
out=Path('results/psplit_rsa_bk321024_plan_iter37.json');assert not out.exists();out.write_text(json.dumps(p,indent=2)+'\n');print('PLAN',sha(out))

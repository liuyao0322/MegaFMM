from pathlib import Path
import json, hashlib
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
init=json.loads(Path('results/psplit_hybrid_s_initial_frozen_identity_iter38.json').read_text())
gatep=Path('results/psplit_hybrid_s_static_review_initial_iter38.json')
gate=json.loads(gatep.read_text())
assert gate['compiled_gate_inherited_from_iteration37'] is False
assert 'PASS_COMPILED_CAN_LAUNCH_41' in gate['status']
for f,h in {**init['frozen_sources'],**init['frozen_binaries']}.items():assert sha(f)==h,f
gb={'path':str(gatep),'sha256':sha(gatep),'status':gate['status']}
base=json.loads(Path('results/psplit_rsa_bk321024_plan_iter37.json').read_text())
p={k:base[k] for k in ['shapes','dtypes','modes','warmup','trials','groups','correct_trials','rotate_inputs','heuristic_candidates','workspace_limit','reference']}
p.update(schema='megafmm.iter38.full1024-plan.v1',declared_before_measurement=True,reason='Mandatory new41 hybrid S vs retained39 RS-A BK64 prior fastest, matched raw RS-A BK32 40, SS BK32 36, multilaunch Lt1; full20 irrespective quick3percent. H10/H12 iteration36 historical measurements retained, not remeasured in38.')
p['routes']=[{'label':a,'algorithm':b,'variant':c} for a,b,c in [('psplit_hybrid_s41','mega',41),('psplit_rsa39','mega',39),('psplit_rsa_bk3240','mega',40),('psplit36','mega',36),('packed_lt1','lt',1)]]
p['timing_contract']='Each new41 native call uses current AB, producer rawA4/roundedS1/FP32 S2 even-odd planes/finalB7, complete-stage register A24/32 to whole7P fixedU/alphaRNbetaFMA in one kernel with zero global matrix workspace. Each stage two ascendingK16/P and all A/P retained until wait0. Every retained route is a complete invocation in the same binary. Graph fixed captured addresses with fresh contents. Complete GPU event and host 100-call batch末同步wall/100 median; tune/setup/capture excluded. Same-row Dense is independently tuned before capture; no Graph-specific tuning. Original FP64 gates/floors unchanged.'
p['frozen_sources']=init['frozen_sources'];p['frozen_binaries']=init['frozen_binaries']
p['csv_paths']=[f"results/small_{r['label']}_{m}_iter38_full1024.csv" for r in p['routes'] for m in p['modes']]
p['freeze_status']='Independent fresh CAN_LAUNCH41 PASS for final corrected Plan/source7/bin2 before firsttiming; full all-role implicit widths/A+P lifetime/init/pool/typedplane ABI/drain/alias/objectbenchtest checked. Numerical/performance separate.37 disabled.'
p['route_configurations']={k:base['route_configurations'][k] for k in ('39','40','36')}
p['route_configurations']['41']={'actual_name':'megakernel_wgmma_psplit_hybrid_s_64x32','producer_quota':64,'consumer_quotas':[192,192],'launch_bounds_min_ctas':1,'enabled':True,'BM':64,'BN':32,'BK':32,'threads':384,'dynamic_shared_bytes':86144,'predicted_dynamic_shared_bytes':86144,'static_shared_bytes':1024,'bank_bytes':43008,'tail_product_bytes':57344,'tail_bank1_overlap_bytes':14336,'drain_barrier':'nonaligned barrier.sync6,384; collective epochs0/6/0','compiled_gate':gb,'initial_registers_per_thread':168,'initial_owned_CTA_pool_registers':64512,'requested_role_pool_registers':57344,'producer_gpr_highest_including_implicit_by_dtype':{'fp16':41,'bf16':39},'wg0_gpr_highest_including_implicit':124,'wg1_gpr_highest_including_implicit':164}
p['expected_row_count']=20;p['enabled_row_count']=20
out=Path('results/psplit_hybrid_s1024_plan_iter38.json');assert not out.exists();out.write_text(json.dumps(p,indent=2)+'\n');print('PLAN',sha(out))

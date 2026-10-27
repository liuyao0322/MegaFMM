#!/usr/bin/env python3
"""CPU-only final archive/hash validation. Reads files only, never runs CUDA."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent.parent;p='.scratch/psplit_tma_bn16_root_future43_'
H=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
read=lambda suffix:json.loads((R/(p+suffix)).read_text())
b=read('bindings.json');d=read('source_delta.json');m=read('model_attempt1.stdout');v=read('verify_attempt3.stdout');c=read('cpu_summary.json');protocol=read('protocol.json')
for key in ['protected_old_future_sha256','immutable_library_source_sha256','pinned_header_sha256']:
 for name,want in b[key].items():assert H(R/name)==want,(key,name)
assert len(b['protected_old_future_sha256'])==142 and len(b['immutable_library_source_sha256'])==42 and len(b['pinned_header_sha256'])==10
assert sum(name.endswith('.cu') for name in b['immutable_library_source_sha256'])==40
assert d['candidate_sha256']==m['candidate_sha256']==v['candidate_sha256']==c['source_sha256']==H(R/(p+'candidate.cu'))
assert v['api_sha256']==c['api_sha256']==H(R/(p+'api.h'))
assert d['replacement_count']==v['forward_inverse_replacement_count']==20
assert m['negative_control_count']==42 and m['counts']['protocol_schedules']==40
assert [a['actual_exit'] for a in c['actual_CPU_process_attempts']]==[0,0,0,0,0,0]
for a in c['actual_CPU_process_attempts']:
 assert H(R/a['source_path'])==a['source_sha256']
 for ext in ['stdout','stderr','exit']:assert H(R/a[ext+'_path'])==a[ext+'_sha256']
 assert int((R/a['exit_path']).read_text())==a['actual_exit']
f=c['actual_failed_CPU_authoring_attempt'];assert f['actual_shell_exit']==int((R/f['exit_path']).read_text())==1
assert f['model_process_was_started'] is False and f['actual_model_process_exit'] is None
assert H(R/f['combined_output_path'])==f['combined_output_sha256'] and H(R/f['exit_path'])==f['exit_sha256']
assert 'FileNotFoundError' in (R/f['combined_output_path']).read_text()
assert (R/(p+'model_attempt1.stderr')).read_bytes()==b'' and (R/(p+'verify_attempt3.stderr')).read_bytes()==b''
assert H(R/(p+'model.py'))==H(R/(p+'model_attempt1_source.py')) and H(R/(p+'verify.py'))==H(R/(p+'verify_attempt3_source.py'))
for obj in [d,m,v,c,protocol]:
 for key,value in obj.items():
  if key.startswith('actual_') and key not in ['actual_CPU_process_attempts','actual_failed_CPU_authoring_attempt']:assert value is None,(key,value)
assert protocol['expected_shared_geometry_only']['logical_SharedStorage_bytes']==95232
assert protocol['B_descriptor_source_prediction']['leading_encoded']==0 and protocol['B_descriptor_source_prediction']['stride_encoded']==16
assert protocol['source_hypothesis']['required_owned_CTA_pool_words']==37888 and protocol['source_hypothesis']['two_CTA_quota_requests_words']==75776>65536
assert protocol['source_hypothesis']['minimum_blocks_launch_bound']==1 and protocol['source_hypothesis']['two_CTA_configuration_prepared'] is False
assert protocol['native_only_domain']['M_N_K_multiples']==[128,32,128]
print(json.dumps({'status':'PASS_FINAL_CPU_BN16_FREEZE','source_sha256':c['source_sha256'],'api_sha256':c['api_sha256'],'protected_old_future_files':142,'immutable_library_sources':42,'immutable_old_CUDA_files':40,'pinned_headers':10,'protected_zero_drift':True,'source_exact_replacements':20,'CPU_successful_process_exits':[0,0,0,0,0,0],'CPU_authoring_shell_exit':1,'failed_authoring_model_process_started':False,'actual_model_exit':0,'final_CPU_negative_controls':42,'final_CPU_schedule_count':40,'source_quota_min1_only':True,'actual_GPU':None,'actual_performance':None},indent=2))

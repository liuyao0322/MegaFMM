#!/usr/bin/env python3
"""Final CPU archive checks. Reads files only; never compiles or runs CUDA."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent.parent;p='.scratch/psplit_tma_bn64_root_future42_'
H=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
b=json.loads((R/(p+'bindings.json')).read_text());d=json.loads((R/(p+'source_delta.json')).read_text());m=json.loads((R/(p+'model_attempt3.stdout')).read_text());v=json.loads((R/(p+'verify_attempt2.stdout')).read_text());c=json.loads((R/(p+'cpu_summary.json')).read_text());protocol=json.loads((R/(p+'protocol.json')).read_text())
for key in ['protected_old_future_sha256','immutable_library_source_sha256','pinned_header_sha256']:
 for name,want in b[key].items():assert H(R/name)==want,(key,name)
assert len(b['protected_old_future_sha256'])==100 and len(b['immutable_library_source_sha256'])==41
assert d['candidate_sha256']==m['candidate_sha256']==v['candidate_sha256']==c['source_sha256']==H(R/(p+'candidate.cu'))
assert v['api_sha256']==c['api_sha256']==H(R/(p+'api.h'))
assert d['replacement_count']==v['forward_inverse_replacement_count']==21
assert m['negative_control_count']==40 and m['counts']['protocol_schedules']==40
assert [a['actual_exit'] for a in c['actual_CPU_attempts']]==[0,0,1,0,0,0]
for a in c['actual_CPU_attempts']:
 assert H(R/a['source_path'])==a['source_sha256']
 for ext in ['stdout','stderr','exit']:assert H(R/a[ext+'_path'])==a[ext+'_sha256']
 assert int((R/a['exit_path']).read_text())==a['actual_exit']
assert (R/(p+'model_attempt1.stderr')).read_text().endswith('AssertionError: negative accepted: fixed_U_wrong_U7_sign\n')
assert (R/(p+'model_attempt3.stderr')).read_bytes()==b'' and (R/(p+'verify_attempt2.stderr')).read_bytes()==b''
assert H(R/(p+'model.py'))==H(R/(p+'model_attempt3_source.py'))
assert H(R/(p+'verify.py'))==H(R/(p+'verify_attempt2_source.py'))
for x in [d,m,v,c,protocol]:
 for key,value in x.items():
  if key.startswith('actual_') and key not in ['actual_CPU_attempts']:assert value is None,(key,value)
assert protocol['expected_shared_geometry_only']['logical_SharedStorage_bytes']==181248
assert protocol['source_hypothesis']['required_owned_CTA_pool_words']==64512
assert protocol['actual_cuBLAS_speedup'] is None and protocol['actual_latency'] is None
print(json.dumps({'status':'PASS_FINAL_CPU_FREEZE','source_sha256':c['source_sha256'],'api_sha256':c['api_sha256'],'protected_old_future_files':100,'immutable_library_sources':41,'pinned_headers':9,'protected_zero_drift':True,'source_exact_replacements':21,'CPU_model_attempt_exits':[1,0,0],'CPU_source_verify_exits':[0,0],'final_CPU_negative_controls':40,'final_CPU_schedule_count':40,'actual_GPU':None,'actual_performance':None},indent=2))

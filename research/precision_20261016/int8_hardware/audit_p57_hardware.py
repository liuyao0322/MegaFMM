"""Close producer-only unroll control with own evidence and P56 restoration."""
from pathlib import Path
import hashlib,json,re
h=Path(__file__).resolve().parent;repo=h.parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gates(file):
 text=(h/file).read_text()
 checks=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('CHECK ')]
 inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('INSIDE_EQUAL ')]
 assert len(checks)==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in checks)
 assert len(inside)==78 and all(x['errors']=='0' and x['elements']=='1024' for x in inside)
 return checks
checks=gates('p57_balanced.log');old=gates('p56_balanced.log')
assert all(x['corrected_ef']==y['corrected_ef'] for x,y in zip(checks,old))
results={}
for mode,suffix in [('ordinary','abba'),('graph','graph_abba')]:
 data=json.loads((h/f'p57_{suffix}.json').read_text())
 assert data['correctness_checks']==312 and data['independent_numerical_configurations']==78
 assert all(x['relative_reduction']<.01 for x in data['summary'])
 for arm in ['a0','b0','b1','a1']:
  assert all(x['graph']==str(int(mode=='graph')) for x in gates(f'p57_{suffix}_{arm}.log'))
 results[mode]=data['summary']
layout=json.loads((h/'probe_p57_layout.json').read_text())
assert layout['configurations']==6 and layout['errors']==0
assert layout['native_uint16_comparisons_including_cta_reuse']==415236096
assert all(x['consumer_regs']=='128' for x in layout['resources'])
profile=json.loads((h/'profile_p57_stats.json').read_text())
assert [x['launch__registers_per_thread']['value'] for x in profile['kernels']]==['52','128']
for k in profile['kernels']:
 for key in ['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']:assert float(k[key]['value'])==0
static=json.loads((h/'p56_p57_static.json').read_text())
assert all(v==('preprocess' in k) for k,v in static['normalized_instruction_text_equal'].items())
assert [x['static_instructions'] for x in static['rows'] if '9corrected' in x['function']]==[2048,2048,2904,2904]
for name in ['pack.cuh','compare.cu','pow2.cuh']:assert (h/'p57'/name).read_bytes()==(h/'p56'/name).read_bytes()
c=(h/'p57/corrected.cuh').read_text()
assert c.replace(' if(tid==0){\n  #pragma unroll 7\n',' if(tid==0){\n')==(h/'p56/corrected.cuh').read_text()
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:assert (h/name).read_bytes()==(h/'p56'/name).read_bytes()
assert digest(h/'bin/compare')==digest(h/'bin/compare_p56')
gates('p57_restore_p56.log')
restore=['p57_restore_p56.log','p57_restore_p56.json','p57_restore_cli_preflight.log','p56_manifest.json','bin/compare_p56']
restore += ['p56/'+n for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']]
(h/'p57_restore_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):digest(h/f) for f in restore},indent=2)+'\n')
out=dict(candidate='P57',goal_achieved=False,performance_accepted=False,retained='P56',standard_original_fp64_gates=78,exact_inside_exponents=79872,printed_per_case_ef_matches_p56=True,full_output_bit_equality_proven=False,max_ef={dt:max(float(x['corrected_ef']) for x in checks if x['dtype']==dt) for dt in ['bf16','fp16']},own_pre_mma_layout=layout,performance=results,independent_checks_per_mode=312,independent_inside_exponents_per_mode=319488,own_profile=profile,static_machine_code=static,fresh_dual_stream_reuse_acceptance=False,sanitizer_acceptance=False,restore='Four working sources/binary byte-exact P56 from14d9720; own new-process78 gates/79872inside pass. Initial unsupported --balanced argument exited before GPU work; preserved separately.',nominal_transfer=dict(bytes_whole_grid=138412032,tma_commands_whole_grid=11264,mma_source_per_cta=480,k128_batches_per_cta=28,slots=4,consumer_shared_bytes=196736,workspace_bytes=11804032,run_kernels=2),scope='Original contiguous1024^3 FP16/BF16 to FP32, alpha1 beta0, exactinside/all7main8cross/fullK512/RN and fences unchanged. One added producer pragma; no unique latency attribution, no floor; profiling timings excluded from performance and not summed.')
(h/'p57_hardware.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p57_hardware.py','STATIC_PRODUCER_HARDWARE.md','p57_hardware.json','p57_manifest.json','p57_proposal_manifest.json','p57_layout_manifest.json','p57_abba_manifest.json','p57_graph_abba_manifest.json','p57_static_manifest.json','profile_p57_manifest.json','p57_restore_manifest.json']
(h/'p57_hardware_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):digest(h/f) for f in files},indent=2)+'\n')
print(json.dumps(dict(candidate='P57',performance_accepted=False,retained='P56',checks_per_mode=312,own_fresh_stress=False,own_sanitizers=False,goal_achieved=False),indent=2))

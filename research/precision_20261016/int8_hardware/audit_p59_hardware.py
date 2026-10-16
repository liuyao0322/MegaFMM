"""Close rejected global-plane fusion with only its own executed evidence."""
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
checks=gates('p59_balanced.log');old=gates('p56_balanced.log')
assert all(x['corrected_ef']==y['corrected_ef'] for x,y in zip(checks,old))
results={}
for control in ['', 'control_']:
 results[control or 'retention']={}
 for mode,suffix in [('ordinary','abba'),('graph','graph_abba')]:
  data=json.loads((h/f'p59_{control}{suffix}.json').read_text())
  assert data['correctness_checks']==312 and data['independent_numerical_configurations']==78
  assert all(x['relative_reduction']<.03 for x in data['summary'])
  if control:assert all(x['relative_reduction']<0 for x in data['summary'])
  for arm in ['a0','b0','b1','a1']:
   assert all(x['graph']==str(int(mode=='graph')) for x in gates(f'p59_{control}{suffix}_{arm}.log'))
  results[control or 'retention'][mode]=data['summary']
layout=json.loads((h/'probe_p59_layout.json').read_text())
assert layout['configurations']==6 and layout['errors']==0
assert layout['native_uint16_comparisons_including_cta_reuse']==415236096
assert layout['inside_exponent_comparisons']==6144
assert all(x['fused_regs']=='114' for x in layout['resources'])
profile=json.loads((h/'profile_p59_stats.json').read_text())
assert len(profile['kernels'])==1 and profile['kernels'][0]['launch__registers_per_thread']['value']=='114'
for key in ['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']:assert float(profile['kernels'][0][key]['value'])==0
static=json.loads((h/'p58_p59_static.json').read_text())
assert all(static['normalized_instruction_text_equal'].values())
assert [x['static_instructions'] for x in static['rows'] if '12fused_global' in x['function']]==[5280,5312]
for name in ['pack.cuh','compare.cu','pow2.cuh']:assert (h/'p59'/name).read_bytes()==(h/'preflight_p59_01/p59'/name).read_bytes()
initial=(h/'preflight_p59_01/p59/corrected.cuh').read_text()
assert initial.replace('(std::is_same_v<T,cute::half_t>?"fp16":"bf16")','(std::is_same_v<T,cute::half_t> ? "fp16" : "bf16")')==(h/'p59/corrected.cuh').read_text()
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:assert (h/name).read_bytes()==(h/'p56'/name).read_bytes()
assert digest(h/'bin/compare')==digest(h/'bin/compare_p56')
gates('p59_restore_p56.log')
restore=['p59_restore_p56.log','p59_restore_p56.json','p56_manifest.json','bin/compare_p56']+['p56/'+n for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']]
(h/'p59_restore_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):digest(h/f) for f in restore},indent=2)+'\n')
out=dict(candidate='P59',goal_achieved=False,performance_accepted=False,research_retained=False,retained='P56',standard_original_fp64_gates=78,exact_inside_exponents=79872,printed_per_case_ef_matches_p56=True,full_output_bit_equality_proven=False,max_ef={dt:max(float(x['corrected_ef']) for x in checks if x['dtype']==dt) for dt in ['bf16','fp16']},own_pre_mma_layout=layout,performance=results,independent_checks_per_comparison_per_mode=312,independent_inside_exponents_per_comparison_per_mode=319488,own_profile=profile,static_machine_code=static,fresh_dual_stream_reuse_acceptance=False,sanitizer_acceptance=False,stress_and_tool_sources_prepared_but_not_executed=True,restore='Four working sources/binary byte-exact P56 from14d9720; own new-process78 gates/79872inside pass.',nominal_transfer=dict(bytes_whole_grid=138412032,tma_commands_whole_grid=11264,mma_source_per_cta=480,k128_batches_per_cta=28,slots=4,shared_bytes=196736,workspace_bytes=11804032,run_kernels_balanced=1,cooperative_grid=128,logical_pack_tasks=384,grid_barriers=2),scope='Original contiguous1024^3 FP16/BF16 to FP32, alpha1 beta0. P58 exact preparation + global-plane publication + P56 producer/consume in one cooperative kernel. exactinside/all7main8cross/fullK512/RN/fences preserved. Source consumer identity and matching historical standalone device code do not prove fused output bit equality or unique latency cause; profiling excluded from performance, no floor.')
(h/'p59_hardware.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p59_hardware.py','GLOBAL_PHASE_FUSION_HARDWARE.md','p59_hardware.json','p59_manifest.json','p59_proposal_manifest.json','p59_layout_manifest.json','p59_abba_manifest.json','p59_graph_abba_manifest.json','p59_control_abba_manifest.json','p59_control_graph_abba_manifest.json','p59_static_manifest.json','profile_p59_manifest.json','p59_restore_manifest.json','p59_compile_preflight_manifest.json','p59_stress_source_manifest.json']
(h/'p59_hardware_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):digest(h/f) for f in files},indent=2)+'\n')
print(json.dumps(dict(candidate='P59',performance_accepted=False,retained='P56',checks_per_comparison_per_mode=312,own_fresh_stress=False,own_sanitizers=False,goal_achieved=False),indent=2))

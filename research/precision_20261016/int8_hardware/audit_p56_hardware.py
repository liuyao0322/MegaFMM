"""Retain P56 research reference only after its own matrix, timing and tool gates."""
from pathlib import Path
import hashlib,json,re
h=Path(__file__).resolve().parent;repo=h.parents[2]
proposal=json.loads((h/'p56_proposal_cpu.json').read_text());assert proposal['event_identity'] and proposal['event_count']==672 and not proposal['gpu_executed']
c=(h/'p56/corrected.cuh').read_text();old=(h/'p51/corrected.cuh').read_text()
assert c.replace(' #pragma unroll 7\n for(int step=0;step<7;++step)',' #pragma unroll 1\n for(int step=0;step<7;++step)')==old
for n in ['pack.cuh','compare.cu','pow2.cuh']:assert (h/'p56'/n).read_bytes()==(h/'p51'/n).read_bytes()
for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:assert (h/n).read_bytes()==(h/'p56'/n).read_bytes()
assert (h/'bin/compare').read_bytes()==(h/'bin/compare_p56').read_bytes()
layout=json.loads((h/'probe_p56_layout.json').read_text());assert layout['configurations']==6 and layout['errors']==0 and layout['native_uint16_comparisons_including_cta_reuse']==415236096
j=json.loads((h/'p56_balanced.json').read_text());ref=json.loads((h/'p51_balanced.json').read_text());assert len(j['checks'])==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in j['checks'])
assert all(x['corrected_ef']==y['corrected_ef'] for x,y in zip(j['checks'],ref['checks']))
validation=json.loads((h/'p56_validation.json').read_text());assert validation['independent_stress_configurations']==168 and validation['independent_stress_calls']==1344
assert validation['sanitizer_cases_per_tool']==56 and validation['sanitizer_calls_per_tool']==112
performance={}
for mode,suffix in [('ordinary','abba'),('graph','graph_abba')]:
 result=json.loads((h/f'p56_{suffix}.json').read_text());assert result['correctness_checks']==312 and all(x['relative_reduction']>.03 for x in result['summary'])
 for arm in ['a0','b0','b1','a1']:
  text=(h/f'p56_{suffix}_{arm}.log').read_text();checks=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('CHECK ')]
  inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('INSIDE_EQUAL ')]
  assert len(checks)==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in checks)
  assert len(inside)==78 and all(x['errors']=='0' and x['elements']=='1024' for x in inside)
 performance[mode]=result['summary']
static=json.loads((h/'p51_p56_static.json').read_text());assert len(static['rows'])==8
for row in static['rows']:
 if '9corrected' in row['function']:assert row['static_instructions']==(1600 if row['candidate']=='p51' else 2048)
profile=json.loads((h/'profile_p56_stats.json').read_text());assert len(profile['kernels'])==2 and profile['kernels'][1]['launch__registers_per_thread']['value']=='128'
for k in profile['kernels']:
 for n in ['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']:assert float(k[n]['value'])==0
out=dict(candidate='P56',goal_achieved=False,production_performance_accepted=False,research_retained=True,retained='P56',historical_fallback='P51',source_changes='Exactly one consumer pragma unroll1 to7; all other source bytes identical P51',standard_original_fp64_gates=78,exact_inside_exponents_per_mode=79872,printed_per_case_ef_matches_p51=True,full_output_bit_equality_proven=False,own_pre_mma_layout=layout,own_validation=validation,independent_checks_per_mode=312,independent_inside_exponents_per_mode=319488,performance=performance,own_static=static,own_profile=profile,nominal_transfer=dict(bytes_whole_grid=138412032,tma_commands_whole_grid=11264,mma_source_per_cta=480,k128_batches_per_cta=28,slots=4,slot_bytes=49152,consumer_shared_bytes=196736,workspace_bytes=11804032,run_kernels=2,all_source_counts_unchanged_p51=True,interpretation='Source counts, not measured physical traffic or a floor. Unroll changes generated scheduling/register allocation/code size together, no unique attribution of gain. All profiler-process timings excluded from ABBA.'),scope='contiguous1024^3 alpha1 beta0 original FP16/BF16 input/FP32 output; exactinside/all7main8cross/fullK512/RN graph preserved. Own probes, ordinary/Graph, fresh streams/guards/scoped tools only; production unchanged and end-to-end target unachieved.')
(h/'p56_hardware.json').write_text(json.dumps(out,indent=2)+'\n')
f=[h/n for n in ['audit_p56_hardware.py','STATIC_LEAF_SCHEDULE_HARDWARE.md','p56_hardware.json','p56_proposal_manifest.json','p56_manifest.json','p56_layout_manifest.json','p56_static_manifest.json','p56_validation_manifest.json','p56_abba_manifest.json','p56_graph_abba_manifest.json','profile_p56_manifest.json']]
(h/'p56_hardware_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in f},indent=2)+'\n')
print(json.dumps(dict(candidate='P56',research_retained=True,production_performance_accepted=False,goal_achieved=False,stress=168,tool_cases_each=56,performance=performance),indent=2))

"""Own P58 resident-grid control; do not transfer P56 stress/tool acceptance."""
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
checks=gates('p58_balanced.log');old=gates('p56_balanced.log')
assert all(x['corrected_ef']==y['corrected_ef'] for x,y in zip(checks,old))
results={}
for mode,suffix in [('ordinary','abba'),('graph','graph_abba')]:
 data=json.loads((h/f'p58_{suffix}.json').read_text())
 assert data['correctness_checks']==312 and data['independent_numerical_configurations']==78
 assert all(x['relative_reduction']<.03 for x in data['summary'])
 for arm in ['a0','b0','b1','a1']:
  assert all(x['graph']==str(int(mode=='graph')) for x in gates(f'p58_{suffix}_{arm}.log'))
 results[mode]=data['summary']
pack=json.loads((h/'probe_p58_pack.json').read_text());edges=json.loads((h/'probe_p58_edges.json').read_text())
assert pack['configurations']==54 and edges['configurations']==18
assert pack['bitwise_errors']==edges['bitwise_errors']==0
profile=json.loads((h/'profile_p58_stats.json').read_text())
assert [x['launch__registers_per_thread']['value'] for x in profile['kernels']]==['52','128']
for k in profile['kernels']:
 for key in ['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']:assert float(k[key]['value'])==0
static=json.loads((h/'p56_p58_static.json').read_text())
assert all(v==('9corrected' in k) for k,v in static['normalized_instruction_text_equal'].items())
for name in ['corrected.cuh','compare.cu','pow2.cuh']:assert (h/'p58'/name).read_bytes()==(h/'p56'/name).read_bytes()
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:assert (h/name).read_bytes()==(h/'p56'/name).read_bytes()
assert digest(h/'bin/compare')==digest(h/'bin/compare_p56')
gates('p58_restore_p56.log')
coop=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in (h/'p58_balanced.log').read_text().splitlines() if l.startswith('COOPERATIVE_INIT ')]
assert len(coop)==2 and all(x['grid']=='128' and x['active_blocks_per_sm']=='3' and x['shared_bytes']=='65536' and x['sm_count']=='132' for x in coop)
restore=['p58_restore_p56.log','p58_restore_p56.json','p56_manifest.json','bin/compare_p56']+['p56/'+n for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']]
(h/'p58_restore_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):digest(h/f) for f in restore},indent=2)+'\n')
out=dict(candidate='P58',goal_achieved=False,performance_accepted=False,retained='P56',standard_original_fp64_gates=78,exact_inside_exponents=79872,printed_per_case_ef_matches_p56=True,full_output_bit_equality_proven=False,max_ef={dt:max(float(x['corrected_ef']) for x in checks if x['dtype']==dt) for dt in ['bf16','fp16']},own_pack_bits=pack,own_edge_bits=edges,cooperative_capacity=coop,performance=results,independent_checks_per_mode=312,independent_inside_exponents_per_mode=319488,own_profile=profile,static_machine_code=static,fresh_dual_stream_reuse_acceptance=False,sanitizer_acceptance=False,restore='Four working sources/binary byte-exact P56 from14d9720; own new-process78 gates/79872inside pass.',nominal_transfer=dict(bytes_whole_grid=138412032,tma_commands_whole_grid=11264,mma_source_per_cta=480,k128_batches_per_cta=28,slots=4,consumer_shared_bytes=196736,workspace_bytes=11804032,run_kernels=2,cooperative_grid=128,logical_pack_tasks=384),scope='Original contiguous1024^3 FP16/BF16 to FP32, alpha1 beta0, exactinside/all7main8cross/fullK512/RN and consumer unchanged. Reduced resident prep grid/three logical tasks perCTA with identical input/output coordinate and math expressions; no unique latency attribution, no floor; profiling excluded from performance and not summed.')
(h/'p58_hardware.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p58_hardware.py','RESIDENT_PACK_HARDWARE.md','p58_hardware.json','p58_manifest.json','p58_proposal_manifest.json','p58_pack_bits_manifest.json','p58_edges_manifest.json','p58_abba_manifest.json','p58_graph_abba_manifest.json','p58_static_manifest.json','profile_p58_manifest.json','p58_restore_manifest.json']
(h/'p58_hardware_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):digest(h/f) for f in files},indent=2)+'\n')
print(json.dumps(dict(candidate='P58',performance_accepted=False,retained='P56',checks_per_mode=312,own_fresh_stress=False,own_sanitizers=False,goal_achieved=False),indent=2))

"""Close the paired native storage control using only its own executed evidence."""
from pathlib import Path
import hashlib,json,re
h=Path(__file__).resolve().parent;repo=h.parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gates(name):
 text=(h/name).read_text();rows=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('CHECK ')]
 inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('INSIDE_EQUAL ')]
 assert len(rows)==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in rows)
 assert len(inside)==78 and all(x['errors']=='0' and x['elements']=='1024' for x in inside)
 return rows
checks=gates('p60_balanced.log');old=gates('p56_balanced.log')
assert all(x['corrected_ef']==y['corrected_ef'] for x,y in zip(checks,old))
results={}
for mode,suffix in [('ordinary','abba'),('graph','graph_abba')]:
 data=json.loads((h/f'p60_{suffix}.json').read_text())
 assert data['correctness_checks']==312 and all(x['relative_reduction']<.03 for x in data['summary'])
 for arm in ['a0','b0','b1','a1']:assert all(x['graph']==str(int(mode=='graph')) for x in gates(f'p60_{suffix}_{arm}.log'))
 results[mode]=data['summary']
pack=json.loads((h/'probe_p60_pack.json').read_text())
assert pack['packing_configurations']==84 and pack['conversion_native_comparisons']==4194304
assert pack['native_uint16_comparisons']==484442112 and pack['bitwise_errors']==pack['guard_errors']==0
profile=json.loads((h/'profile_p60_stats.json').read_text())
assert len(profile['kernels'])==2
assert [x['launch__registers_per_thread']['value'] for x in profile['kernels']]==['37','128']
for row in profile['kernels']:
 for key in ['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']:assert float(row[key]['value'])==0
static=json.loads((h/'p56_p60_static.json').read_text())
assert all(v for k,v in static['normalized_instruction_text_equal'].items() if '9corrected' in k)
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:assert (h/name).read_bytes()==(h/'p56'/name).read_bytes()
assert digest(h/'bin/compare')==digest(h/'bin/compare_p56')
gates('p60_restore_p56.log')
restore=['p60_restore_p56.log','p60_restore_p56.json','p56_manifest.json','bin/compare_p56']+['p56/'+n for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']]
(h/'p60_restore_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):digest(h/f) for f in restore},indent=2)+'\n')
out=dict(candidate='P60',goal_achieved=False,performance_accepted=False,research_retained=False,retained='P56',standard_original_fp64_gates=78,exact_inside_exponents=79872,printed_per_case_ef_matches_p56=True,full_output_bit_equality_proven=False,own_pre_mma_pack=pack,performance=results,independent_checks_per_mode=312,independent_inside_per_mode=319488,own_profile=profile,static_machine_code=static,fresh_dual_stream_reuse_acceptance=False,sanitizer_acceptance=False,stress_and_tool_sources_prepared_but_not_executed=True,restore='Four working sources/binary byte-exact P56 from14d9720, new-process78/79872inside pass',scope='Original1024^3 contiguous FP16/BF16 to FP32 alpha1 beta0; paired storage/conversion, componentwise FP32 RN and exact inside/all7main8cross/consumer preserved; all complete-call costs timed, unlocked/noCI/no floor')
(h/'p60_hardware.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p60_hardware.py','PAIR_NATIVE_PACK_HARDWARE.md','p60_hardware.json','p60_manifest.json','p60_pack_manifest.json','p60_abba_manifest.json','p60_graph_abba_manifest.json','p60_static_manifest.json','profile_p60_manifest.json','p60_restore_manifest.json','p60_stress_source_manifest.json','p60_compile_preflight_manifest.json']
(h/'p60_hardware_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):digest(h/f) for f in files},indent=2)+'\n')
print(json.dumps(dict(candidate='P60',retained='P56',performance_accepted=False,own_pre_mma_bits=484442112,goal_achieved=False),indent=2))

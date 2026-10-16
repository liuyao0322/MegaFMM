"""Close rejected P55 with only its own evidence, and byte-exact P51 restoration."""
from pathlib import Path
import hashlib,json,re
h=Path(__file__).resolve().parent;repo=h.parents[2]
pack=json.loads((h/'probe_p55_pack.json').read_text());layout=json.loads((h/'probe_p55_layout.json').read_text());extra=json.loads((h/'extra_p55.json').read_text())
assert pack['configurations']==96 and pack['errors']==0
assert pack['native_uint16_comparisons']==352321536 and pack['int8_byte_comparisons']==452984832
assert pack['inside_exponents']==98304 and pack['quant_exponents']==110592
assert pack['scalar']['scaling_cases']=='6611' and pack['scalar']['scaling_bit_errors']=='0'
assert pack['scalar']['native_exponent_cases']=='64384' and pack['scalar']['exponent_errors']=='0'
assert layout['configurations']==6 and layout['errors']==0
assert layout['native_uint16_comparisons_including_cta_reuse']==264241152 and layout['int8_comparisons_including_cta_reuse']==301989888
assert extra['original_gates']==84 and extra['original_gate_failures']==0 and extra['inside_exponents']==86016
assert all(v['fixed_floor_failures']==0 for v in extra['summary'].values())
def gates(file):
 text=(h/file).read_text();checks=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('CHECK ')]
 inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('INSIDE_EQUAL ')]
 assert len(checks)==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in checks)
 assert len(inside)==78 and all(x['errors']=='0' and x['elements']=='1024' for x in inside)
 return checks
checks=gates('p55_balanced.log');results={}
for mode,suffix in [('ordinary','abba'),('graph','graph_abba')]:
 result=json.loads((h/f'p55_{suffix}.json').read_text());assert result['correctness_checks']==312 and result['independent_numerical_configurations']==78
 assert all(x['relative_reduction']<-.6 for x in result['summary'])
 for arm in ['a0','b0','b1','a1']:
  armchecks=gates(f'p55_{suffix}_{arm}.log')
  assert all(x['graph']==str(int(mode=='graph')) for x in armchecks)
 results[mode]=result['summary']
profile=json.loads((h/'profile_p55_stats.json').read_text());assert len(profile['kernels'])==2
assert [k['launch__registers_per_thread']['value'] for k in profile['kernels']]==['40','182']
for k in profile['kernels']:
 for n in ['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']:assert float(k[n]['value'])==0
for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:assert (h/n).read_bytes()==(h/'p51'/n).read_bytes()
assert (h/'bin/compare').read_bytes()==(h/'bin/compare_p51').read_bytes()
assert (h/'p55/pow2.cuh').read_bytes()==(h/'p51/pow2.cuh').read_bytes()
gates('p55_restore_p51.log')
p=(h/'p55/pack.cuh').read_text();c=(h/'p55/corrected.cuh').read_text()
for needle in ['WorkBytes==12329472','sizeof(DirectPackShared)==55808','__launch_bounds__(768,1)']:assert needle in p
for needle in ['sizeof(Shared)==197248','s.ae[ap][(mn/64)/8]+s.be[bp][(mn%64)/8]','uint32_t integer[16]={};','warpgroup_wait<1>();']:assert needle in c
assert 'meta_common' not in c
out=dict(candidate='P55',goal_achieved=False,performance_accepted=False,retained='P51',ordinary_original_fp64_checks=78,equal_inside_exponents=79872,max_ef={dt:max(float(x['corrected_ef']) for x in checks if x['dtype']==dt) for dt in ['bf16','fp16']},pre_mma_pack=pack,pre_mma_layout=layout,extra_original_input=extra,independent_checks_per_mode=312,independent_inside_exponents_per_mode=319488,performance=results,own_profile=profile,fresh_dual_stream_reuse_acceptance=False,sanitizer_acceptance=False,nominal_transfer=dict(bytes_whole_grid=138412032,tma_commands_whole_grid=15360,mma_instructions_per_cta=352,k128_batches_per_cta=48,workspace_bytes=12329472,run_kernels=2,interpretation='source counts, not measured physical traffic or a floor. Prep/quantization/transpose/consumer schedule/metadata/addition order change together; no unique causal attribution. All profiler-process times excluded from ABBA and not summed.'),restore='Four working sources/binary byte-exact P51; own fresh process78 original gates and79872inside pass',scope='contiguous1024^3 alpha1 beta0, original FP16/BF16 input/FP32 output, exactinside/7main8cross/fullK512, originalFP64AB/gates. Own extra84 is single-stream/workspace only; no transferred certificate.')
(h/'p55_hardware.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p55_hardware.py','DIRECT_FINE_QUANT_HARDWARE.md','p55_hardware.json','p55_manifest.json','p55_pack_manifest.json','p55_layout_manifest.json','p55_extra_manifest.json','p55_abba_manifest.json','p55_graph_abba_manifest.json','profile_p55_manifest.json','p55_restore_manifest.json','LITERATURE_UPDATE_P55.md']
(h/'p55_hardware_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(dict(candidate='P55',performance_accepted=False,retained='P51',standard=78,extra_new_original_input=84,checks_per_mode=312,own_dual_stream_stress=False,own_sanitizers=False,goal_achieved=False),indent=2))

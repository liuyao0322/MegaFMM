"""Reject P53 without transferring any other candidate's stress/tool certificate."""
from pathlib import Path
import hashlib,json,re
h=Path(__file__).resolve().parent;repo=h.parents[2]
layout=json.loads((h/'probe_p53_layout.json').read_text())
assert layout['configurations']==6 and layout['bit_errors']==0 and layout['native_uint16_comparisons_including_cta_reuse']==415236096
assert layout['recombine_output_bit_comparisons']==3145728 and layout['recombine_bit_errors']==0
j=json.loads((h/'p53_balanced.json').read_text());assert len(j['checks'])==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in j['checks'])
ref=json.loads((h/'p51_balanced.json').read_text());assert all(x['corrected_ef']==y['corrected_ef'] for x,y in zip(j['checks'],ref['checks']))
results={}
for mode,suffix in [('ordinary','abba'),('graph','graph_abba')]:
 result=json.loads((h/f'p53_{suffix}.json').read_text());assert result['correctness_checks']==312 and all(x['relative_reduction']<-.3 for x in result['summary'])
 for arm in ['a0','b0','b1','a1']:
  text=(h/f'p53_{suffix}_{arm}.log').read_text()
  checks=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('CHECK ')]
  inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('INSIDE_EQUAL ')]
  assert len(checks)==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in checks)
  assert len(inside)==78 and all(x['errors']=='0' and x['elements']=='1024' for x in inside)
 results[mode]=result['summary']
profile=json.loads((h/'profile_p53_stats.json').read_text());assert len(profile['kernels'])==3
assert profile['kernels'][1]['launch__registers_per_thread']['value']=='74'
assert profile['kernels'][2]['launch__registers_per_thread']['value']=='18'
for k in profile['kernels']:
 for name in ['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']:assert float(k[name]['value'])==0
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:assert (h/name).read_bytes()==(h/'p51'/name).read_bytes()
for name in ['compare.cu','pow2.cuh']:assert (h/'p53'/name).read_bytes()==(h/'p51'/name).read_bytes()
assert (h/'bin/compare').read_bytes()==(h/'bin/compare_p51').read_bytes()
r=json.loads((h/'p53_restore_p51.json').read_text());assert len(r['checks'])==78 and all(x['pass']=='1' for x in r['checks'])
source=(h/'p53/corrected.cuh').read_text()
for needle in ['<<<dim3(16,8,7),256,sizeof(Shared),stream>>>','recombine<<<1024,256,0,stream>>>','sizeof(Shared)==98432','float d[16]={},r[16]={};','for(int k=0;k<4;++k)']:
 assert needle in source,needle
assert 'float c[4][16]' not in source
out=dict(candidate='P53',goal_achieved=False,performance_accepted=False,retained='P51',ordinary_original_fp64_checks=78,equal_inside_exponents=79872,independent_checks_per_mode=312,independent_exact_exponents_per_mode=319488,printed_per_case_ef_matches_p51=True,full_output_bit_equality_proven=False,layout=layout,performance=results,own_profile=profile,fresh_stream_guard_acceptance=False,sanitizer_acceptance=False,nominal_transfer=dict(bytes_whole_grid=138412032,tma_commands=11264,source_payload_same_as_p51=True,additional_leaf_workspace_bytes=7340032,total_workspace_bytes=19144064,run_kernels=3,interpretation='source box counts, not measured physical traffic; shared stage depth/grid/temporary output change together, no unique causal attribution or physical floor; profiler timings excluded and not summed into ABBA decomposition'),restore='Four working sources/binary restored byte-exact P51, own fresh process78gates/exactinside passes',scope='contiguous1024^3 alpha1 beta0, original FP16/BF16 inputs/FP32output; originalFP64AB/gates, exact inside, seven main/eightcrossdots; no certificate transfer')
(h/'p53_hardware.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p53_hardware.py','LEAF_PARALLEL_HARDWARE.md','p53_hardware.json','p53_manifest.json','p53_layout_manifest.json','p53_abba_manifest.json','p53_graph_abba_manifest.json','profile_p53_manifest.json','p53_restore_manifest.json','p53_recombine_model_manifest.json']
(h/'p53_hardware_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(dict(candidate='P53',performance_accepted=False,retained='P51',checks_per_mode=312,layout_bit_errors=0,recombine_gpu_bit_errors=0,fresh_stream_tools=False,goal_achieved=False),indent=2))

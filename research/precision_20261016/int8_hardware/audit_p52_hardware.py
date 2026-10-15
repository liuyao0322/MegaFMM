"""Reject P52 performance without transferring P51 stress/tool certificates."""
from pathlib import Path
import hashlib,json,re
h=Path(__file__).resolve().parent;repo=h.parents[2]
layout=json.loads((h/'probe_p52_layout.json').read_text());assert layout['configurations']==6 and layout['bit_errors']==0 and layout['native_uint16_comparisons_including_cta_reuse']==692060160
j=json.loads((h/'p52_balanced.json').read_text());assert len(j['checks'])==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in j['checks'])
ref=json.loads((h/'p51_balanced.json').read_text());assert all(x['corrected_ef']==y['corrected_ef'] for x,y in zip(j['checks'],ref['checks']))
results={}
for mode,suffix in [('ordinary','abba'),('graph','graph_abba')]:
 result=json.loads((h/f'p52_{suffix}.json').read_text());assert result['correctness_checks']==312 and all(x['relative_reduction']<0 for x in result['summary'])
 for arm in ['a0','b0','b1','a1']:
  text=(h/f'p52_{suffix}_{arm}.log').read_text()
  checks=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('CHECK ')]
  inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('INSIDE_EQUAL ')]
  assert len(checks)==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in checks)
  assert len(inside)==78 and all(x['errors']=='0' and x['elements']=='1024' for x in inside)
 results[mode]=result['summary']
profile=json.loads((h/'profile_p52_stats.json').read_text())
assert profile['kernels'][1]['launch__registers_per_thread']['value']=='95'
for k in profile['kernels']:
 for name in ['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']:assert float(k[name]['value'])==0
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:assert (h/name).read_bytes()==(h/'p51'/name).read_bytes()
assert (h/'bin/compare').read_bytes()==(h/'bin/compare_p51').read_bytes()
r=json.loads((h/'p52_restore_p51.json').read_text());assert len(r['checks'])==78 and all(x['pass']=='1' for x in r['checks'])
initial=(h/'preflight_p52_01/corrected.cuh').read_text();final=(h/'p52/corrected.cuh').read_text()
assert initial.replace('(std::is_same_v<T,cute::half_t>?"fp16":"bf16")','(std::is_same_v<T,cute::half_t> ? "fp16" : "bf16")')==final
out=dict(candidate='P52',goal_achieved=False,performance_accepted=False,retained='P51',ordinary_original_fp64_checks=78,equal_inside_exponents=79872,independent_checks_per_mode=312,independent_exact_exponents_per_mode=319488,printed_per_case_ef_matches_p51=True,full_output_bit_equality_proven=False,layout=layout,performance=results,own_profile=profile,fresh_stream_guard_acceptance=False,sanitizer_acceptance=False,preflight='Initial host-print ternary compile failed2, no binary/device execution; spaced ternary retry succeeds, arithmetic/TMA unchanged. Original proposal and archival source/log hashes preserved.',nominal_transfer=dict(p51_bytes_per_cta=1081344,p51_ctas=128,p52_bytes_per_cta=901120,p52_ctas=256,whole_grid_byte_ratio=5/3,interpretation='source/TMA box payload count, not measured physical traffic; all mathematical dot work retained. Shape/instruction count, ring depth and byte reuse change together, no unique causal attribution or floor.'),restore='Four working sources restored from committedP51 and binary copied byte-exact; own fresh process78gates passes',scope='contiguous1024^3 alpha1 beta0, original FP16/BF16 inputs/FP32output; originalFP64AB/gates, exact inside, seven main/eightcrossdots; no certificate transfer')
(h/'p52_hardware.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p52_hardware.py','CONSUMER_N16_HARDWARE.md','p52_hardware.json','p52_manifest.json','p52_layout_manifest.json','p52_abba_manifest.json','p52_graph_abba_manifest.json','profile_p52_manifest.json','p52_restore_manifest.json','p52_preflight_manifest.json']
(h/'p52_hardware_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(dict(candidate='P52',performance_accepted=False,retained='P51',checks_per_mode=312,layout_bit_errors=0,fresh_stream_tools=False,goal_achieved=False),indent=2))

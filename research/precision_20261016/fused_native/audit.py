"""Keep rejected native fusion evidence separate from historical P47 controls, retained P51 and production."""
import hashlib,json,re
from pathlib import Path
r=Path(__file__).resolve().parent;repo=r.parents[2];h=r.parent/'int8_hardware'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest_files=sorted(r.glob('n[0-9]*_manifest.json'))
for manifest in manifest_files:
 for name,sha in json.loads(manifest.read_text()).items():assert digest(repo/name)==sha,(manifest,name)
for name,sha in json.loads((r.parent/'source_manifest.json').read_text())['production_unchanged_sha256'].items():
 assert digest(repo/name)==sha,name
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:
 assert digest(h/name)==digest(h/'p51'/name),name
assert digest(h/'bin/compare')==digest(h/'bin/compare_p51')
validation=json.loads((h/'p51_validation.json').read_text())
assert validation['independent_stress_configurations']==144 and validation['sanitizer_cases_per_tool']==48
results={}
for label in ['n1','n2','n3']:
 result=json.loads((r/f'{label}_summary.json').read_text())
 assert result['performance_accepted']==False and result['goal_achieved']==False
 assert not result['fresh_stream_guard_acceptance'] and not result['sanitizer_acceptance']
 assert result['correctness_checks_per_mode']==312 and result['exact_inside_exponents_per_mode']==319488
 assert all(x[f'{label}_over_p47']>1 for x in result['summary'])
 layout=json.loads((r/f'{label}_layout_summary.json').read_text())
 assert layout['configurations']==54 and layout['native_uint16_comparisons_including_cta_reuse']==3737124864
 assert layout['exact_inside_exponents']==55296 and layout['bit_errors']==0 and layout['pre_mma_only']
 for mode in ['ordinary','graph']:
  for arm in ['a0','b0','b1','a1']:
   text=(r/f'{label}_{mode}_abba_{arm}.log').read_text()
   checks=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('CHECK ')]
   inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('INSIDE_EQUAL ')]
   assert len(checks)==78 and all(x['pass']=='1' and x['nonfinite']=='0' for x in checks)
   assert len(inside)==78 and all(x['errors']=='0' for x in inside)
 profile=json.loads((r/f'{label}_profile_stats.json').read_text())['metrics']
 for key in ['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']:
  assert float(profile[key]['value'])==0
 results[label]=dict(precision_checks_per_mode=312,exact_exponents_per_mode=319488,layout=layout,summary=result['summary'],performance_accepted=False,fresh_stream_guard_acceptance=False,sanitizer_acceptance=False,own_profile=profile)
failed=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in (r/'preflight_01/n1_layout.log').read_text().splitlines() if l.startswith('NATIVE_LAYOUT ')]
assert len(failed)==54 and any(x['pass']=='0' for x in failed)
# These source checks are a storage-control audit, not N3 hardware acceptance.
n2=(r/'n2/prototype.cuh').read_text();n3=(r/'n3/prototype.cuh').read_text()
assert 'int cached_inside[1024]' in n3 and 'sizeof(Shared)==188544' in n3
assert 'produce<T>(s,&map_a,&map_b,s.cached_inside,m0,n0)' in n3
out=dict(status='progress',goal_achieved=False,production_unchanged=True,retained_candidate='P51',manifest_files=len(manifest_files),candidates=results,n1_initial_layout_preflight_failed_before_mma=True,n3='Own original-input and layout gates pass; slower than P47 in independent ordinary/Graph ABBA, no fresh stream/guard or sanitizer acceptance')
(r/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status=out['status'],goal_achieved=False,production_unchanged=True,retained_candidate='P51',manifests_checked=len(manifest_files),rejected=['N1','N2','N3']),indent=2))

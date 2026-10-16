"""Only P60's own pre-MMA conversion, plane, alignment and guard evidence."""
from pathlib import Path
import hashlib,json,re
h=Path(__file__).resolve().parent;repo=h.parents[2]
text=(h/'probe_p60_pack.log').read_text()
def rows(tag):return [dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith(tag+' ')]
oracle=rows('PAIR_CONVERSION');pack=rows('PAIR_PACK');resources=rows('PAIR_RESOURCE')
assert len(oracle)==2 and all(x['high_errors']==x['low_errors']=='0' and x['pass']=='1' and int(x['swapped_lane_negative'])>0 for x in oracle)
assert len(pack)==84 and len(resources)==2
assert all(x['pass']=='1' and all(x[k]=='0' for k in ['a_errors','b_errors','exponent_errors','a_guard_errors','b_guard_errors','workspace_guard_errors']) for x in pack)
assert sum(x['input_offset_bytes']=='0' and int(x['pattern'])<9 for x in pack)==54
assert sum(x['input_offset_bytes']=='0' and int(x['pattern'])>=9 for x in pack)==18
assert sum(x['input_offset_bytes']=='2' for x in pack)==12
assert all(int(x['outside_normal'])>0 for x in pack if x['dtype']=='bf16' and int(x['pattern'])>=9)
capacity=rows('COOPERATIVE_INIT');assert len(capacity)==4
assert all(x['supported']=='1' and int(x['active_blocks_per_sm'])*int(x['sm_count'])>=384 and x['shared_bytes']=='65536' for x in capacity)
for name in ['corrected.cuh','compare.cu','pow2.cuh']:assert (h/'p60'/name).read_bytes()==(h/'p56'/name).read_bytes()
old=(h/'p56/pack.cuh').read_text();new=(h/'p60/pack.cuh').read_text();key='template<class T,bool SideA,bool Balance> __global__'
assert new[new.index(key):]==old[old.index(key):]
build=(h/'probe_p60_pack_build.log').read_text()
assert not re.search(r'warning|error:|C7515',build,re.I)
assert not re.search(r'[1-9][0-9]* bytes (?:stack frame|spill stores|spill loads)',build)
out=dict(candidate='P60',scope=__doc__,conversion_pairs=sum(int(x['pairs']) for x in oracle),conversion_native_comparisons=sum(int(x['native_comparisons']) for x in oracle),packing_configurations=84,aligned_standard=54,aligned_edge=18,misaligned_input=12,native_uint16_comparisons=sum(int(x['native_values']) for x in pack),inside_comparisons=84*1024,bitwise_errors=0,guard_errors=0,resources=resources,cooperative_capacity=capacity,conversion_rows=oracle,rows=pack,matrix_acceptance=False,performance_acceptance=False,sanitizer_acceptance=False)
(h/'probe_p60_pack.json').write_text(json.dumps(out,indent=2)+'\n')
files=['audit_p60_pack.py','probe_p60_pack.json','probe_p60_pack.log','probe_p60_pack_build.log','bin/probe_p60_pack','p60_probe_source_manifest.json','p60_source_manifest.json','p60_compile_preflight_manifest.json']
(h/'p60_pack_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['rows','conversion_rows','cooperative_capacity']},indent=2))

"""P60 paired coordinate, alignment and broadcast model before C++ generation."""
from pathlib import Path
import collections,hashlib,json
h=Path(__file__).resolve().parent;repo=h.parents[2]
checks=0;coords={};fallback=0
for side,rows,cols,tiles in [('A',16,128,128),('B',32,32,256)]:
 visited=collections.Counter()
 for tile in range(tiles):
  r0=tile//(512//cols)*rows;c0=tile%(512//cols)*cols
  for tid in range(256):
   lane=tid%32;warp=tid//32
   for j in range(rows*cols//512):
    i=2*(tid+256*j);r=r0+i//cols;c=c0+i%cols
    assert 0<=r<512 and 0<=c<c+1<512 and c%2==0
    for q in [0,512,512*1024,512*1024+512]:
     byte=(r*1024+c+q)*2;assert byte%4==0
     assert (byte+2)%4==2;fallback+=1
    for plane in range(11):assert ((plane*512*512+r*512+c)*2)%4==0
    if side=='B':
     src_lane=lane//16+2*j
     pos=r0+2*warp+(src_lane%2)+16*((src_lane//2)%2)
     assert pos==r
    else:
     assert c==c0+2*(tid%64) and c+1<512
    visited[r,c]+=1;visited[r,c+1]+=1;checks+=1
 assert len(visited)==262144 and set(visited.values())=={1}
 coords[side]=len(visited)
for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:assert (h/n).read_bytes()==(h/'p56'/n).read_bytes()
out=dict(candidate='P60',proposal_only=True,cpp_generated=False,gpu_executed=False,retained='P56',paired_coordinates=checks,covered_elements=coords,original_input_alignment_preserved=True,input_offset2_requires_scalar_fallback=fallback,raw_packing_scalar_values=2097152,raw_packing_aligned_words=1048576,native_plane_values=5767168,native_plane_words=2883584,native_plane_bytes=11534336,inside_reduction_unchanged=True,cooperative_grid=384,threads=256,shared_bytes=65536,workspace_bytes=11804032,run_kernels=2,actual_capacity_checked=False,hardware_accepted=False,scope='CPU integer indices, alignment/fallback decisions and literal B factor broadcast only; no floating conversion, GPU arithmetic, memory race, full matrix or performance acceptance')
(h/'p60_proposal_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
files=[Path(__file__),h/'PAIR_NATIVE_PACK_DESIGN.md',h/'LITERATURE_UPDATE_P59.md',h/'p60_proposal_cpu.json',h/'p59_hardware_manifest.json',h/'p56_hardware_manifest.json',repo/'third_party/cutlass/include/cutlass/numeric_conversion.h',repo/'third_party/cutlass/include/cutlass/version.h']
(h/'p60_proposal_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

"""Independent P58 task/element/factor coverage before code generation."""
from pathlib import Path
import collections,hashlib,json
h=Path(__file__).resolve().parent;repo=h.parents[2]
tasks=[('B',block+offset) for block in range(128) for offset in [0,128]]
tasks += [('A',block) for block in range(128)]
assert len(set(tasks))==384
assert set(tasks)=={(side,tile) for side,count in [('A',128),('B',256)] for tile in range(count)}
counts={};factors=0
for side,rows,cols in [('A',16,128),('B',32,32)]:
 visits=collections.Counter()
 for s,tile in tasks:
  if s!=side:continue
  r0=tile//(512//cols)*rows;c0=tile%(512//cols)*cols
  for tid in range(256):
   pos=c0+tid%cols if side=='A' else r0+tid//32+8*(tid%4)
   assert 0<=pos<512
   for j in range(rows*cols//256):
    i=tid+256*j;r=r0+i//cols;c=c0+i%cols
    visits[r,c]+=1
    # B fast path selects lane j's factor; A uses each lane's own factor.
    factor=pos if side=='A' else r0+tid//32+8*j
    assert factor==(c if side=='A' else r)
    factors+=2
 assert len(visits)==262144 and set(visits.values())=={1}
 counts[side]=len(visits)
assert {8*b+t for b in range(128) for t in range(8)}==set(range(1024))
for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:assert (h/n).read_bytes()==(h/'p56'/n).read_bytes()
out=dict(candidate='P58',proposal_only=True,cpp_generated=False,gpu_executed=False,retained='P56',resident_grid=128,threads=256,logical_tasks=384,tasks_per_cta=3,covered_coordinates=counts,balanced_factor_indices=factors,inside_exponents=1024,shared_bytes=65536,workspace_bytes=11804032,run_kernels=2,consumer_byte_unchanged=True,actual_capacity_checked=False,hardware_accepted=False,scope='CPU task/element/fast-factor indexing only, no cast/arithmetic/GPU/latency acceptance')
(h/'p58_proposal_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
files=[Path(__file__),h/'RESIDENT_PACK_DESIGN.md',h/'p58_proposal_cpu.json',h/'p56_hardware_manifest.json']
(h/'p58_proposal_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

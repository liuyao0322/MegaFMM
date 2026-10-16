"""Pre-generation CPU identity of dynamic/static seven-leaf operation schedules."""
from pathlib import Path
import hashlib,json
h=Path(__file__).resolve().parent;repo=h.parents[2]
order=[0,1,5,6,4,2,3]
def old_product(step):return step if step<2 else 5 if step==2 else 6 if step==3 else 4 if step==4 else 2 if step==5 else 3
def schedule(products):
 events=[]
 for step,p in enumerate(products):
  events += [('init',step,'d/r'),('compiler_fence',step,'d/r'),('hardware_arrive',step)]
  for k in range(4):
   seq=4*step+k;slot=seq%4;phase=seq//4%2
   events += [('ready',slot,phase)]
   for inner in range(8):events.append(('mma_main',p,k*128+inner*16))
   if p==2 or p>=4:
    for inner in range(8):events.append(('mma_alow',p,10 if p==2 else p+3,k*128+inner*16))
   if p==3 or p>=4:
    for inner in range(8):events.append(('mma_blow',p,10 if p==3 else p+3,k*128+inner*16))
   events.append(('commit',seq))
   if k>0:events += [('wait',1),('warp_sync',),('free',(seq-1)%4)]
  events += [('wait',0),('compiler_fence',step,'d/r'),('warp_sync',),('free',(4*step+3)%4)]
  if p>=2:events += [('restore_cross',p,'mulRN'),('correct_leaf',p,'addRN')]
  targets={0:(0,1,2,3),1:(0,),5:(1,2,3),6:(2,3),4:(1,3),2:(1,),3:(2,)}[p]
  for target in targets:events.append(('recombine',p,target,'set' if p==0 else 'subRN' if p==3 else 'addRN'))
 return events
old=schedule([old_product(i) for i in range(7)]);new=schedule(order);assert old==new
assert sum(e[0].startswith('mma_') for e in old)==480
assert sum(e[0]=='free' for e in old)==28
assert sum(e[0]=='wait' for e in old)==28
assert sum(e[0]=='hardware_arrive' for e in old)==7
# Every slot release is known complete in abstract committed-group order.
seq=-1;completed=-1;released=[]
for e in old:
 if e[0]=='commit':seq=e[1]
 if e[0]=='wait':completed=seq-e[1]
 if e[0]=='free':
  target=len(released);assert target<=completed and target%4==e[1];released.append(target)
assert released==list(range(28))
coords={(n+r)*512+m+c for n in range(0,512,32) for m in range(0,512,64) for r in range(32) for c in range(64)};assert coords==set(range(262144))
p=(h/'p51/corrected.cuh').read_text();needle=' #pragma unroll 1\n for(int step=0;step<7;++step)';assert p.count(needle)==1
assert 'Stages=4' in (h/'p51/pack.cuh').read_text()
for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:assert (h/n).read_bytes()==(h/'p51'/n).read_bytes()
out=dict(candidate='P56',proposal_only=True,gpu_executed=False,cpp_generated=False,retained='P51',event_count=len(old),event_identity=True,source_mmas_per_cta=480,ready_batches_per_cta=28,slot_releases_per_cta=len(released),hardware_arrive_per_cta=7,output_coordinate_coverage=len(coords),workspace_bytes=11804032,run_kernels=2,changed_directives=1,changed_geometry=False,unchanged_numerical_dependency_graph=True,actual_compiler_unroll_verified=False,scope='CPU abstract schedule/coordinate equivalence and byte-exact retained source only; not actual WGMMA, rounding, race, capacity, matrix or performance evidence')
(h/'p56_proposal_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
files=[h/'STATIC_LEAF_SCHEDULE_DESIGN.md',Path(__file__),h/'p56_proposal_cpu.json',h/'LITERATURE_UPDATE_P55.md',h/'p55_hardware_manifest.json',h/'p51_manifest.json']
(h/'p56_proposal_manifest.json').write_text(json.dumps({str(p.resolve().relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

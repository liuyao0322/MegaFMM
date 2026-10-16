"""P57 before code generation: independent TMA plane/byte/phase enumeration."""
from pathlib import Path
import json,hashlib
h=Path(__file__).resolve().parent;repo=h.parents[2]
def product(step):return step if step<2 else 5 if step==2 else 6 if step==3 else 4 if step==4 else 2 if step==5 else 3
def dynamic():
 events=[]
 for step in range(7):
  p=product(step);low=10 if p<4 else p+3
  for k in range(4):
   seq=step*4+k;slot=seq%4
   if seq>=4:events.append(('waitfree',slot,(seq-4)//4%2))
   tx=[('B',p,0,16384,k*128),('A',p,32768,8192,k*128)]
   if p==2 or p>=4:tx.append(('A',low,40960,8192,k*128))
   if p==3 or p>=4:tx.append(('B',low,16384,16384,k*128))
   events.append(('ready',slot,seq//4%2,sum(x[3] for x in tx),tuple(tx)))
 return events
def static():
 # Independent literal plane and expected byte table, same order as source dispatch.
 events=[]
 for step,(p,al,bl,size) in enumerate([(0,None,None,24576),(1,None,None,24576),(5,8,8,49152),(6,9,9,49152),(4,7,7,49152),(2,10,None,32768),(3,None,10,40960)]):
  for k in range(4):
   seq=step*4+k;slot=seq%4
   if seq>=4:events.append(('waitfree',slot,(seq//4-1)%2))
   tx=[('B',p,0,16384,k*128),('A',p,32768,8192,k*128)]
   if al is not None:tx.append(('A',al,40960,8192,k*128))
   if bl is not None:tx.append(('B',bl,16384,16384,k*128))
   assert sum(x[3] for x in tx)==size
   events.append(('ready',slot,step%2,size,tuple(tx)))
 return events
old,new=dynamic(),static();assert old==new
ready=[x for x in old if x[0]=='ready'];assert len(ready)==28 and sum(len(x[4]) for x in ready)==88 and sum(x[3] for x in ready)==1081344
for e in ready:
 ranges=[(t[2],t[2]+t[3]) for t in e[4]]
 assert all(0<=a<b<=49152 for a,b in ranges)
 assert all(b<=c or d<=a for i,(a,b) in enumerate(ranges) for c,d in ranges[i+1:])
for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:assert (h/n).read_bytes()==(h/'p56'/n).read_bytes()
out=dict(candidate='P57',proposal_only=True,gpu_executed=False,cpp_generated=False,retained='P56',event_identity=True,producer_events=len(old),ready_batches_per_cta=28,tma_commands_per_cta=88,tma_bytes_per_cta=1081344,tma_bytes_whole_grid=138412032,mma_source_per_cta=480,workspace_bytes=11804032,run_kernels=2,consumer_byte_unchanged=True,scope='Independent CPU literal TMA plane/transaction/byte/phase schedule, storage nonoverlap; no actual compiler, WGMMA, race, matrix or performance acceptance')
(h/'p57_proposal_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
f=[h/'STATIC_PRODUCER_DESIGN.md',Path(__file__),h/'p57_proposal_cpu.json',h/'p56_hardware_manifest.json']
(h/'p57_proposal_manifest.json').write_text(json.dumps({str(p.resolve().relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in f},indent=2)+'\n');print(json.dumps(out,indent=2))

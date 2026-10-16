"""P61 complete input and exact-max reduction-set CPU proof before generation."""
from pathlib import Path
from array import array
import hashlib,json
h=Path(__file__).resolve().parent;repo=h.parents[2]
seen_a=array('B',[0])*(1024*1024);seen_b=array('B',[0])*(1024*1024)
sectors={8:0,16:0}
for columns in [8,16]:
 for block in range(1024//columns):
  for warp in range(8):
   for j in range(1024//(256//columns)):
    addresses=set()
    for lane in range(32):
     tid=warp*32+lane;r=tid//columns+(256//columns)*j;k=block*columns+tid%columns
     addresses.add((r*1024+k)*2//32)
    sectors[columns]+=len(addresses)
assert sectors=={8:131072,16:65536}
for block in range(64):
 for tid in range(256):
  for j in range(64):
   r=tid//16+16*j;k=block*16+tid%16;seen_a[r*1024+k]+=1
 for warp in range(8):
  for kr in [warp,warp+8]:
   for lane in range(32):
    for j in range(32):seen_b[(block*16+kr)*1024+lane+32*j]+=1
 # The xor16 pair then eight-warp combine covers all rows for every output column.
 for column in range(16):
  rows={((warp*32+lane)//16+16*j) for warp in range(8) for lane in [column,column+16] for j in range(64)}
  assert rows==set(range(1024))
 assert set(range(block*16,block*16+16))<=set(range(1024))
assert min(seen_a)==max(seen_a)==min(seen_b)==max(seen_b)==1
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:assert (h/name).read_bytes()==(h/'p56'/name).read_bytes()
out=dict(candidate='P61',proposal_only=True,cpp_generated=False,gpu_executed=False,retained='P56',input_positions_each_side=1048576,coverage_min=1,coverage_max=1,complete_k_exponents=1024,reduction_ctas=64,columns_per_reduction_cta=16,launch_ctas=384,threads=256,scratch_reduction_bytes=(128+16)*4,shared_bytes=65536,grid_barriers=1,workspace_bytes=11804032,modeled_a_sector_address_visits={str(k):v for k,v in sectors.items()},actual_capacity_checked=False,hardware_accepted=False,scope=__doc__+' Sector model is address grouping only, not dynamic cache/DRAM transactions or performance.')
(h/'p61_proposal_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
files=[Path(__file__),h/'INSIDE_K16_DESIGN.md',h/'p61_proposal_cpu.json',h/'p60_hardware_manifest.json',h/'p56_hardware_manifest.json']
(h/'p61_proposal_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

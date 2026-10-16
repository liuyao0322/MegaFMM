"""P59 linear grid, storage and phase constraints before source generation."""
from pathlib import Path
import collections,hashlib,json
h=Path(__file__).resolve().parent;repo=h.parents[2]
p58=json.loads((h/'p58_proposal_cpu.json').read_text())
assert p58['logical_tasks']==384 and p58['covered_coordinates']=={'A':262144,'B':262144}
coords=collections.Counter()
for block in range(128):
 m0=(block//16)*64;n0=(block%16)*32
 for row in range(n0,n0+32):
  for col in range(m0,m0+64):coords[row,col]+=1
assert len(coords)==262144 and set(coords.values())=={1}
assert 65536<=196736 and 196736+1024+128==197888
for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:assert (h/n).read_bytes()==(h/'p56'/n).read_bytes()
out=dict(candidate='P59',proposal_only=True,cpp_generated=False,gpu_executed=False,retained='P56',control='P58',cooperative_grid=128,threads=256,logical_pack_tasks=384,grid_barriers=2,unconditional_all_threads=True,shared_bytes=196736,probe_shared_with_guards=197888,workspace_bytes=11804032,run_kernels_balanced=1,descriptors=2,coordinates=len(coords),actual_capacity_checked=False,hardware_accepted=False,phase_order=['exactinside','grid.sync','pack_all_global_planes_and_writer_proxy_fences','grid.sync','initialize_ready_free','block.sync','P56producer_and_consume'],scope='CPU coordinate/storage/phase schedule only; no cross-proxy hardware/WGMMA/precision/performance certificate')
(h/'p59_proposal_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
files=[Path(__file__),h/'GLOBAL_PHASE_FUSION_DESIGN.md',h/'p59_proposal_cpu.json',h/'p58_hardware_manifest.json',h/'p56_hardware_manifest.json']
(h/'p59_proposal_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

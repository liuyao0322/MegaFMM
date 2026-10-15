"""N3 single-factor storage control, using frozen N2 sources."""
from pathlib import Path
r=Path(__file__).resolve().parent;out=r/'n3';out.mkdir(exist_ok=True)
s=(r/'n2/prototype.cuh').read_text()
s=s.replace('uint64_t raw_ready[Slots],leaf_ready[Slots],leaf_free[Slots];','uint64_t raw_ready[Slots],leaf_ready[Slots],leaf_free[Slots];\n int cached_inside[1024];')
s=s.replace('sizeof(Shared)==184448','sizeof(Shared)==188544')
s=s.replace('proxy_fence();__syncthreads();\n int m0=', 'for(int k=threadIdx.x;k<1024;k+=256)s.cached_inside[k]=inside[k];\n proxy_fence();__syncthreads();\n int m0=')
s=s.replace('produce<T>(s,&map_a,&map_b,inside,m0,n0)','produce<T>(s,&map_a,&map_b,s.cached_inside,m0,n0)')
(out/'prototype.cuh').write_text(s)
for name in ['compare.cu','probe.cu','build.sh','build_probe.sh']:
 s=(r/'n2'/name).read_text()
 if name=='probe.cu':s=s.replace('104003+trial*367+pattern*71','105019+trial*373+pattern*73')
 if name.endswith('.sh'):s=s.replace('fused_native/n2','fused_native/n3')
 (out/name).write_text(s)

"""Instrument frozen P33 warp-leader phase timestamps, independent diagnostic only."""
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p33';target=h/'trace_p33';target.mkdir(exist_ok=False)
s=(base/'pack.cuh').read_text()
helper='''
// Diagnostic only. PTX globaltimer is target-specific; on this H100 record its
// nanosecond values for warp leaders, with no extra CTA/grid barriers.
static_assert(288*8*6*sizeof(uint64_t)<=PartialBytes,"trace fits unused partial workspace");
__device__ __forceinline__ void phase_stamp(float* partial,int phase){
 if((threadIdx.x&31)==0){
  uint64_t now;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
  reinterpret_cast<uint64_t*>(partial)[(blockIdx.x*8+threadIdx.x/32)*6+phase]=now;
 }
}
'''
pos=s.index('// All288CTAs participate');s=s[:pos]+helper+'\n'+s[pos:]
a=s.index('void preprocess_cooperative_columns(');b=s.index('\ntemplate<class T> void init_cooperative_preprocess',a);body=s[a:b]
old=' auto grid=cooperative_groups::this_grid();\n if(blockIdx.x<128){'
new=''' auto grid=cooperative_groups::this_grid();
 phase_stamp(partial,0);
 if(blockIdx.x>=128&&(threadIdx.x&31)==0){
  auto p=reinterpret_cast<uint64_t*>(partial)+(blockIdx.x*8+threadIdx.x/32)*6;
  p[1]=p[2]=p[3]=0;
 }
 if(blockIdx.x<128){'''
assert body.count(old)==1;body=body.replace(old,new)
old='  if(lane<8)ma[warp*8+lane]=am;\n  float bm=0;'
new='  if(lane<8)ma[warp*8+lane]=am;\n  phase_stamp(partial,1);\n  float bm=0;'
assert body.count(old)==1;body=body.replace(old,new)
old='  if(lane==0)mb[warp]=bm;\n  __syncthreads();'
new='  if(lane==0)mb[warp]=bm;\n  phase_stamp(partial,2);\n  __syncthreads();'
assert body.count(old)==1;body=body.replace(old,new)
old='''  }
 }
 grid.sync();
'''
new='''  }
  phase_stamp(partial,3);
 }
 grid.sync();
 phase_stamp(partial,4);
'''
assert body.count(old)==1;body=body.replace(old,new)
pos=body.rfind('\n}');body=body[:pos]+'\n phase_stamp(partial,5);\n'+body[pos:]
s=s[:a]+body+s[b:];(target/'pack.cuh').write_text(s)
for name in ['corrected.cuh','pow2.cuh']:(target/name).write_text((base/name).read_text())
print('trace source generated from frozen P33; math/fences/barriers unchanged, timestamps only')

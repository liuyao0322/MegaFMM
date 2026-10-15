"""P41: full-warp unsigned-bit maximum only in proved nonnegative domains."""
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p39'
s=(base/'pack.cuh').read_text()
anchor='// Keep only two FP32 transformed planes live. All phases share one scratch area.'
helper='''// Local absolute maxima start at +0 and contain no NaN. Unsigned binary32
// bit order equals value order here, including positive subnormals and +inf.
__device__ __forceinline__ float nonnegative_warp_max(float x){
 return __uint_as_float(__reduce_max_sync(0xffffffff,__float_as_uint(x)));
}

'''
assert s.count(anchor)==1;s=s.replace(anchor,helper+anchor)
old='for(int step=16;step;step/=2)hi=fmaxf(hi,__shfl_xor_sync(0xffffffff,hi,step));'
assert s.count(old)==1;s=s.replace(old,'hi=nonnegative_warp_max(hi);')
old=''' for(int step=16;step;step/=2){
  hi0=fmaxf(hi0,__shfl_xor_sync(0xffffffff,hi0,step));
  hi1=fmaxf(hi1,__shfl_xor_sync(0xffffffff,hi1,step));
 }'''
new=''' hi0=nonnegative_warp_max(hi0);
 hi1=nonnegative_warp_max(hi1);'''
assert s.count(old)==1;s=s.replace(old,new)
# Restrict the inside change to the actually dispatched cooperative reducer.
begin=s.index('void preprocess_cooperative_columns(')
old='for(int step=16;step;step/=2)bm=fmaxf(bm,__shfl_xor_sync(0xffffffff,bm,step));'
assert s[begin:].count(old)==1
s=s[:begin]+s[begin:].replace(old,'bm=nonnegative_warp_max(bm);')
(h/'pack.cuh').write_text(s)
for name in ['corrected.cuh','compare.cu','pow2.cuh']:(h/name).write_text((base/name).read_text())
print('generated P41 on frozen P39, no P40 lifetime changes; exact full-warp maxima')

"""P34 isolate eligibility to single-dot groups, retain all paired baseline dots."""
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p33'
s=(base/'corrected.cuh').read_text()
old='bool common=__shfl_sync(0xffffffff,int(meta_common),owner)!=0;'
assert s.count(old)==1
s=s.replace(old,'bool common=false;\n   if(step<9)common=__shfl_sync(0xffffffff,int(meta_common),owner)!=0;')
a=s.index('   if(common){');b=s.index('   }else{\n   #pragma unroll 1\n   for(int k=0;k<4;++k){',a)
fast=s[a:b]
c=fast.index('    if(step>=9){');d=fast.index('     warpgroup_commit_batch();',c)
fast=fast[:c]+fast[d:]
s=s[:a]+fast+s[b:]
old='bool use=same&&bounded;eligible+=use;paired+=use&&g>=2;exceptional+=!bounded;'
new='bool use=same&&bounded&&g<2;eligible+=use;paired+=use&&g>=2;exceptional+=!bounded;'
assert s.count(old)==1
s=s.replace(old,new)
(h/'corrected.cuh').write_text(s)
for name in ['pack.cuh','compare.cu','pow2.cuh']:
    (h/name).write_text((base/name).read_text())
print('generated P34 single-dot-only common-scale restoration from frozen P33')

"""P33 isolate integer wait/slot retirement, retaining P32 scale predicates."""
from pathlib import Path

h=Path(__file__).resolve().parent;base=h/'p32'
s=(base/'corrected.cuh').read_text()
a=s.index('   bool common=__shfl_sync')
b=s.index('   }else{\n   #pragma unroll 1\n   for(int k=0;k<4;++k){',a)
fast=s[a:b]
old='''     warpgroup_commit_batch();
     if(k>0){
      warpgroup_wait<1>();
      asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
      if((tid&31)==0)Barrier::arrive(&s.free[(seq-1)%Stages]);
     }
'''
new='''     warpgroup_commit_batch();warpgroup_wait<0>();
     asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
     if((tid&31)==0)Barrier::arrive(&s.free[slot]);
'''
assert fast.count(old)==1
fast=fast.replace(old,new)
old='''    }
    warpgroup_wait<0>();
    #pragma unroll'''
assert fast.count(old)==1
fast=fast.replace(old,'''    }
    #pragma unroll''')
old='''    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
    if((tid&31)==0)Barrier::arrive(&s.free[(step*4+3)%Stages]);
'''
assert fast.count(old)==1
fast=fast.replace(old,'')
(h/'corrected.cuh').write_text(s[:a]+fast+s[b:])
for name in ['pack.cuh','compare.cu','pow2.cuh']:
    (h/name).write_text((base/name).read_text())
print('generated common-scale wait0 retirement control from frozen P32')

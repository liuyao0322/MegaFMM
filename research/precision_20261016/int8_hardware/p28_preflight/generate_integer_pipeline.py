"""P28: two static integer fragments on a measured P27-style consumer.

Only overlap correction K128 computation and restoration. Scale metadata, FP32
addition order, native path, producer, shared slots and all eight dots remain.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True)
args = ap.parse_args()
h = Path(__file__).resolve().parent
base = h / args.base
rows = json.loads((base / f'{args.base}_balanced.json').read_text())['checks']
assert len(rows) == 78 and all(x['pass'] == '1' for x in rows)
s = (base / 'corrected.cuh').read_text()
a = s.index('  }else{\n   #pragma unroll 1\n   for(int k=0;k<4;++k){')
b = s.index('  // Product/term', a)
old = s[a:b]
mma_a = old.index('    #pragma unroll\n    for(int inner=0;inner<4;++inner)')
mma_b = old.index('    warpgroup_commit_batch();warpgroup_wait<0>();')
mma = old[mma_a:mma_b]
restore_a = old.index('    #pragma unroll\n    for(int j=0;j<16;++j)warpgroup_fence_operand(integer[j]);', mma_b)
restore_b = old.index('    asm volatile("bar.warp.sync', restore_a)
restore = old[restore_a:restore_b]
replacement = '''  }else{
   uint32_t integer0[16]={},integer1[16]={};
   #pragma unroll
   for(int j=0;j<16;++j){warpgroup_fence_operand(integer0[j]);warpgroup_fence_operand(integer1[j]);}
   warpgroup_arrive();
   auto issue=[&](auto index,uint32_t (&integer)[16]){
    constexpr int k=decltype(index)::value;
    int seq=step*4+k,slot=seq%Stages;
    Barrier::wait(&s.ready[slot],seq/Stages%2);asm volatile("" ::: "memory");
    uint64_t offset=slot*24576/16;
''' + mma + '''    warpgroup_commit_batch();
   };
   auto retire=[&](auto index,uint32_t (&integer)[16]){
    constexpr int k=decltype(index)::value;
    int slot=(step*4+k)%Stages;
''' + restore + '''    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
    if((tid&31)==0)Barrier::arrive(&s.free[slot]);
   };
   // Only the predecessor is read/released after wait1. Its fragment is
   // overwritten by the next ScaleOutZero MMA, with no intervening CPU write.
   issue(std::integral_constant<int,0>{},integer0);
   issue(std::integral_constant<int,1>{},integer1);
   warpgroup_wait<1>();retire(std::integral_constant<int,0>{},integer0);
   issue(std::integral_constant<int,2>{},integer0);
   warpgroup_wait<1>();retire(std::integral_constant<int,1>{},integer1);
   issue(std::integral_constant<int,3>{},integer1);
   warpgroup_wait<1>();retire(std::integral_constant<int,2>{},integer0);
   warpgroup_wait<0>();retire(std::integral_constant<int,3>{},integer1);
  }
'''
(h / 'corrected.cuh').write_text(s[:a] + replacement + s[b:])
for name in ['pack.cuh', 'compare.cu']:
    (h / name).write_text((base / name).read_text())
print('generated static integer pipeline from', args.base)

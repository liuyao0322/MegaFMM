"""Clone consumer scheduling and add sparse per-CTA timestamps; no timing verdict."""
from pathlib import Path

r=Path(__file__).resolve().parent
s=(r/'corrected.cuh').read_text()
s=s.replace('namespace int8_research {','''namespace int8_research {
__device__ __forceinline__ void stamp(uint64_t* trace,int seq,int point){
 if(trace && threadIdx.x==128){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  int cta=blockIdx.y*gridDim.x+blockIdx.x;trace[(cta*60+seq)*6+point]=t;}
}
''',1)
s=s.replace('const int* eb,float* D){','const int* eb,float* D,uint64_t* trace){')
s=s.replace('int seq=step*4+k,slot=seq%Stages;\n   Barrier::wait','int seq=step*4+k,slot=seq%Stages;\n   stamp(trace,seq,0);\n   Barrier::wait',1)
s=s.replace('uint64_t offset=slot*24576/16;','stamp(trace,seq,1);\n   uint64_t offset=slot*24576/16;',1)
s=s.replace('warpgroup_commit_batch();warpgroup_wait<0>();','stamp(trace,seq,2);warpgroup_commit_batch();warpgroup_wait<0>();stamp(trace,seq,3);')
s=s.replace('asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");','stamp(trace,seq,4);\n   asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");',1)
s=s.replace('if((tid&31)==0)Barrier::arrive(&s.free[slot]);','if((tid&31)==0)Barrier::arrive(&s.free[slot]);\n   stamp(trace,seq,5);',1)
s=s.replace('consume<T>(s,tid,m0,n0,ea,eb,D);','consume<T>(s,tid,m0,n0,ea,eb,D,trace);')
s=s.replace('w.ea,w.eb,D);','w.ea,w.eb,D,nullptr);')
assert s.count('stamp(trace,seq,0)')==1 and s.count('stamp(trace,seq,3)')==2
(r/'traced_corrected.cuh').write_text('// Generated diagnostic clone. Timing perturbation must be reported.\n'+s)

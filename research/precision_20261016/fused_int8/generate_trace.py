"""Instrument frozen F9 producer/consumer intervals; do not alter arithmetic or barriers."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'f9/prototype.cuh').read_text()
s=s.replace('namespace fused_int8 {','''namespace fused_int8 {
constexpr int TraceRole=136,TraceCTA=272;
__device__ __forceinline__ void stamp(uint64_t* trace,int role,int index){
 if(trace && threadIdx.x==(role?128:0)){
  uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  trace[(blockIdx.y*gridDim.x+blockIdx.x)*TraceCTA+role*TraceRole+index]=t;
 }
}
__device__ __forceinline__ void leaf_stamp(uint64_t* trace,int role,int seq,int point){stamp(trace,role,seq*6+point);}
__device__ __forceinline__ void raw_stamp(uint64_t* trace,int role,int k,int point){stamp(trace,role,120+k*4+point);}
''',1)
s=s.replace('void consume_leaf(Shared& s,int seq,int tid,float c[4][16]){','void consume_leaf(Shared& s,int seq,int tid,float c[4][16],uint64_t* trace){')
s=s.replace('int slot=seq%2;Barrier::wait(&s.ready[slot],seq/2%2);asm volatile("" ::: "memory");float d[16]={};','int slot=seq%2;leaf_stamp(trace,1,seq,0);Barrier::wait(&s.ready[slot],seq/2%2);asm volatile("" ::: "memory");leaf_stamp(trace,1,seq,1);float d[16]={};')
s=s.replace('native_product<T>(s.slot[slot].bytes+16384,s.slot[slot].bytes,d);combine<Prod>(c,d);','native_product<T>(s.slot[slot].bytes+16384,s.slot[slot].bytes,d);combine<Prod>(c,d);leaf_stamp(trace,1,seq,2);')
s=s.replace('correction<T,Prod,0>(s.slot[slot],d);combine<Prod>(c,d);','correction<T,Prod,0>(s.slot[slot],d);combine<Prod>(c,d);leaf_stamp(trace,1,seq,3);')
s=s.replace('if constexpr(Prod!=2&&Prod!=3){correction<T,Prod,1>(s.slot[slot],d);combine<Prod>(c,d);}','if constexpr(Prod!=2&&Prod!=3){correction<T,Prod,1>(s.slot[slot],d);combine<Prod>(c,d);}leaf_stamp(trace,1,seq,4);')
s=s.replace('if((tid&31)==0)Barrier::arrive(&s.free[slot]);','if((tid&31)==0)Barrier::arrive(&s.free[slot]);leaf_stamp(trace,1,seq,5);')
s=s.replace('CUtensorMap B,float* D){','CUtensorMap B,float* D,uint64_t* trace){',1)
s=s.replace('   if(tid==0){\n    if(k){','   raw_stamp(trace,0,k,0);\n   if(tid==0){\n    if(k){',1)
s=s.replace('    TxBarrier::arrive_and_expect_tx(&s.raw_ready,98304);','    raw_stamp(trace,0,k,1);\n    TxBarrier::arrive_and_expect_tx(&s.raw_ready,98304);',1)
s=s.replace('asm volatile("" ::: "memory");choose_inside<T>(s,tid);','asm volatile("" ::: "memory");raw_stamp(trace,0,k,2);choose_inside<T>(s,tid);raw_stamp(trace,0,k,3);',1)
s=s.replace('for(int leaf=0;leaf<5;++leaf){int seq=k*5+leaf,slot=seq%2;','for(int leaf=0;leaf<5;++leaf){int seq=k*5+leaf,slot=seq%2;leaf_stamp(trace,0,seq,0);',1)
s=s.replace('    switch(leaf){','    leaf_stamp(trace,0,seq,1);\n    switch(leaf){',1)
s=s.replace('    if(tid==0)Barrier::arrive(&s.ready[slot]);','    leaf_stamp(trace,0,seq,2);if(tid==0)Barrier::arrive(&s.ready[slot]);leaf_stamp(trace,0,seq,3);',1)
s=s.replace('   Barrier::wait(&s.raw_ready,k%2);asm volatile("" ::: "memory");float d[16]={};','   raw_stamp(trace,1,k,0);Barrier::wait(&s.raw_ready,k%2);asm volatile("" ::: "memory");raw_stamp(trace,1,k,1);float d[16]={};',1)
s=s.replace('   asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");if((tid&31)==0)Barrier::arrive(&s.raw_free);','   raw_stamp(trace,1,k,2);asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");if((tid&31)==0)Barrier::arrive(&s.raw_free);raw_stamp(trace,1,k,3);',1)
s=s.replace(',tid,c);',',tid,c,trace);').replace('>>>(a,b,D);','>>>(a,b,D,nullptr);')
assert s.count('leaf_stamp(trace,1,seq,5)')==1 and s.count('raw_stamp(trace,0,k,3)')==1
(r/'traced_f9.cuh').write_text('// Generated from frozen F9; diagnostic timing only.\n'+s)

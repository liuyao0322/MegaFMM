"""P32: precompute exact common-scale flags in metadata-owning lanes.

P31 fast accumulation, fallback, producer and host-only diagnostic are unchanged.
"""
from pathlib import Path

h=Path(__file__).resolve().parent
base=h/'p31'
s=(base/'corrected.cuh').read_text()
anchor=' float c[4][16]={};'
assert s.count(anchor)==1
s=s.replace(anchor,''' // Each four-lane group owns its four K128 exponents. Compute agreement
 // once before any MMA, instead of eight exponent broadcasts in every step.
 bool meta_common=meta_e0>=-126&&meta_e0<=103&&meta_e1>=-126&&meta_e1<=103;
 #pragma unroll
 for(int delta=1;delta<4;++delta){
  int other0=__shfl_xor_sync(0xffffffff,meta_e0,delta),other1=__shfl_xor_sync(0xffffffff,meta_e1,delta);
  meta_common=meta_common&&(meta_e0==other0)&&(meta_e1==other1);
 }
'''+anchor)
old='''   int owner=(step-7)*4;
   int e0=__shfl_sync(0xffffffff,meta_e0,owner),e1=__shfl_sync(0xffffffff,meta_e1,owner);
   bool common=e0>=-126&&e0<=103&&e1>=-126&&e1<=103;
   #pragma unroll
   for(int kk=1;kk<4;++kk){
    int ek0=__shfl_sync(0xffffffff,meta_e0,owner+kk),ek1=__shfl_sync(0xffffffff,meta_e1,owner+kk);
    common=common&&(ek0==e0)&&(ek1==e1);
   }
   if(common){'''
new='''   int owner=(step-7)*4;
   bool common=__shfl_sync(0xffffffff,int(meta_common),owner)!=0;
   if(common){
    int e0=__shfl_sync(0xffffffff,meta_e0,owner),e1=__shfl_sync(0xffffffff,meta_e1,owner);'''
assert s.count(old)==1
s=s.replace(old,new)
(h/'corrected.cuh').write_text(s)
for name in ['pack.cuh','pow2.cuh']:
    (h/name).write_text((base/name).read_text())
c=(base/'compare.cu').read_text()
anchor=' printf("SUMMARY dtype=%s balance=%d checks=%d failures=%d'
assert c.count(anchor)==1
position=c.index(anchor)
c=c[:position]+''' // Recomputed metadata for the timing input, after every accepted timing arm.
 int8_research::diagnose_uniform_scales(dt,-1,-1,int8_research::Work(workspace.p),stream.s);
'''+c[position:]
(h/'compare.cu').write_text(c)
print('generated precomputed lane-group common-scale flags from frozen P31')

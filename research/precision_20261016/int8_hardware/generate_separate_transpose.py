"""P40: independent transpose arrays for both paired planes, no shared reuse."""
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p39'
s=(base/'pack.cuh').read_text()
old='int8_t transpose[SideA?1:2][SideA?1:128*36]'
new='int8_t transpose[SideA?1:4][SideA?1:128*36]'
assert s.count(old)==1;s=s.replace(old,new)
start=s.index('template<class T,bool SideA,int QP> __device__ __forceinline__')
end=s.index('template<class T,bool SideA,int NP0,int QP0,int NP1,int QP1>',start)
helper=s[start:end]
helper=helper.replace('template<class T,bool SideA,int QP>','template<class T,bool SideA,int QP,int TransposeBase>')
helper=helper.replace('void pair_quant_store(', 'void pair_quant_stage(')
footer=helper.index(' if constexpr(!SideA){\n  __syncthreads();')
helper=helper[:footer]+' // No shared buffer is reused before this CTA exits.\n}\n\n'
helper=helper.replace('s.transpose[0][at]','s.transpose[TransposeBase][at]').replace('s.transpose[1][at]','s.transpose[TransposeBase+1][at]')
s=s[:start]+helper+s[end:]
old=''' pair_quant_store<T,SideA,QP0>(0,s.exponent,q,s,r0,c0);
 pair_quant_store<T,SideA,QP1>(1,s.exponent_second,q,s,r0,c0);'''
new=''' pair_quant_stage<T,SideA,QP0,0>(0,s.exponent,q,s,r0,c0);
 pair_quant_stage<T,SideA,QP1,2>(1,s.exponent_second,q,s,r0,c0);
 if constexpr(!SideA){
  // Both planes have distinct staging arrays. Publish once before cross-warp reads.
  __syncthreads();
  for(int i=tid;i<Rows*Cols;i+=256){int c=i/Rows,r=i%Rows,at=(c0+c)*512+r0+r;
   q[QP0*Plane+at]=s.transpose[0][r*36+c];
   q[(QP0+4)*Plane+at]=s.transpose[1][r*36+c];
   q[QP1*Plane+at]=s.transpose[2][r*36+c];
   q[(QP1+4)*Plane+at]=s.transpose[3][r*36+c];
  }
 }
 // No subsequent readers/writers of this shared scratch in paired groups.
'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('static_assert(sizeof(PairPackShared<true>)>=sizeof(PairPackShared<false>));',
'''static_assert(sizeof(PairPackShared<true>)==65612);
static_assert(sizeof(PairPackShared<false>)==51272);
static_assert(sizeof(PairPackShared<true>)>=sizeof(PairPackShared<false>));''')
(h/'pack.cuh').write_text(s)
for name in ['corrected.cuh','compare.cu','pow2.cuh']:(h/name).write_text((base/name).read_text())
print('generated P40 separate transpose arrays; paired A/B storage barriers2/3')

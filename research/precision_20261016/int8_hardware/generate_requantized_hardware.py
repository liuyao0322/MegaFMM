"""P35: append a globally ordered in-place INT8 requantization phase to P33."""
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p33'
s=(base/'pack.cuh').read_text()
helper='''
// One CTA owns one complete (plane,row-tile) of physical K-contiguous INT8.
// B is already transposed by packing, so both sides use contiguous K512 rows.
// Old exponents are snapshotted before any metadata overwrite by this CTA.
template<bool SideA> __device__ __forceinline__
void requantize_common(int8_t* q,int* exponent,int logical,int* scratch){
 constexpr int Rows=SideA?64:32,Tiles=SideA?8:16,Meta=SideA?32:64;
 int plane=logical/Tiles,tile=logical%Tiles,tid=threadIdx.x;
 if(tid==0){
  int common=-2147483647;
  #pragma unroll
  for(int k=0;k<4;++k){
   int e=exponent[plane*Meta+(SideA?tile*4+k:k*16+tile)];
   scratch[k]=e;common=max(common,e);
  }
  scratch[4]=common;
 }
 __syncthreads();
 int common=scratch[4];
 #pragma unroll 4
 for(int i=tid;i<Rows*512;i+=256){
  int k=(i%512)/128,at=plane*Plane+tile*Rows*512+i;
  int value=__float2int_rn(scale_pow2(float(q[at]),scratch[k]-common));
  q[at]=int8_t(max(-127,min(127,value)));
 }
 // Every output byte has one writer; exponent loads after the block barrier
 // access the snapshot only. No other CTA owns these four metadata entries.
 if(tid<4)exponent[plane*Meta+(SideA?tile*4+tid:tid*16+tile)]=common;
 asm volatile("fence.proxy.async.global;" ::: "memory");
}

'''
pos=s.index('// All288CTAs participate');s=s[:pos]+helper+s[pos:]
a=s.index('void preprocess_cooperative_columns(');b=s.index('\ntemplate<class T> void init_cooperative_preprocess',a)
body=s[a:b]
old=''' }
}
''';assert body.endswith(old+'\n') or body.endswith(old)
end=body.rfind('\n}')
body=body[:end]+'''
 // All original quantized bytes and exponents must be globally visible before
 // any CTA reads and rewrites a complete K512 tile. All288CTAs participate.
 grid.sync();
 if(blockIdx.x<72)requantize_common<true>(qa,ea,blockIdx.x,reinterpret_cast<int*>(scratch_buffer));
 else if(blockIdx.x<216)requantize_common<false>(qb,eb,blockIdx.x-72,reinterpret_cast<int*>(scratch_buffer));
'''+body[end:]
s=s[:a]+body+s[b:]
(h/'pack.cuh').write_text(s)
for name in ['corrected.cuh','compare.cu','pow2.cuh']:(h/name).write_text((base/name).read_text())
print('generated P35 commonK512 INT8 requantization; native planes and consumer frozen P33')

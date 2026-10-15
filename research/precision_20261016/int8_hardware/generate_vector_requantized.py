"""P36: same doubleRN algorithm, one scale and 32-bit loads per thread."""
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p35'
s=(base/'pack.cuh').read_text()
a=s.index(' int common=scratch[4];',s.index('void requantize_common('));b=s.index(' // Every output byte',a)
s=s[:a]+''' int common=scratch[4];
 // With word stride256, the K128 index is invariant for this thread.
 int k=(tid%128)/32,shift=min(common-scratch[k],8);
 // For every signed INT8, shifting by >=8 rounds to zero (including -128's
 // half tie). Thus cap8 is exact and the scale is always a normal FP32 power.
 float scale=__int_as_float((127-shift)<<23);
 auto* words=reinterpret_cast<uint32_t*>(q);
 #pragma unroll 4
 for(int i=tid;i<Rows*512/4;i+=256){
  int at=(plane*Plane+tile*Rows*512)/4+i;
  uint32_t input=words[at],output=0;
  #pragma unroll
  for(int byte=0;byte<4;++byte){
   int value=int(int8_t(input>>(8*byte)));
   int rounded=__float2int_rn(__fmul_rn(float(value),scale));
   output|=uint32_t(uint8_t(rounded))<<(8*byte);
  }
  words[at]=output;
 }
'''+s[b:]
(h/'pack.cuh').write_text(s)
for name in ['corrected.cuh','pow2.cuh']:(h/name).write_text((base/name).read_text())
s=(base/'compare.cu').read_text()
old=' // Recomputed metadata for the timing input, after every accepted timing arm.\n int8_research::diagnose_uniform_scales'
new=' // Diagnostic stage probes above overwrite metadata with their older standalone\n // packers. Recompute the actual complete preparation outside all timed arms.\n prep();\n int8_research::diagnose_uniform_scales'
assert s.count(old)==1
(h/'compare.cu').write_text(s.replace(old,new))
print('generated P36 vectorized exact INT8 requantization; full prep diagnostic restored')

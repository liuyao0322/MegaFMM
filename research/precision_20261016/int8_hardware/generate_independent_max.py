"""P38 exact column reduction with four independent max dependency chains."""
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p33'
s=(base/'pack.cuh').read_text()
old='''  float am=0;
  #pragma unroll 4
  for(int row=tid/8;row<1024;row+=32)
   am=fmaxf(am,fabsf(float(A[row*1024+k0+tid%8])));'''
new='''  float am0=0,am1=0,am2=0,am3=0;
  #pragma unroll 1
  for(int row=tid/8;row<1024;row+=128){
   am0=fmaxf(am0,fabsf(float(A[row*1024+k0+tid%8])));
   am1=fmaxf(am1,fabsf(float(A[(row+32)*1024+k0+tid%8])));
   am2=fmaxf(am2,fabsf(float(A[(row+64)*1024+k0+tid%8])));
   am3=fmaxf(am3,fabsf(float(A[(row+96)*1024+k0+tid%8])));
  }
  float am=fmaxf(fmaxf(am0,am1),fmaxf(am2,am3));'''
assert s.count(old)==1;s=s.replace(old,new)
old='''  float bm=0;
  #pragma unroll 4
  for(int col=lane;col<1024;col+=32)
   bm=fmaxf(bm,fabsf(float(B[(k0+warp)*1024+col])));'''
new='''  float bm0=0,bm1=0,bm2=0,bm3=0;
  #pragma unroll 1
  for(int col=lane;col<1024;col+=128){
   bm0=fmaxf(bm0,fabsf(float(B[(k0+warp)*1024+col])));
   bm1=fmaxf(bm1,fabsf(float(B[(k0+warp)*1024+col+32])));
   bm2=fmaxf(bm2,fabsf(float(B[(k0+warp)*1024+col+64])));
   bm3=fmaxf(bm3,fabsf(float(B[(k0+warp)*1024+col+96])));
  }
  float bm=fmaxf(fmaxf(bm0,bm1),fmaxf(bm2,bm3));'''
assert s.count(old)==1;s=s.replace(old,new)
(h/'pack.cuh').write_text(s)
for name in ['corrected.cuh','compare.cu','pow2.cuh']:(h/name).write_text((base/name).read_text())
print('generated P38 independent max chains, same exact column inputs and all packing math')

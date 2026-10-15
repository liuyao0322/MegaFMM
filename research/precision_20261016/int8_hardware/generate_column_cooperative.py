"""P30: full per-column maxima in one CTA, one cooperative phase barrier.

Frozen P29 packing and P27 consumer math are preserved. The128active reduction
CTAs each own8columns. Partial workspace stays reserved but is unused here.
"""
from pathlib import Path

h=Path(__file__).resolve().parent
base=h/'p29'
s=(base/'pack.cuh').read_text()
start=s.index(' if(blockIdx.x<256){',s.index('void preprocess_cooperative('))
end=s.index(' if(blockIdx.x<192){',start)
new=''' if(blockIdx.x<128){
  int tid=threadIdx.x,lane=tid%32,warp=tid/32,k0=blockIdx.x*8;
  float am=0;
  #pragma unroll 4
  for(int row=tid/8;row<1024;row+=32)
   am=fmaxf(am,fabsf(float(A[row*1024+k0+tid%8])));
  am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,8));
  am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,16));
  float* ma=reinterpret_cast<float*>(scratch_buffer);float* mb=ma+64;
  if(lane<8)ma[warp*8+lane]=am;
  float bm=0;
  #pragma unroll 4
  for(int col=lane;col<1024;col+=32)
   bm=fmaxf(bm,fabsf(float(B[(k0+warp)*1024+col])));
  for(int step=16;step;step/=2)bm=fmaxf(bm,__shfl_xor_sync(0xffffffff,bm,step));
  if(lane==0)mb[warp]=bm;
  __syncthreads();
  if(tid<8){
   float x=0,y=mb[tid];
   #pragma unroll
   for(int w=0;w<8;++w)x=fmaxf(x,ma[w*8+tid]);
   inside[k0+tid]=(x>0&&y>0)?__float2int_rn(.5f*(log2f(x)-log2f(y))):0;
  }
 }
 grid.sync();

'''
s=s[:start]+new+s[end:]
s=s.replace('// All288CTAs participate in both grid barriers. Scratch is reused only after\n// the grid has published partial maxima and then finalized all1024 exponents.',
'''// All288CTAs participate in the grid barrier. The first128CTAs each finish
// all8column exponents before the barrier permits inside reads/scratch reuse.''')
s=s.replace('preprocess_cooperative','preprocess_cooperative_columns')
(h/'pack.cuh').write_text(s)
for name in ['corrected.cuh','compare.cu','pow2.cuh']:
    (h/name).write_text((base/name).read_text())
print('generated single-CTA exact8column reduction from frozen P29')

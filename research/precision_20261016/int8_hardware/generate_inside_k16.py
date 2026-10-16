"""P61 isolates exact inside reduction geometry; native packing stays P56."""
from pathlib import Path
import hashlib,json
h=Path(__file__).resolve().parent;repo=h.parents[2];out=h/'p61';out.mkdir(exist_ok=True)
s=(h/'p56/pack.cuh').read_text()
start=s.index(' if(blockIdx.x<128){',s.index('void preprocess_cooperative_columns'))
end=s.index('\n grid.sync();',start)
s=s[:start]+''' if(blockIdx.x<64){
  int tid=threadIdx.x,lane=tid%32,warp=tid/32,k0=blockIdx.x*16;
  float am=0;
  #pragma unroll 4
  for(int row=tid/16;row<1024;row+=16)
   am=fmaxf(am,fabsf(float(A[row*1024+k0+tid%16])));
  am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,16));
  float* ma=reinterpret_cast<float*>(scratch_buffer);float* mb=ma+128;
  if(lane<16)ma[warp*16+lane]=am;
  #pragma unroll
  for(int kr=warp;kr<16;kr+=8){
   float bm=0;
   #pragma unroll 4
   for(int col=lane;col<1024;col+=32)
    bm=fmaxf(bm,fabsf(float(B[(k0+kr)*1024+col])));
   for(int step=16;step;step/=2)bm=fmaxf(bm,__shfl_xor_sync(0xffffffff,bm,step));
   if(lane==0)mb[kr]=bm;
  }
  __syncthreads();
  if(tid<16){
   float x=0,y=mb[tid];
   #pragma unroll
   for(int w=0;w<8;++w)x=fmaxf(x,ma[w*16+tid]);
   inside[k0+tid]=(x>0&&y>0)?__float2int_rn(.5f*(log2f(x)-log2f(y))):0;
  }
 }'''+s[end:]
s=s.replace('// All384CTAs participate in the grid barrier. The first128CTAs each finish\n// all8column exponents before the barrier permits inside reads/scratch reuse.','// All384CTAs participate; first64CTAs finish16complete column exponents each.\n// Native packing and the unconditional grid barrier remain P56.')
(out/'pack.cuh').write_text(s)
for n in ['corrected.cuh','compare.cu','pow2.cuh']:(out/n).write_bytes((h/'p56'/n).read_bytes())
old=(h/'p56/pack.cuh').read_text();key='\n grid.sync();'
assert s[s.index(key,s.index('void preprocess_cooperative_columns')):]==old[old.index(key,old.index('void preprocess_cooperative_columns')):]
key='template<class T,bool SideA,bool Balance,int Group>'
assert s[s.index(key):s.index('// All384CTAs')]==old[old.index(key):old.index('// All384CTAs')]
files=[Path(__file__),h/'p61_proposal_manifest.json']+[out/n for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']]
(h/'p61_source_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print('P61 isolated exact inside K16 reduction generated; working P56 unchanged')

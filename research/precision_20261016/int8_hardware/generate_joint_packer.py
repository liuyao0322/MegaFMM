"""Combine the two frozen shared-staged producer launches without new math."""
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
# Address-map preflight; actual device behavior still needs the full GPU gate.
for side_a, tiles in ((False, 64), (True, 32)):
    seen = [(((block-192)//32 if side_a else block//64), block % tiles)
            for block in range(288) if (block >= 192) == side_a]
    assert len(seen) == 3*tiles and len(set(seen)) == 3*tiles
    assert set(seen) == {(group, tile) for group in range(3) for tile in range(tiles)}
s = (base / 'pack.cuh').read_text()
needle = 'template<class T,bool Balance> void preprocess('
assert s.count(needle) == 1
joint = '''// B occupies0..191, A192..287.192 is divisible by both tile counts,
// so pack_pair's blockIdx modulo Tiles still selects the original tile.
template<class T,bool Balance> __global__ __launch_bounds__(256,1)
void pack_joint(const T* __restrict__ A,const T* __restrict__ B,
 T* __restrict__ ha,T* __restrict__ hb,int8_t* __restrict__ qa,int8_t* __restrict__ qb,
 int* __restrict__ ea,int* __restrict__ eb,const int* __restrict__ inside){
 extern __shared__ __align__(128) unsigned char scratch_buffer[];
 if(blockIdx.x<192){
  auto& s=*reinterpret_cast<PairPackShared<false>*>(scratch_buffer);
  int group=blockIdx.x/64;
  if(group==0)pack_pair<T,false,Balance,0>(B,hb,qb,eb,inside,s);
  else if(group==1)pack_pair<T,false,Balance,1>(B,hb,qb,eb,inside,s);
  else pack_pair<T,false,Balance,2>(B,hb,qb,eb,inside,s);
 }else{
  auto& s=*reinterpret_cast<PairPackShared<true>*>(scratch_buffer);
  int group=(blockIdx.x-192)/32;
  if(group==0)pack_pair<T,true,Balance,0>(A,ha,qa,ea,inside,s);
  else if(group==1)pack_pair<T,true,Balance,1>(A,ha,qa,ea,inside,s);
  else pack_pair<T,true,Balance,2>(A,ha,qa,ea,inside,s);
 }
}
static_assert(sizeof(PairPackShared<true>)>=sizeof(PairPackShared<false>));

'''
s = s.replace(needle, joint+needle)
old = ''' pack<T,true,Balance><<<3*32,256,sizeof(PairPackShared<true>),stream>>>(A,static_cast<T*>(w.ha),w.qa,w.ea,w.inside);
 pack<T,false,Balance><<<3*64,256,sizeof(PairPackShared<false>),stream>>>(B,static_cast<T*>(w.hb),w.qb,w.eb,w.inside);'''
assert s.count(old) == 1
s = s.replace(old, ''' pack_joint<T,Balance><<<288,256,sizeof(PairPackShared<true>),stream>>>(
  A,B,static_cast<T*>(w.ha),static_cast<T*>(w.hb),w.qa,w.qb,w.ea,w.eb,w.inside);''')
(h / 'pack.cuh').write_text(s)
s = (base / 'corrected.cuh').read_text()
needle = 'template<class T,bool Balance> void init_shared_packer(){'
assert s.count(needle) == 1
s = s.replace(needle, needle+'''
 check(cudaFuncSetAttribute(pack_joint<T,Balance>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(PairPackShared<true>)),"joint pack dynamic shared");''')
(h / 'corrected.cuh').write_text(s)
# Standalone A/B component probes remain diagnostic; actual preprocess uses joint.
(h / 'compare.cu').write_text((base / 'compare.cu').read_text())
print('generated joint packer from', args.base)

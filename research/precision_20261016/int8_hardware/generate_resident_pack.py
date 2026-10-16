"""Isolated P58 reduced resident grid, unchanged mathematical expressions."""
from pathlib import Path
import hashlib,json,re
h=Path(__file__).resolve().parent;repo=h.parents[2];out=h/'p58';out.mkdir(exist_ok=True)
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:
 s=(h/'p56'/name).read_text()
 if name=='pack.cuh':
  assert s.count('PairPackShared<SideA>& s){')==1
  s=s.replace('PairPackShared<SideA>& s){','PairPackShared<SideA>& s,int logical_block){')
  s=s.replace('int tile=blockIdx.x%Tiles','int tile=logical_block%Tiles')
  s,n=re.subn(r'(pack_pair<[^\n]+\([^;]+,(?:s|scratch))\);',r'\1,blockIdx.x);',s)
  assert n==15,n
  start=s.index(' if(blockIdx.x<256){',s.index('void preprocess_cooperative_columns'))
  end=s.index('\n}\n\ntemplate<class T> void init_cooperative_preprocess',start)
  s=s[:start]+''' auto& bs=*reinterpret_cast<PairPackShared<false>*>(scratch_buffer);
 pack_pair<T,false,Balance,0>(B,hb,qb,eb,inside,bs,blockIdx.x);
 pack_pair<T,false,Balance,0>(B,hb,qb,eb,inside,bs,blockIdx.x+128);
 auto& as=*reinterpret_cast<PairPackShared<true>*>(scratch_buffer);
 pack_pair<T,true,Balance,0>(A,ha,qa,ea,inside,as,blockIdx.x);'''+s[end:]
  s=s.replace('// All384CTAs participate in the grid barrier. The first128CTAs each finish','// All128CTAs participate in the grid barrier. All128CTAs each finish')
  s=s.replace('grid=384','grid=128').replace('active*sms<384','active*sms<128').replace('for384CTAs','for128CTAs').replace('dim3(384)','dim3(128)')
 (out/name).write_text(s)
files=[Path(__file__),h/'p58_proposal_manifest.json']+[out/n for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']]
(h/'p58_source_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print('P58 isolated sources generated; working P56 unchanged')

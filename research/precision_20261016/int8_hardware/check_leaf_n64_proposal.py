"""Isolated CPU P54 generation; leave working retained P51 untouched."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tempfile
r=Path(__file__).resolve().parent
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='megafmm_p54_proposal_') as td:
 t=Path(td);shutil.copytree(r/'p53',t/'p53');shutil.copy2(r/'generate_leaf_n64.py',t/'generate_leaf_n64.py')
 subprocess.run(['python3',str(t/'generate_leaf_n64.py')],check=True)
 same={n:digest(t/n)==digest(r/'p53'/n) for n in ['pack.cuh','compare.cu','pow2.cuh']}
 assert all(same.values())
 s=(t/'corrected.cuh').read_text();old=(r/'p53/corrected.cuh').read_text()
 assert s[s.index('__global__ void recombine'):s.index('void encode(')]==old[old.index('__global__ void recombine'):old.index('void encode(')]
 assert 'float c[4][16]' not in s
 assert s.count('<<<dim3(8,8,7),256,sizeof(Shared),stream>>>')==2
 assert 'sizeof(Shared)==131200' in s and 'float d[32]={},r[32]={};' in s
 assert s.count(','.join(f'd[{i}]' for i in range(32))+',scale);')==2
 for needle in ['s.operand[slot]+49152','n0=blockIdx.x*64','Shape<_64,_128>','Shape<_64,_16>','box[1]=64;box[2]=2;','ConsumerStages=2','for(int k=0;k<4;++k)','if(p==2||p>=4)','if(p==3||p>=4)','warpgroup_wait<1>();','warpgroup_wait<0>();']:
  assert needle in s,needle
 assert s.count('recombine<<<1024,256,0,stream>>>')==1
 out=dict(scope='isolated CPU prepared P54 proposal from rejected frozen P53; working P51 retained; no P54 application/build/device measurement',unchanged=same,recombine_kernel_byte_identical_to_p53=True,generated_source_sha256={n:digest(t/n) for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']},consumer_shared_arithmetic_bytes=131200,leaf_grid=448,total_workspace_bytes=19144064,run_kernel_count=3,nominal_tma_payload_bytes=92274688,nominal_tma_commands=5632,native_mma_instructions_whole_grid=30720,gpu_executed=False)
(r/'p54_proposal_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

"""Isolated CPU source generation checks; no candidate application or GPU work."""
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

r=Path(__file__).resolve().parent
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']
with tempfile.TemporaryDirectory(prefix='megafmm_p53_proposal_') as td:
 t=Path(td)
 shutil.copytree(r/'p51',t/'p51')
 for n in ['generate_leaf_parallel.py','leaf_parallel_consumer.inc']:
  shutil.copy2(r/n,t/n)
 subprocess.run(['python3',str(t/'generate_leaf_parallel.py')],check=True)
 same={n:digest(t/n)==digest(r/'p51'/n) for n in ['compare.cu','pow2.cuh']}
 assert all(same.values())
 old=(r/'p51/pack.cuh').read_text()
 line=next(x for x in old.splitlines(keepends=True) if x.startswith('constexpr size_t WorkBytes='))
 expected=old.replace(line,line.replace('WorkBytes','BaseWorkBytes')+'constexpr size_t LeafBytes=7*Plane*sizeof(float),WorkBytes=BaseWorkBytes+LeafBytes;\n')
 assert (t/'pack.cuh').read_text()==expected
 s=(t/'corrected.cuh').read_text()
 assert 'sizeof(Shared)==98432' in s and 'constexpr int ConsumerStages=2;' in s
 assert s.count('<<<dim3(16,8,7),256,sizeof(Shared),stream>>>')==2
 assert s.count('recombine<<<1024,256,0,stream>>>')==1
 assert 'float c[4][16]' not in s
 oldconsumer=(r/'p51/corrected.cuh').read_text()
 original_mma=oldconsumer[oldconsumer.index('template<class T> __device__ __forceinline__ void float_mma'):oldconsumer.index('template<class T> __device__ __forceinline__ void consume')]
 assert original_mma in s
 for needle in ['for(int k=0;k<4;++k)','float d[16]={},r[16]={};','warpgroup_wait<1>();','warpgroup_wait<0>();','if(p==2||p>=4)','if(p==3||p>=4)','p=blockIdx.z']:
  assert needle in s,needle
 out=dict(scope='isolated CPU proposal generation only; working P51 retained, no P53 application/build/device measurement',unchanged=same,pack_change='workspace constants only',generated_source_sha256={n:digest(t/n) for n in names},consumer_shared_arithmetic_bytes=98432,base_workspace_bytes=11804032,additional_leaf_bytes=7340032,total_workspace_bytes=19144064,leaf_grid=896,run_kernel_count=3,gpu_executed=False)
(r/'p53_proposal_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

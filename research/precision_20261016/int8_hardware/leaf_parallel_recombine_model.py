"""CPU FP32 recombination equivalence only; not matrix/hardware acceptance."""
import hashlib,json
from pathlib import Path
import numpy as np
h=Path(__file__).resolve().parent;repo=h.parents[2]
rng=np.random.default_rng(117089);n=262144
p=np.ldexp(rng.uniform(-1,1,(7,n)),rng.integers(-50,51,(7,n))).astype(np.float32)
# Fresh correlated leaf values exercise cancellation as well as independent scales.
p[1,n//4:n//2]=-p[0,n//4:n//2]
p[5,n//2:3*n//4]=-p[0,n//2:3*n//4]
p[:,::257]=np.float32(-0.0);p[:,::263]=np.float32(0.0)
c=np.repeat(p[0,None,:],4,axis=0)
c[0]=c[0]+p[1]
for q in [1,2,3]:c[q]=c[q]+p[5]
for q in [2,3]:c[q]=c[q]+p[6]
for q in [1,3]:c[q]=c[q]+p[4]
c[1]=c[1]+p[2];c[2]=c[2]-p[3]
t=p[0]+p[5];u=t+p[6]
z=np.stack([p[0]+p[1],(t+p[4])+p[2],u-p[3],u+p[4]])
errors=int(np.count_nonzero(c.view(np.uint32)!=z.view(np.uint32)))
negative=p[0]+(p[5]+(p[4]+p[2]))
negative_errors=int(np.count_nonzero(c[1].view(np.uint32)!=negative.view(np.uint32)))
assert errors==0 and np.isfinite(c).all() and np.isfinite(z).all() and negative_errors>0
out=dict(scope='CPU float32 corrected-leaf recombination only. New independent/scaled/canceling leaf values and signed zeros; no original-input matrix, WGMMA, performance or sanitizer certificate.',seed=117089,leaf_tuples=n,output_bit_comparisons=4*n,bit_errors=errors,nonfinite_outputs=0,changed_order_negative_control_differences=negative_errors,arithmetic='C11=RN(P0+P1),t=RN(P0+P5),u=RN(t+P6),C12=RN(RN(t+P4)+P2),C21=RN(u-P3),C22=RN(u+P4); preserves P51 scalar RN dependency graph')
(h/'leaf_parallel_recombine_model.json').write_text(json.dumps(out,indent=2)+'\n')
files=['leaf_parallel_recombine_model.py','leaf_parallel_recombine_model.json','p51_manifest.json']
(h/'p53_recombine_model_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

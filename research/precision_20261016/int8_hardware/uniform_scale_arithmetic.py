"""CPU-only FP32 arithmetic control; no matrix, WGMMA or timing acceptance."""
import json
from pathlib import Path
import numpy as np

h=Path(__file__).resolve().parent
rng=np.random.default_rng(29117)
rows=[]
for dots in [1,2]:
    bound=dots*128*127*127
    edges=np.array([[0,0,0,0],[bound]*4,[-bound]*4,
                    [bound,bound,-bound,-bound],[bound,-bound,1,-1],
                    [bound,1,-bound,-1],[1,2,3,4],[-1,-2,-3,-4]],dtype=np.int64)
    values=np.concatenate([edges,rng.integers(-bound,bound+1,size=(20000,4),dtype=np.int64)])
    total=values.sum(axis=1)
    assert max(abs(total))<2**24 and 4*bound<2**24
    for exponent in range(-126,104):
        scale=np.ldexp(np.float32(1),exponent,dtype=np.float32)
        sequential=np.zeros(len(values),dtype=np.float32)
        for k in range(4):
            restored=np.multiply(values[:,k].astype(np.float32),scale,dtype=np.float32)
            sequential=np.add(sequential,restored,dtype=np.float32)
        merged=np.multiply(total.astype(np.float32),scale,dtype=np.float32)
        errors=int(np.count_nonzero(sequential.view(np.uint32)!=merged.view(np.uint32)))
        assert errors==0 and np.isfinite(merged).all()
        rows.append(dict(dots_per_k128=dots,exponent=exponent,generated_integer_tuples=len(values),bitwise_errors=errors))
control_old=sum(float(np.ldexp(np.float32(1),e,dtype=np.float32)) for e in [-1,0,1,2])
control_wrong=float(np.float32(4)*np.float32(.5))
assert control_old!=control_wrong
out=dict(scope='CPU FP32 restoration arithmetic only; generated integer tuples repeated over230normal scales, not independent matrix inputs or hardware acceptance',
         seed=29117,generated_tuples_per_dot_count=len(values),scale_exponents_per_dot_count=230,
         scaled_arithmetic_checks=sum(x['generated_integer_tuples'] for x in rows),
         max_paired_k512_integer_bound=2*512*127*127,bitwise_errors=0,
         differing_scale_negative_control=dict(original=control_old,invalid_merged=control_wrong),rows=rows)
(h/'uniform_scale_arithmetic.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

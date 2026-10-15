"""Enumerate dense/sparse FP8 corrections; ideal FP64 leaves, not GPU timing."""
import itertools
import json
from pathlib import Path
import numpy as np
from screen import q,blocks,schemes,evaluate
from fp8_correction import fp8


def prune(v,axis):
    a=v if axis==1 else v.T
    z=a.copy().reshape(-1,4)
    idx=np.argsort(np.abs(z),axis=1)[:,:2]
    np.put_along_axis(z,idx,0,axis=1)
    a=z.reshape(a.shape)
    return a if axis==1 else a.T


def main():
    output=[]
    for dtype in ['fp16','bf16']:
        prepared=[]
        for seed in [201,202,203]:
            rng=np.random.default_rng(seed)
            a=q(rng.uniform(-1,1,(1024,1024)),dtype)
            b=q(rng.uniform(-1,1,(1024,1024)),dtype)
            aa,ab,ac,ad=blocks(a);ba,bb,bc,bd=blocks(b)
            ss=[ac+ad,ac+ad-aa,aa-ac,ab-(ac+ad-aa)]
            tt=[bb-ba,bd-(bb-ba),bd-bb,(bd-(bb-ba))-bc]
            sh=[q(v,dtype) for v in ss];th=[q(v,dtype) for v in tt]
            sl=[v-h for v,h in zip(ss,sh)];tl=[v-h for v,h in zip(tt,th)]
            pairs=[(sl[3],bd,1),(ad,tl[3],0),(sl[0],th[0],1),(sh[0],tl[0],0),(sl[1],th[1],1),(sh[1],tl[1],0),(sl[2],th[2],1),(sh[2],tl[2],0)]
            signs=[(0,1,0,0),(0,0,-1,0),(0,1,0,1),(0,1,0,1),(0,1,1,1),(0,1,1,1),(0,0,1,1),(0,0,1,1)]
            ref=a@b;out=evaluate(a,b,schemes()['winograd'],dtype);deltas=[]
            for (left,right,residual_side),sg in zip(pairs,signs):
                dense=fp8(left)@fp8(right)
                sparse=(fp8(prune(left,1))@fp8(right)) if residual_side else (fp8(left)@fp8(prune(right,0)))
                def expand(x):return np.block([[sg[0]*x,sg[1]*x],[sg[2]*x,sg[3]*x]])
                out+=expand(sparse)
                deltas.append(expand(dense-sparse).ravel())
            err=(out-ref).ravel();deltas=np.asarray(deltas)
            prepared.append((float(err@err),deltas@err,deltas@deltas.T,float(np.sum(ref*ref))))
        for count in range(9):
            best=None
            for subset in itertools.combinations(range(8),count):
                v=np.zeros(8);v[list(subset)]=1
                errs=[np.sqrt(max(0,e+2*v@d+v@g@v)/den) for e,d,g,den in prepared]
                row=dict(dtype=dtype,dense_terms=count,sparse_terms=8-count,dense_indices=list(subset),worst_ef=float(max(errs)),seed_errors=[float(x) for x in errs],equivalent_fp16_products=7+count/2+(8-count)/4)
                if best is None or row['worst_ef']<best['worst_ef']:best=row
            output.append(best);print(best,flush=True)
    Path('research/precision_20261015/sparse_mix.json').write_text(json.dumps(dict(scope='all 256 subsets of dense versus optimal magnitude-pruned 2:4 residual correction; globally scaled E4M3; FP64 leaves; equivalent products is arithmetic-only model, not latency',rows=output),indent=2)+'\n')


if __name__=='__main__':main()

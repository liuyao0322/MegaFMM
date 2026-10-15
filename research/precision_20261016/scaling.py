"""Power-of-two inside/outside scaling of the uncorrected rank-seven formula.

Based on Ballard et al., https://arxiv.org/html/1507.00687 sections 6.1-6.3.
Models exact scaling and FP64 leaves, so it favors the proposed intervention.
"""
import argparse
import json
from pathlib import Path
import torch
from ablate import inputs, transforms, leaves, recombine


def pow2(x):
    return torch.exp2(torch.round(torch.log2(torch.where(x>0,x,1.))))


def outside(a,b):
    da=pow2(a.abs().amax(1))
    db=pow2(b.abs().amax(0))
    return a/da[:,None],b/db[None,:],da[:,None]*db[None,:]


def inside(a,b):
    # D[k]=sqrt(max_i |A_ik| / max_j |B_kj|), A'=A/D, B'=D B.
    av=a.abs().amax(0)
    bv=b.abs().amax(1)
    d=pow2(torch.sqrt(torch.where((av>0)&(bv>0),av/bv,1.)))
    return a/d[None,:],b*d[:,None]


def sw(a,b,dtype):
    s,t=transforms(a,b)
    s,t=[[v.to(dtype).double() for v in vs] for vs in (s,t)]
    return recombine(leaves(a,b,s,t))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='research/precision_20261016/scaling.json')
    args=ap.parse_args()
    rows=[]
    for dt in [torch.float16,torch.bfloat16]:
        for pattern in ['uniform','range16','cancellation','block_scaled','bf16_range','inside_mismatch']:
            for seed in [701,702,703]:
                a,b=inputs(1024,seed,pattern,dt)
                ref=a@b
                den=torch.linalg.vector_norm(ref)
                def ef(x):return float((torch.linalg.vector_norm(x-ref)/den).item())
                aa,bb,scale=outside(a,b)
                ai,bi=inside(a,b)
                aoi,boi=inside(aa,bb)
                aio,bio,scaleio=outside(ai,bi)
                row=dict(dtype=str(dt).split('.')[-1],pattern=pattern,seed=seed,
                         unscaled=ef(sw(a,b,dt)),outside=ef(sw(aa,bb,dt)*scale),
                         inside=ef(sw(ai,bi,dt)),outside_inside=ef(sw(aoi,boi,dt)*scale),
                         inside_outside=ef(sw(aio,bio,dt)*scaleio))
                # These invariants guard the scaling implementation independently
                # of the fast formula. The input-reference contract is unchanged.
                assert ef((aa@bb)*scale)<2e-12
                assert ef(ai@bi)<2e-12
                rows.append(row)
                print(json.dumps(row),flush=True)
                Path(args.output).write_text(json.dumps(dict(scope=__doc__,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()

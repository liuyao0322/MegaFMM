"""Test FP8 first-order residual correction numerics, with FP64 leaf dots.

Not production acceptance or a performance measurement. Uses per-operand
power-of-two scaling, whose reduction/conversion cost is not modeled here.
"""
import json
from pathlib import Path
import numpy as np
from screen import q, blocks, schemes, evaluate


def fp8(x, mantissa=3):
    # E4M3FN (448 max) or E5M2 (57344 max), with one global power-of-two scale.
    limit=448 if mantissa==3 else 57344
    minsub=2.**(-9 if mantissa==3 else -16)
    peak=float(np.max(np.abs(x)))
    scale=2.**np.ceil(np.log2(peak/limit)) if peak else 1.
    v=x/scale
    m,e=np.frexp(v)
    step=np.maximum(np.ldexp(np.ones_like(v),e-(mantissa+1)),minsub)
    out=np.clip(np.rint(v/step)*step,-limit,limit)*scale
    return out


def main():
    rows=[]
    for dtype in ['fp16','bf16']:
        for seed in [1,2,3]:
            rng=np.random.default_rng(seed)
            a=q(rng.uniform(-1,1,(1024,1024)),dtype)
            b=q(rng.uniform(-1,1,(1024,1024)),dtype)
            aa,ab,ac,ad=blocks(a);ba,bb,bc,bd=blocks(b)
            ss=[ac+ad,ac+ad-aa,aa-ac,ab-(ac+ad-aa)]
            tt=[bb-ba,bd-(bb-ba),bd-bb,(bd-(bb-ba))-bc]
            sh=[q(v,dtype) for v in ss];th=[q(v,dtype) for v in tt]
            sl=[v-h for v,h in zip(ss,sh)];tl=[v-h for v,h in zip(tt,th)]
            ref=a@b
            raw=evaluate(a,b,schemes()['winograd'],dtype)
            pairs=[(sl[3],bd),(ad,tl[3]),(sl[0],th[0]),(sh[0],tl[0]),(sl[1],th[1]),(sh[1],tl[1]),(sl[2],th[2]),(sh[2],tl[2])]
            signs=[(0,1,0,0),(0,0,-1,0),(0,1,0,1),(0,1,0,1),(0,1,1,1),(0,1,1,1),(0,0,1,1),(0,0,1,1)]
            for mantissa in [3,2]:
                out=raw.copy();obs=[]
                for (left,right),sg in zip(pairs,signs):
                    corr=fp8(left,mantissa)@fp8(right,mantissa)
                    out += np.block([[sg[0]*corr,sg[1]*corr],[sg[2]*corr,sg[3]*corr]])
                for name,rs in [('S',sl),('T',tl)]:
                    for i,v in enumerate(rs):
                        n=v.reshape(-1,4)
                        # Optimal 2:4 retention of largest residuals in each row group.
                        keep=np.sort(np.abs(n),axis=1)[:,2:]
                        kept=float(np.sum(keep*keep));total=float(np.sum(v*v))
                        obs.append(dict(name=f'{name}{i+1}',zero_fraction=float(np.mean(v==0)),discarded_2of4_energy=1-kept/total if total else 0))
                row=dict(input=dtype,seed=seed,correction_format=f'E{4 if mantissa==3 else 5}M{mantissa}',ef=float(np.linalg.norm(out-ref)/np.linalg.norm(ref)),residual_stats=obs)
                rows.append(row);print({k:v for k,v in row.items() if k!='residual_stats'},flush=True)
    Path('research/precision_20261015/fp8_correction.json').write_text(json.dumps(dict(scope='diagnostic only; globally scaled FP8 correction operands; FP64 leaves/recombination; no GPU latency/acceptance claim',rows=rows),indent=2)+'\n')


if __name__=='__main__':
    main()

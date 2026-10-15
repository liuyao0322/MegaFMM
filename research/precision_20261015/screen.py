"""Transform-quantization diagnostic, NOT GPU acceptance or timing.

Source: https://arxiv.org/html/2506.19405 equations 98,101,102.
FP64 leaves isolate operand rounding; all references use original quantized A/B.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def q(x, dtype):
    x = np.asarray(x, dtype=np.float32)
    if dtype == 'fp16':
        return x.astype(np.float16).astype(np.float64)
    if dtype == 'fp32':
        return x.astype(np.float64)
    u = x.view(np.uint32)
    u = (u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)
    return u.view(np.float32).astype(np.float64)


def schemes():
    s = np.sqrt(3)
    return {
        'DPS98': (
            [[s/2,.5,.5,s/6],[0,0,1,-s/3],[0,1,0,s/3],[0,0,0,-2/s],[-s/2,-.5,.5,-s/2],[-s/2,-.5,.5,s/6],[-s/2,.5,.5,-s/6]],
            [[0,2/s,0,0],[-1,s/3,0,0],[0,s/3,0,-1],[.5,-s/6,s/2,-.5],[-.5,s/2,-s/2,-.5],[.5,s/6,s/2,.5],[.5,s/6,-s/2,-.5]],
            [[s/6,.5,.5,s/2],[-s/3,0,-1,0],[s/3,-1,0,0],[s/6,-.5,-.5,s/2],[s/2,-.5,.5,s/2],[-s/6,-.5,.5,s/2],[-2/s,0,0,0]]),
        'DPS101': (
            [[0,-1,1,0],[1,.5,-.5,-.25],[0,0,1,-.5],[0,1,0,-.5],[0,0,1,.5],[1,-.5,.5,-.25],[0,1,0,.5]],
            [[1,0,0,-1],[1,.5,0,0],[0,.5,0,-1],[.5,.25,-1,-.5],[0,.5,0,1],[1,-.5,0,0],[.5,-.25,1,-.5]],
            [[0,1,1,0],[.5,1,0,0],[.25,-.5,-.5,1],[-.5,0,1,0],[.25,.5,.5,1],[.5,-1,0,0],[.5,0,1,0]]),
        'DPS102': (
            [[4/9,-8/9,-8/9,-4/9],[0,5/9,0,10/9],[8/9,-2/3,0,0],[4/9,2/9,8/9,4/9],[0,-10/9,0,0],[4/9,-1/3,-8/9,2/3],[-4/9,-2/9,8/9,4/9]],
            [[-3/5,4/5,-4/5,-3/5],[0,.5,0,-1],[-1,.5,0,0],[0,5/4,0,0],[3/5,-3/10,4/5,-2/5],[2/5,3/10,-4/5,-3/5],[-3/5,-9/20,-4/5,-3/5]],
            [[9/20,9/10,9/10,-9/20],[0,0,27/40,-9/10],[-9/8,0,-9/16,0],[9/20,9/10,9/40,9/20],[-27/40,9/10,27/80,-9/20],[0,0,-9/8,0],[9/20,9/10,-9/40,-9/20]]),
        'classic': (
            [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]],
            [[1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],[0,0,0,1],[1,1,0,0],[0,0,1,1]],
            [[1,0,0,1],[0,0,1,-1],[0,1,0,1],[1,0,1,0],[-1,1,0,0],[0,0,0,1],[1,0,0,0]]),
        'winograd': (
            [[1,0,0,0],[0,1,0,0],[1,1,-1,-1],[0,0,0,1],[0,0,1,1],[-1,0,1,1],[1,0,-1,0]],
            [[1,0,0,0],[0,0,1,0],[0,0,0,1],[1,-1,-1,1],[-1,1,0,0],[1,-1,0,1],[0,-1,0,1]],
            [[1,1,1,1],[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,1,0,1],[0,1,1,1],[0,0,1,1]])
    }


def validate(l, r, w):
    tensor = np.einsum('ri,rj,rk->kij', l, r, w)
    target = np.zeros((4,4,4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                target[2*i+j,2*i+k,2*k+j] = 1
    err = float(np.max(np.abs(tensor-target)))
    assert err < 1e-13, err
    return err


def blocks(a):
    n = a.shape[0]//2
    return [a[:n,:n],a[:n,n:],a[n:,:n],a[n:,n:]]


def evaluate(a,b,scheme,dtype,extra_bits=0):
    l,r,w = [np.asarray(v,dtype=np.float64) for v in scheme]
    ab,bb = blocks(a),blocks(b)
    outputs = [np.zeros_like(ab[0]) for _ in range(4)]
    for lr,rr,wr in zip(l,r,w):
        left = sum(c*v for c,v in zip(lr,ab) if c)
        right = sum(c*v for c,v in zip(rr,bb) if c)
        if extra_bits:
            # Hypothetical significand budget, not an available instruction.
            def quant_bits(x):
                m,e=np.frexp(x)
                p=11+extra_bits
                return np.ldexp(np.rint(m*2**p)/2**p,e)
            left,right=quant_bits(left),quant_bits(right)
        elif dtype != 'exact':
            left,right=q(left,dtype),q(right,dtype)
        p = left@right
        for k,c in enumerate(wr):
            if c:
                outputs[k] += c*p
    return np.block([[outputs[0],outputs[1]],[outputs[2],outputs[3]]])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--size',type=int,default=1024)
    ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--output',default='research/precision_20261015/screen.json')
    args=ap.parse_args()
    formulas=schemes()
    checks={name:validate(*[np.asarray(v) for v in scheme]) for name,scheme in formulas.items()}
    rows=[]
    for dtype in ['fp16','bf16']:
        for seed in range(1,args.seeds+1):
            rng=np.random.default_rng(seed)
            a=q(rng.uniform(-1,1,(args.size,args.size)),dtype)
            b=q(rng.uniform(-1,1,(args.size,args.size)),dtype)
            ref=a@b
            den=np.linalg.norm(ref)
            for name,scheme in formulas.items():
                for operand in ([dtype,'fp16'] if dtype=='bf16' else [dtype]):
                    out=evaluate(a,b,scheme,operand)
                    row=dict(input=dtype,seed=seed,formula=name,operand=operand,ef=float(np.linalg.norm(out-ref)/den))
                    rows.append(row)
                    print(row,flush=True)
            if dtype=='fp16':
                for bits in [1,2,3]:
                    out=evaluate(a,b,formulas['winograd'],dtype,extra_bits=bits)
                    row=dict(input=dtype,seed=seed,formula='winograd_hypothetical_extra_bits',operand=f'fp16+{bits}',ef=float(np.linalg.norm(out-ref)/den))
                    rows.append(row)
                    print(row,flush=True)
    Path(args.output).write_text(json.dumps(dict(scope='diagnostic only; FP64 leaf products and recombination; not GPU acceptance or performance',size=args.size,tensor_identity_checks=checks,rows=rows),indent=2)+'\n')


if __name__=='__main__':
    main()

"""NEW numerical screen: block-scaled INT8 first-order compensation.

No FP8 kernels or production changes. Exact FP64 dots of rounded INT8 operands
model an exact INT32 dot (K=128 bounds its magnitude below 2^31), followed by
ideal rescaling. This does NOT measure conversion/synchronization/runtime costs.
"""
import argparse
import json
from pathlib import Path
import torch
from ablate import inputs, transforms, leaves, recombine, correction_pairs, lift, SIGNS
from scaling import inside


def quant_block(x, rows, cols):
    m,n=x.shape
    tiles=x.reshape(m//rows,rows,n//cols,cols)
    peak=tiles.abs().amax((1,3),keepdim=True)
    scale=torch.exp2(torch.ceil(torch.log2(torch.where(peak>0,peak/127.,1.))))
    integer=torch.round(tiles/scale).clamp(-127,127)
    assert float(integer.abs().max())<=127
    return (integer*scale).reshape(m,n)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--patterns',default='uniform,positive,range16,cancellation,block_scaled,bf16_range,inside_mismatch')
    ap.add_argument('--inside-scale',action='store_true')
    ap.add_argument('--output',default='research/precision_20261016/int8_model.json')
    args=ap.parse_args()
    rows=[]
    for dt in [torch.float16,torch.bfloat16]:
        for pat in args.patterns.split(','):
            for seed in [901,902,903]:
                a,b=inputs(1024,seed,pat,dt)
                ref=a@b
                den=torch.linalg.vector_norm(ref)
                if args.inside_scale:
                    a,b=inside(a,b)
                s,t=transforms(a,b)
                sh,th=[[v.to(dt).double() for v in vs] for vs in (s,t)]
                sl,tl=[[v-h for v,h in zip(vs,hs)] for vs,hs in ((s,sh),(t,th))]
                raw=recombine(leaves(a,b,sh,th))
                pairs=correction_pairs(a,b,sh,th,sl,tl)
                result=raw.clone()
                for (x,y),sg in zip(pairs,SIGNS):
                    # Main 64x32 leaf output, BK=128. Each operand's scale is
                    # constant over the shared K tile, so undo the product of
                    # the two scales after each exact INT32 partial product.
                    qx,qy=quant_block(x,64,128),quant_block(y,128,32)
                    result+=lift(qx@qy,sg)
                err=float((torch.linalg.vector_norm(result-ref)/den).item())
                rawerr=float((torch.linalg.vector_norm(raw-ref)/den).item())
                row=dict(dtype=str(dt).split('.')[-1],pattern=pat,seed=seed,
                         input_scaling='power2_inside' if args.inside_scale else 'none',
                         raw_ef=rawerr,int8_first_order_ef=err,
                         fixed_floor=1e-4 if dt==torch.float16 else 1e-3,
                         pass_fixed_floor=err<=(1e-4 if dt==torch.float16 else 1e-3))
                rows.append(row);print(json.dumps(row),flush=True)
                Path(args.output).write_text(json.dumps(dict(scope=__doc__,block_a=[64,128],block_b=[128,32],max_integer_partial=127*127*128,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()

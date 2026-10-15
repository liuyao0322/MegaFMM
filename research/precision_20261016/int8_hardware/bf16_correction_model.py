"""M11: original native main products, eight BF16 correction dots.

Three paired numerical arms: existing K128 INT8, BF16 low without scaling,
and BF16 low scaled by fixed2^16 then restored once per correction leaf.
Original rounded-input FP64 reference, TF32 disabled, both input dtypes.
FP32 batched dot/add modeling is not exact Hopper WGMMA simulation or speed.
"""
import argparse
import json
from pathlib import Path
import torch
from wide_quant_model import (PATTERNS, GROUPS, tile_a, tile_b, original_inputs,
                              alternating_inputs, blocks, SIGNS, quant_pair,
                              run as wide_control)

ARMS=('int8_k128','bf16_low_unscaled','bf16_low_scale16')
ALL_PATTERNS=PATTERNS+('tiny','tiny_cancellation')

def inputs(dt,pattern,seed):
    if pattern=='alternating_inside_mismatch':
        return alternating_inputs(1024,seed,pattern,dt)
    base='cancellation' if pattern in ('cancellation_k_bands','tiny_cancellation') else 'uniform' if pattern in ('balanced_k_bands','tiny') else pattern
    a,b=original_inputs(1024,seed,base,dt)
    if pattern in ('balanced_k_bands','cancellation_k_bands'):
        e=torch.tensor([0,4,-4,2],device='cuda').repeat_interleave(128).repeat(2)
        factor=torch.exp2(e.double());a=a*factor[None,:];b=b*factor[:,None]
    if pattern in ('tiny','tiny_cancellation'):
        factor=2.**(-60 if dt==torch.bfloat16 else -10);a=a*factor;b=b*factor
    return a.to(dt).double(),b.to(dt).double()

def run(dt,pattern,seed):
    a,b=inputs(dt,pattern,seed);ref=a@b
    af,bf=a.float(),b.float();am,bm=af.abs().amax(0),bf.abs().amax(1)
    e=torch.where((am>0)&(bm>0),torch.round(.5*(torch.log2(am)-torch.log2(bm))),0).int()
    aa=blocks(torch.ldexp(af,-e[None,:]));bb=blocks(torch.ldexp(bf,e[:,None]))
    s1,t1=aa[2]+aa[3],bb[1]-bb[0];s2,t2=s1-aa[0],bb[3]-t1
    ss,tt=[s1,s2,aa[0]-aa[2],aa[1]-s2],[t1,t2,bb[3]-bb[1],t2-bb[2]]
    ah,bh=[x.to(dt).float() for x in aa],[x.to(dt).float() for x in bb]
    sh,th=[x.to(dt).float() for x in ss],[x.to(dt).float() for x in tt]
    native=((ah[0],bh[0]),(ah[1],bh[2]),(sh[3],bh[3]),(ah[3],th[3]),(sh[0],th[0]),(sh[1],th[1]),(sh[2],th[2]))
    main=[torch.zeros((128,64,32),device='cuda') for _ in native]
    correction={arm:[torch.zeros_like(main[0]) for _ in GROUPS] for arm in ARMS}
    nonfinite_operands={arm:0 for arm in ARMS};saturated=max_integer=0
    bits=11 if dt==torch.float16 else 8
    for offset in range(0,512,128):
        for j,(x,y) in enumerate(native):main[j]=main[j]+torch.bmm(tile_a(x,offset,128),tile_b(y,offset,128))
        sq=[quant_pair(tile_a(h,offset,128),tile_a(x-h,offset,128),bits,True) for x,h in zip(ss,sh)]
        tq=[quant_pair(tile_b(h,offset,128),tile_b(x-h,offset,128),bits,True) for x,h in zip(tt,th)]
        ar=tile_a(ah[3],offset,128);br=tile_b(bh[3],offset,128)
        aq=quant_pair(ar,torch.zeros_like(ar),bits,False);bq=quant_pair(br,torch.zeros_like(br),bits,False)
        pairs=[(sq[3],1,bq,0),(aq,0,tq[3],1)]
        for j in range(3):pairs.extend(((sq[j],1,tq[j],0),(sq[j],0,tq[j],1)))
        saturated+=sum(x[4] for x in sq+tq)
        for j,group in enumerate(GROUPS):
            x,xi,y,yi=pairs[group[0]];integer=torch.bmm(x[xi],y[yi]);scale=x[2+xi]*y[2+yi]
            if len(group)==2:
                xx,xxi,yy,yyi=pairs[group[1]];assert torch.equal(scale,xx[2+xxi]*yy[2+yyi])
                integer=integer+torch.bmm(xx[xxi],yy[yyi])
            max_integer=max(max_integer,int(integer.abs().max().item()))
            correction['int8_k128'][j]=correction['int8_k128'][j]+integer*scale
        for arm,power in [('bf16_low_unscaled',0),('bf16_low_scale16',16)]:
            def high(x):return x.to(torch.bfloat16).float()
            def low(x):return (x*(2.**power)).to(torch.bfloat16).float()
            sa=[high(tile_a(x,offset,128)) for x in sh];ta=[high(tile_b(x,offset,128)) for x in th]
            sl=[low(tile_a(x-h,offset,128)) for x,h in zip(ss,sh)];tl=[low(tile_b(x-h,offset,128)) for x,h in zip(tt,th)]
            rr,rrb=high(ar),high(br)
            nonfinite_operands[arm]+=sum(int((~torch.isfinite(x)).sum().item()) for x in sa+ta+sl+tl+[rr,rrb])
            pairs=[(sl[3],rrb),(rr,tl[3])]
            for j in range(3):pairs.extend(((sl[j],ta[j]),(sa[j],tl[j])))
            for j,group in enumerate(GROUPS):
                value=torch.bmm(*pairs[group[0]])
                if len(group)==2:value=value+torch.bmm(*pairs[group[1]])
                correction[arm][j]=correction[arm][j]+value
    u2=main[0]+main[5];u3=u2+main[6]
    native_output=[main[0]+main[1],(u2+main[4])+main[2],u3-main[3],u3+main[4]]
    def untile(x):return x.reshape(8,16,64,32).permute(0,2,1,3).reshape(512,512)
    rows=[]
    for arm in ARMS:
        c=list(native_output)
        for d,group in zip(correction[arm],GROUPS):
            if arm=='bf16_low_scale16':d=d*(2.**-16)
            for j,sign in enumerate(SIGNS[group[0]]):
                if sign:c[j]=c[j]+d if sign>0 else c[j]-d
        out=torch.cat([torch.cat([untile(c[0]),untile(c[1])],1),torch.cat([untile(c[2]),untile(c[3])],1)],0).double()
        ef=float((torch.linalg.vector_norm(out-ref)/torch.linalg.vector_norm(ref)).item());nf=int((~torch.isfinite(out)).sum().item())
        floor=1e-4 if dt==torch.float16 else 1e-3
        rows.append(dict(dtype=str(dt).split('.')[-1],pattern=pattern,seed=seed,arm=arm,
            ef=ef,nonfinite=nf,nonfinite_correction_operands=nonfinite_operands[arm],floor=floor,
            pass_fixed_floor=nf==0 and nonfinite_operands[arm]==0 and ef<=floor,
            saturated_low_values=saturated if arm=='int8_k128' else None,
            max_absolute_integer=max_integer if arm=='int8_k128' else None))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    x=torch.tensor(2.**127+2.**119,device='cuda',dtype=torch.float32)
    h=x.to(torch.bfloat16).float();lo=x-h
    assert torch.isfinite(x) and torch.isfinite(h) and not torch.isfinite((lo*(2.**16)).to(torch.bfloat16))
    print('RANGE_NEGATIVE_CONTROL '+json.dumps(dict(finite_transform=True,finite_native_high=True,scaled_low_bf16_nonfinite=True)),flush=True)
    rows=[]
    for dt in (torch.float16,torch.bfloat16):
        for pattern in ALL_PATTERNS:
            for seed in (6301,6302,6303):
                fresh=run(dt,pattern,seed)
                if pattern=='uniform' and seed==6301:
                    old=wide_control(dt,pattern,seed)[0]
                    assert fresh[0]['ef']==old['ef'] and fresh[0]['max_absolute_integer']==old['max_absolute_integer']
                    print('CONTROL '+json.dumps(dict(dtype=str(dt),seed=seed,baseline_matches_m9=True)),flush=True)
                rows+=fresh
                for row in fresh:print(json.dumps(row),flush=True)
                args.output.write_text(json.dumps(dict(scope=__doc__,rows=rows),indent=2)+'\n')
    return 0 if all(x['pass_fixed_floor'] for x in rows) else 2
if __name__=='__main__':raise SystemExit(main())

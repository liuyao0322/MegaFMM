"""M6 screen: globally shared K128 inside scales, matching P9 quantization tile sizes.

No production or kernel performance claim. Batched FP32 leaves approximate native
Hopper accumulation; exact-integer FP32 dots (<2^22) model INT32 correction dots.
TF32 is disabled. Reference is original rounded A/B FP64 AB.
"""
import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ablate import inputs, blocks, transforms, SIGNS


def quant_pair(high, low, mantissa, fixed_low):
    peak=high.abs().amax((-2,-1),keepdim=True)
    eh=torch.ceil(torch.log2(torch.where(peak>0,peak/127,torch.ones_like(peak))))
    sh=torch.exp2(eh)
    if fixed_low:
        sl=torch.exp2(eh-mantissa)
    else:
        peak=low.abs().amax((-2,-1),keepdim=True)
        sl=torch.exp2(torch.ceil(torch.log2(torch.where(peak>0,peak/127,torch.ones_like(peak)))))
    qh=torch.round(high/sh);ql=torch.round(low/sl)
    saturation=int((ql.abs()>127).sum().item())
    return qh.clamp(-127,127),ql.clamp(-127,127),sh,sl,saturation


def tile_a(x,k):
    return x[:,k*128:(k+1)*128].reshape(8,64,128)[:,None].expand(8,16,64,128).reshape(128,64,128).float()


def tile_b(x,k):
    return x[k*128:(k+1)*128,:].reshape(128,16,32).permute(1,0,2)[None].expand(8,16,128,32).reshape(128,128,32).float()


def run(dt,pattern,seed,fixed_low,interleave,low_extra_bits=0):
    a,b=inputs(1024,seed,pattern,dt);ref=a@b
    a0,b0=blocks(a),blocks(b)
    p=[torch.zeros((128,64,32),device='cuda') for _ in range(7)]
    corr=[torch.zeros_like(p[0]) for _ in range(8)]
    fused=[torch.zeros_like(p[0]) for _ in range(4)]
    saturated=0
    for k in range(4):
        aa=[tile_a(x,k) for x in a0];bb=[tile_b(x,k) for x in b0]
        ex=[]
        for ai,bi in [((0,2),(0,1)),((1,3),(2,3))]:
            am=torch.maximum(*(aa[i].abs().amax((-2,-1),keepdim=True) for i in ai))
            bm=torch.maximum(*(bb[i].abs().amax((-2,-1),keepdim=True) for i in bi))
            # One maximum over all row/column tiles for each original K half.
            am=am.amax().expand_as(am);bm=bm.amax().expand_as(bm)
            ex.append(torch.where((am>0)&(bm>0),torch.round(.5*(torch.log2(am)-torch.log2(bm))),0))
        aa=[torch.ldexp(x,-ex[i%2].int()) for i,x in enumerate(aa)]
        bb=[torch.ldexp(x,ex[i//2].int()) for i,x in enumerate(bb)]
        s1=aa[2]+aa[3];s2=s1-aa[0];t1=bb[1]-bb[0];t2=bb[3]-t1
        ss=[s1,s2,aa[0]-aa[2],aa[1]-s2];tt=[t1,t2,bb[3]-bb[1],t2-bb[2]]
        ah=[x.to(dt).float() for x in aa];bh=[x.to(dt).float() for x in bb]
        sh=[x.to(dt).float() for x in ss];th=[x.to(dt).float() for x in tt]
        main=[(ah[0],bh[0]),(ah[1],bh[2]),(sh[3],bh[3]),(ah[3],th[3]),(sh[0],th[0]),(sh[1],th[1]),(sh[2],th[2])]
        partial=[torch.bmm(x,y) for x,y in main]
        for j,d in enumerate(partial):p[j]=p[j]+d
        low_bits=(10 if dt==torch.float16 else 7)+low_extra_bits
        sq=[quant_pair(h,x-h,low_bits,fixed_low) for x,h in zip(ss,sh)]
        tq=[quant_pair(h,x-h,low_bits,fixed_low) for x,h in zip(tt,th)]
        ar=quant_pair(ah[3],torch.zeros_like(ah[3]),10,False)
        br=quant_pair(bh[3],torch.zeros_like(bh[3]),10,False)
        saturated+=sum(x[4] for x in sq+tq)
        pairs=[(sq[3],1,br,0),(ar,0,tq[3],1)]
        for j in range(3):pairs.extend([(sq[j],1,tq[j],0),(sq[j],0,tq[j],1)])
        correction_partial=[]
        for j,(x,xi,y,yi) in enumerate(pairs):
            d=torch.bmm(x[xi],y[yi])
            # Separate RN multiply/add; each dot is an exactly representable integer.
            cd=d*(x[2+xi]*y[2+yi]);corr[j]=corr[j]+cd;correction_partial.append(cd)
        if interleave:
            # On-chip schedule: each K128 then P0,P1,P5,P6,P4,P2,P3.
            # Corrections immediately follow their native product.
            destinations=[(0,1,2,3),(0,),(1,),(2,),(1,3),(1,2,3),(2,3)]
            associated={5:(4,5),6:(6,7),4:(2,3),2:(0,),3:(1,)}
            for prod in (0,1,5,6,4,2,3):
                for j in destinations[prod]:
                    fused[j]=fused[j]-partial[prod] if prod==3 else fused[j]+partial[prod]
                terms=associated.get(prod,())
                # Matched relative scales permit exact INT32 pair summation,
                # followed by a single FP32 rescaling (rather than two rounds).
                if fixed_low and len(terms)==2:
                    i,j=terms
                    x,xi,y,yi=pairs[i];xx,xxi,yy,yyi=pairs[j]
                    scale=x[2+xi]*y[2+yi]
                    assert torch.equal(scale,xx[2+xxi]*yy[2+yyi])
                    cd=(torch.bmm(x[xi],y[yi])+torch.bmm(xx[xxi],yy[yyi]))*scale
                    for dest,sg in enumerate(SIGNS[i]):
                        if sg:fused[dest]=fused[dest]+cd if sg>0 else fused[dest]-cd
                else:
                    for term in terms:
                        for dest,sg in enumerate(SIGNS[term]):
                            if sg:fused[dest]=fused[dest]+correction_partial[term] if sg>0 else fused[dest]-correction_partial[term]
    u2=p[0]+p[5];u3=u2+p[6]
    c=[p[0]+p[1],(u2+p[4])+p[2],u3-p[3],u3+p[4]]
    for d,sign in zip(corr,SIGNS):
        for j,sg in enumerate(sign):
            if sg:c[j]=c[j]+d if sg>0 else c[j]-d
    if interleave:c=fused
    def untile(x):return x.reshape(8,16,64,32).permute(0,2,1,3).reshape(512,512)
    out=torch.cat([torch.cat([untile(c[0]),untile(c[1])],1),torch.cat([untile(c[2]),untile(c[3])],1)],0).double()
    ef=float((torch.linalg.vector_norm(out-ref)/torch.linalg.vector_norm(ref)).item())
    nonfinite=int((~torch.isfinite(out)).sum().item())
    floor=1e-4 if dt==torch.float16 else 1e-3
    return dict(dtype=str(dt).split('.')[-1],pattern=pattern,seed=seed,low_scale='relative' if fixed_low else 'adaptive',low_extra_bits=low_extra_bits,accumulation='K128_interleaved' if interleave else 'leaf_then_recombine',ef=ef,nonfinite=nonfinite,saturated_low_values=saturated,floor=floor,pass_fixed_floor=nonfinite==0 and ef<=floor)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixed-low',action='store_true');ap.add_argument('--low-extra-bits',type=int,default=0,choices=[0,1]);ap.add_argument('--interleave',action='store_true');ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[]
    for dt in (torch.float16,torch.bfloat16):
        for pattern in ('uniform','positive','range16','cancellation','block_scaled','bf16_range','inside_mismatch'):
            for seed in (1701,1702,1703):
                row=run(dt,pattern,seed,args.fixed_low,args.interleave,args.low_extra_bits);rows.append(row);print(json.dumps(row),flush=True)
                args.output.write_text(json.dumps(dict(scope=__doc__,block_a=[64,128],block_b=[128,32],inside='global K128, separate original K halves',rows=rows),indent=2)+'\n')
    return 0 if all(x['pass_fixed_floor'] for x in rows) else 2


if __name__=='__main__':raise SystemExit(main())

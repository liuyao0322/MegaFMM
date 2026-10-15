"""M8: per-column inside, relative low scales, paired INT8 dots before rescaling.

Controlled three-arm numerical screen; seven FP32 modeled native leaves plus eight
integer dots. Paired arm combines three pairs while still integer, leaving five
correction leaves. Not WGMMA simulation or hardware/performance acceptance.
Original rounded-input FP64 reference, fixed dtype floors and TF32 disabled.
"""
import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ablate import inputs as original_inputs, blocks, SIGNS
from alternating_inside_model import inputs as alternating_inputs
from global_k128_model import tile_a,tile_b,quant_pair

ARMS=('adaptive_unpaired','relative_unpaired','relative_paired')
GROUPS=((0,),(1,),(2,3),(4,5),(6,7))


def run(dt,pattern,seed):
    fn=alternating_inputs if pattern=='alternating_inside_mismatch' else original_inputs
    a,b=fn(1024,seed,pattern,dt);ref=a@b
    af,bf=a.float(),b.float()
    am=af.abs().amax(0);bm=bf.abs().amax(1)
    exponent=torch.where((am>0)&(bm>0),torch.round(.5*(torch.log2(am)-torch.log2(bm))),0).int()
    af=torch.ldexp(af,-exponent[None,:]);bf=torch.ldexp(bf,exponent[:,None])
    aa0,bb0=blocks(af),blocks(bf)
    main=[torch.zeros((128,64,32),device='cuda') for _ in range(7)]
    correction={arm:[torch.zeros_like(main[0]) for _ in (GROUPS if arm=='relative_paired' else range(8))] for arm in ARMS}
    saturated={arm:0 for arm in ARMS};max_integer=0
    for k in range(4):
        aa=[tile_a(x,k) for x in aa0];bb=[tile_b(x,k) for x in bb0]
        s1=aa[2]+aa[3];s2=s1-aa[0];t1=bb[1]-bb[0];t2=bb[3]-t1
        ss=[s1,s2,aa[0]-aa[2],aa[1]-s2];tt=[t1,t2,bb[3]-bb[1],t2-bb[2]]
        ah=[x.to(dt).float() for x in aa];bh=[x.to(dt).float() for x in bb]
        sh=[x.to(dt).float() for x in ss];th=[x.to(dt).float() for x in tt]
        native=((ah[0],bh[0]),(ah[1],bh[2]),(sh[3],bh[3]),(ah[3],th[3]),(sh[0],th[0]),(sh[1],th[1]),(sh[2],th[2]))
        for j,(x,y) in enumerate(native):main[j]=main[j]+torch.bmm(x,y)
        for fixed in (False,True):
            bits=11 if dt==torch.float16 else 8
            sq=[quant_pair(h,x-h,bits,fixed) for x,h in zip(ss,sh)]
            tq=[quant_pair(h,x-h,bits,fixed) for x,h in zip(tt,th)]
            ar=quant_pair(ah[3],torch.zeros_like(ah[3]),bits,False)
            br=quant_pair(bh[3],torch.zeros_like(bh[3]),bits,False)
            pairs=[(sq[3],1,br,0),(ar,0,tq[3],1)]
            for j in range(3):pairs.extend(((sq[j],1,tq[j],0),(sq[j],0,tq[j],1)))
            integers=[];scales=[]
            for x,xi,y,yi in pairs:
                integers.append(torch.bmm(x[xi],y[yi]));scales.append(x[2+xi]*y[2+yi])
            arm='relative_unpaired' if fixed else 'adaptive_unpaired'
            saturated[arm]+=sum(x[4] for x in sq+tq)
            for j,(integer,scale) in enumerate(zip(integers,scales)):
                correction[arm][j]=correction[arm][j]+integer*scale
            if fixed:
                saturated['relative_paired']=saturated[arm]
                for j,group in enumerate(GROUPS):
                    integer=integers[group[0]]
                    if len(group)==2:
                        other=group[1]
                        assert SIGNS[group[0]]==SIGNS[other]
                        assert torch.equal(scales[group[0]],scales[other])
                        integer=integer+integers[other]
                    max_integer=max(max_integer,int(integer.abs().max().item()))
                    assert max_integer<=2*128*127**2<2**22
                    correction['relative_paired'][j]=correction['relative_paired'][j]+integer*scales[group[0]]
    u2=main[0]+main[5];u3=u2+main[6]
    native_output=[main[0]+main[1],(u2+main[4])+main[2],u3-main[3],u3+main[4]]
    def untile(x):return x.reshape(8,16,64,32).permute(0,2,1,3).reshape(512,512)
    rows=[]
    for arm in ARMS:
        c=list(native_output)
        signs=[SIGNS[g[0]] for g in GROUPS] if arm=='relative_paired' else SIGNS
        for d,sign in zip(correction[arm],signs):
            for j,sg in enumerate(sign):
                if sg:c[j]=c[j]+d if sg>0 else c[j]-d
        out=torch.cat([torch.cat([untile(c[0]),untile(c[1])],1),torch.cat([untile(c[2]),untile(c[3])],1)],0).double()
        ef=float((torch.linalg.vector_norm(out-ref)/torch.linalg.vector_norm(ref)).item());nonfinite=int((~torch.isfinite(out)).sum().item())
        floor=1e-4 if dt==torch.float16 else 1e-3
        rows.append(dict(dtype=str(dt).split('.')[-1],pattern=pattern,seed=seed,arm=arm,inside='per_column',accumulation='leaf_then_recombine',correction_leaves=len(correction[arm]),ef=ef,nonfinite=nonfinite,saturated_low_values=saturated[arm],max_absolute_paired_integer=max_integer if arm=='relative_paired' else None,floor=floor,pass_fixed_floor=nonfinite==0 and ef<=floor))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[]
    for dt in (torch.float16,torch.bfloat16):
        for pattern in ('uniform','positive','range16','cancellation','block_scaled','bf16_range','inside_mismatch','alternating_inside_mismatch'):
            for seed in (3901,3902,3903):
                new=run(dt,pattern,seed);rows.extend(new)
                for row in new:print(json.dumps(row),flush=True)
                args.output.write_text(json.dumps(dict(scope=__doc__,block_a=[64,128],block_b=[128,32],rows=rows),indent=2)+'\n')
    return 0 if all(x['pass_fixed_floor'] for x in rows) else 2

if __name__=='__main__':raise SystemExit(main())

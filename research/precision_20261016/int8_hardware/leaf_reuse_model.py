"""M13: isolate FP32 recombination order for native-high operand reuse.

Both arms share seven native main products and all eight scaled native residual
cross terms. TF32-disabled FP32 bmm/add is a numerical screen, not exact Hopper
WGMMA accumulation or a latency measurement. Original rounded-input FP64 AB.
"""
import argparse
import json
from pathlib import Path
import torch
from native_correction_model import inputs, ALL_PATTERNS, run as m12_control
from wide_quant_model import blocks, tile_a, tile_b, GROUPS, SIGNS

ARMS=('native_low_scaled', 'native_leaf_scaled')

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
    correction=[torch.zeros_like(main[0]) for _ in GROUPS]
    bits=11 if dt==torch.float16 else 8
    nf_operands=0
    for offset in range(0,512,128):
        for j,(x,y) in enumerate(native):
            main[j]=main[j]+torch.bmm(tile_a(x,offset,128),tile_b(y,offset,128))
        sa=[tile_a(x,offset,128) for x in sh];ta=[tile_b(x,offset,128) for x in th]
        sl=[(tile_a(x-h,offset,128)*(2.**bits)).to(dt).float() for x,h in zip(ss,sh)]
        tl=[(tile_b(x-h,offset,128)*(2.**bits)).to(dt).float() for x,h in zip(tt,th)]
        rr,rrb=tile_a(ah[3],offset,128),tile_b(bh[3],offset,128)
        nf_operands+=sum(int((~torch.isfinite(x)).sum().item()) for x in sa+ta+sl+tl+[rr,rrb])
        pairs=[(sl[3],rrb),(rr,tl[3])]
        for j in range(3):pairs.extend(((sl[j],ta[j]),(sa[j],tl[j])))
        for j,group in enumerate(GROUPS):
            value=torch.bmm(*pairs[group[0]])
            if len(group)==2:value=value+torch.bmm(*pairs[group[1]])
            correction[j]=correction[j]+value
    correction=[d*(2.**-bits) for d in correction]
    def recombine(m):
        u2=m[0]+m[5];u3=u2+m[6]
        return [m[0]+m[1],(u2+m[4])+m[2],u3-m[3],u3+m[4]]
    old=recombine(main)
    for d,group in zip(correction,GROUPS):
        for j,sign in enumerate(SIGNS[group[0]]):
            if sign:old[j]=old[j]+d if sign>0 else old[j]-d
    leaf=list(main)
    for p,d in zip((2,3,4,5,6),correction):leaf[p]=leaf[p]+d
    new=recombine(leaf)
    def untile(x):return x.reshape(8,16,64,32).permute(0,2,1,3).reshape(512,512)
    outs=[];rows=[]
    dense_ef=float((torch.linalg.vector_norm((af@bf).double()-ref)/torch.linalg.vector_norm(ref)).item())
    for arm,c in zip(ARMS,(old,new)):
        out=torch.cat([torch.cat([untile(c[0]),untile(c[1])],1),torch.cat([untile(c[2]),untile(c[3])],1)],0).double()
        outs.append(out)
        ef=float((torch.linalg.vector_norm(out-ref)/torch.linalg.vector_norm(ref)).item());nf=int((~torch.isfinite(out)).sum().item())
        floor=1e-4 if dt==torch.float16 else 1e-3
        rows.append(dict(dtype=str(dt).split('.')[-1],pattern=pattern,seed=seed,arm=arm,
            ef=ef,nonfinite=nf,nonfinite_correction_operands=nf_operands,floor=floor,dense_ef=dense_ef,
            pass_original_gate=nf==0 and nf_operands==0 and ef<=3*dense_ef+floor,
            pass_fixed_floor=nf==0 and nf_operands==0 and ef<=floor))
    delta=float((torch.linalg.vector_norm(outs[1]-outs[0])/torch.linalg.vector_norm(ref)).item())
    for row in rows:row['paired_output_delta_ef']=delta
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[];controls=[]
    for dt in (torch.float16,torch.bfloat16):
        for pattern in ALL_PATTERNS:
            for seed in (8301,8302,8303):
                fresh=run(dt,pattern,seed)
                if pattern=='uniform' and seed==8301:
                    old=m12_control(dt,pattern,seed)[2]
                    assert fresh[0]['ef']==old['ef'] and fresh[0]['nonfinite_correction_operands']==old['nonfinite_correction_operands']
                    control=dict(dtype=str(dt),seed=seed,native_arm_matches_m12=True)
                    controls.append(control);print('CONTROL '+json.dumps(control),flush=True)
                rows+=fresh
                for row in fresh:print(json.dumps(row),flush=True)
                args.output.write_text(json.dumps(dict(scope=__doc__,rows=rows,controls=controls),indent=2)+'\n')
    return 0 if all(x['pass_fixed_floor'] and x['pass_original_gate'] for x in rows) else 2
if __name__=='__main__':raise SystemExit(main())

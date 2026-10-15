"""M14: K64 native fusion numerical screen, not WGMMA or timing evidence.

Four paired arms isolate K granularity and Winograd recombination: full K128
corrected leaves (M13 control), full K64 leaves, K64 chunk-output accumulation,
and K64 streaming product accumulation with only four persistent output tiles.
Both original dtypes, exact column scaling, all seven main/eight cross dots.
"""
import argparse
import json
from pathlib import Path
import torch
from native_correction_model import inputs, ALL_PATTERNS
from leaf_reuse_model import run as m13_control
from wide_quant_model import blocks, tile_a, tile_b, GROUPS

ARMS=('full_k128','full_k64','chunk_k64','stream_k64')

def recombine(m):
    u2=m[0]+m[5];u3=u2+m[6]
    return [m[0]+m[1],(u2+m[4])+m[2],u3-m[3],u3+m[4]]

def corrected(main,corr,power):
    leaf=list(main)
    for p,d in zip((2,3,4,5,6),corr):leaf[p]=leaf[p]+d*(2.**-power)
    return leaf

def run(dt,pattern,seed):
    a,b=inputs(dt,pattern,seed);ref=a@b;af,bf=a.float(),b.float()
    am,bm=af.abs().amax(0),bf.abs().amax(1)
    e=torch.where((am>0)&(bm>0),torch.round(.5*(torch.log2(am)-torch.log2(bm))),0).int()
    aa=blocks(torch.ldexp(af,-e[None,:]));bb=blocks(torch.ldexp(bf,e[:,None]))
    s1,t1=aa[2]+aa[3],bb[1]-bb[0];s2,t2=s1-aa[0],bb[3]-t1
    ss,tt=[s1,s2,aa[0]-aa[2],aa[1]-s2],[t1,t2,bb[3]-bb[1],t2-bb[2]]
    ah,bh=[x.to(dt).float() for x in aa],[x.to(dt).float() for x in bb]
    sh,th=[x.to(dt).float() for x in ss],[x.to(dt).float() for x in tt]
    native=((ah[0],bh[0]),(ah[1],bh[2]),(sh[3],bh[3]),(ah[3],th[3]),(sh[0],th[0]),(sh[1],th[1]),(sh[2],th[2]))
    bits=11 if dt==torch.float16 else 8
    output={};nonfinite={}
    for width in (128,64):
        mains=[torch.zeros((128,64,32),device='cuda') for _ in native]
        corrections=[torch.zeros_like(mains[0]) for _ in GROUPS]
        chunk=stream=None;nf_operands=0
        for offset in range(0,512,width):
            main=[torch.bmm(tile_a(x,offset,width),tile_b(y,offset,width)) for x,y in native]
            sa=[tile_a(x,offset,width) for x in sh];ta=[tile_b(x,offset,width) for x in th]
            sl=[(tile_a(x-h,offset,width)*(2.**bits)).to(dt).float() for x,h in zip(ss,sh)]
            tl=[(tile_b(x-h,offset,width)*(2.**bits)).to(dt).float() for x,h in zip(tt,th)]
            rr,rrb=tile_a(ah[3],offset,width),tile_b(bh[3],offset,width)
            nf_operands+=sum(int((~torch.isfinite(x)).sum().item()) for x in sa+ta+sl+tl+[rr,rrb])
            pairs=[(sl[3],rrb),(rr,tl[3])]
            for j in range(3):pairs.extend(((sl[j],ta[j]),(sa[j],tl[j])))
            corr=[]
            for group in GROUPS:
                d=torch.bmm(*pairs[group[0]])
                if len(group)==2:d=d+torch.bmm(*pairs[group[1]])
                corr.append(d)
            mains=[x+y for x,y in zip(mains,main)]
            corrections=[x+y for x,y in zip(corrections,corr)]
            if width==64:
                leaf=corrected(main,corr,bits);part=recombine(leaf)
                chunk=part if chunk is None else [x+y for x,y in zip(chunk,part)]
                # Exact streaming order0,1,5,6,4,2,3, with correction before use.
                stream=list(leaf[0] for _ in range(4)) if stream is None else [x+leaf[0] for x in stream]
                stream[0]=stream[0]+leaf[1]
                for j in (1,2,3):stream[j]=stream[j]+leaf[5]
                for j in (2,3):stream[j]=stream[j]+leaf[6]
                for j in (1,3):stream[j]=stream[j]+leaf[4]
                stream[1]=stream[1]+leaf[2];stream[2]=stream[2]-leaf[3]
        full=f'full_k{width}';output[full]=recombine(corrected(mains,corrections,bits));nonfinite[full]=nf_operands
        if width==64:
            output['chunk_k64']=chunk;output['stream_k64']=stream
            nonfinite['chunk_k64']=nonfinite['stream_k64']=nf_operands
    def untile(x):return x.reshape(8,16,64,32).permute(0,2,1,3).reshape(512,512)
    norm=torch.linalg.vector_norm(ref)
    dense_ef=float((torch.linalg.vector_norm((af@bf).double()-ref)/norm).item())
    outputs={arm:torch.cat([torch.cat([untile(c[0]),untile(c[1])],1),torch.cat([untile(c[2]),untile(c[3])],1)],0).double() for arm,c in output.items()}
    rows=[]
    for arm in ARMS:
        out=outputs[arm];ef=float((torch.linalg.vector_norm(out-ref)/norm).item());nf=int((~torch.isfinite(out)).sum().item())
        floor=1e-4 if dt==torch.float16 else 1e-3
        delta=float((torch.linalg.vector_norm(out-outputs['full_k128'])/norm).item())
        rows.append(dict(dtype=str(dt).split('.')[-1],pattern=pattern,seed=seed,arm=arm,ef=ef,
            nonfinite=nf,nonfinite_correction_operands=nonfinite[arm],floor=floor,dense_ef=dense_ef,
            delta_to_full_k128_ef=delta,
            pass_fixed_floor=nf==0 and nonfinite[arm]==0 and ef<=floor,
            pass_original_gate=nf==0 and nonfinite[arm]==0 and ef<=3*dense_ef+floor))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[];controls=[]
    for dt in (torch.float16,torch.bfloat16):
        for pattern in ALL_PATTERNS:
            for seed in (9301,9302,9303):
                fresh=run(dt,pattern,seed)
                if pattern=='uniform' and seed==9301:
                    old=m13_control(dt,pattern,seed)[1]
                    assert fresh[0]['ef']==old['ef']
                    c=dict(dtype=str(dt),seed=seed,full_k128_matches_m13_leaf=True)
                    controls.append(c);print('CONTROL '+json.dumps(c),flush=True)
                rows+=fresh
                for row in fresh:print(json.dumps(row),flush=True)
                args.output.write_text(json.dumps(dict(scope=__doc__,rows=rows,controls=controls),indent=2)+'\n')
    return 0 if all(x['pass_original_gate'] and x['pass_fixed_floor'] for x in rows) else 2
if __name__=='__main__':raise SystemExit(main())

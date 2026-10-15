"""Actual H100 FP8 residual products; diagnostic, not production latency.

Main products use FP32 cuBLAS with TF32 disabled; residual products use
torch._scaled_mm with E4M3 inputs and FP32 output, fast accumulation disabled.
"""
import json
from pathlib import Path
import torch


def main():
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    rows=[]
    timings=[]
    for dtype in [torch.float16,torch.bfloat16]:
        for seed in [101,102,103]:
            gen=torch.Generator(device='cuda').manual_seed(seed)
            a=(torch.rand((1024,1024),generator=gen,device='cuda')*2-1).to(dtype).float()
            b=(torch.rand((1024,1024),generator=gen,device='cuda')*2-1).to(dtype).float()
            aa,ab,ac,ad=a[:512,:512],a[:512,512:],a[512:,:512],a[512:,512:]
            ba,bb,bc,bd=b[:512,:512],b[:512,512:],b[512:,:512],b[512:,512:]
            ss=[ac+ad,ac+ad-aa,aa-ac,ab-(ac+ad-aa)]
            tt=[bb-ba,bd-(bb-ba),bd-bb,(bd-(bb-ba))-bc]
            sh=[v.to(dtype).float() for v in ss];th=[v.to(dtype).float() for v in tt]
            sl=[v-h for v,h in zip(ss,sh)];tl=[v-h for v,h in zip(tt,th)]
            p=[aa@ba,ab@bc,sh[3]@bd,ad@th[3],sh[0]@th[0],sh[1]@th[1],sh[2]@th[2]]
            u2=p[0]+p[5];u3=u2+p[6];u4=u2+p[4]
            raw=torch.cat([torch.cat([p[0]+p[1],u4+p[2]],1),torch.cat([u3-p[3],u3+p[4]],1)],0)
            pairs=[(sl[3],bd),(ad,tl[3]),(sl[0],th[0]),(sh[0],tl[0]),(sl[1],th[1]),(sh[1],tl[1]),(sl[2],th[2]),(sh[2],tl[2])]
            signs=[(0,1,0,0),(0,0,-1,0),(0,1,0,1),(0,1,0,1),(0,1,1,1),(0,1,1,1),(0,0,1,1),(0,0,1,1)]
            ref=a.double()@b.double()
            # Mixed subset selected exclusively on separate seeds 201,202,203.
            for mode in ['dense','residual_2of4','mixed4_dense']:
                out=raw.clone()
                for term,((left,right),sg) in enumerate(zip(pairs,signs)):
                    if mode=='residual_2of4' or (mode=='mixed4_dense' and term not in [3,4,5,6]):
                        # Prune residual operand only, grouping along its contraction axis.
                        is_left = term in [0,2,4,6]
                        v=left if is_left else right.T
                        groups=v.contiguous().view(-1,4)
                        idx=groups.abs().topk(2,dim=1).indices
                        keep=torch.zeros_like(groups).scatter_(1,idx,groups.gather(1,idx)).view_as(v)
                        if is_left: left=keep
                        else: right=keep.T
                    def encode(v):
                        peak=v.abs().amax().clamp_min(1e-30)
                        scale=torch.exp2(torch.ceil(torch.log2(peak/448))).reshape(1)
                        return (v/scale).to(torch.float8_e4m3fn),scale
                    lv,ls=encode(left);rv,rs=encode(right)
                    result=torch._scaled_mm(lv.contiguous(),rv.T.contiguous().T,scale_a=ls,scale_b=rs,out_dtype=torch.float32,use_fast_accum=False)
                    for k,c in enumerate(sg):
                        if c: out[(k//2)*512:(k//2+1)*512,(k%2)*512:(k%2+1)*512].add_(result,alpha=c)
                ef=float((torch.linalg.norm(out.double()-ref)/torch.linalg.norm(ref)).item())
                row=dict(dtype=str(dtype),seed=seed,correction=mode,ef=ef,nonfinite=int((~out.isfinite()).sum().item()))
                rows.append(row);print(row,flush=True)
            if seed==103:
                # Isolate a naive append-only correction chain. FP8 preparation is
                # explicitly excluded, so this is not a complete-call benchmark.
                prepared=[]
                for left,right in pairs:
                    lv,ls=encode(left);rv,rs=encode(right)
                    prepared.append((lv.contiguous(),rv.T.contiguous().T,ls,rs))
                def correction_chain():
                    return [torch._scaled_mm(lv,rv,scale_a=ls,scale_b=rs,out_dtype=torch.float32,use_fast_accum=False) for lv,rv,ls,rs in prepared]
                for _ in range(10): correction_chain()
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    retained=correction_chain()
                for _ in range(20):graph.replay()
                samples=[]
                for _ in range(5):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100):graph.replay()
                    end.record();end.synchronize()
                    samples.append(start.elapsed_time(end)*1000/100)
                timing=dict(dtype=str(dtype),scope='eight prequantized FP8 GEMMs only; graph replay; excludes encoding, main products, output recombination',group_us=samples,median_us=sorted(samples)[2])
                timings.append(timing);print(timing,flush=True)
    Path('research/precision_20261015/hardware_fp8.json').write_text(json.dumps(dict(scope='actual FP8 Tensor Core correction but FP32 main products; not production implementation or acceptance; timing only for explicitly scoped append-only chain',device=torch.cuda.get_device_name(),torch=torch.__version__,rows=rows,timings=timings),indent=2)+'\n')


if __name__=='__main__':
    main()

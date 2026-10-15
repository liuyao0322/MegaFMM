"""Single-kernel eight FP8 products, prequantized inputs; timing diagnostic."""
import json
from pathlib import Path
import torch
import triton
import triton.language as tl


@triton.jit
def products(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0);batch=tl.program_id(1)
    mi=tile//tl.cdiv(N,BN);ni=tile%tl.cdiv(N,BN)
    rows=mi*BM+tl.arange(0,BM)
    cols=ni*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for k in range(tl.cdiv(K,BK)):
        av=tl.load(A+batch*M*K+rows[:,None]*K+(kk[None,:]+k*BK))
        bv=tl.load(B+batch*N*K+cols[None,:]*K+(kk[:,None]+k*BK))
        acc=tl.dot(av,bv,acc,max_num_imprecise_acc=0)
    tl.store(C+batch*M*N+rows[:,None]*N+cols[None,:],acc)


def main():
    torch.manual_seed(401)
    a=torch.randn((8,512,512),device='cuda').to(torch.float8_e4m3fn)
    b=torch.randn((8,512,512),device='cuda').to(torch.float8_e4m3fn)
    out=torch.empty((8,512,512),device='cuda',dtype=torch.float32)
    one=torch.ones((1,),device='cuda')
    ref=torch.stack([torch._scaled_mm(a[i],b[i].T,scale_a=one,scale_b=one,out_dtype=torch.float32,use_fast_accum=False) for i in range(8)])
    exact=torch.bmm(a.double(),b.double().transpose(1,2))
    print(dict(cublas_relative_frobenius=float((torch.linalg.norm(ref.double()-exact)/torch.linalg.norm(exact)).item())),flush=True)
    rows=[]
    for bm,bn,bk,warps,stages in [(64,64,64,4,3),(64,128,64,4,3),(128,64,64,4,3),(128,128,64,4,3),(128,128,128,4,3),(128,128,128,8,3),(64,128,128,4,3),(64,64,128,4,3),(64,128,256,4,3),(128,64,128,4,3)]:
        def launch():products[(triton.cdiv(512,bm)*triton.cdiv(512,bn),8)](a,b,out,512,512,512,bm,bn,bk,num_warps=warps,num_stages=stages)
        launch();torch.cuda.synchronize()
        err=float((out-ref).abs().max().item())
        rel=float((torch.linalg.norm(out.double()-exact)/torch.linalg.norm(exact)).item())
        # This is a correction-only primitive: <0.2% error in a ~1e-3
        # correction is a screening budget, not the production GEMM gate.
        assert rel<.002,(err,rel)
        for _ in range(10):launch()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):launch()
        for _ in range(20):graph.replay()
        times=[]
        for _ in range(5):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):graph.replay()
            end.record();end.synchronize();times.append(start.elapsed_time(end)*10)
        row=dict(bm=bm,bn=bn,bk=bk,warps=warps,stages=stages,maxabs_vs_cublas=err,relative_frobenius_vs_fp64=rel,median_us=sorted(times)[2],samples_us=times)
        rows.append(row);print(row,flush=True)
    Path('research/precision_20261015/grouped_fp8.json').write_text(json.dumps(dict(scope='one kernel for eight prequantized 512-cubed FP8 products only; excludes transformation, scaling, original seven main products and output recombination; no complete-call claim',torch=torch.__version__,triton=triton.__version__,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()

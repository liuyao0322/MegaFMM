"""Stage and residual ablations against FP64 AB of original rounded inputs.

This is an independent numerical model, not production-kernel acceptance or
latency measurement. Torch float32 GEMMs disable TF32; FP64 leaves isolate casts.
"""
import argparse
import itertools
import json
from pathlib import Path
import torch


def blocks(x):
    h = x.shape[0] // 2
    return [x[:h, :h], x[:h, h:], x[h:, :h], x[h:, h:]]


def transforms(a, b):
    a11, a12, a21, a22 = blocks(a)
    b11, b12, b21, b22 = blocks(b)
    s1 = a21 + a22
    s2 = s1 - a11
    t1 = b12 - b11
    t2 = b22 - t1
    return [s1, s2, a11-a21, a12-s2], [t1, t2, b22-b12, t2-b21]


def leaves(a, b, s, t):
    a11, a12, _, a22 = blocks(a)
    b11, _, b21, b22 = blocks(b)
    return [a11@b11, a12@b21, s[3]@b22, a22@t[3],
            s[0]@t[0], s[1]@t[1], s[2]@t[2]]


def recombine(p):
    u2 = p[0]+p[5]
    u3 = u2+p[6]
    return torch.cat([torch.cat([p[0]+p[1], (u2+p[4])+p[2]], 1),
                      torch.cat([u3-p[3], u3+p[4]], 1)], 0)


SIGNS = [(0,1,0,0), (0,0,-1,0), (0,1,0,1), (0,1,0,1),
         (0,1,1,1), (0,1,1,1), (0,0,1,1), (0,0,1,1)]
NAMES = ['S4', 'T4', 'S1', 'T1', 'S2', 'T2', 'S3', 'T3']


def lift(x, signs):
    return torch.cat([torch.cat([signs[0]*x, signs[1]*x], 1),
                      torch.cat([signs[2]*x, signs[3]*x], 1)], 0)


def correction_pairs(a, b, sh, th, sl, tl):
    return [(sl[3], blocks(b)[3]), (blocks(a)[3], tl[3]),
            (sl[0], th[0]), (sh[0], tl[0]), (sl[1], th[1]),
            (sh[1], tl[1]), (sl[2], th[2]), (sh[2], tl[2])]


def q8(x, scale=None):
    if scale is None:
        peak = x.abs().max().item()
        scale = 2.0**__import__('math').ceil(__import__('math').log2(peak/448)) if peak else 1.
    return (x/scale).to(torch.float8_e4m3fn).double()*scale


def inputs(n, seed, pattern, dtype):
    g = torch.Generator(device='cuda').manual_seed(seed)
    a = torch.rand((n,n), generator=g, device='cuda', dtype=torch.float64)*2-1
    b = torch.rand((n,n), generator=g, device='cuda', dtype=torch.float64)*2-1
    if pattern == 'positive':
        a, b = a.abs(), b.abs()
    elif pattern == 'range16':
        a *= torch.exp2(torch.randint(-4,5,a.shape,generator=g,device='cuda'))
        b *= torch.exp2(torch.randint(-4,5,b.shape,generator=g,device='cuda'))
    elif pattern == 'cancellation':
        a[:,n//2:] = -a[:,:n//2] + 1/256
        b[n//2:,:] = b[:n//2,:]
    elif pattern == 'block_scaled':
        a[:n//2,:] *= 64
        a[n//2:,:] /= 64
        b[:,:n//2] /= 64
        b[:,n//2:] *= 64
    elif pattern == 'inside_mismatch':
        a[:,n//2:] *= 2.**-12
        b[:n//2,:] *= 2.**-12
    elif pattern == 'bf16_range':
        if dtype == torch.bfloat16:
            a *= 2.**30
            b *= 2.**-30
        else:
            a *= 2.**10
            b *= 2.**-10
    return a.to(dtype).double(), b.to(dtype).double()


def run(n, seed, pattern, dtype, legacy_representations=False):
    a,b = inputs(n,seed,pattern,dtype)
    ref = a@b
    den = torch.linalg.vector_norm(ref)
    def ef(x):
        return float((torch.linalg.vector_norm(x.double()-ref)/den).item())
    s,t = transforms(a,b)
    sf,tf = transforms(a.float(),b.float())
    sh,th = [[v.to(dtype).double() for v in vs] for vs in (sf,tf)]
    sl,tl = [[v-h for v,h in zip(vs,hs)] for vs,hs in ((s,sh),(t,th))]
    p = leaves(a,b,sh,th)
    raw = recombine(p)
    p32 = leaves(a.float(),b.float(),[v.float() for v in sh],[v.float() for v in th])
    dense32 = a.float()@b.float()
    floor = 1e-3 if dtype == torch.bfloat16 else 1e-4
    gate = 3*ef(dense32)+floor
    stages = {
        'exact_transforms_leaves_recombine': ef(recombine(leaves(a,b,s,t))),
        'fp32_transform_only': ef(recombine(leaves(a,b,[v.double() for v in sf],[v.double() for v in tf]))),
        'operand_cast_only': ef(raw),
        'fp32_leaf_no_operand_cast': ef(recombine([v.double() for v in leaves(a.float(),b.float(),sf,tf)])),
        'fp32_leaf_with_operand_cast': ef(recombine([v.double() for v in p32])),
        'fp32_recombine_only': ef(recombine([v.float() for v in p])),
        'all_fp32_with_operand_cast': ef(recombine(p32)),
        'dense_fp32': ef(dense32),
    }
    pairs = correction_pairs(a,b,sh,th,sl,tl)
    corrections = [lift(x@y,sg) for (x,y),sg in zip(pairs,SIGNS)]
    first = raw + sum(corrections)
    second = sum(lift(sl[i]@tl[i],SIGNS[2+2*i]) for i in range(3))
    assert ef(first+second)<2e-12, (pattern,ef(first+second))
    basis = torch.stack([v.reshape(-1) for v in corrections])
    error = (raw-ref).reshape(-1)
    gram = ((basis@basis.T)/den.square()).cpu()
    cross = ((basis@error)/den.square()).cpu()
    e2 = float((error@error/den.square()).item())
    subsets = []
    for k in range(9):
        best = None
        for ids in itertools.combinations(range(8),k):
            mask = torch.zeros(8,dtype=torch.float64)
            mask[list(ids)] = 1
            val = max(0., e2+2*float(mask@cross)+float(mask@gram@mask))**.5
            if best is None or val<best['ef']:
                best = {'terms': [NAMES[i] for i in ids], 'ef': val}
        subsets.append({'count':k,**best})
    row = dict(dtype=str(dtype).split('.')[-1],n=n,seed=seed,pattern=pattern,gate=gate,
               stages=stages,first_order_exact=ef(first),second_order_norm=float((torch.linalg.vector_norm(second)/den).item()),
               minimal_subset_by_count=subsets,
               single_term_norms={k:float(gram[i,i].sqrt()) for i,k in enumerate(NAMES)},
               residual_cross_gram=gram.tolist(),
               transform_fp32_maxabs=max(float((v-vf.double()).abs().max()) for v,vf in zip(s+t,sf+tf)),
               cast_error_quadrants=[float((torch.linalg.vector_norm(v)/den).item()) for v in blocks(raw-ref)])
    # Retained only to explain historical raw fields. Do not rerun this stopped
    # research direction as part of normal stage-ablation reproduction.
    if legacy_representations:
        adaptive = raw + sum(lift(q8(x)@q8(y),sg) for (x,y),sg in zip(pairs,SIGNS))
        residual_left = [True,False,True,False,True,False,True,False]
        residual_scale = 2.**(-17 if dtype == torch.float16 else -14)
        fixed = raw + sum(lift(q8(x,residual_scale if rl else 2.**-6) @
                              q8(y,2.**-6 if rl else residual_scale),sg)
                          for (x,y),sg,rl in zip(pairs,SIGNS,residual_left))
        half_s,half_t = [[v.half().double() for v in vs] for vs in (sf,tf)]
        half_all = recombine(leaves(a.half().double(),b.half().double(),half_s,half_t))
        row.update(fp8_adaptive=ef(adaptive),fp8_fixed=ef(fixed),all_fp16_operands=ef(half_all))
    return row


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--size',type=int,default=1024)
    ap.add_argument('--seeds',default='701,702,703')
    ap.add_argument('--patterns',default='uniform,positive,range16,cancellation,block_scaled,bf16_range')
    ap.add_argument('--output',default='research/precision_20261016/ablation.json')
    ap.add_argument('--legacy-representations',action='store_true',help='Historical models only; stopped research, excluded by default')
    args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for dt in (torch.float16,torch.bfloat16):
        for pattern in args.patterns.split(','):
            for seed in map(int,args.seeds.split(',')):
                row=run(args.size,seed,pattern,dt,args.legacy_representations)
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ['residual_cross_gram','minimal_subset_by_count']}),flush=True)
                Path(args.output).write_text(json.dumps(dict(scope=__doc__,torch=torch.__version__,device=torch.cuda.get_device_name(),rows=rows),indent=2,allow_nan=True)+'\n')


if __name__=='__main__':
    main()

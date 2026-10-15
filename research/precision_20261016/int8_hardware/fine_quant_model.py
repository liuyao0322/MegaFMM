"""M15: fullK512 INT8 correction geometry x recombination-order ablation.

Original rounded FP16/BF16 inputs/FP64 AB, all seven native main/eight cross
dots. TF32-disabled FP32 bmm/add is numerical modeling, not Hopper accumulation,
layout, synchronization, kernel speed or production acceptance evidence.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from native_correction_model import inputs as old_inputs, ALL_PATTERNS
from wide_quant_model import blocks, tile_a, tile_b, GROUPS, SIGNS, run as m9_control
from leaf_reuse_model import run as m13_control

PATTERNS = ALL_PATTERNS + ('rowcol_scale_bands', 'rowcol_cancellation')
SEEDS = (123107, 123108, 123109)
ARMS = ('coarse_post', 'coarse_leaf', 'fine_post', 'fine_leaf', 'native_leaf')
BOUND = 2*512*127**2
assert BOUND == 16516096 and BOUND < 2**24 < 2**31


def inputs(dt, pattern, seed):
    if pattern not in ('rowcol_scale_bands', 'rowcol_cancellation'):
        return old_inputs(dt, pattern, seed)
    base = 'cancellation' if pattern == 'rowcol_cancellation' else 'uniform'
    a, b = old_inputs(dt, base, seed)
    e = torch.tensor([-4, 0, 4, -2], device='cuda').repeat_interleave(8).repeat(32)
    f = torch.exp2(e.double())
    return (a*f[:, None]).to(dt).double(), (b*f[None, :]).to(dt).double()


def recombine(m):
    t = m[0]+m[5]
    u = t+m[6]
    return [m[0]+m[1], (t+m[4])+m[2], u-m[3], u+m[4]]


def output(main, correction, order):
    if order == 'leaf':
        leaf = list(main)
        for p, d in zip((2, 3, 4, 5, 6), correction):
            leaf[p] = leaf[p]+d
        c = recombine(leaf)
    else:
        c = recombine(main)
        for d, group in zip(correction, GROUPS):
            for j, sign in enumerate(SIGNS[group[0]]):
                if sign:
                    c[j] = c[j]+d if sign > 0 else c[j]-d
    def untile(x):
        return x.reshape(8, 16, 64, 32).permute(0, 2, 1, 3).reshape(512, 512)
    return torch.cat([torch.cat([untile(c[0]), untile(c[1])], 1),
                      torch.cat([untile(c[2]), untile(c[3])], 1)], 0).double()


def quantize(high, low, bits, group, side):
    if side == 'a':
        peak = high.reshape(512//group, group, 512).abs().amax((1, 2))
        peak = peak.repeat_interleave(group)[:, None]
    else:
        peak = high.reshape(512, 512//group, group).abs().amax((0, 2))
        peak = peak.repeat_interleave(group)[None, :]
    e = torch.ceil(torch.log2(torch.where(peak > 0, peak/127, torch.ones_like(peak))))
    sh = torch.exp2(e)
    sl = torch.exp2(e-bits)
    qh, ql = torch.round(high/sh), torch.round(low/sl)
    sat = sum(int((q.abs() > 127).sum().item()) for q in (qh, ql))
    nf = sum(int((~torch.isfinite(q)).sum().item()) for q in (high, low, qh, ql, sh, sl))
    return (qh.clamp(-127, 127), ql.clamp(-127, 127), sh, sl, sat, nf)


def scale_tile(x, side):
    if side == 'a':
        return x.reshape(8, 64)[:, None, :, None].expand(8, 16, 64, 1).reshape(128, 64, 1)
    return x.reshape(16, 32)[None, :, None, :].expand(8, 16, 1, 32).reshape(128, 1, 32)


def integer_corrections(ss, tt, sh, th, ar, br, bits, ga, gb):
    sq = [quantize(h, x-h, bits, ga, 'a') for x, h in zip(ss, sh)]
    tq = [quantize(h, x-h, bits, gb, 'b') for x, h in zip(tt, th)]
    aq = quantize(ar, torch.zeros_like(ar), bits, ga, 'a')
    bq = quantize(br, torch.zeros_like(br), bits, gb, 'b')
    allq = sq+tq+[aq, bq]
    sat = sum(x[4] for x in allq)
    nf = sum(x[5] for x in allq)
    pairs = [(sq[3], 1, bq, 0), (aq, 0, tq[3], 1)]
    for j in range(3):
        pairs.extend(((sq[j], 1, tq[j], 0), (sq[j], 0, tq[j], 1)))
    corrections = []
    max_integer = underflow_scales = 0
    for group in GROUPS:
        x, xi, y, yi = pairs[group[0]]
        integer = torch.bmm(tile_a(x[xi], 0, 512), tile_b(y[yi], 0, 512))
        scale = scale_tile(x[2+xi], 'a')*scale_tile(y[2+yi], 'b')
        if len(group) == 2:
            xx, xxi, yy, yyi = pairs[group[1]]
            assert SIGNS[group[0]] == SIGNS[group[1]]
            assert torch.equal(scale, scale_tile(xx[2+xxi], 'a')*scale_tile(yy[2+yyi], 'b'))
            integer = integer+torch.bmm(tile_a(xx[xxi], 0, 512), tile_b(yy[yyi], 0, 512))
        max_integer = max(max_integer, int(integer.abs().max().item()))
        assert max_integer <= BOUND
        assert torch.equal(integer, torch.round(integer))
        underflow_scales += int((scale == 0).sum().item())
        corrections.append(integer*scale)
    return corrections, dict(saturated_values=sat, nonfinite_correction_operands=nf,
                             max_absolute_integer=max_integer, zero_product_scales=underflow_scales)


def run(dt, pattern, seed):
    a, b = inputs(dt, pattern, seed)
    ref = a@b
    den = torch.linalg.vector_norm(ref)
    af, bf = a.float(), b.float()
    am, bm = af.abs().amax(0), bf.abs().amax(1)
    e = torch.where((am > 0) & (bm > 0), torch.round(.5*(torch.log2(am)-torch.log2(bm))), 0).int()
    cpu_e = torch.tensor([round(.5*(math.log2(x)-math.log2(y))) if x > 0 and y > 0 else 0
                          for x, y in zip(am.tolist(), bm.tolist())], device='cuda', dtype=torch.int32)
    inside_errors = int((e != cpu_e).sum().item())
    aa, bb = blocks(torch.ldexp(af, -e[None, :])), blocks(torch.ldexp(bf, e[:, None]))
    s1, t1 = aa[2]+aa[3], bb[1]-bb[0]
    s2, t2 = s1-aa[0], bb[3]-t1
    ss, tt = [s1, s2, aa[0]-aa[2], aa[1]-s2], [t1, t2, bb[3]-bb[1], t2-bb[2]]
    ah, bh = [x.to(dt).float() for x in aa], [x.to(dt).float() for x in bb]
    sh, th = [x.to(dt).float() for x in ss], [x.to(dt).float() for x in tt]
    native = ((ah[0], bh[0]), (ah[1], bh[2]), (sh[3], bh[3]), (ah[3], th[3]),
              (sh[0], th[0]), (sh[1], th[1]), (sh[2], th[2]))
    bits = 11 if dt == torch.float16 else 8
    main = [torch.zeros((128, 64, 32), device='cuda') for _ in native]
    nc = [torch.zeros_like(main[0]) for _ in GROUPS]
    nnf = 0
    for offset in range(0, 512, 128):
        for j, (x, y) in enumerate(native):
            main[j] = main[j]+torch.bmm(tile_a(x, offset, 128), tile_b(y, offset, 128))
        sa, ta = [tile_a(x, offset, 128) for x in sh], [tile_b(x, offset, 128) for x in th]
        sl = [(tile_a(x-h, offset, 128)*2.**bits).to(dt).float() for x, h in zip(ss, sh)]
        tl = [(tile_b(x-h, offset, 128)*2.**bits).to(dt).float() for x, h in zip(tt, th)]
        rr, rrb = tile_a(ah[3], offset, 128), tile_b(bh[3], offset, 128)
        nnf += sum(int((~torch.isfinite(x)).sum().item()) for x in sa+ta+sl+tl+[rr, rrb])
        pairs = [(sl[3], rrb), (rr, tl[3])]
        for j in range(3):
            pairs.extend(((sl[j], ta[j]), (sa[j], tl[j])))
        for j, group in enumerate(GROUPS):
            value = torch.bmm(*pairs[group[0]])
            if len(group) == 2:
                value = value+torch.bmm(*pairs[group[1]])
            nc[j] = nc[j]+value
    nc = [x*2.**-bits for x in nc]
    floor = 1e-4 if dt == torch.float16 else 1e-3
    dense_ef = float((torch.linalg.vector_norm((af@bf).double()-ref)/den).item())
    rows, outputs = [], {}
    for label, ga, gb in [('coarse', 64, 32), ('fine', 8, 8), ('native', None, None)]:
        if label == 'native':
            correction, stats = nc, dict(saturated_values=None, nonfinite_correction_operands=nnf,
                                         max_absolute_integer=None, zero_product_scales=None)
        else:
            correction, stats = integer_corrections(ss, tt, sh, th, ah[3], bh[3], bits, ga, gb)
        for order in (('leaf',) if label == 'native' else ('post', 'leaf')):
            out = output(main, correction, order)
            ef = float((torch.linalg.vector_norm(out-ref)/den).item())
            nf = int((~torch.isfinite(out)).sum().item())
            valid = nf == 0 and stats['nonfinite_correction_operands'] == 0 and inside_errors == 0
            valid = valid and stats['saturated_values'] in (None, 0)
            arm = label+'_'+order
            outputs[arm] = out
            rows.append(dict(dtype=str(dt).split('.')[-1], pattern=pattern, seed=seed, arm=arm,
                             quant_k=512 if label != 'native' else None, group_a=ga, group_b=gb,
                             ef=ef, dense_ef=dense_ef, gate=3*dense_ef+floor, floor=floor,
                             nonfinite=nf, inside_exponents=1024, inside_cpu_errors=inside_errors,
                             pass_original_gate=valid and ef <= 3*dense_ef+floor,
                             pass_fixed_floor=valid and ef <= floor, **stats))
    for row in rows:
        row['delta_from_native_leaf_ef'] = float((torch.linalg.vector_norm(
            outputs[row['arm']]-outputs['native_leaf'])/den).item())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, controls = [], []
    for dt in (torch.float16, torch.bfloat16):
        for pattern in PATTERNS:
            for seed in SEEDS:
                fresh = run(dt, pattern, seed)
                if pattern == 'uniform' and seed == SEEDS[0]:
                    old9 = [x for x in m9_control(dt, pattern, seed) if x['quant_k'] == 512][0]
                    old13 = [x for x in m13_control(dt, pattern, seed) if x['arm'] == 'native_leaf_scaled'][0]
                    by_arm = {x['arm']: x for x in fresh}
                    assert by_arm['coarse_post']['ef'] == old9['ef']
                    assert by_arm['coarse_post']['max_absolute_integer'] == old9['max_absolute_integer']
                    assert by_arm['native_leaf']['ef'] == old13['ef']
                    control = dict(dtype=str(dt), seed=seed, coarse_post_matches_m9=True,
                                   native_leaf_matches_m13=True)
                    controls.append(control)
                    print('CONTROL '+json.dumps(control), flush=True)
                rows.extend(fresh)
                for row in fresh:
                    print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(dict(scope=__doc__, torch=torch.__version__,
                    device=torch.cuda.get_device_name(), integer_absolute_bound=BOUND,
                    configurations=len(rows)//len(ARMS), rows=rows, controls=controls), indent=2)+'\n')
    return 0 if all(x['pass_fixed_floor'] and x['pass_original_gate'] for x in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())

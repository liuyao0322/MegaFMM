"""M9: widen only INT8 quantization/accumulation K, retaining per-column inside.

Three controlled widths128/256/512 on the same fresh original rounded inputs.
Native leaves always accumulate four FP32 modeled K128 slabs, independent of arm.
Each correction pair is accumulated while integer across its whole quantization
slab before one FP32 rescale. Not hardware, synchronization or latency evidence.
"""
import argparse
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablate import inputs as original_inputs, blocks, SIGNS
from alternating_inside_model import inputs as alternating_inputs
from global_k128_model import quant_pair
from paired_column_model import run as reference_control

GROUPS = ((0,), (1,), (2, 3), (4, 5), (6, 7))
PATTERNS = ('uniform', 'positive', 'range16', 'cancellation', 'block_scaled',
            'bf16_range', 'inside_mismatch', 'alternating_inside_mismatch',
            'balanced_k_bands', 'cancellation_k_bands')


def tile_a(x, offset, width):
    return (x[:, offset:offset+width].reshape(8, 64, width)[:, None]
            .expand(8, 16, 64, width).reshape(128, 64, width))


def tile_b(x, offset, width):
    return (x[offset:offset+width, :].reshape(width, 16, 32)
            .permute(1, 0, 2)[None].expand(8, 16, width, 32)
            .reshape(128, width, 32))


def run(dt, pattern, seed):
    if pattern == 'alternating_inside_mismatch':
        a, b = alternating_inputs(1024, seed, pattern, dt)
    elif pattern in ('balanced_k_bands', 'cancellation_k_bands'):
        base_pattern = 'cancellation' if pattern.startswith('cancellation') else 'uniform'
        a, b = original_inputs(1024, seed, base_pattern, dt)
        powers = torch.tensor([0, 4, -4, 2], device='cuda').repeat_interleave(128).repeat(2)
        factor = torch.exp2(powers.double())
        a = (a*factor[None, :]).to(dt).double()
        b = (b*factor[:, None]).to(dt).double()
    else:
        a, b = original_inputs(1024, seed, pattern, dt)
    ref = a@b
    af, bf = a.float(), b.float()
    am, bm = af.abs().amax(0), bf.abs().amax(1)
    exponent = torch.where((am > 0) & (bm > 0),
                           torch.round(.5*(torch.log2(am)-torch.log2(bm))), 0).int()
    aa = blocks(torch.ldexp(af, -exponent[None, :]))
    bb = blocks(torch.ldexp(bf, exponent[:, None]))
    s1, t1 = aa[2]+aa[3], bb[1]-bb[0]
    s2, t2 = s1-aa[0], bb[3]-t1
    ss, tt = [s1, s2, aa[0]-aa[2], aa[1]-s2], [t1, t2, bb[3]-bb[1], t2-bb[2]]
    ah, bh = [x.to(dt).float() for x in aa], [x.to(dt).float() for x in bb]
    sh, th = [x.to(dt).float() for x in ss], [x.to(dt).float() for x in tt]
    native = ((ah[0], bh[0]), (ah[1], bh[2]), (sh[3], bh[3]), (ah[3], th[3]),
              (sh[0], th[0]), (sh[1], th[1]), (sh[2], th[2]))
    main = [torch.zeros((128, 64, 32), device='cuda') for _ in native]
    for offset in range(0, 512, 128):
        for j, (x, y) in enumerate(native):
            main[j] = main[j]+torch.bmm(tile_a(x, offset, 128), tile_b(y, offset, 128))
    u2 = main[0]+main[5]
    u3 = u2+main[6]
    native_output = [main[0]+main[1], (u2+main[4])+main[2], u3-main[3], u3+main[4]]
    rows = []
    for width in (128, 256, 512):
        correction = [torch.zeros_like(main[0]) for _ in GROUPS]
        saturated = max_integer = 0
        bits = 11 if dt == torch.float16 else 8
        for offset in range(0, 512, width):
            sq = [quant_pair(tile_a(h, offset, width), tile_a(x-h, offset, width), bits, True)
                  for x, h in zip(ss, sh)]
            tq = [quant_pair(tile_b(h, offset, width), tile_b(x-h, offset, width), bits, True)
                  for x, h in zip(tt, th)]
            ar = tile_a(ah[3], offset, width)
            br = tile_b(bh[3], offset, width)
            ar = quant_pair(ar, torch.zeros_like(ar), bits, False)
            br = quant_pair(br, torch.zeros_like(br), bits, False)
            saturated += sum(x[4] for x in sq+tq)
            pairs = [(sq[3], 1, br, 0), (ar, 0, tq[3], 1)]
            for j in range(3):
                pairs.extend(((sq[j], 1, tq[j], 0), (sq[j], 0, tq[j], 1)))
            for j, group in enumerate(GROUPS):
                x, xi, y, yi = pairs[group[0]]
                integer = torch.bmm(x[xi], y[yi])
                scale = x[2+xi]*y[2+yi]
                if len(group) == 2:
                    xx, xxi, yy, yyi = pairs[group[1]]
                    assert SIGNS[group[0]] == SIGNS[group[1]]
                    assert torch.equal(scale, xx[2+xxi]*yy[2+yyi])
                    integer = integer+torch.bmm(xx[xxi], yy[yyi])
                max_integer = max(max_integer, int(integer.abs().max().item()))
                assert max_integer <= 2*width*127**2 < 2**24
                correction[j] = correction[j]+integer*scale
        c = list(native_output)
        for d, group in zip(correction, GROUPS):
            for j, sign in enumerate(SIGNS[group[0]]):
                if sign:
                    c[j] = c[j]+d if sign > 0 else c[j]-d
        def untile(x):
            return x.reshape(8, 16, 64, 32).permute(0, 2, 1, 3).reshape(512, 512)
        out = torch.cat([torch.cat([untile(c[0]), untile(c[1])], 1),
                         torch.cat([untile(c[2]), untile(c[3])], 1)], 0).double()
        ef = float((torch.linalg.vector_norm(out-ref)/torch.linalg.vector_norm(ref)).item())
        nonfinite = int((~torch.isfinite(out)).sum().item())
        floor = 1e-4 if dt == torch.float16 else 1e-3
        rows.append(dict(dtype=str(dt).split('.')[-1], pattern=pattern, seed=seed,
                         quant_k=width, block_a=[64, width], block_b=[width, 32],
                         inside='per_column', ef=ef, nonfinite=nonfinite,
                         saturated_low_values=saturated, max_absolute_integer=max_integer,
                         floor=floor, pass_fixed_floor=nonfinite == 0 and ef <= floor))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    for dt in (torch.float16, torch.bfloat16):
        for pattern in PATTERNS:
            for seed in (4801, 4802, 4803):
                fresh = run(dt, pattern, seed)
                if pattern == 'uniform' and seed == 4801:
                    control = [x for x in reference_control(dt, pattern, seed)
                               if x['arm'] == 'relative_paired'][0]
                    assert fresh[0]['ef'] == control['ef']
                    assert fresh[0]['max_absolute_integer'] == control['max_absolute_paired_integer']
                    print('CONTROL '+json.dumps(dict(dtype=str(dt), seed=seed,
                          same_input=True, quant_k128_equals_m8=True)), flush=True)
                rows.extend(fresh)
                for row in fresh:
                    print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(dict(scope=__doc__, rows=rows), indent=2)+'\n')
    return 0 if all(x['pass_fixed_floor'] for x in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())

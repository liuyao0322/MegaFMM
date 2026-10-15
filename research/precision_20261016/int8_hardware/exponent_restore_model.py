"""M16: paired product versus combined-exponent restoration; numerical only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import fine_quant_model as m15

SEEDS = (124139, 124140, 124141)
original_integer_corrections = m15.integer_corrections


def integer_corrections(mode):
    def evaluate(ss, tt, sh, th, ar, br, bits, ga, gb):
        sq = [m15.quantize(h, x-h, bits, ga, 'a') for x, h in zip(ss, sh)]
        tq = [m15.quantize(h, x-h, bits, gb, 'b') for x, h in zip(tt, th)]
        aq = m15.quantize(ar, torch.zeros_like(ar), bits, ga, 'a')
        bq = m15.quantize(br, torch.zeros_like(br), bits, gb, 'b')
        pairs = [(sq[3], 1, bq, 0), (aq, 0, tq[3], 1)]
        for j in range(3):
            pairs.extend(((sq[j], 1, tq[j], 0), (sq[j], 0, tq[j], 1)))
        allq = sq+tq+[aq, bq]
        stats = dict(saturated_values=sum(x[4] for x in allq),
                     nonfinite_correction_operands=sum(x[5] for x in allq),
                     max_absolute_integer=0, zero_product_scales=0,
                     nonzero_integer_at_zero_scale=0, lost_representable_nonzero=0,
                     correction_bit_differences=0)
        corrections = []
        for group in m15.GROUPS:
            x, xi, y, yi = pairs[group[0]]
            integer = torch.bmm(m15.tile_a(x[xi], 0, 512), m15.tile_b(y[yi], 0, 512))
            sa, sb = m15.scale_tile(x[2+xi], 'a'), m15.scale_tile(y[2+yi], 'b')
            ea, eb = torch.log2(sa).int(), torch.log2(sb).int()
            assert torch.equal(torch.exp2(ea.float()), sa) and torch.equal(torch.exp2(eb.float()), sb)
            e = ea+eb
            if len(group) == 2:
                xx, xxi, yy, yyi = pairs[group[1]]
                assert torch.equal(e, torch.log2(m15.scale_tile(xx[2+xxi], 'a')).int()+
                                   torch.log2(m15.scale_tile(yy[2+yyi], 'b')).int())
                integer = integer+torch.bmm(m15.tile_a(xx[xxi], 0, 512), m15.tile_b(yy[yyi], 0, 512))
            maximum = int(integer.abs().max().item())
            assert maximum <= m15.BOUND and torch.equal(integer, torch.round(integer))
            assert int(e.min().item()) >= -500 and int(e.max().item()) <= 500
            scale = sa*sb
            product = integer*scale
            combined = torch.ldexp(integer.double(), e).float()
            stats['max_absolute_integer'] = max(stats['max_absolute_integer'], maximum)
            stats['zero_product_scales'] += int((scale == 0).sum().item())
            stats['nonzero_integer_at_zero_scale'] += int(((scale == 0) & (integer != 0)).sum().item())
            stats['lost_representable_nonzero'] += int(((product == 0) & (combined != 0) & torch.isfinite(combined)).sum().item())
            stats['correction_bit_differences'] += int((product.contiguous().view(torch.int32) != combined.contiguous().view(torch.int32)).sum().item())
            corrections.append(product if mode == 'product' else combined)
        return corrections, stats
    return evaluate


def scalar_control():
    integers = np.array([0, 1, -1, 127, -127, 65535, -65535, m15.BOUND, -m15.BOUND], dtype=np.float64)
    exponents = np.arange(-300, 301, dtype=np.int32)
    with np.errstate(over='ignore', under='ignore'):
        expected = np.ldexp(integers[:, None], exponents[None, :]).astype(np.float32)
    actual = torch.ldexp(torch.tensor(integers, device='cuda')[:, None],
                         torch.tensor(exponents, device='cuda')[None, :]).float().cpu().numpy()
    errors = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
    integer = torch.tensor(2.**20, device='cuda', dtype=torch.float32)
    e = torch.tensor(-152, device='cuda', dtype=torch.int32)
    old = integer*torch.exp2(e.float())
    new = torch.ldexp(integer.double(), e).float()
    assert errors == 0 and old.item() == 0 and new.item() == 2.**-132
    return dict(cases=actual.size, bit_errors=errors, combined_model='FP64 exact power-of-two scaling then FP32',
                negative_product_value=old.item(), negative_combined_value=new.item(), native_cuda_operation_tested=False)


def paired_run(dt, pattern, seed, replay, controls):
    rows = []
    for mode in ['product', 'exponent']:
        m15.integer_corrections = integer_corrections(mode)
        try:
            fresh = m15.run(dt, pattern, seed)
        finally:
            m15.integer_corrections = original_integer_corrections
        if mode == 'product' and pattern == 'uniform' and seed == SEEDS[0]:
            old = m15.run(dt, pattern, seed)
            assert all(all(a[k] == b[k] for k in ['arm', 'ef', 'max_absolute_integer', 'zero_product_scales']) for a, b in zip(fresh, old))
            controls.append(dict(dtype=str(dt), seed=seed, product_matches_unmodified_m15=True))
        if mode == 'product' and replay:
            old = [r for r in json.loads((Path(__file__).parent/'fine_quant_model.json').read_text())['rows']
                   if r['dtype'] == 'bfloat16' and r['pattern'] == pattern and r['seed'] == seed]
            assert len(old) == 5 and all(all(a[k] == b[k] for k in ['arm', 'ef', 'zero_product_scales']) for a, b in zip(fresh, old))
            controls.append(dict(dtype=str(dt), seed=seed, product_replay_matches_archived_m15=True))
        for row in fresh:
            row.update(restoration=mode, replayed_m15_input=replay)
        rows.extend(fresh)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scalar = scalar_control()
    print('SCALAR '+json.dumps(scalar), flush=True)
    rows, controls = [], []
    configurations = [(dt, pattern, seed, False) for dt in (torch.float16, torch.bfloat16)
                      for pattern in m15.PATTERNS for seed in SEEDS]
    configurations.append((torch.bfloat16, 'tiny_cancellation', 123107, True))
    for dt, pattern, seed, replay in configurations:
        fresh = paired_run(dt, pattern, seed, replay, controls)
        rows.extend(fresh)
        for row in fresh:
            print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(dict(scope=__doc__, scalar=scalar, controls=controls,
            fresh_configurations=84, replayed_configurations=1, rows=rows), indent=2)+'\n')
    assert len(rows) == 850 and len(controls) == 3
    return 0 if all(r['pass_original_gate'] and
                    (r['pass_fixed_floor'] or r['arm'].startswith('coarse')) for r in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())

"""Freeze M12's paired numerical screen, independently of hardware acceptance."""
import hashlib
import json
from pathlib import Path

h = Path(__file__).resolve().parent
repo = h.parents[2]
rows = json.loads((h / 'native_correction_model.json').read_text())['rows']
assert len(rows) == 216
keys = {(x['dtype'], x['pattern'], x['seed']) for x in rows}
assert len(keys) == 72 and {x['seed'] for x in rows} == {7301, 7302, 7303}
log = (h / 'native_correction_model.log').read_text().splitlines()
controls = [json.loads(x.removeprefix('CONTROL ')) for x in log if x.startswith('CONTROL ')]
assert len(controls) == 2 and all(x['int8_and_bf16_match_m11'] for x in controls)
negative = [json.loads(x.removeprefix('RANGE_CONTROL ')) for x in log if x.startswith('RANGE_CONTROL ')]
assert len(negative) == 1 and all(negative[0].values())
arms = ['int8_k128', 'bf16_low_unscaled', 'native_low_scaled']
summary = []
for dt in ['float16', 'bfloat16']:
    for arm in arms:
        r = [x for x in rows if x['dtype'] == dt and x['arm'] == arm]
        assert len(r) == 36 and len({(x['pattern'], x['seed']) for x in r}) == 36
        assert all(x['floor'] == (1e-4 if dt == 'float16' else 1e-3) for x in r)
        worst = max(r, key=lambda x: x['ef'])
        summary.append(dict(dtype=dt, arm=arm, configurations=36,
            passes=sum(x['pass_fixed_floor'] for x in r),
            nonfinite=sum(x['nonfinite'] for x in r),
            nonfinite_correction_operand_observations=sum(x['nonfinite_correction_operands'] for x in r),
            max_ef=worst['ef'], worst_pattern=worst['pattern'], worst_seed=worst['seed']))
paired = []
for dt in ['float16', 'bfloat16']:
    ratios = []
    scale_differences = []
    for key in sorted(k for k in keys if k[0] == dt):
        r = {x['arm']: x for x in rows if (x['dtype'], x['pattern'], x['seed']) == key}
        assert set(r) == set(arms)
        assert r['int8_k128']['saturated_low_values'] == 0
        assert r['int8_k128']['max_absolute_integer'] < 2**22
        ratios.append(r['int8_k128']['ef'] / r['bf16_low_unscaled']['ef'])
        scale_differences.append(abs(r['native_low_scaled']['ef'] - r['bf16_low_unscaled']['ef']))
    paired.append(dict(dtype=dt, paired_configurations=len(ratios),
        unscaled_lower_error_configurations=sum(x > 1 for x in ratios),
        baseline_to_unscaled_ef_ratio_min=min(ratios),
        baseline_to_unscaled_ef_ratio_max=max(ratios),
        max_absolute_ef_difference_native_vs_bf16=max(scale_differences)))
scalar = [json.loads(x.removeprefix('SCALAR_RANGE ')) for x in log if x.startswith('SCALAR_RANGE ')]
assert len(scalar) == 2 and sum(x['sampled_transforms'] for x in scalar) == 515072
assert all(x['scaled_low_nonfinite'] == 0 and x['excluded_nonfinite_native_high'] == 4 for x in scalar)
out = dict(scope='72 fresh matrix configurations, three paired numerical arms; FP32 bmm/add with TF32 disabled, not exact Hopper WGMMA rounding or hardware acceptance',
    configurations=72, paired_arm_rows=216, summary=summary, paired=paired,
    same_input_controls=controls, scalar_range_controls=scalar, fixed16_vs_native8_range_control=negative[0],
    range_limit='Native p11/8 scale avoids fixed16 finite-high residual overflow; no underflow or arbitrary transform/native-high finiteness theorem. Midpoint-neighbor sampling is not exhaustive FP32 interior.',
    hardware_accepted=False, performance_accepted=False)
(h / 'm12_summary.json').write_text(json.dumps(out, indent=2) + '\n')
files = ['native_correction_model.py', 'native_correction_model.json', 'native_correction_model.log',
    'summarize_native_correction_model.py', 'generate_native_correction_model.py', 'm12_summary.json', 'NATIVE_CORRECTION_DESIGN.md', 'bf16_correction_model.py', 'm11_manifest.json',
    'wide_quant_model.py', 'paired_column_model.py', 'alternating_inside_model.py',
    'global_k128_model.py', '../ablate.py']
(h / 'm12_manifest.json').write_text(json.dumps({str((h / f).relative_to(repo)):
    hashlib.sha256((h / f).read_bytes()).hexdigest() for f in files}, indent=2) + '\n')
print(json.dumps(out, indent=2))

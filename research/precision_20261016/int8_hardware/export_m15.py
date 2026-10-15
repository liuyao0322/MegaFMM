"""Freeze M15 including its six stricter-floor negative rows; no speed claim."""
from pathlib import Path
import hashlib
import json

h = Path(__file__).resolve().parent
repo = h.parents[2]
j = json.loads((h/'fine_quant_model.json').read_text())
rows = j['rows']
assert j['configurations'] == 84 and len(rows) == 420
assert len({(r['dtype'], r['pattern'], r['seed']) for r in rows}) == 84
assert len(j['controls']) == 2 and all(r['coarse_post_matches_m9'] and r['native_leaf_matches_m13'] for r in j['controls'])
assert all(r['pass_original_gate'] and r['nonfinite'] == 0 and r['nonfinite_correction_operands'] == 0 and r['inside_cpu_errors'] == 0 for r in rows)
assert all(r['saturated_values'] in (None, 0) for r in rows)
summary = []
for dt in ['float16', 'bfloat16']:
    for arm in ['coarse_post', 'coarse_leaf', 'fine_post', 'fine_leaf', 'native_leaf']:
        group = [r for r in rows if r['dtype'] == dt and r['arm'] == arm]
        assert len(group) == 42
        expected = 39 if dt == 'float16' and arm.startswith('coarse') else 42
        assert sum(r['pass_fixed_floor'] for r in group) == expected
        worst = max(group, key=lambda r: r['ef'])
        summary.append(dict(dtype=dt, arm=arm, configurations=42,
            original_gate_passes=42, fixed_floor_passes=expected, max_ef=worst['ef'],
            worst_pattern=worst['pattern'], worst_seed=worst['seed'],
            max_integer=max((r['max_absolute_integer'] or 0) for r in group),
            zero_product_scales=sum(r['zero_product_scales'] or 0 for r in group)))
negative = [r for r in rows if not r['pass_fixed_floor']]
assert len(negative) == 6 and all(r['dtype'] == 'float16' and r['pattern'] == 'rowcol_cancellation' and r['arm'].startswith('coarse') for r in negative)
paired = []
for dt in ['float16', 'bfloat16']:
    for order in ['post', 'leaf']:
        c = {(r['pattern'], r['seed']): r for r in rows if r['dtype'] == dt and r['arm'] == 'coarse_'+order}
        f = {(r['pattern'], r['seed']): r for r in rows if r['dtype'] == dt and r['arm'] == 'fine_'+order}
        changes = [f[k]['ef']-c[k]['ef'] for k in c]
        paired.append(dict(dtype=dt, order=order, lower_error=sum(x < 0 for x in changes),
                           equal_error=sum(x == 0 for x in changes), higher_error=sum(x > 0 for x in changes),
                           min_ef_change=min(changes), max_ef_change=max(changes)))
out = dict(scope=j['scope'], configurations=84, rows=420, model_exit_code=2,
           exit_reason='Six coarse FP16 stricter-fixed-floor negative rows, all original gates pass; full run completed, not a crash.',
           summary=summary, paired_geometry=paired, negative_fixed_floor_rows=negative,
           fine_original_and_fixed_passes=84, inside_cpu_exponents=84*1024,
           inside_cpu_errors=0, integer_absolute_bound=j['integer_absolute_bound'],
           goal_achieved=False, hardware_accepted=False, performance_measured=False,
           limitation='BF16 tiny_cancellation seed123107 has zero rounded scale products; count alone does not establish whether an omitted correction was nonzero. Requires targeted restoration ablation before hardware design.')
(h/'m15_summary.json').write_text(json.dumps(out, indent=2)+'\n')
files = ['fine_quant_model.py', 'fine_quant_model.json', 'fine_quant_model.log',
         'run_fine_quant_model.sh', 'FINE_QUANT_SCREEN_DESIGN.md', 'm15_premeasurement_manifest.json',
         'export_m15.py', 'm15_summary.json', 'FINE_QUANT_SCREEN_REPORT.md']
(h/'m15_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):
    hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files}, indent=2)+'\n')
print(json.dumps(dict(configurations=84, rows=420, original_passes=420,
                     fixed_floor_negative_rows=6, fine_passes=84, hardware_accepted=False)))

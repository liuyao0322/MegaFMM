"""Freeze M16's negative causal result without claiming a native CUDA fix."""
from pathlib import Path
import hashlib
import json

h = Path(__file__).resolve().parent
repo = h.parents[2]
j = json.loads((h/'exponent_restore_model.json').read_text())
rows = j['rows']
assert len(rows) == 850 and j['fresh_configurations'] == 84 and j['replayed_configurations'] == 1
assert j['scalar']['cases'] == 5409 and j['scalar']['bit_errors'] == 0
assert j['scalar']['negative_product_value'] == 0 and j['scalar']['negative_combined_value'] == 2.**-132
assert len(j['controls']) == 3
assert all(r['pass_original_gate'] and r['nonfinite'] == 0 and r['nonfinite_correction_operands'] == 0 and r['inside_cpu_errors'] == 0 for r in rows)
assert all(r['saturated_values'] in (None, 0) for r in rows)
assert all(r['pass_fixed_floor'] for r in rows if not r['arm'].startswith('coarse'))
assert all(r.get('nonzero_integer_at_zero_scale', 0) == 0 and r.get('lost_representable_nonzero', 0) == 0 and r.get('correction_bit_differences', 0) == 0 for r in rows)
key = lambda r: (r['dtype'], r['pattern'], r['seed'], r['arm'])
old = {key(r): r for r in rows if r['restoration'] == 'product'}
new = {key(r): r for r in rows if r['restoration'] == 'exponent'}
assert len(old) == len(new) == 425 and old.keys() == new.keys()
assert all(old[k]['ef'] == new[k]['ef'] and old[k]['delta_from_native_leaf_ef'] == new[k]['delta_from_native_leaf_ef'] for k in old)
summary = []
for dt in ['float16', 'bfloat16']:
    for arm in ['coarse_post', 'coarse_leaf', 'fine_post', 'fine_leaf', 'native_leaf']:
        rs = [r for r in rows if r['dtype'] == dt and r['arm'] == arm and r['restoration'] == 'exponent' and not r['replayed_m15_input']]
        assert len(rs) == 42
        worst = max(rs, key=lambda r: r['ef'])
        summary.append(dict(dtype=dt, arm=arm, fresh_configurations=42, original_gate_passes=42,
            fixed_floor_passes=sum(r['pass_fixed_floor'] for r in rs), max_ef=worst['ef'],
            worst_pattern=worst['pattern'], worst_seed=worst['seed']))
out = dict(scope=j['scope'], fresh_configurations=84, replayed_configurations=1, rows=850,
           scalar=j['scalar'], summary=summary, model_exit_code=0, controls=j['controls'],
           paired_arm_ef_equalities=425, nonzero_integer_at_zero_product_scale=0,
           lost_representable_nonzero=0, correction_bit_differences=0,
           inside_cpu_exponents=85*1024, inside_cpu_errors=0,
           fine_original_and_fixed_fresh_passes=84,
           conclusion='Combined exponents protect a constructed scalar case, but no tested matrix correction or eF changed. M15 zero scale products coincide with zero integer accumulations; not the measured matrix error source.',
           goal_achieved=False, hardware_accepted=False, performance_measured=False)
(h/'m16_summary.json').write_text(json.dumps(out, indent=2)+'\n')
files = ['EXPONENT_RESTORE_SCREEN_DESIGN.md', 'exponent_restore_model.py',
         'run_exponent_restore_model.sh', 'exponent_restore_model.log', 'exponent_restore_model.json',
         'm16_premeasurement_manifest.json', 'export_m16.py', 'm16_summary.json',
         'EXPONENT_RESTORE_SCREEN_REPORT.md']
(h/'m16_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):
    hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files}, indent=2)+'\n')
print(json.dumps(dict(fresh_configurations=84, replay=1, rows=850, scalar_bit_errors=0,
                     paired_arm_ef_equalities=425, correction_bit_differences=0, hardware_accepted=False)))

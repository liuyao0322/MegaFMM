#!/usr/bin/env python3
"""Measure the complete MNK suite repeatedly; never count a fallback as a win.

The registry is evidence, not an automatically trusted runtime cache. Plan::tune
still validates and times candidates against pointers on the current device.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--shapes', help='M,N,K;...; default: all bench/shapes.json entries')
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--rotate-inputs', type=int, default=2)
p.add_argument('--margin', type=float, default=0.03)
p.add_argument('--graph', action='store_true')
p.add_argument('--output-dir', type=Path)
a = p.parse_args()
if a.repeats < 2 or a.rotate_inputs < 1 or a.margin < 0:
    p.error('requires repeats>=2, rotate-inputs>=1, margin>=0')
spec = json.loads((ROOT / 'bench/shapes.json').read_text())
shapes = a.shapes or ';'.join(','.join(map(str, s)) for group in
                            ('squares', 'rectangles', 'boundaries') for s in spec[group])
dest = a.output_dir or ROOT / 'results' / f'mnk_acceptance_{time.strftime("%Y%m%d_%H%M%S")}'
dest = dest.resolve()
dest.mkdir(parents=True, exist_ok=True)
def binary_digest():
    return hashlib.sha256((ROOT / 'build/megafmm_bench').read_bytes()).hexdigest()

binary_sha256 = binary_digest()
source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
source_sha256 = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in list((ROOT / 'solution').glob('*')) + [ROOT / 'include/megafmm/gemm.h'] if f.is_file()}
observations = {}
processes = []
pattern = re.compile(r'TIME shape=(\d+),(\d+),(\d+) dtype=(\w+) group=(\d+) case=(dense|solution) event_ms=([\deE.+-]+)')
for repeat in range(a.repeats):
    if binary_digest() != binary_sha256:
        raise RuntimeError('benchmark binary changed during acceptance; do not combine different builds')
    csv_path = dest / f'run{repeat}.csv'
    log_path = dest / f'run{repeat}.log'
    cmd = ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench',
           '--algorithm', 'tuned', '--shapes', shapes, '--dtype', 'both',
           '--trials', str(spec['trials']), '--warmup', str(spec['warmup']),
           '--groups', str(spec['groups']), '--correct-trials', str(spec['correctness_trials']),
           '--rotate-inputs', str(a.rotate_inputs), '--output', str(csv_path)]
    if a.graph:
        cmd.append('--graph')
    print(f'Running independent sweep {repeat+1}/{a.repeats}: {log_path}', flush=True)
    with log_path.open('w') as out:
        result = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    processes.append({'repeat': repeat, 'exit_code': result.returncode, 'command': cmd})
    if result.returncode not in (0, 2):
        break
    groups = {}
    for m, n, k, dtype, group, case, elapsed in pattern.findall(log_path.read_text()):
        key = (m, n, k, dtype)
        groups.setdefault(key, {}).setdefault(group, {})[case] = float(elapsed)
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            key = tuple(row[k] for k in ('m', 'n', 'k', 'dtype'))
            config = row['solution_lt_config']
            selection = config.split('selection=', 1)[1].split(';', 1)[0]
            pairs = [v['dense']/v['solution'] for v in groups.get(key, {}).values()
                     if 'dense' in v and 'solution' in v and v['solution'] > 0]
            record = {
                'repeat': repeat, 'selection': selection, 'actual': row['actual'],
                'numerical_pass': row['numerical_pass'] == '1',
                'correct_trials': int(row['correct_trials']),
                'median_speedup': float(row['speedup']),
                'paired_speedups': pairs,
                'solution_ms': float(row['solution_event_ms']),
                'dense_ms': float(row['dense_event_ms']),
                'config': config,
            }
            observations.setdefault(key, []).append(record)
    print(f'Sweep {repeat+1}: exit={result.returncode}', flush=True)

entries = []
for key, rows in observations.items():
    pairs = [v for r in rows for v in r['paired_speedups']]
    precision = all(r['numerical_pass'] and r['correct_trials'] >= spec['correctness_trials'] for r in rows)
    custom = all(r['selection'] != 'dense_cublaslt' for r in rows)
    complete = len(rows) == a.repeats and len(pairs) == a.repeats * spec['groups']
    finite = all(math.isfinite(x) for x in pairs)
    stable = complete and precision and custom and finite and min(pairs) >= 1 + a.margin
    entry = dict(zip(('m', 'n', 'k'), map(int, key[:3])), dtype=key[3],
                 status='stable_win' if stable else 'unresolved',
                 complete=complete, precision_pass=precision, custom_every_repeat=custom,
                 min_paired_speedup=min(pairs) if pairs else None,
                 median_paired_speedup=statistics.median(pairs) if pairs else None,
                 observations=rows)
    entries.append(entry)
expected = 2 * len(shapes.split(';'))
wins = sum(e['status'] == 'stable_win' for e in entries)
report = {
    'binary_sha256': binary_sha256, 'source_commit': source_commit, 'source_sha256': source_sha256,
    'schema_version': 1, 'shapes': shapes, 'mode': 'graph' if a.graph else 'launch',
    'rotate_inputs': a.rotate_inputs, 'repeats_required': a.repeats,
    'trials': spec['trials'], 'groups': spec['groups'],
    'correctness_trials': spec['correctness_trials'], 'margin': a.margin,
    'expected_combinations': expected, 'observed_combinations': len(entries),
    'stable_wins': wins, 'objective_achieved': wins == expected and len(entries) == expected,
    'processes': processes, 'entries': entries,
}
(dest / 'registry.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(f'{wins}/{expected} stable custom-kernel wins. Evidence: {dest / "registry.json"}', flush=True)
for e in entries:
    print(f"{e['m']}x{e['n']}x{e['k']} {e['dtype']}: {e['status']} "
          f"min_paired_speedup={e['min_paired_speedup']}", flush=True)
raise SystemExit(0 if report['objective_achieved'] else 2)

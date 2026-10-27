#!/usr/bin/env python3
"""Audit a public strict SWTuned shape sweep and retain a readable report."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--result-dir', required=True, type=Path)
    parser.add_argument('--build-dir', required=True, type=Path)
    parser.add_argument('--append-iteration', type=int)
    args = parser.parse_args()
    root = args.result_dir
    config = json.loads(args.config.read_text())
    shapes = [tuple(s) for group in ('squares', 'rectangles', 'boundaries')
              for s in config[group]]
    rows = list(csv.DictReader((root / 'current.csv').open()))
    log = (root / 'driver.log').read_text()
    errors = []
    if (root / 'exit.status').read_text().strip() != '0':
        errors.append('benchmark process failed')
    attempts = json.loads((root / 'attempts.json').read_text()) if (root / 'attempts.json').exists() else None
    rejections = [r for r in (attempts or []) if r['status'] == 'precision_rejected']
    expected = [s for s in shapes if list(s) not in [r['shape'] for r in rejections]]
    if attempts and ([tuple(r['shape']) for r in attempts] != shapes or any(r['status'] == 'error' for r in attempts)):
        errors.append('not all declared shapes have an accepted or precision-rejected outcome')
    if [tuple(int(r[d]) for d in ('m', 'n', 'k')) for r in rows] != expected:
        errors.append('CSV shapes do not match the declared suite in order')
    if len(re.findall(r'^CHECK ', log, re.M)) != 3 * len(rows):
        errors.append('incorrect fresh correctness-check count')
    if len(re.findall(r'^TIME ', log, re.M)) != 10 * len(rows):
        errors.append('incorrect timing group count')
    if re.search(r'^CHECK .* pass=0', log, re.M):
        errors.append('at least one fresh numerical check failed')
    cache = (args.build_dir / 'CMakeCache.txt').read_text()
    if not re.search(r'^MEGAFMM_FP32_CONFIG_HEADER:[^=]*=\s*$', cache, re.M):
        errors.append('build does not use the default template registry')
    data = []
    for row in rows:
        shape = tuple(int(row[d]) for d in ('m', 'n', 'k'))
        if any(row[name] != '1' for name in ('numerical_pass', 'precision_required', 'accepted')):
            errors.append(f'{shape}: numerical gate not accepted')
        if row['nonfinite'] != '0':
            errors.append(f'{shape}: nonfinite output')
        if (row['dtype'], row['requested'], row['actual'], row['mode'], row['rotate_inputs'],
                row['groups'], row['trials'], row['correct_trials']) != (
                'fp32', 'sw-tuned-strict', 'sw_tuned', 'launch', '2', '5', '100', '3'):
            errors.append(f'{shape}: unexpected benchmark protocol')
        selected = re.search(r'selected_id[=:](\d+)', row['solution_lt_config'])
        dense = 1000 * float(row['dense_event_ms'])
        solution = 1000 * float(row['solution_event_ms'])
        data.append(dict(m=shape[0], n=shape[1], k=shape[2], solution_us=solution,
                         cublaslt_us=dense, speedup=dense / solution,
                         selected_id=int(selected[1]) if selected else None,
                         solution_cv=float(row['solution_cv']), dense_cv=float(row['dense_cv'])))
    tuning = list(csv.DictReader((root / 'current.csv.sw_tuning.csv').open()))
    for rejection in rejections:
        rejected_rows = [r for r in tuning if [int(r[d]) for d in ('m', 'n', 'k')] == rejection['shape']]
        if len(rejected_rows) != 13 or any(r['status'] != 'precision_rejected' or r['numerical_pass'] != '0' for r in rejected_rows):
            errors.append(f'{rejection["shape"]}: incomplete strict precision rejection evidence')
    group_data = {}
    for label, selected in [('all', data),
                            *[(f'K={k}', [r for r in data if r['k'] == k])
                              for k in sorted({r['k'] for r in data})]]:
        group_data[label] = dict(count=len(selected), wins=sum(r['speedup'] > 1 for r in selected),
                                geometric_mean=math.exp(statistics.mean(math.log(r['speedup']) for r in selected)),
                                minimum=min(r['speedup'] for r in selected), maximum=max(r['speedup'] for r in selected))
    source_files = sorted(p for p in Path('solution').rglob('*') if p.is_file())
    audit = dict(config=str(args.config), shape_count=len(shapes), measured_rows=len(rows),
                 errors=errors, passed=not errors, groups=group_data, precision_rejections=rejections,
                 protocol=dict(warmup=config['warmup'], groups=5, trials=100, correctness_trials=3,
                               rotate_inputs=2, algorithm='sw-tuned-strict', registry='default',
                               statistic='median of five 100-call group means', timing='CUDA event, complete ordinary launch'),
                 source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                 binary_sha256=hashlib.sha256((args.build_dir / 'megafmm_bench').read_bytes()).hexdigest(),
                 limitations=['One measurement process per new shape; this is a coverage sweep, not independent repeat certification.',
                              'Reports the public tuner choice from the unchanged default registry, not the best direct candidate.',
                              'Only the additional shapes were measured in this run; previous original-suite measurements are separate.'])
    (root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    with (root / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]) if data else [], lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)
    lines = ['# FP32 additional-shape measurement', '',
             'Fixed default kernel registry; public strict SWTuned API versus same-run cuBLASLt.',
             'CUDA-event full-call latency includes input packing and output reconstruction; tune/setup are excluded.',
             '20 warmups, 5 groups × 100 calls, 3 fresh original-input FP64 checks, 2 rotating input sets.',
             '', f'Audit passed: {not errors}. Attempted: {len(shapes)}. Valid timings: {len(data)}. Precision rejected: {len(rejections)}. Wins: {group_data["all"]["wins"]}.', '',
             '| M | N | K | SWTuned μs | cuBLASLt μs | Speedup | Selected ID |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for row in data:
        lines.append(f'| {row["m"]} | {row["n"]} | {row["k"]} | {row["solution_us"]:.3f} | {row["cublaslt_us"]:.3f} | {row["speedup"]:.4f} | {row["selected_id"]} |')
    for rejection in rejections:
        m, n, k = rejection['shape']
        lines.append(f'| {m} | {n} | {k} | precision rejected | — | — | all 13 candidates rejected |')
    lines += ['', 'This is one measurement process per shape. Previous original-suite measurements are separate.',
              'Raw CSV retains setup/tune times, p10/p90, CV, numerical errors, and selected configurations.']
    if errors:
        lines += ['', 'Audit errors:', *[f'- {e}' for e in errors]]
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    if args.append_iteration is not None:
        with Path('ITERATIONS.md').open('a') as stream:
            stream.write(f'\n\n## Measurement {args.append_iteration} — more FP32 problem shapes, fixed kernel pool\n\n'
                         'User corrected the previous parameter-sweep interpretation: measure more M/N/K shapes. '
                         'Added 64 disjoint shapes outside the original14, including tall/wide transpose pairs, varied K, intermediate squares and odd boundaries. '
                         'No solution change or template addition. Default nine packed plus up to four zero-workspace candidates; public strict SWTuned selection. '
                         'Original full-call ordinary launch, fresh pack/reconstruction, 20warmup/5x100/3fresh original-input FP64/2rotate/8GiB maintained. '
                         f'Process/audit success={not errors}; rows={len(data)}/{len(shapes)}, wins={group_data["all"]["wins"]}, '
                         f'geomean={group_data["all"]["geometric_mean"]:.5f}, range={group_data["all"]["minimum"]:.5f}–{group_data["all"]["maximum"]:.5f}. '
                         f'Raw/public tuning records and all timing groups retained in {root}; full table report.md, audit audit.json. '
                         'One process per shape; no independent repeat certification or combined original14 claim.\n')
    print(json.dumps(dict(passed=not errors, errors=errors, groups=group_data), indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())

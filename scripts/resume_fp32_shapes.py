#!/usr/bin/env python3
"""Finish a shape sweep in isolated processes so precision rejections cannot stop it."""
import argparse
import csv
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-dir', required=True, type=Path)
    parser.add_argument('--result-dir', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--binary', required=True)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    shapes = [tuple(s) for group in ('squares', 'rectangles', 'boundaries') for s in config[group]]
    initial = list(csv.DictReader((args.initial_dir / 'current.csv').open()))
    completed = {tuple(int(row[d]) for d in ('m', 'n', 'k')) for row in initial}
    attempts = [dict(shape=list(s), status='accepted', source=str(args.initial_dir))
                for s in shapes if s in completed]
    rows = list(initial)
    tuning = [r for r in csv.DictReader((args.initial_dir / 'current.csv.sw_tuning.csv').open())
              if tuple(int(r[d]) for d in ('m', 'n', 'k')) in completed]
    (args.result_dir / 'driver.log').write_text((args.initial_dir / 'driver.log').read_text())
    for s in shapes:
        if s in completed:
            continue
        name = 'x'.join(map(str, s))
        output = args.result_dir / (name + '.csv')
        argv = ['python3', 'scripts/run_benchmark.py', '--config', str(args.config),
                '--binary', args.binary, '--algorithm', 'sw-tuned-strict', '--dtype', 'fp32',
                '--rotate-inputs', '2', '--shapes', ','.join(map(str, s)), '--output', str(output)]
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (args.result_dir / (name + '.log')).write_text(result.stdout)
        with (args.result_dir / 'driver.log').open('a') as stream:
            stream.write(result.stdout)
        measured = list(csv.DictReader(output.open())) if output.exists() else []
        candidate_path = Path(str(output) + '.sw_tuning.csv')
        candidates = list(csv.DictReader(candidate_path.open())) if candidate_path.exists() else []
        rejected = (result.returncode != 0 and not measured and candidates
                    and all(r['status'] == 'precision_rejected' for r in candidates)
                    and 'ERROR SW tuning: no_precision_valid_candidate' in result.stdout)
        status = 'accepted' if result.returncode == 0 and len(measured) == 1 else 'precision_rejected' if rejected else 'error'
        attempts.append(dict(shape=list(s), status=status, exit_code=result.returncode,
                             source=str(output), candidate_count=len(candidates)))
        rows.extend(measured)
        tuning.extend(candidates)
        print(f'{name}: {status}', flush=True)
    for path, entries in [('current.csv', rows), ('current.csv.sw_tuning.csv', tuning)]:
        with (args.result_dir / path).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(entries)
    (args.result_dir / 'attempts.json').write_text(json.dumps(attempts, indent=2) + '\n')
    failed = any(r['status'] == 'error' for r in attempts)
    (args.result_dir / 'exit.status').write_text(str(int(failed)) + '\n')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())

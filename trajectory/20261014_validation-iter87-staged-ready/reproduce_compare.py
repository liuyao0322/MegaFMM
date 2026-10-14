"""Rebuild isolated iter85/86/87 snapshots and repeat the 1024-only comparison.

Run from the repository with: python3 <this-file> [output-directory]
The GPU launcher uses the existing MEGAFMM_JOB_ID allocation.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(subprocess.check_output(
    ['git', '-C', str(Path(__file__).parent), 'rev-parse', '--show-toplevel'],
    text=True).strip())
out = Path(sys.argv[1] if len(sys.argv) > 1 else
           root / 'results/reproduce_iter87').resolve()
if not os.environ.get('SLURM_STEP_ID'):
    subprocess.run(['bash', str(root / 'scripts/on_gpu.sh'), 'python3',
                    str(Path(__file__).resolve()), str(out)], check=True)
    sys.exit(0)
os.chdir(root)
out.mkdir(parents=True, exist_ok=True)
versions = {'iter85': '2870c75', 'iter86': 'ecb7b5c', 'iter87': 'c9380c3'}
bins = {}
for name, commit in versions.items():
    src = out / name / 'source'
    (src / 'solution').mkdir(parents=True, exist_ok=True)
    for filename in ['CMakeLists.txt', 'solution/internal.h', 'solution/plan.cpp',
                     'solution/scale_output.cu', 'solution/sw_pipeline.cu']:
        (src / filename).write_bytes(subprocess.check_output(
            ['git', 'show', f'{commit}:{filename}']))
    for dirname in ['include', 'bench', 'tests']:
        if not (src / dirname).exists():
            (src / dirname).symlink_to(root / dirname, target_is_directory=True)
    build = out / name / 'build'
    with (out / name / 'build.log').open('w') as log:
        subprocess.run(['cmake', '-S', str(src), '-B', str(build),
                        '-DCMAKE_BUILD_TYPE=Release',
                        f'-DMEGAFMM_CUTLASS_DIR={root / "third_party/cutlass"}'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['cmake', '--build', str(build), '--target',
                        'megafmm_bench', '-j', '4'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    bins[name] = build / 'megafmm_bench'
rows = []
for mode in ['launch', 'graph']:
    for name, repeat in [('iter85', 1), ('iter86', 1), ('iter87', 1),
                         ('iter87', 2), ('iter86', 2), ('iter85', 2)]:
        stem = out / f'{mode}_{name}_{repeat}'
        args = [str(bins[name]), '--algorithm', 'sw', '--shapes',
                '1024,1024,1024', '--dtype', 'both', '--rotate-inputs', '1',
                '--trials', '100', '--warmup', '20', '--groups', '5',
                '--correct-trials', '3', '--output', str(stem) + '.csv']
        if mode == 'graph':
            args.append('--graph')
        with Path(str(stem) + '.log').open('w') as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
        with Path(str(stem) + '.csv').open() as f:
            for row in csv.DictReader(f):
                assert row['numerical_pass'] == '1' and row['nonfinite'] == '0'
                result = dict(mode=mode, version=name, repeat=repeat,
                              dtype=row['dtype'],
                              sw_us=float(row['solution_event_ms']) * 1000,
                              dense_us=float(row['dense_event_ms']) * 1000)
                rows.append(result)
                print(result, flush=True)
(out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')

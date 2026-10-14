#!/usr/bin/env python3
"""Owner-run iteration45 serial full timing; each inner exit retained verbatim."""
from pathlib import Path
import hashlib, json, subprocess, sys
PLAN_PATH = 'results/psplit_tma_bn16_min21024_plan_iter45.json'
IDENTITY_PATH = 'results/psplit_tma_bn16_min2_measured_identity_iter45.json'
PLAN_SHA = '9c89ac6d7b1b6aa15741347121a0162c12ccfeb2a9667ef120ac9d0ff66c2c55'
IDENTITY_SHA = 'cb8ea442360c820b7a2054a9e4f20a83b407196b496c1cfb7f06a6b3d99ba15b'
RUNS = [{'path': 'results/small_psplit_tma_bn16_min2_49_launch_iter45_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '49', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_tma_bn16_min2_49_launch_iter45_full1024.csv']}, {'path': 'results/small_psplit_tma_bn16_min2_49_graph_iter45_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '49', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_tma_bn16_min2_49_graph_iter45_full1024.csv', '--graph']}, {'path': 'results/small_psplit_tma_bn16_48_launch_iter45_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '48', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_tma_bn16_48_launch_iter45_full1024.csv']}, {'path': 'results/small_psplit_tma_bn16_48_graph_iter45_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '48', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_tma_bn16_48_graph_iter45_full1024.csv', '--graph']}, {'path': 'results/small_psplit_selective_tma_a45_launch_iter45_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '45', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_selective_tma_a45_launch_iter45_full1024.csv']}, {'path': 'results/small_psplit_selective_tma_a45_graph_iter45_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '45', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_selective_tma_a45_graph_iter45_full1024.csv', '--graph']}, {'path': 'results/small_psplit_rsa39_launch_iter45_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '39', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_rsa39_launch_iter45_full1024.csv']}, {'path': 'results/small_psplit_rsa39_graph_iter45_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '39', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_rsa39_graph_iter45_full1024.csv', '--graph']}, {'path': 'results/small_packed_lt1_launch_iter45_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'lt', '--lt-variant', '1', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_packed_lt1_launch_iter45_full1024.csv']}, {'path': 'results/small_packed_lt1_graph_iter45_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'lt', '--lt-variant', '1', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_packed_lt1_graph_iter45_full1024.csv', '--graph']}]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    if sha(PLAN_PATH) != PLAN_SHA or sha(IDENTITY_PATH) != IDENTITY_SHA:
        raise RuntimeError('Frozen plan/identity bytes changed before timing')
    plan = json.loads(Path(PLAN_PATH).read_text())
    identity = json.loads(Path(IDENTITY_PATH).read_text())
    for path, wanted in {**plan['frozen_sources'], **plan['frozen_binaries']}.items():
        if sha(path) != wanted: raise RuntimeError('Frozen source/bin changed: '+path)
    gate = identity['independent_compiled_gate']
    if sha(gate['path']) != gate['sha256']: raise RuntimeError('Frozen compiled gate changed')
    if sha(identity['linked_identity_path']) != identity['linked_identity_sha256']:
        raise RuntimeError('Frozen linked identity changed')
    for filename in plan['csv_paths']:
        for suffix in ('.csv', '.log', '.exit'):
            if Path(filename).with_suffix(suffix).exists():
                raise RuntimeError('One-time full timing artifact already exists: '+filename)
    inner_exits = []
    for run in RUNS:
        path, cmd = Path(run['path']), run['command']
        print('START', str(path), flush=True)
        with path.with_suffix('.log').open('x') as output:
            output.write('COMMAND '+json.dumps(cmd)+'\n'); output.flush()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                output.write(line); output.flush()
                if line.startswith(('CHECK', 'FAIL', 'Error')): print(line, end='', flush=True)
            code = proc.wait()
        with path.with_suffix('.exit').open('x') as output: output.write(str(code)+'\n')
        inner_exits.append(code)
        print('DONE', str(path), 'exit', code, flush=True)
        if code not in (0, 2):
            print('TERMINAL_RUNNER_EXIT', code, 'INNER_EXITS', inner_exits, flush=True)
            return code
    print('TERMINAL_RUNNER_EXIT 0 INNER_EXITS', inner_exits, flush=True)
    return 0
if __name__ == '__main__': sys.exit(main())

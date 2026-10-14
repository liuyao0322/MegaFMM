#!/usr/bin/env python3
"""Owner-run iteration47 serial full timing; each inner exit retained verbatim."""
from pathlib import Path
import hashlib, json, subprocess, sys
PLAN_PATH = 'results/psplit_tma_inplace_ss1024_plan_iter47.json'
IDENTITY_PATH = 'results/psplit_tma_inplace_ss_measured_identity_iter47.json'
PLAN_SHA = '6a52716d0f2391f7a4a6e0b51b8aa43afb9142b3e0b348346926a56bb156dafb'
IDENTITY_SHA = '5c63549c06580f0fe58d3f05d5d17c30e4e88ffa026bc32cb0c064d71ef4e52b'
RUNS = [{'path': 'results/small_psplit_tma_inplace_ss51_launch_iter47_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '51', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_tma_inplace_ss51_launch_iter47_full1024.csv']}, {'path': 'results/small_psplit_tma_inplace_ss51_graph_iter47_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '51', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_tma_inplace_ss51_graph_iter47_full1024.csv', '--graph']}, {'path': 'results/small_psplit_tma_inplace_ss52_launch_iter47_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '52', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_tma_inplace_ss52_launch_iter47_full1024.csv']}, {'path': 'results/small_psplit_tma_inplace_ss52_graph_iter47_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '52', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_tma_inplace_ss52_graph_iter47_full1024.csv', '--graph']}, {'path': 'results/small_psplit_selective_tma_a45_launch_iter47_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '45', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_selective_tma_a45_launch_iter47_full1024.csv']}, {'path': 'results/small_psplit_selective_tma_a45_graph_iter47_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '45', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_selective_tma_a45_graph_iter47_full1024.csv', '--graph']}, {'path': 'results/small_psplit_tma_bn16_k32_window50_launch_iter47_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '50', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_tma_bn16_k32_window50_launch_iter47_full1024.csv']}, {'path': 'results/small_psplit_tma_bn16_k32_window50_graph_iter47_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '50', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_tma_bn16_k32_window50_graph_iter47_full1024.csv', '--graph']}, {'path': 'results/small_psplit_rsa39_launch_iter47_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '39', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_rsa39_launch_iter47_full1024.csv']}, {'path': 'results/small_psplit_rsa39_graph_iter47_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'mega', '--mega-variant', '39', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_psplit_rsa39_graph_iter47_full1024.csv', '--graph']}, {'path': 'results/small_packed_lt1_launch_iter47_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'lt', '--lt-variant', '1', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_packed_lt1_launch_iter47_full1024.csv']}, {'path': 'results/small_packed_lt1_graph_iter47_full1024.csv', 'command': ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', 'lt', '--lt-variant', '1', '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20', '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', 'results/small_packed_lt1_graph_iter47_full1024.csv', '--graph']}]
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

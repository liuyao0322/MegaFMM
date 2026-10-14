#!/usr/bin/env python3
"""Generate owner-run serial full benchmark script text; never runs a benchmark."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_runs(plan):
    require(plan['expected_row_count'] == 20 and len(plan['routes']) == 5 and len(plan['csv_paths']) == 10,
            'Runner requires frozen mandatory20/ten-CSV plan')
    require([(r['algorithm'], r['variant']) for r in plan['routes']] ==
            [('mega', 47), ('mega', 45), ('mega', 39), ('mega', 36), ('lt', 1)], 'Wrong five-route domain')
    require(tuple(plan[k] for k in ('warmup', 'trials', 'groups', 'correct_trials', 'rotate_inputs', 'heuristic_candidates',
                                  'workspace_limit')) == (20, 100, 5, 3, 1, 32, 8589934592), 'Changed mandatory protocol')
    require(plan['shapes'] == [[1024, 1024, 1024]] and plan['dtypes'] == ['bf16', 'fp16'] and
            plan['modes'] == ['launch', 'graph'], 'Wrong shape/dtype/mode domain')
    runs = []
    for route in plan['routes']:
        if route['algorithm'] == 'mega':
            enabled = plan['route_configurations'][str(route['variant'])].get('enabled')
            require(isinstance(enabled, bool), 'Enabled must be explicit bool')
            require(route['variant'] == 47 or enabled, 'Retained control cannot be omitted')
            if not enabled:
                continue
        for mode in plan['modes']:
            path = f"results/small_{route['label']}_{mode}_iter43_full1024.csv"
            require(path in plan['csv_paths'], 'Runner path differs from frozen plan')
            cmd = ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench', '--algorithm', route['algorithm'],
                '--mega-variant' if route['algorithm'] == 'mega' else '--lt-variant', str(route['variant']),
                '--shapes', '1024,1024,1024', '--dtype', 'both', '--trials', '100', '--warmup', '20',
                '--groups', '5', '--correct-trials', '3', '--rotate-inputs', '1', '--output', path]
            if mode == 'graph':
                cmd.append('--graph')
            runs.append({'path': path, 'command': cmd})
    require(len(runs) * 2 == plan['enabled_row_count'], 'Enabled row count differs from serial runner selection')
    return runs


def generate(plan_path, identity_path):
    plan, identity = json.loads(Path(plan_path).read_text()), json.loads(Path(identity_path).read_text())
    require(identity['predeclared_plan'] == {'path': str(plan_path), 'sha256': sha(plan_path)} and
            identity['predeclared_plan_sha256'] == sha(plan_path), 'Identity must bind this frozen plan')
    require(identity['frozen_sources'] == plan['frozen_sources'] and identity['frozen_binaries'] == plan['frozen_binaries'] and
            identity['route_configurations'] == plan['route_configurations'] and
            identity['independent_compiled_gate'] == plan['independent_compiled_gate'], 'Runner identity/gate/plan freeze mismatch')
    select_runs(plan)
    return '''#!/usr/bin/env python3
"""Owner-run iteration43 serial full timing; each inner exit retained verbatim."""
from pathlib import Path
import hashlib, json, subprocess, sys
PLAN_PATH = ''' + repr(str(plan_path)) + '''
IDENTITY_PATH = ''' + repr(str(identity_path)) + '''
PLAN_SHA = ''' + repr(sha(plan_path)) + '''
IDENTITY_SHA = ''' + repr(sha(identity_path)) + '''
RUNS = ''' + repr(select_runs(plan)) + '''
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
            output.write('COMMAND '+json.dumps(cmd)+'\\n'); output.flush()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                output.write(line); output.flush()
                if line.startswith(('CHECK', 'FAIL', 'Error')): print(line, end='', flush=True)
            code = proc.wait()
        with path.with_suffix('.exit').open('x') as output: output.write(str(code)+'\\n')
        inner_exits.append(code)
        print('DONE', str(path), 'exit', code, flush=True)
        if code not in (0, 2):
            print('TERMINAL_RUNNER_EXIT', code, 'INNER_EXITS', inner_exits, flush=True)
            return code
    print('TERMINAL_RUNNER_EXIT 0 INNER_EXITS', inner_exits, flush=True)
    return 0
if __name__ == '__main__': sys.exit(main())
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True); parser.add_argument('--identity', required=True)
    parser.add_argument('--output', help='Owner-selected runner script path; default stdout')
    args = parser.parse_args(); content = generate(args.plan, args.identity)
    if args.output:
        with Path(args.output).open('x') as stream: stream.write(content)
    else:
        print(content, end='')

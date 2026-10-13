from pathlib import Path
import subprocess, sys, json

shapes = json.loads(Path('bench/small_shapes.json').read_text())['cli_shapes']
full = json.loads(Path('results/psplit_hybrid_s1024_plan_iter38.json').read_text())
runs = [] # All five retained or independently gated routes, full1024; quick41 is the mandatory bench.sh archive.
for route in full['routes']:
    option = {'lt':'--lt-variant','mega':'--mega-variant','hopper':'--hopper-variant'}[route['algorithm']]
    args = ['--algorithm', route['algorithm'], option, str(route['variant'])]
    for mode in full['modes']:
        runs.append((f"small_{route['label']}_{mode}_iter38_full1024", args,
                     '1024,1024,1024', 100, 3, mode == 'graph'))
for name, args, shape, trials, fresh, graph in runs:
    path = Path('results') / (name + '.csv')
    assert not path.exists(), str(path)
    cmd = ['bash', 'scripts/on_gpu.sh', './build/megafmm_bench'] + args + [
        '--shapes', shape, '--dtype', 'both', '--trials', str(trials),
        '--warmup', '20', '--groups', '5', '--correct-trials', str(fresh),
        '--rotate-inputs', '1', '--output', str(path)] + (['--graph'] if graph else [])
    print('START', name, flush=True)
    with path.with_suffix('.log').open('w') as output:
        output.write('COMMAND ' + json.dumps(cmd) + '\n')
        output.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            output.write(line)
            output.flush()
            if line.startswith(('CHECK', 'FAIL', 'Error')):
                print(line, end='', flush=True)
        code = proc.wait()
    path.with_suffix('.exit').write_text(str(code) + '\n')
    print('DONE', name, 'exit', code, flush=True)
    if code not in (0, 2):
        sys.exit(code)

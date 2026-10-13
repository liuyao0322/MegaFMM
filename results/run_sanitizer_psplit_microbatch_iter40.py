import subprocess, sys, json
from pathlib import Path

san = '/share/app/cuda/cuda-12.8/compute-sanitizer/compute-sanitizer'
version = subprocess.check_output([san, '--version'], stderr=subprocess.STDOUT, text=True)
for variant in ([int(sys.argv[1])] if len(sys.argv) > 1 else [44]):
    for tool in ([sys.argv[2]] if len(sys.argv) > 2 else ['memcheck', 'racecheck', 'synccheck']):
        cmd = ['bash', 'scripts/on_gpu.sh', san, '--tool', tool, '--error-exitcode', '99']
        if tool == 'memcheck':
            cmd += ['--leak-check', 'full']
        if tool == 'racecheck':
            cmd += ['--racecheck-report', 'all']
        cmd += ['./build/test_correctness', '--smoke-candidates', '--mega-only', '--mega-variant', str(variant)]
        path = Path(f'results/sanitizer_psplit_microbatch{variant}_{tool}_iter40.txt')
        assert not path.exists(), str(path)
        print('START', variant, tool, flush=True)
        with path.open('w') as output:
            output.write(version + 'COMMAND ' + json.dumps(cmd) + '\n')
            output.flush()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                output.write(line)
                output.flush()
                if 'SUMMARY:' in line or line.startswith(('FAIL', 'Error')):
                    print(line, end='', flush=True)
            code = proc.wait()
        path.with_suffix('.exit').write_text(str(code) + '\n')
        print('DONE', variant, tool, 'exit', code, flush=True)
        if code:
            sys.exit(code)

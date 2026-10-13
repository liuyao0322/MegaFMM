from pathlib import Path
import json, subprocess, sys

def run(name, command, accepted=(0,)):
    path=Path('results')/(name+'.txt')
    assert not path.exists(),str(path)
    print('START',name,flush=True)
    with path.open('w') as output:
        output.write('COMMAND '+json.dumps(command)+'\n'); output.flush()
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:
            output.write(line); output.flush()
            if line.startswith(('FAIL','Error','NUMERICAL_GATE')) or 'SUMMARY:' in line:
                print(line,end='',flush=True)
        code=proc.wait()
    path.with_suffix('.exit').write_text(str(code)+'\n')
    print('DONE',name,'exit',code,flush=True)
    if code not in accepted: sys.exit(code)

run('strict_psplit_rsa_bk32_iter37',['bash','scripts/on_gpu.sh','./build/test_correctness','--strict-psplit-rsa-bk32-only'],(0,2))
run('disabled_psplit_rsa_bk3237_cli_iter37',['bash','scripts/on_gpu.sh','./build/test_correctness','--smoke-candidates','--mega-only','--mega-variant','37'],(1,))
code=subprocess.call(['python3','.scratch/run_sanitizer_psplit_rsa_bk32_iter37.py'])
sys.exit(code)

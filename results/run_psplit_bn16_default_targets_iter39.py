from pathlib import Path
import subprocess,json,sys

def run(name,args):
    path=Path('results')/(name+'.txt');assert not path.exists(),path
    print('START',name,flush=True)
    cmd=['bash','scripts/on_gpu.sh','./build/test_correctness']+args
    with path.open('w') as out:
        out.write('COMMAND '+json.dumps(cmd)+'\n');out.flush()
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in p.stdout:
            out.write(line);out.flush()
            if line.startswith('FAIL'):print(line,end='',flush=True)
        code=p.wait()
    path.with_suffix('.exit').write_text(str(code)+'\n')
    text=path.read_text();print('DONE',name,code,'CASE',sum(s.startswith('CASE ') for s in text.splitlines()),'PASS',sum(s.startswith('PASS') for s in text.splitlines()),flush=True)
    if code:sys.exit(code)

run('correctness_psplit_bn16_iter39',[])
for variant in (42,):
    run(f'correctness_psplit_bn16{variant}_targeted_iter39',['--smoke-candidates','--mega-only','--mega-variant',str(variant)])

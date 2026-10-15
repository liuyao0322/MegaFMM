"""Exact BF16 cooperative preprocessing disassembly; static counts only."""
import argparse
from collections import Counter
import hashlib
import json
import re
import subprocess
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('labels',nargs='+');args=ap.parse_args()
h=Path(__file__).resolve().parent;repo=h.parents[2]
name='_ZN13int8_research30preprocess_cooperative_columnsIN7cutlass10bfloat16_tEEEvPKT_S5_PS3_S6_PaS7_PiS8_S8_Pf'
for label in args.labels:
    binary=h/'bin'/('compare_'+label)
    result=subprocess.run(['/share/app/cuda/cuda-12.8/bin/cuobjdump','--dump-sass','--function',name,str(binary)],capture_output=True,text=True,check=True)
    assert re.findall(r'^\s*Function\s*:\s*(\S+)',result.stdout,re.M)==[name]
    ops=[]
    for line in result.stdout.splitlines():
        m=re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?(?:U?P)\d+\s+)?([A-Z][A-Z0-9_.]*)',line)
        if m:ops.append(m.group(1))
    assert ops
    raw=h/'ncu'/(label+'_preprocess_sass.log');raw.write_text(result.stdout)
    stats=dict(scope='one exact frozen BF16 cooperative preprocessing function; static instructions, not dynamic counts, no latency inference',label=label,function=name,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),instructions=len(ops),opcodes=dict(sorted(Counter(ops).items())),
        warnings_in_other_cubins=result.stderr.count('not found'))
    target=h/(label+'_preprocess_static_stats.json');target.write_text(json.dumps(stats,indent=2)+'\n')
    files=[Path(__file__).resolve(),binary,h/(label+'_manifest.json'),target,raw]
    (h/(label+'_preprocess_static_manifest.json')).write_text(json.dumps({str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
    print(label,dict(instructions=len(ops),shuffle=sum(x.startswith('SHFL') for x in ops),redux=sum(x.startswith('REDUX') for x in ops),barriers=sum(x.startswith('BAR.SYNC') for x in ops)))

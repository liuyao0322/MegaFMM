"""CPU disassembly of one exact frozen BF16 consumer; static counts only."""
import argparse
from collections import Counter
import hashlib
import json
import re
import subprocess
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('labels',nargs='+');args=ap.parse_args()
h=Path(__file__).resolve().parent;repo=h.parents[2]
name='_ZN13int8_research9correctedIN7cutlass10bfloat16_tEEEv14CUtensorMap_stS3_S3_S3_PKiS5_Pf'
for label in args.labels:
    binary=h/'bin'/('compare_'+label)
    result=subprocess.run(['/share/app/cuda/cuda-12.8/bin/cuobjdump','--dump-sass','--function',name,str(binary)],capture_output=True,text=True,check=True)
    functions=re.findall(r'^\s*Function\s*:\s*(\S+)',result.stdout,re.M)
    assert functions==[name],functions
    ops=[]
    for line in result.stdout.splitlines():
        match=re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?(?:U?P)\d+\s+)?([A-Z][A-Z0-9_.]*)',line)
        if match:ops.append(match.group(1))
    assert ops
    raw=h/'ncu'/(label+'_corrected_sass.log');raw.write_text(result.stdout)
    stats=dict(scope='one exact frozen BF16 corrected function; static instructions, not dynamic execution counts',label=label,function=name,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),instructions=len(ops),
        opcodes=dict(sorted(Counter(ops).items())),
        warning_scope='cuobjdump may report requested function absent from other cubins; exact target found once',
        warnings_in_other_cubins=result.stderr.count('not found'))
    target=h/(label+'_static_stats.json');target.write_text(json.dumps(stats,indent=2)+'\n')
    files=[h/'export_uniform_static.py',binary,h/(label+'_manifest.json'),target,raw]
    (h/(label+'_static_manifest.json')).write_text(json.dumps({str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
    print(label,'static_instructions',len(ops),'SHFL',sum(x.startswith('SHFL') for x in ops))

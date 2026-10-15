"""Static instruction counts only; never infer executed counts or latency."""
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[2]
tool='/share/app/cuda/cuda-12.4/bin/cuobjdump'
rows=[];instructions={}
for label in ['p46','p47']:
    s=subprocess.check_output([tool,'--dump-sass',str(h/'bin'/f'compare_{label}')],text=True)
    sections=re.split(r'Function : (\S+)\n',s)
    for i in range(1,len(sections),2):
        name,body=sections[i:i+2]
        if 'preprocess_cooperative' not in name and '9corrected' not in name:continue
        inst=[re.sub(r'\s+',' ',x.strip()) for x in re.findall(r'/\*[0-9a-f]+\*/\s+([^;\n]+);',body)]
        op=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)',body)
        assert len(inst)==len(op)
        instructions[label,name]=inst
        rows.append(dict(candidate=label,function=name,instructions=len(op),
            LDS=sum(x.startswith('LDS') for x in op),STS=sum(x.startswith('STS') for x in op),
            BAR=sum(x.startswith('BAR') for x in op)))
assert len(rows)==8
consumers=[n for label,n in instructions if label=='p46' and '9corrected' in n]
assert len(consumers)==2
matches={n:instructions['p46',n]==instructions['p47',n] for n in consumers}
assert all(matches.values())
out=dict(scope=__doc__,tool_version=subprocess.check_output([tool,'--version'],text=True),
    rows=rows,consumer_normalized_instruction_text_identical=matches)
(h/'p46_p47_static.json').write_text(json.dumps(out,indent=2)+'\n')
files=['inspect_p46_p47_sass.py','p46_p47_static.json','bin/compare_p46','bin/compare_p47','p46_manifest.json','p47_manifest.json']
(h/'p47_static_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))

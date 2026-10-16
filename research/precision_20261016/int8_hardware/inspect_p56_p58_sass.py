"""Exact frozen binaries, static instruction evidence only, never dynamic counts."""
from pathlib import Path
import collections,hashlib,json,re,subprocess
h=Path(__file__).resolve().parent;repo=h.parents[2];tool='/share/app/cuda/cuda-12.4/bin/cuobjdump';rows=[];inst={};raw=[]
for label in ['p56','p58']:
 out=subprocess.check_output([tool,'--dump-sass',str(h/'bin'/f'compare_{label}')],text=True)
 sections=re.split(r'Function : (\S+)\n',out)
 found=0
 for i in range(1,len(sections),2):
  name,body=sections[i:i+2]
  if '9corrected' not in name and '30preprocess_cooperative_columns' not in name:continue
  found+=1;ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)',body)
  seq=[re.sub(r'\s+',' ',x.strip()) for x in re.findall(r'/\*[0-9a-f]+\*/\s+([^;\n]+);',body)]
  assert len(seq)==len(ops)
  raw.append(f'Candidate {label}\nFunction : {name}\n'+body)
  inst[label,name]=seq
  rows.append(dict(candidate=label,function=name,static_instructions=len(ops),static_mma_instructions=sum('GMMA' in x for x in ops),static_bra=sum(x.startswith('BRA') for x in ops),opcodes=dict(sorted(collections.Counter(ops).items()))))
 assert found==4
names=[n for lab,n in inst if lab=='p56'];matches={n:inst['p56',n]==inst['p58',n] for n in names}
assert all(matches[n] for n in names if '9corrected' in n),'Unchanged consumer source did not produce identical instruction text'
# Producer may already be expanded by the compiler; report identity rather than assuming change.
out=dict(scope=__doc__,rows=rows,normalized_instruction_text_equal=matches,interpretation='Resident-grid packing control; equality or difference of exact instruction text provides compiler evidence, not executed counts or proof of latency/numerical behavior. Consumer instruction text equality alone is not a fresh matrix/sanitizer certificate.')
(h/'p56_p58_static.json').write_text(json.dumps(out,indent=2)+'\n')
(h/'ncu/p56_p58_static_sass.log').write_text('\n'.join(raw))
f=[h/n for n in ['inspect_p56_p58_sass.py','p56_p58_static.json','ncu/p56_p58_static_sass.log','bin/compare_p56','bin/compare_p58','p56_manifest.json','p58_manifest.json']]
(h/'p58_static_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in f},indent=2)+'\n')
print(json.dumps(dict(rows=[{k:v for k,v in x.items() if k!='opcodes'} for x in rows],normalized_instruction_text_equal=matches),indent=2))

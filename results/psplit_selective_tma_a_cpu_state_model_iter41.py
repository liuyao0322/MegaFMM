from pathlib import Path
import ast,sys,json,hashlib
source=Path('.scratch/psplit_selective_tmaA_root_future40_model.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='57284d4f3ecee1039bf955901ad70e443d6786980dc57997de58824c55703d77'
text=source.read_text(); tree=ast.parse(text); retained=[]; stripped=[]
for node in tree.body:
 statement=ast.get_source_segment(text,node) or ''
 if 'assert all(not p.exists()' in statement or (isinstance(node,ast.Expr) and '.write_text(' in statement):stripped.append(node.lineno);continue
 retained.append(node)
tree.body=retained
# All writes denied, including frozen-prefix writers; stdout handled by parent process.
def deny(event,args):
 if event=='open' and args:
  mode=args[1] or ''; flags=args[2] if len(args)>2 else 0
  if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(1|2|64|512|1024)): raise RuntimeError('CPU41 all file writes denied')
sys.addaudithook(deny)
exec(compile(tree,str(source),'exec'),{'__file__':str(source),'__name__':'__cpu41_readonly_model__'})

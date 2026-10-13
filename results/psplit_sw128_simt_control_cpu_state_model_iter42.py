from pathlib import Path
import ast,sys,hashlib,json
source=Path('.scratch/psplit_sw128_simt_control_root_future41_model.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='14d9f9914ea1665ca4970debea5f48de7c753e30f43706ef53a1631395f1c7fa'
text=source.read_text(); tree=ast.parse(text); replaced=[]
for node in tree.body:
 segment=ast.get_source_segment(text,node) or ''
 if isinstance(node,ast.For) and 'library_source_before_sha256' in segment:
  node.iter=ast.Tuple(elts=[ast.Constant('protected_future_before_sha256'),ast.Constant('pinned_header_before_sha256')],ctx=ast.Load()); replaced.append(node.lineno)
assert len(replaced)==1
def deny(event,args):
 if event=='open' and args:
  mode=args[1] or '';flags=args[2] if len(args)>2 else 0
  if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(1|2|64|512|1024)):raise RuntimeError('CPU42 all writes denied')
sys.addaudithook(deny)
exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),{'__file__':str(source),'__name__':'__cpu42_readonly_model__'})

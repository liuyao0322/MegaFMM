"""Protected CPU re-evaluation of frozen literal models, with AST write operations removed.
Original artifact-writing top-level code is never executed unchanged. All actual writes
are refused at process level. Original code/outputs/candidates/headers remain frozen.
"""
from pathlib import Path
import ast,sys,json,hashlib
sys.dont_write_bytecode=True
kind=sys.argv[1];names={'raw':'psplit_bn16_future38','ss':'psplit_bn16_ss_future38','tail':'psplit_bn16_tailbanks_future38'};name=names[kind];src=Path('.scratch',name+'_model.py')
protected={}
for stem in names.values():
 m=Path('.scratch',stem+'_manifest.json');protected[str(m)]=hashlib.sha256(m.read_bytes()).hexdigest();protected.update(json.loads(m.read_text())['files'])
for f,h in protected.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
class NoWrites(ast.NodeTransformer):
 removed=0
 def visit_Assert(self,node):
  if 'exists' in ast.unparse(node.test) and ast.unparse(node.test).startswith('not '):self.removed+=1;return ast.Pass()
  return self.generic_visit(node)
 def visit_Call(self,node):
  if isinstance(node.func,ast.Attribute) and node.func.attr in ('write_text','write_bytes'):
   self.removed+=1;return ast.copy_location(ast.Constant(value=None),node)
  return self.generic_visit(node)
t=NoWrites();tree=ast.fix_missing_locations(t.visit(ast.parse(src.read_text())));assert t.removed>0
print('OWNER_CPU_REPLAY original',str(src),'sha',hashlib.sha256(src.read_bytes()).hexdigest(),'AST_writer_operations_or_existing_output_asserts_removed',t.removed,flush=True)
def refuse(event,args):
 if event=='open' and len(args)>1:
  mode=args[1] or '';flags=args[2] if len(args)>2 else 0
  if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(1|2|64|512|1024)):raise RuntimeError('Iteration39 CPU replay rejects ALL file writes')
sys.addaudithook(refuse)
ns={'__name__':'__owner_protected_ast_replay__','__file__':str(src.resolve())};exec(compile(tree,str(src),'exec'),ns)
for f,h in protected.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
print('OWNER_CPU_PROTECTED_FROZEN_BYTES_UNCHANGED',len(protected),flush=True)

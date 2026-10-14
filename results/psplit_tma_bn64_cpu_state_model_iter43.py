from pathlib import Path
import sys,hashlib
source=Path('.scratch/psplit_tma_bn64_root_future42_model.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='04f26e951f684481e42c069a765460215a2f19eacf3626c0e0b2e2c3d3731b36'
def deny(event,args):
 if event=='open' and args:
  mode=args[1] or '';flags=args[2] if len(args)>2 else 0
  if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(1|2|64|512|1024)):raise RuntimeError('CPU43 all writes denied')
sys.addaudithook(deny)
exec(compile(source.read_text(),str(source),'exec'),{'__file__':str(source.resolve()),'__name__':'__cpu43_readonly_model__'})

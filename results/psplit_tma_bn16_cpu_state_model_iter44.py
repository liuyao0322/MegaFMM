from pathlib import Path
import sys,hashlib
sys.dont_write_bytecode=True
source=Path('.scratch/psplit_tma_bn16_root_future43_model.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='232033b38530e4f4383efcbef3a4acb394f7a26ed51000a687e9e18c049c7212'
def deny(event,args):
 if event=="open" and args:
  mode=args[1] or ""; flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
  if (isinstance(mode,str) and any(c in mode for c in "wax+")) or flags&(1|2|64|512|1024):raise RuntimeError("CPU readonly model refuses every file write")
sys.addaudithook(deny)
exec(compile(source.read_text(),str(source),"exec"),{"__file__":str(source.resolve()),"__name__":"__cpu44_readonly_model__"})

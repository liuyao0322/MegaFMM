from pathlib import Path
import hashlib,json,os,sys
sys.dont_write_bytecode=True
root=Path('/share/home/liuyao/workspace/MegaFMM')
models={'hybrid':('.scratch/psplit_rsA_hybridS_future37_model.py','5bc9d8a360177fdbc84a3fefb7f0e295f985ff17240a952cbb9ed9058a299dba'),'planar':('.scratch/psplit_rsA_hybridS_planar_future37_model.py','a3ea1c358db0c8e0ee3328ba77df1d714c786ff518b9fc5a8eb445de186b41c9')}
path,expected=models[sys.argv[1]];original=root/path;raw=original.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
mp=root/'.scratch/psplit_rsA_hybridS_future37_manifest.json';m=json.loads(mp.read_text());protect={**m['files'],str(mp.relative_to(root)):hashlib.sha256(mp.read_bytes()).hexdigest()}
for p,h in protect.items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h
def refuse(event,args):
 if event=='open':
  mode,flags=args[1],args[2]
  if (isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
   raise PermissionError('Iteration38 CPU execution rejects all filesystem writes')
sys.addaudithook(refuse)
scope={'__name__':'__owner_iteration38_protected_definitions__','__file__':str(original)}
exec(compile(raw,str(original),'exec'),scope)
# This particular frozen model main only prints stdout and already refuses writes.
# No parent model imports or historical artifact-writing main is called.
scope['main']()
for p,h in protect.items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h

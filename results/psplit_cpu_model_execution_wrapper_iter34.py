from pathlib import Path
import runpy,hashlib,sys
root=Path.cwd();original_write=Path.write_text
mapping={root/'.scratch/cooperative_st_once_psplit_review.json':root/'results/psplit_cpu_model_snapshot_iter34.json',root/'.scratch/psplit_asymmetric_review.json':root/'results/psplit_asymmetric_cpu_model_snapshot_iter34.json',root/'.scratch/psplit_asymmetric_semantic.diff':root/'results/psplit_asymmetric_cpu_semantic_iter34.diff'}
protected=list(mapping)+[root/'.scratch/cooperative_st_once_psplit_candidate.cu',root/'.scratch/cooperative_st_once_psplit_model.py',root/'.scratch/psplit_asymmetric_candidate.cu',root/'.scratch/psplit_asymmetric_model.py']
old={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
def redirected(self,*args,**kwargs):return original_write(mapping.get(self.resolve(),self),*args,**kwargs)
Path.write_text=redirected
for name in ([sys.argv[1]] if len(sys.argv)>1 else ['cooperative_st_once_psplit_model.py','psplit_asymmetric_model.py']):
 print('ACTUAL_ORIGINAL_MODEL',name,'sourcehash',hashlib.sha256((root/'.scratch'/name).read_bytes()).hexdigest(),'OUTPUT_WRITES_REDIRECTED_ONLY',flush=True)
 runpy.run_path(str(root/'.scratch'/name),run_name='__main__')
assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in old.items())
print('FROZEN_MODELS_AND_PRIOR_SNAPSHOTS_BYTE_UNCHANGED',flush=True)

from pathlib import Path
import subprocess,json,sys,hashlib
p=Path('results/small_psplit_selective_tma_a_environment_iter41.txt');assert not p.exists()
cmd=['bash','scripts/on_gpu.sh','python3','results/psplit_selective_tma_a_environment_runner_iter41.py']
print('START_POST_VALIDATION_ENVIRONMENT',flush=True)
with p.open('w') as f:f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush();code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
p.with_suffix('.exit').write_text(str(code)+'\n');print('DONE_ENVIRONMENT',code,flush=True)
if code:sys.exit(code)
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();probe='results/psplit_selective_tma_a_function_occupancy_probe_iter41.txt';j=json.load(open(probe))
meta={'path':str(p),'exit_path':str(p.with_suffix('.exit')),'sha256':h(p),'exit':code,'allocation':107488,'host':'g08','GPU':'NVIDIA H100 80GB HBM3','gpu_uuid':'GPU-78def847-b6c1-9dd4-ff30-269a9ae1a73f','SMs':132,'SMs_evidence':'actual archived DriverAPI cuDeviceGetAttribute','application_cuda':'12.4.99','driver':'575.51.03','CUTLASS':'3.5.1 commit f7b19de32c5d1f3cedfc735c2849f12b537522ee','sanitizer_path':'/share/app/cuda/cuda-12.8/compute-sanitizer/compute-sanitizer','sanitizer_version':'2025.1.0.0 build35583870','clocks_locked':False,'snapshot_scope':'Actual readonly postvalidation snapshot; no kernel launch or within-group clock stability/residency proof.','function_attributes_and_occupancy_API':{'path':probe,'sha256':h(probe),'exit':int(Path(probe).with_suffix('.exit').read_text()),'records':j['records']}}
Path('results/psplit_selective_tma_a_environment_metadata_iter41.json').write_text(json.dumps(meta,indent=2)+'\n')

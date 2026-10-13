from pathlib import Path
import subprocess,json,hashlib
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=Path('results/psplit_rsa_bk321024_plan_iter37.json');plan=json.loads(p.read_text())
init=json.loads(Path('results/psplit_rsa_bk32_initial_frozen_identity_iter37.json').read_text());gate=plan['route_configurations']['40']['compiled_gate']
for f,h in {**init['frozen_sources'],**init['frozen_binaries']}.items():assert sha(f)==h,f
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
for f,h in init['frozen_sources'].items():assert hashlib.sha256(subprocess.check_output(['git','show',head+':'+f])).hexdigest()==h,f
assert sha(gate['path'])==gate['sha256'];archive=Path('.scratch/psplit_rsa_bk32_quick_archive_iter37.txt').read_text().strip();assert Path(archive,'results.csv').exists()
obj={'schema':'megafmm.iter37.rsa-bk32-measured-identity.v1','base_checkpoint':init['base_checkpoint'],'measurement_source_commit':head,'frozen_sources':init['frozen_sources'],'frozen_binaries':init['frozen_binaries'],'predeclared_plan':{'path':str(p),'sha256':sha(p)},'predeclared_plan_sha256':sha(p),'quick_archive':archive,'independent_compiled_gate':gate,'linked_identity_path':init['linked_identity_path'],'linked_identity_sha256':init['linked_identity_sha256'],'compiled_scope':'Independent fresh CAN_LAUNCH40 PASS for final source548 and bench49e5/test16ee. I168 owned64512>=57344; role FP16 39/124/141 BF16 39/150/141; no STACK/LOCAL/spill/newC7512; all-stage A24/32 and P48/64 generic pending accesses0. Actual61568dynamic+1024static/384 and bothbankdrain before crossbank tailSTS. Original numerical certification separate;37 remainsdisabled. Earlier pregate pending snapshot preserved as chronological initial artifact.','metadata_scope':'Measuredidentity created ONCE after AKO quicklogger/Git closure, before any mandatoryfull timing. Plan/gate/quick/link/source7/bin2 included initially; no postmeasurement metadata append.'}
out=Path('results/psplit_rsa_bk32_measured_identity_iter37.json');assert not out.exists();out.write_text(json.dumps(obj,indent=2)+'\n');print('MEASURED_SOURCE',head,'IDENTITY_SHA',sha(out),'PLAN_SHA',sha(p))

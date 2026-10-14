from pathlib import Path
import hashlib,json
p=Path("results/small_psplit_tma_bn16_min2_profile_iter45.json")
e=Path("results/psplit_tma_bn16_min2_precision_future_evidence_iter45.json")
j=json.loads(p.read_text());v=json.loads(e.read_text())
for path,wanted in v["frozen_cpu_only_files"].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==wanted,path
j["precision_future_evidence"]={"artifact_path":str(e),"artifact_sha256":hashlib.sha256(e.read_bytes()).hexdigest(),**v}
assert j["precision_certified"] is False and j["goal_complete"] is False
p.write_text(json.dumps(j,indent=2)+"\n")
print("PROFILE_SEPARATE_CPU_FUTURE_APPENDED",len(v["frozen_cpu_only_files"]))

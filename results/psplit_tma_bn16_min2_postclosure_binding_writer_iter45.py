from pathlib import Path
import json,hashlib
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=Path("results/small_psplit_tma_bn16_min2_profile_iter45.json");j=json.loads(p.read_text())
for path,wanted in j["raw_sha256"].items():assert sha(path)==wanted,path
omitted={p.name,"psplit_tma_bn16_min2_data_audit_iter45.json","psplit_tma_bn16_min2_static_review_final_iter45.json","psplit_tma_bn16_min2_static_review_final_iter45.md"}
new={str(q):sha(q) for q in Path("results").iterdir() if q.is_file() and "iter45" in q.name and q.name not in omitted}
j["raw_sha256"].update(new)
p.write_text(json.dumps(j,indent=2)+"\n")
print("POSTCLOSURE_RAW_BINDINGS_COMPLETE",len(j["raw_sha256"]))

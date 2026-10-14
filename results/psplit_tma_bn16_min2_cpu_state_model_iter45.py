import pathlib,json,hashlib,itertools
pref=".scratch/psplit_tma_bn16_min2_root_future44_"
def sha(b):return hashlib.sha256(b).hexdigest()
delta=json.loads(pathlib.Path(pref+"source_delta.json").read_text())
parent=pathlib.Path(delta["parent_path"]).read_text()
assert sha(parent.encode())==delta["parent_sha256"]
src=parent
for op in delta["steps"]:
 assert sha(src.encode())==op["before_sha256"]
 assert src.count(op["old"])==op["count"]
 src=src.replace(op["old"],op["new"])
 assert sha(src.encode())==op["after_sha256"]
candidate=pathlib.Path(pref+"candidate.cu").read_text()
assert src==candidate and sha(src.encode())==delta["candidate_sha256"]
for op in reversed(delta["steps"]):
 assert src.count(op["new"])==op["count"]
 src=src.replace(op["new"],op["old"])
assert src==parent
assert candidate.count("warpgroup_reg_dealloc<40>();")==1
assert candidate.count("warpgroup_reg_alloc<88>();")==1
assert candidate.count("warpgroup_reg_alloc<112>();")==1
assert "__launch_bounds__(384,2)" in candidate
assert "if (tid < 256) {\n      cutlass::arch::warpgroup_reg_alloc<88>();\n      clear(q0)" in candidate
assert "} else {\n      cutlass::arch::warpgroup_reg_alloc<112>();\n      clear(q0)" in candidate
protected=json.loads(pathlib.Path(pref+"protected_base.json").read_text())
for key in ["future","old_cuda"]:
 for p,e in protected[key].items():
  assert sha(pathlib.Path(p).read_bytes())==e,(key,p)
assert sha(pathlib.Path("PLAN.md").read_bytes())==protected["PLAN"]
# Algebraic allocation model only: CTA-local initial owned pool and role actions.
def viable(initial,q=(40,88,112),ctas=2):
 return initial%8==0 and all(a%8==0 for a in q) and q[0]<=initial<=min(q[1:]) and 384*initial>=128*sum(q) and ctas*384*initial<=65536 and ctas*128*sum(q)<=65536
assert viable(80) and 128*(40+88+112)==30720
release=128*(80-40);acquire=[128*(88-80),128*(112-80)]
assert release==sum(acquire)==5120
# Two independent CTA pools, all720 orderings of six decrement/increment requests.
completed=0;blocked_requests=0
events=[(c,r) for c in range(2) for r in range(3)]
for order in itertools.permutations(events):
 free=[0,0];remaining=list(order);guard=0
 while remaining:
  advanced=False
  for c,r in list(remaining):
   if r==0:
    free[c]+=release;remaining.remove((c,r));advanced=True
   elif free[c]>=acquire[r-1]:
    free[c]-=acquire[r-1];remaining.remove((c,r));advanced=True
   else:blocked_requests+=1
  guard+=1
  assert advanced and guard<=3
 assert free==[0,0]
 completed+=1
assert completed==720
negatives=[
 ("insufficient_initial72",72,(40,88,112),2),
 ("two_startup_initial88",88,(40,88,112),2),
 ("both_consumer96_112",80,(40,96,112),2),
 ("producer48_pool",80,(48,88,112),2),
 ("qWG0belowinitial",80,(40,72,112),2),
 ("qWG1belowinitial",80,(40,88,72),2),
 ("nonhardwarequota87",80,(40,87,112),2),
 ("nonhardwareinitial81",81,(40,88,112),2),
 ("three_CTA_pool",80,(40,88,112),3),
 ("old128_128_min2",80,(40,128,128),2)
]
for name,i,q,c in negatives:
 assert not viable(i,q,c),name
p=json.loads(pathlib.Path(pref+"protocol.json").read_text())
assert all(p[k] is None for k in ["actual_initial_allocation","actual_shared_sizeof","actual_residency","actual_GPR_fit","actual_STACK_LOCAL_spill","actual_CAN","actual_correctness","actual_precision","actual_latency","actual_speedup"])
print(json.dumps({"status":"PASS_CPU_SOURCE_AND_REGISTER_ARITHMETIC_ONLY","candidate_sha256":sha(candidate.encode()),"parent_sha256":sha(parent.encode()),"ordered_delta_steps":len(delta["steps"]),"exact_reverse":True,"protected_future_files":len(protected["future"]),"protected_old_CUDA":len(protected["old_cuda"]),"allocation_schedules":completed,"temporarily_blocked_requests":blocked_requests,"negative_fixtures_rejected":len(negatives),"source_quota_pool":30720,"source_two_pool":61440,"actual_GPU":None,"actual_CAN":None,"actual_precision":None,"actual_latency":None,"issues":[]}))

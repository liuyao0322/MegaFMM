from pathlib import Path
import json
p=json.loads(Path("results/small_psplit_tma_bn16_min2_profile_iter45.json").read_text())
m=json.loads(Path("results/small_psplit_tma_bn16_min2_measurements_iter45.json").read_text())
v=p["validation"];assert len(m["rows"])==20 and len(p["sanitizers"])==3
assert all(x["exit"]==0 for x in p["sanitizers"])
text="\n## Iteration45 actual closed evidence\n\n"
text+="- ONLY49 BM64 BN16 BK64/minCTA2/q40-88-112 uses four ascending K16 atom-local A windows while wholeK FP32 P, fixed U, original RN DAG/final onecasts and alphaRN/betaFMA remain unchanged. Initial whole-stage49 REG80 STACK24/spill40FP16/36BF16 and newC7512 forbade any native launch; revision1 atom-local A repaired those resources. Fresh actual I80 owns30720=request30720, all-role implicit maxima37/77/71 and zero STACK/LOCAL/spill; legal initial-owned commonP R32..55 zeros precede INC, no actual grown registers used. Dynamic95232 and occupancy API prediction2 are recorded; measured residency remains null.\n"
text+="- Samebinary FULL20 declares49/48/45/39/Lt1 × two dtype × ordinary/Graph,20warm100calls5groups3fresh/32Lt8GiB/rotate1; all five groups and original60FAIL/NF0 retained. Authority CSV ranks own complete SW latency, not changes in independent Dense denominator.\n"
for dtype in ("bf16","fp16"):
 for mode in ("launch","graph"):
  rs={str(r["variant"]):r for r in (x["csv_row"] for x in m["rows"]) if r["dtype"]==dtype and r["mode"]==mode and str(r["variant"]) in ("49","48","45","39")}
  a,b,c=rs["49"],rs["48"],rs["45"]
  f=lambda x:float(x["solution_event_ms"])*1000
  text+=f"- {dtype}/{mode}:49 {f(a):.9f}us vs48 {f(b):.9f}us vs45 {f(c):.9f}us.49/48 latency reduction {(1-f(a)/f(b))*100:.6f}%;49/45 increase {(f(a)/f(c)-1)*100:.6f}%.45 is the fastest measured native route, not necessarily fastest among all routes.\n"
text+=f"- Actual default CASE/PASS/BEGIN {v['default']['CASE']}/{v['default']['PASS']}/{v['default']['BEGIN']} and selected49 {v['selected49']['CASE']}/{v['selected49']['PASS']}/{v['selected49']['BEGIN']} each exit0. Strict original30 gates {v['strict']['pass']}PASS/{v['strict']['fail']}FAIL/nonfinite{v['strict']['nonfinite']}; three serial CUDA12.8 sanitizers exit0 with zero reported errors/hazards/leaks within their tested scopes. Full1920 numerical source domain was not executed; structural/cross-atom bit fixtures do not certify original FP64 precision.\n"
text+="- Plan/onceidentity were created after quick source Git65a2292 and before full timing. SourceGit binds seven executed sources; later prestatusGit binds the SHA-pinned plan/identity. No hindsight identity append or timing rerun. Auto remains Dense, precision_certified=false and goal_complete=false; no automatic iteration46/public50. Full report: results/1024_MEGAKERNEL_PSPLIT_TMA_BN16_MIN2_REPORT.md.\n"
text+="- Separate ROOT CPU-only precision future:30 frozen files,27cases/111schemes; mixed BF16-to-FP16 reduced error on7/7 random cases but bounded exact finite cancellation still fails BF16 gate≈0.001 with mixed eF9.7169 and hi+lo15 omitting lo*lo eF0.06917. Large/tiny stress includes nonfinite/eF1. Original failed CPU helper exit1 retained. No CUDA/GPU/performance/precision-policy change or certification follows from that proxy evidence; results/psplit_tma_bn16_min2_precision_future_evidence_iter45.json.\n"
for name in ("HINTS.md","ITERATIONS.md"):
 q=Path(name);assert "Iteration45 actual closed evidence" not in q.read_text();q.write_text(q.read_text()+text)
print("CLOSED_EVIDENCE_APPENDED")

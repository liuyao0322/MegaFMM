from pathlib import Path
import csv,re
p=Path('results/iter384');log=(p/'driver.log').read_text();rows=list(csv.DictReader((p/'current.csv').open())) if (p/'current.csv').exists() else []
archive=re.findall(r'Trajectory saved to: (\S+)',log)
s='\n### Iteration384 — cross-product SW accumulator pipeline\n\n'
s+='- **Change:** Add native64x8/64x16,BK128,S8,K512 candidates600000–600001. Each of the original seven products retains an independent FP32 accumulator; one pending WGMMA group spans product boundaries and a final wait precedes original-order reconstruction. Original transform/TMA producer and BF16/FP16 input types unchanged, no compensation/product planes/fullAB route. All old native/batch/restored1024/scalar candidates remain.\n'
s+='- **Validation:** Requested clean build and all native/batch/small/cross-product prechecks. Cross precheck covers28shape/config/type combinations,140calls+28nativefractional bitwise checks,Graph/range/guards/originalintegerFP64. Actual completed scope is in raw logs; isolated sanitizer evidence is separate and does not certify full28.\n'
s+=f'- **Benchmark:** Original14shapes x2dtype, observed{len(rows)}/28rows;100calls x5groups,20warmups,2rotating bindings,3fresh original-inputFP64 checks. Accepted={sum(r.get("accepted")=="1" for r in rows)}, historical gate passes={sum(r.get("numerical_pass")=="1" for r in rows)}.\n\n'
s+='| M,N,K | dtype | cuBLASLt us | SW us | speedup | SW eF | selected |\n|---|---|---:|---:|---:|---:|---|\n'
for r in rows:
 selected=re.search(r';selected_id=(-?\d+);',r['solution_lt_config'])
 s+=f'| {r["m"]},{r["n"]},{r["k"]} | {r["dtype"]} | {float(r["dense_event_ms"])*1000:.3f} | {float(r["solution_event_ms"])*1000:.3f} | {float(r["speedup"]):.4f} | {float(r["solution_ef"]):.8g} | {selected[1] if selected else "unknown"} |\n'
s+='\n- **Analysis:** Research showed512 gains but still belowLt. This formal integration is not completion of the original all28 cuBLAS acceleration goal. Confirm actual candidate preservation and group medians, then independently repeat any gains. Errors are reported against original inputs and are not claimed improved.\n'
s+='- **Next:** Audit2328candidate records/all2316previouscandidates/sourceidentity/selectedmedian, selection/API/Graph/workspace checks and scoped independent repeats. Continue broad shape optimization.\n'
s+=f'- **Archive:** {archive[-1] if archive else "missing; diagnose only after this iteration is logged and committed"}\n'
f=Path('ITERATIONS.md');f.write_text(f.read_text()+s);print(s)

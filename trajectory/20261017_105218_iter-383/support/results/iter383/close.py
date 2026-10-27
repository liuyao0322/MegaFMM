from pathlib import Path
import csv,re
p=Path('results/iter383');log=(p/'driver.log').read_text();rows=list(csv.DictReader((p/'current.csv').open())) if (p/'current.csv').exists() else []
archive=re.findall(r'Trajectory saved to: (\S+)',log)
s='\n### Iteration383 — native narrow tiles and exact short-K candidates\n\n'
s+='- **Change:** Add six native small-aligned candidates, retaining all existing native/batch/restored1024/scalar routes. Native64x16 avoids raw-quadrant copies; three exactK2 variants fully unroll producer and consumer atK512. All aligned M multiple64/N multiple128/K multiple256 eligible; exactK2 variants require K512. No split-product candidate promoted: research ABBA regressed. BF16 stays BF16, exactly7products, unchanged FP32 reconstruction, all-finite measured selection/error reporting.\n'
s+='- **Validation:** Full clean build; native, seven-batch, and new narrow/short-K candidate prechecks requested, including thin/odd/partial-tile/persistent cases, changed-inputGraph, BF high-range, guards, independent integer original-FP64. See raw logs for actual completed scope; no independent-repeat/sanitizer certification inferred.\n'
s+=f'- **Benchmark:** Full14-shape x2dtype requested, observed{len(rows)}/28 rows;100calls x5groups,20warmups,2rotating bindings,3fresh original-inputFP64 trials. Finite accepted={sum(r.get("accepted")=="1" for r in rows)}, old gate passes={sum(r.get("numerical_pass")=="1" for r in rows)}.\n\n'
s+='| M,N,K | dtype | cuBLASLt us | SW us | speedup | SW eF | selected |\n|---|---|---:|---:|---:|---:|---|\n'
for r in rows:
 selected=re.search(r';selected_id=(-?\d+);',r['solution_lt_config'])
 s+=f'| {r["m"]},{r["n"]},{r["k"]} | {r["dtype"]} | {float(r["dense_event_ms"])*1000:.3f} | {float(r["solution_event_ms"])*1000:.3f} | {float(r["speedup"]):.4f} | {float(r["solution_ef"]):.8g} | {selected[1] if selected else "unknown"} |\n'
s+='\n- **Analysis:** Measured small-kernel candidate integration, not proof of all28 cuBLAS wins. Compare actual rows and group consistency; original errors remain visible and are not claimed improved.\n- **Next:** Audit2316candidate records, old-pool preservation and winner medians, then selection/API/Graph/workspace tests. Independent repeats and any sanitizer scope must be reported separately.\n'
s+=f'- **Archive:** {archive[-1] if archive else "missing: inspect driver log only after logging/committing this iteration"}\n'
f=Path('ITERATIONS.md');f.write_text(f.read_text()+s);print(s)

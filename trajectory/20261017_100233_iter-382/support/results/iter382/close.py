from pathlib import Path
import csv,re
p=Path('results/iter382');log=(p/'driver.log').read_text();rows=list(csv.DictReader((p/'current.csv').open())) if (p/'current.csv').exists() else []
archive=re.findall(r'Trajectory saved to: (\S+)',log)
s='\n### Iteration382 — additive native-dtype seven-batch leaf candidates\n\n'
s+='- **Change:** Add four Hopper batched-leaf architectures (L=7 distinct SW half-problems), keeping every old native/restored1024/scalar candidate. Physical half sizes align16; seven FP32 product planes and full pack/compute/reconstruction measured. BF16 stays BF16, no compensation/fullAB substitute. Workspace filtered before setup, actual SM count captured. Selection remains all-finite five-round ranking/top4confirmation with unchanged error reporting. No adaptive-budget change.\n'
s+='- **Validation:** Full clean build; native candidate precheck plus new seven-batch candidate precheck requested, including thin/odd/partial-tile/persistent cases, changed-inputGraph, BF high-range, guards, independent integer original-FP64. See raw logs for actual completed scope; no independent-repeat/sanitizer certification inferred.\n'
s+=f'- **Benchmark:** Full14-shape x2dtype requested, observed{len(rows)}/28 rows;100calls x5groups,20warmups,2rotating bindings,3fresh original-inputFP64 trials. Finite accepted={sum(r.get("accepted")=="1" for r in rows)}, old gate passes={sum(r.get("numerical_pass")=="1" for r in rows)}.\n\n'
s+='| M,N,K | dtype | cuBLASLt us | SW us | speedup | SW eF | selected |\n|---|---|---:|---:|---:|---:|---|\n'
for r in rows:
 selected=re.search(r';selected_id=(-?\d+);',r['solution_lt_config'])
 s+=f'| {r["m"]},{r["n"]},{r["k"]} | {r["dtype"]} | {float(r["dense_event_ms"])*1000:.3f} | {float(r["solution_event_ms"])*1000:.3f} | {float(r["speedup"]):.4f} | {float(r["solution_ef"]):.8g} | {selected[1] if selected else "unknown"} |\n'
s+='\n- **Analysis:** Incremental candidate integration, not proof of all28 cuBLAS wins. Compare actual rows and group consistency; original errors remain visible and are not claimed improved.\n- **Next:** Audit2232candidate records, old-pool preservation and winner medians, then selection/API/Graph/workspace tests. Independent repeats and any sanitizer scope must be reported separately.\n'
s+=f'- **Archive:** {archive[-1] if archive else "missing: inspect driver log only after logging/committing this iteration"}\n'
f=Path('ITERATIONS.md');f.write_text(f.read_text()+s);print(s)

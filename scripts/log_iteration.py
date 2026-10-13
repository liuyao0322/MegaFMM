#!/usr/bin/env python3
"""Close an AKO iteration immediately after scripts/bench.sh returns."""
import argparse
import csv
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("number", type=int)
for name in ("title", "hypothesis", "changes", "analysis", "next", "status"):
    p.add_argument("--" + name, required=True)
a = p.parse_args()
rows = list(csv.DictReader(Path("results/current.csv").open()))
archives = sorted(Path("trajectory").glob(f"*_iter-{a.number}"))
archive = str(archives[-1]) if archives else "missing trajectory"
speed = sum(float(r["speedup"]) for r in rows) / len(rows) if rows else 0
runtime = sum(float(r["solution_event_ms"]) for r in rows) / len(rows) if rows else 0
if a.status == "auto":
    import math
    baseline = {tuple(r[k] for k in ("m", "n", "k", "dtype", "requested", "variant", "mode")): r
                for r in csv.DictReader(Path("results/baseline.csv").open())}
    gains = []
    for r in rows:
        key = tuple(r[k] for k in ("m", "n", "k", "dtype", "requested", "variant", "mode"))
        if key in baseline:
            gains.append(float(baseline[key]["solution_event_ms"]) / float(r["solution_event_ms"]))
    gain = math.exp(sum(math.log(g) for g in gains) / len(gains)) if gains else 1
    a.status = "failed" if not rows else "improved" if gain >= 1.03 else "regression" if gain < 1/1.03 else "no-change"
    a.analysis += f" Matched-baseline geometric latency improvement={gain:.4f}x ({len(gains)} rows); certification remains independent."
entry = f"\n### Iter {a.number} — {a.title}\n\n"
entry += f"- **Hypothesis:** {a.hypothesis}\n- **Changes:** {a.changes}\n"
entry += f"- **Bench:** Compiled={'True' if rows else 'False/partial'}; frozen numerical gate={'PASS' if rows and all(r['numerical_pass']=='1' for r in rows) else 'FAIL/unchecked'}. No certification is inferred from runtime gains.\n"
entry += f"- **Scope:** {len(rows)} rows; arithmetic mean of row median latency={runtime:.6f} ms; mean speedup={speed:.4f}x. Compare individual identical shape/dtype/mode rows below.\n"
entry += "\n| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |\n|---|---|---|---|---:|---:|---:|---:|---|\n"
for r in rows:
    entry += f"| {r['m']},{r['n']},{r['k']} | {r['dtype']} | {r['actual']} v{r['variant']} | {r['mode']} | {float(r['dense_event_ms']):.6f} | {float(r['solution_event_ms']):.6f} | {float(r['speedup']):.4f} | {float(r['solution_ef']):.6g} | {r['numerical_pass']} |\n"
entry += f"\n- **Analysis:** {a.analysis}\n- **Next:** {a.next}\n- **Archive:** {archive}/\n"
log = Path("ITERATIONS.md")
s = log.read_text()
marker = "\n## Iterations"
summary = f"| {a.number} | {a.title} | {speed:.4f} | {runtime:.6f} | {a.status} |\n"
s = s.replace(marker, "\n" + summary + marker, 1)
log.write_text(s + entry)
print(f"Logged iter {a.number}: {len(rows)} rows, {archive}")

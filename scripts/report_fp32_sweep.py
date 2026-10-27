#!/usr/bin/env python3
"""Close one immutable FP32 sweep, retaining every timing group and raw result."""
import argparse
import csv
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--shapes", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / "configs.json").read_text())
    path = args.root / "screen/current.csv"
    rows = list(csv.DictReader(path.open())) if path.exists() and path.stat().st_size else []
    log = (args.root / "driver.log").read_text()
    status = int((args.root / "status").read_text())
    counts = {key: len(re.findall("^" + key + " .*$", log, re.M)) for key in ("CHECK", "TIME", "RESULT")}
    expected = len(manifest) * args.shapes
    passed = (status == 0 and len(rows) == expected and
              counts == {"CHECK": 3 * expected, "TIME": 5 * (len(manifest) + 4) * args.shapes, "RESULT": expected} and
              all(r["numerical_pass"] == r["bitwise_match"] == "1" for r in rows))
    groups = {}
    for row in rows:
        shape = tuple(int(row[k]) for k in ("m", "n", "k"))
        groups.setdefault(shape, []).append(row)
    lines = [f"\n\n## Iteration{args.iteration} — opt-in FP32 parameter sweep\n\n",
             f"Explicit template registry: {len(manifest)} entries; no fixed-shape dispatch or default registry expansion. "
             "Original seven-product FP32/RN, kernel bodies, packing per call, binding guards, actual cooperative capacity, "
             "caller stream/workspace, and no run allocation/host synchronization preserved. "
             "Original JSON protocol: 20 warmup, 5x100 timings, 3 fresh original-input FP64 checks, 2 rotating input sets; ordinary full calls.\n\n",
             f"Exit={status}; rows={len(rows)}/{expected}; records={counts}; all_checks_pass={passed}.\n\n",
             "| MxNxK | best existing us | sweep best us | cuBLASLt us | speedup vs existing | speedup vs Lt | sweep ID |\n",
             "|---|---:|---:|---:|---:|---:|---:|\n"]
    summary = []
    for shape, cases in groups.items():
        good = [r for r in cases if r["numerical_pass"] == r["bitwise_match"] == "1"]
        controls = [r for r in good if manifest[int(r["configuration_id"])]["control"]]
        if not controls or not good:
            continue
        best = min(good, key=lambda r: float(r["solution_ms"]))
        old = min(controls, key=lambda r: float(r["solution_ms"]))
        new_us = 1000 * float(best["solution_ms"])
        old_us = 1000 * float(old["solution_ms"])
        lt_us = 1000 * float(best["dense_ms"])
        sid = manifest[int(best["configuration_id"])]["sweep_id"]
        lines.append(f"| {'x'.join(map(str, shape))} | {old_us:.3f} | {new_us:.3f} | {lt_us:.3f} | {old_us/new_us:.4f} | {lt_us/new_us:.4f} | {sid} |\n")
        summary.append(dict(shape=shape, old_us=old_us, new_us=new_us, lt_us=lt_us,
                            vs_existing=old_us/new_us, speedup=lt_us/new_us, sweep_id=sid))
    lines.append("\nScreening evidence only: best-of-pool measurements are separate from public tuner selection and independent final validation. All groups and failed trials retained.\n")
    with Path("ITERATIONS.md").open("a") as stream:
        stream.write("".join(lines))
    report = dict(status=status, rows=len(rows), expected_rows=expected, counts=counts,
                  all_checks_pass=passed, best=summary)
    (args.root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("".join(lines))


if __name__ == "__main__":
    main()

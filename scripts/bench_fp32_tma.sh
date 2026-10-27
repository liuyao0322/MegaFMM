#!/bin/bash
# Existing FP32 full-call benchmark contract, with isolated results/trajectory.
# Build and validate first; this script only measures the resulting executable.
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL="${1:?usage: bash scripts/bench_fp32_tma.sh iter-N}"
[[ "$LABEL" =~ ^iter-[0-9]+$ || "$LABEL" == final ]]
BUILD_DIR="${MEGAFMM_BUILD_DIR:-build-fp32}"
RESULT_DIR="${MEGAFMM_RESULT_DIR:-results/fp32_tma/$LABEL}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAJ_DIR="trajectory/${TIMESTAMP}_fp32-sw_${LABEL}"
mkdir -p "$RESULT_DIR" "$TRAJ_DIR"
snapshot() {
  cp -r solution/* "$TRAJ_DIR/"
  cp -r "$RESULT_DIR" "$TRAJ_DIR/results"
  git rev-parse HEAD > "$TRAJ_DIR/parent_commit.txt"
  find solution -type f -print0 | sort -z | xargs -0 sha256sum > "$TRAJ_DIR/identity.sha256"
  sha256sum "$BUILD_DIR/libmegafmm.a" "$BUILD_DIR/megafmm_bench" >> "$TRAJ_DIR/identity.sha256"
  printf 'Trajectory saved to: %s\n' "$TRAJ_DIR"
}
trap snapshot EXIT
bash scripts/on_gpu.sh bash -c '
  set -euo pipefail
  for mode in launch; do
    extra=()
    for algorithm in sw sw-tuned-strict; do
      "$1/megafmm_bench" --algorithm "$algorithm" --dtype fp32 \
        --shapes "1,1,1;33,65,129;512,512,512;1024,1024,1024;1025,1537,769" \
        --rotate-inputs 2 --correct-trials 3 --trials 20 --warmup 5 --groups 3 \
        --output "$2/${algorithm}_${mode}.csv" "${extra[@]}" \
        > "$2/${algorithm}_${mode}.log" 2>&1
    done
  done
' _ "$BUILD_DIR" "$RESULT_DIR"
/usr/bin/python3 - "$RESULT_DIR" <<'PY'
import csv,sys
from pathlib import Path
for p in sorted(Path(sys.argv[1]).glob('*.csv')):
    if 'sw_tuning' in p.name:
        continue
    rows=list(csv.DictReader(p.open()))
    assert len(rows)==5 and all(r['numerical_pass']=='1' for r in rows),p
    for r in rows:
        print(p.name, f"{r['m']}x{r['n']}x{r['k']}",
              f"SW={float(r['solution_event_ms'])*1000:.3f}us",
              f"Lt={float(r['dense_event_ms'])*1000:.3f}us",
              f"speedup={float(r['speedup']):.4f}",r['solution_lt_config'])
PY

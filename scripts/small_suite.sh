#!/usr/bin/env bash
# Reproducible small-matrix closure after scripts/build.sh has completed.
# Attach to the existing allocation through on_gpu.sh; never allocate a new job.
# Benchmark exit 2 means frozen numerical certification failed: retain every row
# and continue. Any other nonzero status is a runtime failure and aborts with 1.
set -euo pipefail
cd "$(dirname "$0")/.."

all_labels=(
  lt0_launch lt0_graph lt1_launch lt1_graph
  batch5_launch batch5_graph batch6_launch batch6_graph
  warp4_launch warp4_graph packed7_launch packed7_graph
  fast8_launch fast8_graph product9_launch product9_graph
  product10_launch product10_graph narrow11_launch narrow11_graph
  narrow12_launch narrow12_graph
)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf 'Usage: bash scripts/small_suite.sh [LABEL ...]\n'
  printf 'No labels runs all 22 cases; selected labels use the same full protocol.\n'
  printf 'Labels: %s\n' "${all_labels[*]}"
  printf 'Build first. Uses the current MEGAFMM_JOB_ID / existing allocation.\n'
  exit 0
fi
selected=("$@")
if (( ${#selected[@]} == 0 )); then selected=("${all_labels[@]}"); fi
declare -A selected_labels=()
for label in "${selected[@]}"; do
  valid=0
  for allowed in "${all_labels[@]}"; do
    if [[ "$allowed" == "$label" ]]; then valid=1; break; fi
  done
  if (( valid == 0 )); then
    printf 'Unknown measurement label: %s\n' "$label" >&2
    exit 1
  fi
  if [[ -n "${selected_labels[$label]:-}" ]]; then
    printf 'Duplicate measurement label: %s\n' "$label" >&2
    exit 1
  fi
  selected_labels[$label]=1
done
if [[ ! -x build/megafmm_bench ]]; then
  printf 'Missing executable build/megafmm_bench; build before running this suite.\n' >&2
  exit 1
fi

small_shapes=$(python3 - <<'PY'
import json
with open('bench/small_shapes.json', encoding='utf-8') as stream:
    suite = json.load(stream)
shapes = suite['squares'] + suite['rectangles']
expected = ';'.join(','.join(map(str, shape)) for shape in shapes)
assert len(shapes) == suite['shape_count'] == 9, 'expected the predeclared nine-shape suite'
assert all(len(shape) == 3 and all(isinstance(n, int) and n > 0 for n in shape)
           for shape in shapes), 'invalid shape'
assert suite['tuple_order'] == ['M', 'N', 'K'], 'unexpected shape ordering'
assert suite['cli_shapes'] == expected, 'cli_shapes differs from declared shapes'
assert (suite['full']['warmup'], suite['full']['trials'], suite['full']['groups'],
        suite['full']['correctness_trials']) == (20, 100, 5, 3), 'full protocol changed'
print(suite['cli_shapes'])
PY
)
source_commit=$(git rev-parse HEAD)
binary_sha256=$(sha256sum build/megafmm_bench)
binary_sha256=${binary_sha256%% *}
mkdir -p results
metadata=results/small_final_runs.csv

# Preserve unselected rows on a partial rerun, replace selected rows, and reject
# a mixed-binary archive. Selecting all labels deliberately starts a fresh suite.
python3 - "$metadata" "$binary_sha256" "${selected[@]}" <<'PY'
import csv
import sys
from pathlib import Path
path = Path(sys.argv[1])
binary = sys.argv[2]
selected = set(sys.argv[3:])
fields = ['label', 'source_commit', 'binary_sha256', 'exit_code']
retained = []
if len(selected) < 22 and path.is_file() and path.stat().st_size:
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise SystemExit('small_final_runs.csv schema differs; start a complete fresh suite')
        retained = [row for row in reader if row['label'] not in selected]
    if any(row['binary_sha256'] != binary for row in retained):
        raise SystemExit('unselected small_final rows use a different binary; start a complete fresh suite')
with path.open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(retained)
PY

# Clear every selected output before the first run, so an aborted suite leaves
# subsequent selected cases visibly empty rather than exposing stale timings.
for label in "${selected[@]}"; do
  : > "results/small_final_${label}.csv"
  : > "results/small_final_${label}.log"
  : > "results/small_final_${label}_telemetry_before.csv"
  : > "results/small_final_${label}_telemetry_after.csv"
done

common=(--dtype both --shapes "$small_shapes" --trials 100 --warmup 20
        --groups 5 --correct-trials 3 --rotate-inputs 1)
numeric_failure=0
telemetry() {
  local output=$1
  if ! bash scripts/on_gpu.sh bash -c 'exec nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" \
      --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
      --format=csv' > "$output"; then
    printf 'Telemetry failed: %s\n' "$output" >&2
    exit 1
  fi
}
run_case() {
  local label=$1
  shift
  if [[ -z "${selected_labels[$label]:-}" ]]; then return 0; fi
  local current_sha output status
  current_sha=$(sha256sum build/megafmm_bench)
  current_sha=${current_sha%% *}
  if [[ "$current_sha" != "$binary_sha256" ]]; then
    printf 'Benchmark binary changed during the suite; abort before %s.\n' "$label" >&2
    exit 1
  fi
  output="results/small_final_${label}.csv"
  telemetry "results/small_final_${label}_telemetry_before.csv"
  # The actual benchmark status is recorded even when it indicates failure.
  set +e
  bash scripts/on_gpu.sh ./build/megafmm_bench "${common[@]}" "$@" --output "$output" \
    > "results/small_final_${label}.log" 2>&1
  status=$?
  set -e
  printf '%s,%s,%s,%s\n' "$label" "$source_commit" "$binary_sha256" "$status" >> "$metadata"
  telemetry "results/small_final_${label}_telemetry_after.csv"
  current_sha=$(sha256sum build/megafmm_bench)
  current_sha=${current_sha%% *}
  if [[ "$current_sha" != "$binary_sha256" ]]; then
    printf 'Benchmark binary changed while %s ran; invalidate this run and abort.\n' "$label" >&2
    exit 1
  fi
  printf '%s finished: exit=%s; %s\n' "$label" "$status" "$output"
  if [[ "$status" == 2 ]]; then
    numeric_failure=2
  elif [[ "$status" != 0 ]]; then
    printf 'Runtime failure for %s (actual exit=%s); see its log.\n' "$label" "$status" >&2
    exit 1
  fi
}

# Same shape, dtype, workspace contract, and sampling configuration in both modes.
run_case lt0_launch --algorithm lt --lt-variant 0
run_case lt0_graph --algorithm lt --lt-variant 0 --graph
run_case lt1_launch --algorithm lt --lt-variant 1
run_case lt1_graph --algorithm lt --lt-variant 1 --graph
run_case batch5_launch --algorithm hopper --hopper-variant 5
run_case batch5_graph --algorithm hopper --hopper-variant 5 --graph
run_case batch6_launch --algorithm hopper --hopper-variant 6
run_case batch6_graph --algorithm hopper --hopper-variant 6 --graph
run_case warp4_launch --algorithm mega --mega-variant 4
run_case warp4_graph --algorithm mega --mega-variant 4 --graph
run_case packed7_launch --algorithm mega --mega-variant 7
run_case packed7_graph --algorithm mega --mega-variant 7 --graph
run_case fast8_launch --algorithm mega --mega-variant 8
run_case fast8_graph --algorithm mega --mega-variant 8 --graph
# Product variant 9 requires the product-parallel M1 integration and matching build.
run_case product9_launch --algorithm mega --mega-variant 9
run_case product9_graph --algorithm mega --mega-variant 9 --graph
run_case product10_launch --algorithm mega --mega-variant 10
run_case product10_graph --algorithm mega --mega-variant 10 --graph
run_case narrow11_launch --algorithm mega --mega-variant 11
run_case narrow11_graph --algorithm mega --mega-variant 11 --graph
run_case narrow12_launch --algorithm mega --mega-variant 12
run_case narrow12_graph --algorithm mega --mega-variant 12 --graph
exit "$numeric_failure"

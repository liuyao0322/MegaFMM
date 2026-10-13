#!/usr/bin/env bash
# Full measurements. Selected outputs start empty; exit 2 remains explicit.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
primary_shapes=$(python3 - <<'PY'
import json
p=json.load(open('bench/shapes.json'))
print(';'.join(','.join(map(str,s)) for key in ('squares','rectangles','boundaries') for s in p[key]))
PY
)
# Declared before measurement: a diagnostic neighborhood augments the original
# primary set; it does not replace any original square/rectangle/boundary.
neighbor_shapes='12288,12288,12288;14336,14336,14336'
large_shapes='4096,4096,4096;8192,8192,8192;12288,12288,12288;14336,14336,14336;16384,16384,16384'
rotation_shapes='8192,8192,8192;16384,16384,16384'
source_commit=$(git rev-parse HEAD)
selected=("$@")
valid_labels=' lt_launch lt_graph lt_rotating lt_rotating_graph hopper3_launch hopper4_launch mega3_launch hopper3_graph hopper4_graph '
for label in "${selected[@]}"; do
  if [[ "$valid_labels" != *" $label "* ]]; then
    printf 'Unknown measurement label: %s\n' "$label" >&2
    exit 1
  fi
done
if (( ${#selected[@]} == 0 )) || [[ ! -s results/final_runs.csv ]]; then
  printf 'label,source_commit,exit_code\n' > results/final_runs.csv
fi
common=(--dtype both --trials 100 --warmup 20 --groups 5 --correct-trials 3)
numeric_failure=0
run_case() {
  local label=$1
  shift
  if (( ${#selected[@]} > 0 )); then
    local match=0
    local requested
    for requested in "${selected[@]}"; do
      if [[ "$requested" == "$label" ]]; then match=1; break; fi
    done
    if (( match == 0 )); then return; fi
  fi
  local output="results/final_${label}.csv"
  : > "$output"
  bash scripts/on_gpu.sh bash -c 'nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" \
    --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv' > "results/final_${label}_telemetry_before.csv"
  set +e
  bash scripts/on_gpu.sh ./build/megafmm_bench "${common[@]}" "$@" --output "$output" \
    > "results/final_${label}.log" 2>&1
  local status=$?
  set -e
  printf '%s,%s,%s\n' "$label" "$source_commit" "$status" >> results/final_runs.csv
  bash scripts/on_gpu.sh bash -c 'nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" \
    --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv' > "results/final_${label}_telemetry_after.csv"
  printf '%s finished: exit=%s; %s\n' "$label" "$status" "$output"
  if [[ "$status" == 2 ]]; then numeric_failure=2
  elif [[ "$status" != 0 ]]; then exit "$status"; fi
}
run_case lt_launch --algorithm lt --shapes "$primary_shapes;$neighbor_shapes"
run_case lt_graph --algorithm lt --shapes "$large_shapes" --graph
run_case lt_rotating --algorithm lt --shapes "$rotation_shapes" --rotate-inputs 4
run_case lt_rotating_graph --algorithm lt --shapes "$rotation_shapes" --rotate-inputs 4 --graph
run_case hopper3_launch --algorithm hopper --hopper-variant 3 --shapes "$large_shapes"
run_case hopper4_launch --algorithm hopper --hopper-variant 4 --shapes "$large_shapes"
run_case mega3_launch --algorithm mega --mega-variant 3 --shapes '4096,4096,4096;8192,8192,8192'
run_case hopper3_graph --algorithm hopper --hopper-variant 3 --shapes "$large_shapes" --graph
run_case hopper4_graph --algorithm hopper --hopper-variant 4 --shapes "$large_shapes" --graph
exit "$numeric_failure"

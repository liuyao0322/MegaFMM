#!/bin/bash
set -euo pipefail
cd /share/home/liuyao/workspace/MegaFMM
root=trajectory/20261016_sw1024_remeasure
nvidia-smi --query-gpu=name,uuid,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv > "$root/device_before.csv"
for repeat in 0 1 2; do
  for mode in launch graph; do
    flags=()
    if [[ "$mode" == graph ]]; then flags+=(--graph); fi
    set +e
    ./build/megafmm_bench --algorithm sw --shapes 1024,1024,1024 --dtype both --trials 100 --warmup 20 --groups 5 --correct-trials 3 --rotate-inputs 2 --output "$root/${mode}_${repeat}.csv" "${flags[@]}" > "$root/${mode}_${repeat}.log" 2>&1
    result=$?
    set -e
    printf '%s\n' "$result" > "$root/${mode}_${repeat}.exit"
    if [[ "$result" != 0 && "$result" != 2 ]]; then exit "$result"; fi
  done
done
nvidia-smi --query-gpu=name,uuid,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv > "$root/device_after.csv"

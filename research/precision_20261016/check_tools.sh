#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
root=research/precision_20261016
mkdir -p "$root/ncu/tmp"
export TMPDIR="$PWD/$root/ncu/tmp"
/share/app/cuda/cuda-12.8/bin/ncu --target-processes all --profile-from-start off \
  --metrics gpu__time_duration.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__shared_mem_per_block_dynamic \
  --clock-control none --cache-control none --force-overwrite -o "$root/ncu/direct_b128_w1" \
  "$root/validate_b128_w1" --profile > "$root/ncu/profile.log" 2>&1
/share/app/cuda/cuda-12.8/bin/ncu --import "$root/ncu/direct_b128_w1.ncu-rep" --page raw --csv > "$root/ncu/metrics.csv"
/share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  "$root/validate_b128_w1" > "$root/memcheck.log" 2>&1
/share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name kns=direct_quadrants \
  "$root/validate_b128_w1" > "$root/racecheck_scoped.log" 2>&1
/share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool synccheck --error-exitcode 99 \
  --kernel-name kns=direct_quadrants "$root/validate_b128_w1" > "$root/synccheck_scoped.log" 2>&1

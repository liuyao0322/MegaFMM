#!/bin/bash
set -euo pipefail
cd /share/home/liuyao/workspace/MegaFMM
root=trajectory/20261016_sw_slowpoint_analysis
ps -eo comm,args > "$root/processes_before.txt"
# The documented workaround uses an owned TMPDIR; no existing lock is removed.
mkdir -p /tmp/liuyao-megafmm-analysis-ncu
export TMPDIR=/tmp/liuyao-megafmm-analysis-ncu
set +e
/share/app/cuda/cuda-12.8/bin/ncu --profile-from-start off --target-processes all --section SpeedOfLight --section LaunchStats --section Occupancy --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis_Chart --force-overwrite -o "$root/ncu_4096" --csv --log-file "$root/ncu_4096.csv" ./build/megafmm_bench --algorithm sw --shapes 4096,4096,4096 --dtype bf16 --profile > "$root/ncu_4096_driver.log" 2>&1
result=$?
printf '%s\n' "$result" > "$root/ncu_4096.exit"
exit "$result"

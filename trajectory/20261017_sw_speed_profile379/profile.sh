#!/bin/bash
# Sequential diagnostic capture after formal/validation work has completed.
set -euo pipefail
cd /share/home/liuyao/workspace/MegaFMM
root=research/sw_speed_20261017
out="${1:?profile output directory required}"
mkdir -p "$out" /tmp/liuyao-megafmm-speed-ncu
export TMPDIR=/tmp/liuyao-megafmm-speed-ncu
nvcc -O2 -std=c++17 -arch=sm_90a -Iinclude -Isolution "$root/profile.cu" build-sw-speed/libmegafmm.a -lcublas -lcublasLt -lcuda -o "$out/profile"
sha256sum build-sw-speed/libmegafmm.a build-sw-speed/megafmm_bench "$out/profile" > "$out/binary.sha256"
# BF16 precision-valid aligned wide consumer and padded narrow consumer.
for id in 142 160; do
 /share/app/cuda/cuda-12.8/bin/ncu --profile-from-start off --target-processes all --kernel-name 'regex:compute' --launch-count 1 --section SpeedOfLight --section LaunchStats --section Occupancy --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis_Chart --force-overwrite -o "$out/ncu_$id" --csv --log-file "$out/ncu_$id.csv" "$out/profile" "$id" > "$out/ncu_${id}_driver.log" 2>&1
 /share/app/cuda/cuda-12.8/bin/ncu --import "$out/ncu_$id.ncu-rep" --page raw --csv > "$out/ncu_${id}_raw.csv"
done
sha256sum --check "$out/binary.sha256"

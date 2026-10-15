#!/bin/bash
set -euo pipefail
work=research/falcon_layout_fusion/check335
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include "$work/check.cu" -lcuda -o "$work/check" -Xptxas=-v > "$work/build.log" 2>&1
"$work/check" > "$work/stress.log" 2>&1
for checker in memcheck racecheck synccheck; do
 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool "$checker" --error-exitcode 99 "$work/check" --quick > "$work/$checker.log" 2>&1
done
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include "$work/multistream.cu" -lcuda -o "$work/multistream" > "$work/multistream_build.log" 2>&1
"$work/multistream" > "$work/multistream.log" 2>&1

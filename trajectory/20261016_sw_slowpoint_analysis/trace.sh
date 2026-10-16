#!/bin/bash
set -euo pipefail
cd /share/home/liuyao/workspace/MegaFMM
root=trajectory/20261016_sw_slowpoint_analysis
set +e
/share/app/cuda/cuda-12.8/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true --output="$root/timeline" build/megafmm_bench --algorithm sw --shapes '512,512,512;2048,2048,2048;4096,4096,4096;16384,16384,16384;8192,1024,512;1024,8192,512;514,770,258' --dtype bf16 --trials 100 --warmup 20 --groups 5 --correct-trials 3 --rotate-inputs 2 --output "$root/traced_results.csv" > "$root/timeline_driver.log" 2>&1
result=$?
printf '%s\n' "$result" > "$root/timeline.exit"
set -e
if [[ "$result" != 0 && "$result" != 2 ]];then exit "$result";fi
/share/app/cuda/cuda-12.8/bin/nsys export --type sqlite --force-overwrite=true --output "$root/timeline.sqlite" "$root/timeline.nsys-rep" > "$root/timeline_export.log" 2>&1

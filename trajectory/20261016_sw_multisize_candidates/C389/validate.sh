#!/bin/bash
set -euo pipefail
cd /share/home/liuyao/workspace/MegaFMM
root=results/stage389
for tool in memcheck synccheck racecheck; do
 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool "$tool" --error-exitcode 97 "$root/test_sw_pipeline" --hopper-only > "$root/$tool.log" 2>&1
done
cuobjdump --dump-sass "$root/sw_hopper.o" > "$root/sass.txt"

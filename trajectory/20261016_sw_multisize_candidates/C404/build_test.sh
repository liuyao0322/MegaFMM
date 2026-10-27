#!/bin/bash
set -euo pipefail
cd /share/home/liuyao/workspace/MegaFMM
root=results/stage404
nvcc -O3 -DNDEBUG -std=c++17 --expt-relaxed-constexpr -gencode=arch=compute_90a,code=sm_90a -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -Iinclude -Isolution -Ithird_party/cutlass/include -c "$root/sw_hopper.cu" -o "$root/sw_hopper.o" > "$root/build.log" 2>&1
for target in megafmm_bench test_sw_pipeline; do
  object="build/CMakeFiles/$target.dir/bench/benchmark.cu.o"
  if [[ "$target" == test_sw_pipeline ]]; then object="build/CMakeFiles/$target.dir/tests/test_sw_pipeline.cu.o"; fi
  nvcc -gencode=arch=compute_90a,code=sm_90a "$object" "$root/sw_hopper.o" build/libmegafmm.a -L/share/app/cuda/cuda-12.4/targets/x86_64-linux/lib/stubs -lcuda -lcublasLt -lcublas -o "$root/$target" >> "$root/build.log" 2>&1
done
cuobjdump --dump-resource-usage "$root/sw_hopper.o" > "$root/resources.txt"
"$root/test_sw_pipeline" > "$root/test.log" 2>&1
set +e
"$root/megafmm_bench" --algorithm sw --shapes '8192,8192,8192;16384,16384,16384' --dtype both --trials 100 --warmup 20 --groups 5 --correct-trials 3 --rotate-inputs 2 --output "$root/results.csv" > "$root/timing.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$root/timing.exit"
if [[ "$result" != 0 && "$result" != 2 ]]; then exit "$result"; fi

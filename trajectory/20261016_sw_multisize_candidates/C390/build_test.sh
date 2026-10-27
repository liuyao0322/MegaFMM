#!/bin/bash
set -euo pipefail
cd /share/home/liuyao/workspace/MegaFMM
root=results/stage390
nvcc -O3 -DNDEBUG -std=c++17 -gencode=arch=compute_90a,code=sm_90a -Iinclude -c "$root/sw_seven.cu" -o "$root/sw_seven.o" > "$root/build.log" 2>&1
g++ -O3 -DNDEBUG -std=c++17 -Iinclude -Isolution -I/share/app/cuda/cuda-12.4/include -c "$root/plan.cpp" -o "$root/plan.o" >> "$root/build.log" 2>&1
nvcc -gencode=arch=compute_90a,code=sm_90a build/CMakeFiles/megafmm_bench.dir/bench/benchmark.cu.o "$root/plan.o" "$root/sw_seven.o" build/libmegafmm.a -L/share/app/cuda/cuda-12.4/targets/x86_64-linux/lib/stubs -lcuda -lcublasLt -lcublas -o "$root/megafmm_bench" >> "$root/build.log" 2>&1
set +e
"$root/megafmm_bench" --algorithm sw --shapes '4096,4096,4096;8192,8192,8192;16384,16384,16384' --dtype both --trials 100 --warmup 20 --groups 5 --correct-trials 3 --rotate-inputs 2 --output "$root/results.csv" > "$root/timing.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$root/timing.exit"
if [[ "$result" != 0 && "$result" != 2 ]]; then exit "$result"; fi

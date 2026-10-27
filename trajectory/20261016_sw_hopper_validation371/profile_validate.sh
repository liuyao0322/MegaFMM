#!/bin/bash
set -euo pipefail
cd /share/home/liuyao/workspace/MegaFMM
root=results/sw_hopper_validation371
cuobjdump --dump-resource-usage build/CMakeFiles/megafmm.dir/solution/sw_hopper.cu.o > "$root/resources.txt"
set +e
/share/app/cuda/cuda-12.8/bin/ncu --target-processes all --profile-from-start off --set full --force-overwrite -o "$root/profile" --csv --log-file "$root/profile.csv" ./build/megafmm_bench --algorithm sw --shapes 2048,2048,2048 --dtype bf16 --profile > "$root/profile_driver.log" 2>&1
profile_exit=$?
set -e
printf '%s\n' "$profile_exit" > "$root/profile.exit"
for tool in memcheck racecheck synccheck; do
  /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 ./build/test_sw_pipeline --hopper-only > "$root/$tool.log" 2>&1
done

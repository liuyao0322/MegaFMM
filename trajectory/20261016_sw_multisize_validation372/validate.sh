#!/bin/bash
set -euo pipefail
cd /share/home/liuyao/workspace/MegaFMM
root=trajectory/20261016_sw_multisize_validation372
sha256sum solution/* include/megafmm/gemm.h tests/test_sw_pipeline.cu scripts/generate_sw_hopper.py scripts/generate_sw_padded.py > "$root/source.sha256"
sha256sum build/megafmm_bench build/test_sw_pipeline build/libmegafmm.a > "$root/binaries.sha256"
echo "Running four relevant CTest checks"
ctest --test-dir build --output-on-failure -R '^(gpu_correctness|exact_winograd_dag|sw_pipeline_implementation|tuning_confirmation_policy)$' > "$root/ctest.log" 2>&1
echo "CTest passed"
for tool in memcheck synccheck racecheck; do
 echo "Running $tool: 120 guarded fresh direct/Graph calls"
 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool "$tool" --error-exitcode 97 build/test_sw_pipeline --hopper-only > "$root/$tool.log" 2>&1
done
cuobjdump --dump-resource-usage build/CMakeFiles/megafmm.dir/solution/sw_hopper.cu.o > "$root/resources.txt"
for mode in launch graph; do
 for process in 1 2 3; do
  echo "Independent benchmark: $mode process $process / 3"
  args=();if [[ "$mode" == graph ]];then args+=(--graph);fi
  set +e
  build/megafmm_bench --algorithm sw --shapes '1025,1537,769;514,770,258;1,1,1' --dtype both --trials 100 --warmup 20 --groups 5 --correct-trials 3 --rotate-inputs 2 "${args[@]}" --output "$root/${mode}_${process}.csv" > "$root/${mode}_${process}.log" 2>&1
  result=$?
  set -e
  printf '%s\n' "$result" > "$root/${mode}_${process}.exit"
  if [[ "$result" != 0 && "$result" != 2 ]];then exit "$result";fi
 done
done

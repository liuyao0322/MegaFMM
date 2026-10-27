#!/bin/bash
set -euo pipefail
timeout -k 5 300 ctest --test-dir build --output-on-failure -R '^(gpu_correctness|exact_winograd_dag|sw_multisize_implementation|tuned_correctness|tuning_candidates_correctness|tuning_precision_guard)$' > results/mnk369_ctest.log 2>&1
timeout -k 5 120 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool memcheck --error-exitcode 99 ./build/test_tuning_precision > results/mnk369_guard_memcheck.log 2>&1
timeout -k 5 120 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool racecheck --error-exitcode 99 ./build/test_tuning_precision > results/mnk369_guard_racecheck.log 2>&1
timeout -k 5 600 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool memcheck --error-exitcode 99 ./build/test_tuning_candidates --wmma > results/mnk369_memcheck.log 2>&1
timeout -k 5 900 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool racecheck --error-exitcode 99 ./build/test_tuning_candidates --wmma > results/mnk369_racecheck.log 2>&1

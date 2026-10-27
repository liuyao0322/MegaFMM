#!/bin/bash
set -euo pipefail
ctest --test-dir build -R '^(gpu_correctness|exact_winograd_dag|sw_pipeline_implementation|tuning_confirmation_policy)$' --output-on-failure > results/restore_sw1024/ctest.log 2>&1
set +e
./build/test_sw1024 > results/restore_sw1024/precision.log 2>&1
precision_exit=$?
set -e
printf '%s\n' "$precision_exit" > results/restore_sw1024/precision.exit
# Code 2 denotes known numerical failures; code 1 is an implementation error.
if [[ "$precision_exit" != 0 && "$precision_exit" != 2 ]]; then exit "$precision_exit"; fi
set +e
/share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool memcheck --error-exitcode 99 ./build/test_sw1024 --quick > results/restore_sw1024/memcheck.log 2>&1
memcheck_exit=$?
set -e
printf '%s\n' "$memcheck_exit" > results/restore_sw1024/memcheck.exit
if [[ "$memcheck_exit" != 0 && "$memcheck_exit" != 2 ]]; then exit "$memcheck_exit"; fi
rg -q 'ERROR SUMMARY: 0 errors' results/restore_sw1024/memcheck.log

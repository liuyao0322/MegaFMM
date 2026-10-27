#!/bin/bash
set -euo pipefail
for mode in launch graph; do
 args=()
 if [[ "$mode" == graph ]]; then args+=(--graph); fi
 set +e
 ./build/megafmm_bench --algorithm sw --shapes 1024,1024,1024 --dtype both --trials 100 --warmup 20 --groups 5 --correct-trials 3 --output "results/profile_iter97/${mode}.csv" "${args[@]}" > "results/profile_iter97/${mode}.log" 2>&1
 result=$?
 set -e
 [[ $result == 2 ]] || { echo "Unexpected exit $result for $mode"; exit 1; }
done

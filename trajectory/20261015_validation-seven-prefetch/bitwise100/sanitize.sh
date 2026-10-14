#!/bin/bash
set -euo pipefail
for checker in memcheck racecheck synccheck; do
 args=()
 if [[ "$checker" != memcheck ]]; then args+=(--kernel-name kns=sw_cooperative_hilo); fi
 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool "$checker" --error-exitcode 99 "${args[@]}" ./results/opt_seven_round/bitwise100/check --quick > "results/opt_seven_round/bitwise100/${checker}.log" 2>&1
done
cuobjdump --dump-sass results/opt_seven_round/iter100_bench > results/opt_seven_round/profile100/production.sass
cuobjdump --dump-sass results/opt_seven_round/profile100/diagnostics/measure > results/opt_seven_round/profile100/diagnostic.sass

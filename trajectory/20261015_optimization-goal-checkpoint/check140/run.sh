set -eu
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/opt_goal/check140/check.cu -lcuda -o results/opt_goal/check140/check > results/opt_goal/check140/build.log 2>&1
results/opt_goal/check140/check > results/opt_goal/check140/stress.log 2>&1
for checker in memcheck racecheck synccheck; do
 args=()
 if [[ "$checker" != memcheck ]];then args+=(--kernel-name kns=sw_cooperative_hilo);fi
 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool "$checker" --error-exitcode 99 "${args[@]}" results/opt_goal/check140/check --quick > results/opt_goal/check140/$checker.log 2>&1
done

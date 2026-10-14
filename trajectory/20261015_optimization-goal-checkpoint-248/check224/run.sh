set -eu
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/opt_goal/check224/check.cu -lcuda -o results/opt_goal/check224/check > results/opt_goal/check224/build.log 2>&1
results/opt_goal/check224/check > results/opt_goal/check224/stress.log 2>&1
for checker in memcheck racecheck synccheck; do
 args=()
 if [[ "$checker" != memcheck ]];then args+=(--kernel-name kns=sw_cooperative_hilo);fi
 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool "$checker" --error-exitcode 99 "${args[@]}" results/opt_goal/check224/check --quick > results/opt_goal/check224/$checker.log 2>&1
done

nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/opt_goal/check224/multistream.cu -lcuda -o results/opt_goal/check224/multistream > results/opt_goal/check224/multistream_build.log 2>&1
results/opt_goal/check224/multistream > results/opt_goal/check224/multistream.log 2>&1

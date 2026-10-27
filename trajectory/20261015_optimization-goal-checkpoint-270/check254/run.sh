set -eu
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/opt_goal/check254/check.cu -lcuda -o results/opt_goal/check254/check > results/opt_goal/check254/build.log 2>&1
results/opt_goal/check254/check > results/opt_goal/check254/stress.log 2>&1
for checker in memcheck racecheck synccheck; do
 args=()

 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool "$checker" --error-exitcode 99 "${args[@]}" results/opt_goal/check254/check --quick > results/opt_goal/check254/$checker.log 2>&1
done

nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/opt_goal/check254/multistream.cu -lcuda -o results/opt_goal/check254/multistream > results/opt_goal/check254/multistream_build.log 2>&1
results/opt_goal/check254/multistream > results/opt_goal/check254/multistream.log 2>&1

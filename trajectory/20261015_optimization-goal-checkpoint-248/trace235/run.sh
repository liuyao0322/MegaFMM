set -eu
export TMPDIR="$PWD/results/opt_goal/trace235/tmp"
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/opt_goal/trace235/driver.cu -lcuda -o results/opt_goal/trace235/driver > results/opt_goal/trace235/build.log 2>&1
results/opt_goal/trace235/driver > results/opt_goal/trace235/timing.csv 2>results/opt_goal/trace235/stderr.log
cuobjdump --dump-resource-usage results/opt_goal/trace235/driver > results/opt_goal/trace235/resources.log

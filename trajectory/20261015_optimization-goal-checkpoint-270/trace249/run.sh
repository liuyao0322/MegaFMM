set -eu
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/opt_goal/trace249/driver.cu -lcuda -o results/opt_goal/trace249/driver > results/opt_goal/trace249/build.log 2>&1
cuobjdump --dump-resource-usage results/opt_goal/trace249/driver > results/opt_goal/trace249/resources.log
results/opt_goal/trace249/driver > results/opt_goal/trace249/timing.csv 2>results/opt_goal/trace249/stderr.log

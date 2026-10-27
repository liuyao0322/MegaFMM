set -eu
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/opt_goal/phasetrace282/driver.cu -lcuda -o results/opt_goal/phasetrace282/driver > results/opt_goal/phasetrace282/build.log 2>&1
cuobjdump --dump-resource-usage results/opt_goal/phasetrace282/driver > results/opt_goal/phasetrace282/resources.log
results/opt_goal/phasetrace282/driver > results/opt_goal/phasetrace282/timing.csv 2>results/opt_goal/phasetrace282/stderr.log

set -eu
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/opt_goal/profile162/diagnostics/source/measure.cu -lcuda -o results/opt_goal/profile162/diagnostics/measure -Xptxas=-v > results/opt_goal/profile162/diagnostics/build.log 2>&1
results/opt_goal/profile162/diagnostics/measure results/opt_goal/profile162/diagnostics 5 > results/opt_goal/profile162/diagnostics/run.log 2>&1

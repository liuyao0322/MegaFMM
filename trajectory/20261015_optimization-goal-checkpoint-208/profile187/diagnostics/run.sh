set -eu
p=results/opt_goal/profile187/diagnostics
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include "$p/measure.cu" -lcuda -o "$p/measure" -Xptxas=-v > "$p/build.log" 2>&1
"$p/measure" > "$p/records.csv" 2> "$p/run.log"

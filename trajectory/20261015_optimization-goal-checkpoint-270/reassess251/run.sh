set -eu
export TMPDIR="$PWD/results/opt_goal/reassess251/tmp"
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/opt_goal/reassess251/profile.cu -lcuda -o results/opt_goal/reassess251/profile > results/opt_goal/reassess251/build.log 2>&1
/share/app/cuda/cuda-12.8/bin/ncu --profile-from-start off --set full --kernel-name regex:sw_cooperative_hilo --launch-count 6 --clock-control none --cache-control none --force-overwrite -o results/opt_goal/reassess251/full results/opt_goal/reassess251/profile > results/opt_goal/reassess251/full.log 2>&1
/share/app/cuda/cuda-12.8/bin/ncu --import results/opt_goal/reassess251/full.ncu-rep --page raw --csv > results/opt_goal/reassess251/raw.csv
/share/app/cuda/cuda-12.8/bin/ncu --import results/opt_goal/reassess251/full.ncu-rep --page source --print-source sass --csv > results/opt_goal/reassess251/sass.csv

set -eu
export TMPDIR="$PWD/results/opt_goal/reassess277/tmp"
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/opt_goal/reassess277/profile.cu -lcuda -o results/opt_goal/reassess277/profile > results/opt_goal/reassess277/build.log 2>&1
/share/app/cuda/cuda-12.8/bin/ncu --profile-from-start off --set full --kernel-name regex:sw_cooperative_hilo --launch-count 4 --clock-control none --cache-control none --force-overwrite -o results/opt_goal/reassess277/full results/opt_goal/reassess277/profile > results/opt_goal/reassess277/full.log 2>&1
/share/app/cuda/cuda-12.8/bin/ncu --import results/opt_goal/reassess277/full.ncu-rep --page raw --csv > results/opt_goal/reassess277/raw.csv
/share/app/cuda/cuda-12.8/bin/ncu --import results/opt_goal/reassess277/full.ncu-rep --page source --print-source sass --csv > results/opt_goal/reassess277/sass.csv

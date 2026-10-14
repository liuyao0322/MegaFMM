set -eu
export TMPDIR="$PWD/results/opt_goal/reassess292/tmp"
/share/app/cuda/cuda-12.8/bin/ncu --profile-from-start off --set full --kernel-name regex:sw_cooperative_hilo --launch-count 2 --clock-control none --cache-control none --force-overwrite -o results/opt_goal/reassess292/full results/opt_seven_round/iter282_bench --algorithm sw --shapes 1024,1024,1024 --dtype both --profile > results/opt_goal/reassess292/output.log 2>&1
/share/app/cuda/cuda-12.8/bin/ncu --import results/opt_goal/reassess292/full.ncu-rep --page raw --csv > results/opt_goal/reassess292/raw.csv
/share/app/cuda/cuda-12.8/bin/ncu --import results/opt_goal/reassess292/full.ncu-rep --page source --print-source sass --csv > results/opt_goal/reassess292/sass.csv

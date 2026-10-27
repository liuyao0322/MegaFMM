set -eu
export TMPDIR="$PWD/results/opt_goal/nsys252/tmp"
nsys profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --force-overwrite=true -o results/opt_goal/nsys252/full ./build/megafmm_bench --algorithm sw --shapes 1024,1024,1024 --dtype both --profile > results/opt_goal/nsys252/output.log 2>&1

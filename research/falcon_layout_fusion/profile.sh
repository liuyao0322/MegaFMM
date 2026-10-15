#!/bin/bash
set -euo pipefail
base=research/falcon_layout_fusion
mkdir -p "$base/profile/tmp"
export TMPDIR="$PWD/$base/profile/tmp"
metrics=gpu__time_duration.sum,launch__registers_per_thread,smsp__inst_executed.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
for version in iter331 iter332 iter333 iter335; do
 /share/app/cuda/cuda-12.8/bin/ncu --profile-from-start off --kernel-name regex:sw_ --launch-count 4 --clock-control none --cache-control none --metrics "$metrics" --force-overwrite -o "$base/profile/$version" "$base/${version}_bench" --algorithm sw --shapes 1024,1024,1024 --dtype both --profile > "$base/profile/$version.log" 2>&1
 /share/app/cuda/cuda-12.8/bin/ncu --import "$base/profile/$version.ncu-rep" --page raw --csv > "$base/profile/$version.csv"
 echo "$version profile done"
done

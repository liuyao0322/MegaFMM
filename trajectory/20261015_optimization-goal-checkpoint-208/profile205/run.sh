#!/bin/bash
set -eu
export TMPDIR="$PWD/results/opt_goal/profile205/tmp"
/share/app/cuda/cuda-12.8/bin/ncu --profile-from-start off --kernel-name regex:sw_cooperative_hilo --launch-count 2 --clock-control none --cache-control none --metrics gpu__time_duration.sum,launch__registers_per_thread,smsp__inst_executed.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active --csv results/opt_seven_round/iter205_bench --algorithm sw --shapes 1024,1024,1024 --dtype both --profile > results/opt_goal/profile205/raw.csv 2>results/opt_goal/profile205/stderr.log

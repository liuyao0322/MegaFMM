#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/fused_native
mkdir -p "$r/ncu/tmp/profile_n3"
export TMPDIR="$PWD/$r/ncu/tmp/profile_n3"
/share/app/cuda/cuda-12.8/bin/ncu --target-processes all --kernel-name-base demangled \
 --kernel-name 'regex:native_fusion::fused' --launch-count 1 --cache-control none --clock-control none \
 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section LaunchStats \
 --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,dram__bytes.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum \
 -o "$r/ncu/profile_n3" "$r/n3/bin/compare" --quick

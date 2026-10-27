#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
MEGAFMM_JOB_ID="${MEGAFMM_JOB_ID:-107488}"
if [[ -z "${SLURM_STEP_ID:-}" ]]; then
  exec srun --jobid="$MEGAFMM_JOB_ID" --ntasks=1 --cpus-per-task=4 --gres=gpu:nvidia:1 bash scripts/on_gpu.sh "$@"
fi
module load cuda/12.4
export CUDA_MODULE_LOADING=EAGER
exec "$@"

#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/int8_hardware
env -u LD_LIBRARY_PATH /share/home/liuyao/miniconda3/envs/kernelagent/bin/python \
  "$r/fine_quant_model.py" --output "$r/fine_quant_model.json"

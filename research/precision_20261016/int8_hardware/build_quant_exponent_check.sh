#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/int8_hardware
nvcc -O3 -std=c++17 -arch=sm_90a -Xptxas=-v "$r/check_native_quant_exponent.cu" -o "$r/bin/check_native_quant_exponent"

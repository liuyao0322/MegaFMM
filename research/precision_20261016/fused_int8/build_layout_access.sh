#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/fused_int8
g++ -O2 -std=c++17 -Ithird_party/cutlass/include -I/share/app/cuda/cuda-12.4/include "$r/layout_access.cpp" -o "$r/bin/layout_access"
"$r/bin/layout_access"

#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
root=research/precision_20261016/int8_hardware
# Historical P7 validation. Refuse a later working source/binary before overwriting evidence.
python3 - <<'PY'
import hashlib
from pathlib import Path
r=Path('research/precision_20261016/int8_hardware')
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:
 assert (r/name).read_bytes()==(r/'p7'/name).read_bytes(), 'restore matching P7 checkout before validation'
assert hashlib.sha256((r/'bin/compare_p7').read_bytes()).hexdigest()=='98cbe1b765c666960436cd69fc7af70c1098f052b40af34ea607886ab60e026a'
PY
"$root/bin/compare_p7" --graph > "$root/p7_graph.log" 2>&1
bash "$root/build_stress.sh" > "$root/stress_build.log" 2>&1
"$root/bin/stress" > "$root/stress.log" 2>&1
# Full uninstrumented72-case/576-call stress above is the acceptance run.
# Sanitizers use24cases/48calls, all six extended patterns and both streams/dtypes.
for tool in memcheck racecheck synccheck; do
 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 \
  --kernel-name kns=int8_research "$root/bin/stress" --sanitizer > "$root/${tool}.log" 2>&1
done

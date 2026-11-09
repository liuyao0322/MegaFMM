#!/bin/bash
# Run after building/validating the isolated tree. Counts come from shapes.json.
set -euo pipefail
cd "$(dirname "$0")/.."
label="${1:?iteration label required}"
result="${MEGAFMM_RESULT_DIR:-results/tensorcore-$label}"
build="${MEGAFMM_BUILD_DIR:-build}"
mkdir -p "$result"
test ! -e "$result/current.csv"
python3 - "$result" "$build" <<'PY'
import hashlib,json,sys,subprocess
from pathlib import Path
out,build=map(Path,sys.argv[1:]);files=[p for p in Path('solution').rglob('*') if p.is_file()]+[Path('include/megafmm/gemm.h'),Path('bench/benchmark.cu'),Path('bench/shapes.json'),build/'megafmm_bench']
(out/'identity.json').write_text(json.dumps({'parent':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
PY
args=()
[[ -z "${MEGAFMM_SHAPES:-}" ]] || args+=(--shapes "$MEGAFMM_SHAPES")
set +e
python3 scripts/run_benchmark.py --config bench/shapes.json --binary "$build/megafmm_bench" --algorithm sw-tuned --rotate-inputs 2 --output "$result/current.csv" "${args[@]}" > "$result/output.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$result/exitcode"
mkdir -p "trajectory/tensorcore-$label"
cp -r solution "trajectory/tensorcore-$label/"
cp "$result/current.csv" "trajectory/tensorcore-$label/results.csv"
python3 - "$result/current.csv" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
for r in rows:
 print(r['m'],r['n'],r['k'],r['dtype'],'SW_ms='+r['solution_event_ms'],'Lt_ms='+r['dense_event_ms'],'speedup='+r['speedup'],'precision='+r['numerical_pass'],r['solution_lt_config'].split(';selected=',1)[-1].split(';median_ms=')[0])
print('ROWS',len(rows))
PY
exit "$status"

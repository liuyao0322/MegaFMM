"""Complete the formal archive after close/commit and policy audit.

This script does not launch GPU work or infer certification from requested tests.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil

p = Path('results/iter383')
log = (p / 'driver.log').read_text()
audit = json.loads((p / 'audit.json').read_text())
assert audit['rows'] == 28
frozen = json.loads((p / 'source_before_build.json').read_text())['source_sha256']
for name, digest in frozen.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
archive = Path(re.findall(r'Trajectory saved to: (\S+)', log)[-1])
batch = [x for x in log.splitlines() if x.startswith('PASS seven_batch ')]
native = [x for x in log.splitlines() if x.startswith('PASS native SW candidates ')]
small = [x for x in log.splitlines() if x.startswith('PASS small_native ')]
assert len(batch) == 56 and len(native) == 8 and len(small) == 90
expected_small = {(s,d,str(i)) for s in ['128x256x512','512x512x512','512x1024x512','1024x1024x512','512x512x256','512x512x1024','8192x1024x512','1024x8192x512','2048x2048x1024'] for d in ['bf16','fp16'] for i in range(6 if s.endswith('x512') else 3)}
observed_small = {re.search(r'PASS small_native (\S+) (bf16|fp16) config=(\d+)',x).groups() for x in small}
assert observed_small == expected_small
assert all('fresh=5 graph+range+guards+integer_FP64+fractional_native_bitwise' in x for x in small)
assert all('fresh=4 graph+range+guards+full_rank_integer_FP64' in x for x in batch)
native_configs = sum(int(re.search(r'count=(\d+)', x)[1]) for x in native)
assert native_configs == 612
support = archive / 'support'
files = [
    'CMakeLists.txt', 'README.md', 'docs/sw_tuner.md',
    'scripts/bench.sh', 'scripts/on_gpu.sh',
    'tests/test_sw_tuner.cu', 'tests/test_sw_seven_batch.cu', 'tests/test_sw_small_native.cu',
    'include/megafmm/gemm.h', 'bench/benchmark.cu',
    'results/iter383/source_before_build.json',
    'results/iter383/run.sh', 'results/iter383/close.py',
    'results/iter383/audit.py', 'results/iter383/archive.py',
]
digests = {}
for name in files:
    f = Path(name)
    dest = support / f
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, dest)
    digests[name] = hashlib.sha256(f.read_bytes()).hexdigest()
shutil.copy2(p / 'build.log', archive / 'build.log')
precheck = {
    'native_configs': native_configs,
    'native_calls': native_configs * 3,
    'batch_combinations': len(batch),
    'batch_calls': len(batch) * 4,
    'batch_changed_input_graph_calls': len(batch),
    'batch_random_integer_original_fp64_calls': len(batch),
    'scope': 'Prechecks only. No independent timing repeat or sanitizer certification.',
    'small_combinations': len(small),
    'small_calls': len(small)*5,
    'small_matching_fractional_bitwise_calls': len(small),
    'small_changed_input_graph_calls': len(small),
    'small_random_integer_original_fp64_calls': len(small),
    'records': native + batch + small,
}
(archive / 'precheck_audit.json').write_text(json.dumps(precheck, indent=2) + '\n')
(archive / 'support_manifest.json').write_text(json.dumps(digests, indent=2) + '\n')
summary = audit['summary']
report = [
    '# Iteration 383: native narrow tiles and exact two-block loops\n',
    'Six native narrow/short-K fused candidates were added to the existing pool; the three exact-two-block candidates require original K512. '
    'All prior native/restored/scalar candidates remain available. BF16 operands '
    'remain BF16. There are no residual/correction products or full-AB replacements.\n',
    f'Full single-sweep evidence: {audit["rows"]} shape/dtype rows, '
    f'{audit["candidate_records"]} candidate records, '
    f'{audit["fresh_fp64_checks"]} fresh original-input FP64 checks. '
    f'{audit["single_sweep_5group_wins_1p03"]} rows have all five paired timing '
    'groups at least 1.03x cuBLASLt. This is not independent-repeat certification.\n',
    'The original numerical threshold is report-only. Finite output and '
    'implementation checks remain required. Removing a threshold does not improve accuracy.\n',
    f'Completed prechecks: {native_configs * 3} original native calls and '
    f'{len(batch) * 4} batched-leaf calls, {len(small)*5} new narrow/short-K calls and {len(small)} matching-native bitwise comparisons. See precheck_audit.json for '
    'exact observed scope. No sanitizer result is implied.\n',
    '| M,N,K | dtype | SW us | cuBLASLt us | speedup | min paired speedup | eF | selected ID |',
    '|---|---|---:|---:|---:|---:|---:|---:|',
]
for r in summary:
    m, n, k, dtype = r['shape_dtype']
    report.append(f'| {m},{n},{k} | {dtype} | {r["sw_us"]:.3f} | '
                  f'{r["cublas_us"]:.3f} | {r["speedup"]:.4f} | '
                  f'{r["min_group_speedup"]:.4f} | {r["ef"]:.8g} | {r["selected"]} |')
report += ['', 'Raw candidate measurements and confirmation medians are in '
           'results.csv.sw_tuning.csv. Raw full-call timing and original-input '
           'errors are in results.csv and output.txt. Source/support hashes '
           'are in manifest.json and support_manifest.json.', '']
(archive / 'REPORT.md').write_text('\n'.join(report))
print(f'Archived support and actual prechecks in {archive}')

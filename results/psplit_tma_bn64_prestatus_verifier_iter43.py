#!/usr/bin/env python3
"""Readonly checkpoint verifier; stdout remains outside the frozen profile."""
from pathlib import Path
import hashlib, json, subprocess

def sha(data):
    return hashlib.sha256(data).hexdigest()

profile_path = 'results/small_psplit_tma_bn64_profile_iter43.json'
profile = json.loads(Path(profile_path).read_text())
maps = ['source_hashes', 'binaries', 'old_cuda_sha256',
        'retained_old_evidence_sha256', 'prior_csv_sha256', 'raw_sha256']
merged = {}
bindings = 0
for name in maps:
    for path, wanted in profile[name].items():
        bindings += 1
        assert merged.get(path, wanted) == wanted, ('Conflicting maps', path)
        merged[path] = wanted
        assert sha(Path(path).read_bytes()) == wanted, ('Onsite binding', path)
tracked = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD'], text=True).splitlines())
git_checks = 0
for path, wanted in merged.items():
    if path in tracked:
        assert sha(subprocess.check_output(['git', 'show', 'HEAD:' + path])) == wanted, ('Current Git binding', path)
        git_checks += 1
for path, wanted in profile['source_hashes'].items():
    assert sha(subprocess.check_output(['git', 'show', profile['measurement_source_commit'] + ':' + path])) == wanted
primary = [profile_path, 'results/1024_MEGAKERNEL_PSPLIT_TMA_BN64_REPORT.md',
           'results/small_psplit_tma_bn64_measurements_iter43.json',
           'results/psplit_tma_bn64_test_coverage_iter43.json']
primary_sha = {p: sha(Path(p).read_bytes()) for p in primary}
for path, wanted in primary_sha.items():
    assert sha(subprocess.check_output(['git', 'show', 'HEAD:' + path])) == wanted
assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(), 'Git must be clean'
canonical = lambda value: sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
print(json.dumps({'status': 'PASS_READONLY_PRESTATUS_BYTES_GIT',
    'HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'bindings': bindings, 'unique_bindings': len(merged), 'Git_bindings_checked': git_checks,
    'onsite_mismatches': 0, 'primary_sha256': primary_sha,
    'six_maps_digest': canonical({k: profile[k] for k in maps}),
    'nonaudit_digest': canonical({k: v for k, v in profile.items() if k not in {'independent_data_audit', 'independent_static_final_audit'}})}, indent=2))

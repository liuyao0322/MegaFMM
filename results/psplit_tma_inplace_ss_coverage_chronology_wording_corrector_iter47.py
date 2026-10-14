from pathlib import Path
import hashlib, json

path = Path('results/psplit_tma_inplace_ss_test_coverage_iter47.json')
original = path.read_bytes()
original_sha = hashlib.sha256(original).hexdigest()
assert original_sha == '3dffe27be51e2cd82b15d3ba1250ad4bbb597ad3e89527c4c1d7ed306c70171f'
archive = Path('results/psplit_tma_inplace_ss_test_coverage_before_wording_correction_iter47.json')
assert not archive.exists()
archive.write_bytes(original)
value = json.loads(original)
old = 'Historical46 actual adapter provenance remains separate; current47 generator preparedbeforevalidation.'
new = ('Historical46 actual adapter provenance remains separate. Original iteration47 prepared '
       'generator ff766dfb was frozen before validation; the current nested-bindings adapter '
       '5a2988b2 was executed after validation. Original failure exit1 and adapter exit0 '
       'remain preserved. This output wording correction changes no counts, commands or audit flags.')
assert value['limitations'][-1] == old
value['limitations'][-1] = new
corrected = (json.dumps(value, indent=2) + '\n').encode()
path.write_bytes(corrected)
before = json.loads(original)
assert [k for k in before if before[k] != value[k]] == ['limitations']
delta = {
    'schema': 'megafmm.iter47.coverage-chronology-wording-only-delta.v1',
    'CPU_only': True, 'GPU_rerun': False, 'post_validation_metadata_correction': True,
    'original_coverage': {'path': str(archive), 'sha256': original_sha},
    'corrected_coverage': {'path': str(path), 'sha256': hashlib.sha256(corrected).hexdigest()},
    'changed_top_level_keys': ['limitations'], 'changed_list_index': len(value['limitations']) - 1,
    'before': old, 'after': new,
    'scope': 'Only clarifies original prepared generator versus executed postvalidation adapter; all other JSON values remain exactly equal.'
}
target = Path('results/psplit_tma_inplace_ss_coverage_chronology_wording_delta_iter47.json')
assert not target.exists()
target.write_text(json.dumps(delta, indent=2) + '\n')
print('COVERAGE_WORDING_ONLY_CLOSED', delta['corrected_coverage']['sha256'])

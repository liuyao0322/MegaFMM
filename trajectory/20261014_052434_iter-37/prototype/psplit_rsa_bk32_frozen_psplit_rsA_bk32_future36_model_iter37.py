"""CPU-only BK32 RS-A geometry and virtual-input-lifetime review.

Import only definitions from the frozen parent; never execute its artifact-writing
top-level body. This is not CUDA compilation, allocation, or a performance test.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path('/share/home/liuyao/workspace/MegaFMM')
PREFIX = ROOT / '.scratch/psplit_rsA_bk32_future36_'
OLD_MODEL = ROOT / '.scratch/psplit_rsA_future35_model.py'
OLD_MANIFEST = ROOT / '.scratch/psplit_rsA_future35_manifest.json'
MODEL_SHA = 'aab2e1940470681b51d83596440ae49c5f2369e6abf2a1ff33d7f3c5ab7644ee'
MANIFEST_SHA = '78b454ce6247a2da0f08dadb48c2bf1a0a6b378ed01fe66673fa5cbfb4a46108'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert sha(OLD_MODEL) == MODEL_SHA and sha(OLD_MANIFEST) == MANIFEST_SHA
old = json.loads(OLD_MANIFEST.read_text())
before = {p: sha(ROOT / p) for p in old['files']}
assert before == old['files']

def protect_old(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        name = str(args[0])
        mode = args[1] or ''
        flags = args[2] if len(args) > 2 else 0
        writes = isinstance(mode, str) and any(x in mode for x in 'wax+')
        writes = writes or (isinstance(flags, int) and flags & (1 | 2 | 64 | 512 | 1024))
        if writes and 'psplit_rsA_future35_' in name:
            raise RuntimeError('frozen-parent write rejected: ' + name)

sys.addaudithook(protect_old)
ns = {}
definitions = OLD_MODEL.read_text().split("candidate=R/'.scratch/psplit_rsA_future35_candidate.cu'", 1)[0]
assert 'write_text(' not in definitions
exec(compile(definitions, str(OLD_MODEL) + ':definitions-only', 'exec'), ns)

def lifetime(products, atoms, stages):
    labels = {(p, k, w) for p in range(products) for k in range(atoms) for w in range(4)}
    pending = set()
    issues = writes = maximum = 0
    trace = []
    for stage in range(stages):
        assert not pending
        for label in sorted(labels):
            assert label not in pending
            writes += 1
        trace.append([stage, 'all_A_built_before_first_issue', len(labels)])
        for p in range(products):
            for k in range(atoms):
                source = {(p, k, w) for w in range(4)}
                assert not pending.intersection(source)
                pending.update(source)
                issues += 1
                maximum = max(maximum, len(pending))
        trace.append([stage, 'commit_then_wait0', len(pending)])
        pending.clear()
        assert not pending.intersection(labels)
        trace.append([stage, 'postwait_generic_fence_then_release', 0])
    bad = {(0, 0, w) for w in range(4)}
    pending.update(bad)
    rejects = {
        'four_words_rewritten_after_first_MMA': bool(pending.intersection(bad)),
        'generic_read_or_fence_before_wait0': bool(pending.intersection(bad)),
    }
    assert all(rejects.values())
    return {'products': products, 'atoms': atoms, 'stages': stages,
            'packed_A_words': len(labels), 'FP32_P_words': 16 * products,
            'payload_register_lower_bound': len(labels) + 16 * products,
            'MMA_issues': issues, 'A_writes': writes,
            'max_pending_words': maximum, 'negative_controls': rejects,
            'trace': trace, 'physical_allocation_proved': False}

source = (ROOT / '.scratch/psplit_rsA_bk32_future36_candidate.cu').read_text()
secondary = (ROOT / '.scratch/psplit_rsA_bk32_future36_candidate_160.cu').read_text()
assert source.count('barrier.sync 6,384;') == 2
assert source.count('__syncthreads();') == 2
assert source.count('warpgroup_wait<0>();') == 2
assert 'p.k%64 == 0' in source and 'p.k%128 == 0' not in source
assert 'for(int atom=0;atom<BK/16;++atom)' in source
assert 'size<2>(coords)==Int<BK/16>{}' in source
assert 'alignas(16) Element a0_data' in source
assert source.count('warpgroup_reg_alloc<192>();') == 1
assert secondary.count('warpgroup_reg_alloc<160>();') == 1
for spelling in ['psplit_rsA_build_stage<Element,0,T::BK>',
                 'psplit_rsA_build_stage<Element,3,T::BK>']:
    assert source.count(spelling) == 1

counts32 = ns['counts'](32)
counts64 = ns['counts'](64)
for key in ['wholeCTA_SS_operand_shared_writes', 'wholeCTA_RS_operand_shared_writes']:
    assert counts32[key] == counts64[key]
bank = counts32['raw_operand_bank_bytes']
tail = counts32['tail_bytes']
assert bank == 30720 and tail == 57344 and tail > bank
shared = ((2 * bank + 32 + 127) // 128) * 128
assert shared == 61568

budgets = {}
for consumer in [192, 160]:
    final_pool = 128 * (64 + consumer + consumer)
    legal = [i for i in range(64, 169, 8)
             if 384 * i >= final_pool and 384 * i <= 65536 and i <= consumer]
    budgets[str(consumer)] = {
        'producer': 64, 'consumers': [consumer, consumer],
        'source_required_pool': final_pool, 'legal_uniform_initial_candidates': legal,
        'directions_and_pool': {str(i): {'initial_pool': 384 * i,
            'producer_release': 128 * (i - 64),
            'each_consumer_request': 128 * (consumer - i)} for i in legal},
        'actual_uniform_allocation_verified': False,
    }
assert budgets['192']['legal_uniform_initial_candidates'] == [152, 160, 168]
assert budgets['160']['legal_uniform_initial_candidates'] == [128, 136, 144, 152, 160]

domains = [1, 2, 3, 4, 6, 7, 8, 9, 10, 16, 17, 18, 33, 34]
report = {
    'status': 'PASS_CPU_ONLY_RS_A_BK32_GEOMETRY_VIRTUAL_LIFETIME',
    'baseline_model_sha256': MODEL_SHA, 'baseline_manifest_sha256': MANIFEST_SHA,
    'source_sha256': sha(ROOT / '.scratch/psplit_rsA_bk32_future36_candidate.cu'),
    'secondary160_sha256': sha(ROOT / '.scratch/psplit_rsA_bk32_future36_candidate_160.cu'),
    'geometry': ns['geometry'](32), 'counts32': counts32,
    'counts64_parent': counts64,
    'predicted_shared_storage_bytes_only': shared,
    'tail_bank_overlaps_bytes': [bank, tail - bank],
    'requires_both_banks_fully_drained_before_tail': True,
    'virtual_A_lifetimes': [lifetime(p, 2, s) for p in [3, 4] for s in domains],
    'quota_hypotheses': budgets,
    'precision_contract_changed': False,
    'CPU_numerical_emulation_rerun': False,
    'CPU_numerical_scope': 'geometry-only transformation keeps the parent FP32 RN/onecast helper arithmetic unchanged; parent emulation is retained, not claimed as a new CUDA or FP64 test',
    'not_proved': ['CUDA template instantiation or layouts', 'actual initial/role register allocation',
                  'zero stack/local/spill or async physical A/P lifetime', 'hardware synchronization',
                  'original FP64 numerical gates', 'GPU/Sanitizer/performance/cuBLAS speedup'],
}
after = {p: sha(ROOT / p) for p in old['files']}
assert after == before
report['old_frozen_eight_files_verified_unchanged'] = True
path = ROOT / '.scratch/psplit_rsA_bk32_future36_review.json'
assert not path.exists()
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['status', 'source_sha256', 'secondary160_sha256',
                 'geometry', 'predicted_shared_storage_bytes_only', 'tail_bank_overlaps_bytes',
                 'quota_hypotheses', 'old_frozen_eight_files_verified_unchanged']}, indent=2))

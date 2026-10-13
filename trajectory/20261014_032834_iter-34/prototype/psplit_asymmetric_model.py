#!/usr/bin/env python3
"""CPU-only follow-up: source normalization and hypothetical register pools.

No compilation, CUDA calls, GPU queries, or old-file writes. Occupancy arithmetic
is transcribed from CUDA12.4 cuda_occupancy.h with explicit device assumptions.
It is not a certificate of effective initial ownership or physical residency.
"""
from collections import deque
import difflib
import hashlib
import json
from pathlib import Path
import re

ROOT = Path('/share/home/liuyao/workspace/MegaFMM')
PREFIX = ROOT / '.scratch/psplit_asymmetric'
FROZEN = {
    '.scratch/cooperative_st_once_psplit_candidate.cu': '84e56c44deb1585f6be457461dcc9073cad6631316e159249e77dbbb77c78664',
    '.scratch/cooperative_st_once_psplit_model.py': 'd5a159f926b9608a467d39ca806027f14772ce44fc035f078fc337f5ffa6755f',
    '.scratch/cooperative_st_once_psplit_review.json': '17b2e7776520e028ce7f92a72f76bdc3ab63cd1204db5f63071368d85824bc66',
    '.scratch/cooperative_st_once_psplit_protocol.md': 'f737837ad8c7b69b2001aac7b7894276ded587a1ab4d95b5a62847c038e35575',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(source):
    # This candidate's string literals contain no comment delimiters.
    source = re.sub(r'/\*.*?\*/|//[^\n]*', '', source, flags=re.S)
    return re.sub(r'\s+', '', source)


def source_review():
    old = (ROOT / '.scratch/cooperative_st_once_psplit_candidate.cu').read_text()
    new = (PREFIX.with_name(PREFIX.name + '_candidate.cu')).read_text()
    renamed = new.replace('PSplitAsymmetric', 'PSplit').replace('psplit_asymmetric', 'psplit')
    before, after = compact(old), compact(renamed)
    old_quota = 'cutlass::arch::warpgroup_reg_alloc<96>();'
    new_quota = ('if(tid<256)cutlass::arch::warpgroup_reg_alloc<80>();'
                 'elsecutlass::arch::warpgroup_reg_alloc<96>();')
    assert before.count('__launch_bounds__(384,1)') == 1
    assert after.count('__launch_bounds__(384,2)') == 1
    assert before.count(old_quota) == 1 and after.count(new_quota) == 1
    restored = after.replace('__launch_bounds__(384,2)', '__launch_bounds__(384,1)').replace(new_quota, old_quota)
    assert restored == before, 'semantic source changed beyond min2 and WG0 quota'
    assert len(re.findall(r'floatq[0-3]_data\[cosize_v<QLayout>\];', after)) == 4
    assert len(re.findall(r'make_tensor\(make_rmem_ptr\(q[0-3]_data\),QLayout\{\}\)', after)) == 4
    assert 'usingQLayout=typenamedecltype(thr_mma.make_fragment_C(coords))::layout_type;' in after
    # Accumulator value initialization remains after whichever alloc executes.
    assert after.index(new_quota) < after.index('clear(q0);clear(q1);clear(q2);')
    assert after.count('__syncthreads();') == 3
    assert after.count('params.consumer_arv_count=T::AllConsumerThreads;') == 1
    assert 'staticconstexprintAllConsumerThreads=256;' in after
    assert after.count('warpgroup_wait<0>();') == 1
    assert after.count('warpgroup_commit_batch();') == 1
    assert after.count('pipeline.producer_tail(state);') == 1
    assert after.count('gemm(mma,') == 7
    assert after.count('<<<grid,T::Threads,T::SharedBytes,stream>>>') == 1
    assert 'cudaLaunchCooperativeKernel' not in after and 'grid.sync' not in after
    normalized_diff = ''.join(difflib.unified_diff(
        old.splitlines(True), renamed.splitlines(True),
        fromfile='frozen_Psplit96_96', tofile='asymmetric80_96_names_normalized'))
    (PREFIX.with_name(PREFIX.name + '_semantic.diff')).write_text(normalized_diff)
    return {
        'status': 'PASS_EXACT_COMMENT_WHITESPACE_SYMBOL_NORMALIZED_TWO_CHANGE_COMPARISON',
        'semantic_changes': ['__launch_bounds__(384,1) -> (384,2)',
                             'WG0 uniform inc96 -> inc80; WG1 stays inc96'],
        'all_remaining_source_exact_after_normalization': True,
        'plain_four_uninitialized_fragment_arrays_and_rmem_views_preserved': True,
        'alloc_precedes_clear': True,
        'whole_seven_P_fixed_U_original_numeric_intrinsics_unchanged': True,
        'producer_DAG_vector_geometry_and_BF16_FP16_onecast_unchanged': True,
        'union_tail_barrier256_three_CTA_barriers_and_one_launch_unchanged': True,
        'native_metadata_currentAB_helpers_and_fallback_integration_contract_unchanged': True,
        'inherits_frozen_CPU_protocol_by_exact_remaining_source_not_GPU_proof': True,
        'normalized_semantic_diff': str(PREFIX.with_name(PREFIX.name + '_semantic.diff')),
    }


def quota_row(initial, role=(64, 80, 96)):
    legal_immediates = all(24 <= q <= 256 and q % 8 == 0 for q in (*role, initial))
    direction = initial >= role[0] and initial <= role[1] and initial <= role[2]
    pool = 384 * initial
    need = 128 * sum(role)
    return {'effective_initial_per_thread': initial, 'role_quotas': list(role),
            'legal_8_register_immediates': legal_immediates,
            'dec_inc_directions_legal': direction,
            'CTA_initial_pool_registers': pool, 'final_role_pool_registers': need,
            'pool_sufficient': pool >= need,
            'accepted_by_direction_and_pool': legal_immediates and direction and pool >= need,
            'final_free_registers_if_completed': pool - need}


def transfer_model(initial, role=(64, 80, 96)):
    """Exhaust all counts of completed per-warp quota transfers.

    Four identical warps per WG. Aligned instructions are assumed correctly
    uniform; counts abstract transfer completion, not GPU instruction timing.
    No extra CTA pool reserve or SM borrowing is assumed. Fair producer progress
    is required; producer deallocation has no consumer wait predecessor.
    """
    row = quota_row(initial, role)
    if not row['dec_inc_directions_legal']:
        return {'initial': initial, 'role': list(role), 'status': 'INVALID_DIRECTION_NO_MODEL'}
    release = (initial - role[0]) * 32
    acquire = ((role[1] - initial) * 32, (role[2] - initial) * 32)
    start = (0, 0, 0)
    todo = deque([start])
    seen = {start}
    parents = {}
    transitions = 0
    dead = []
    terminals = []
    while todo:
        state = todo.popleft()
        p, c0, c1 = state
        free = release * p - acquire[0] * c0 - acquire[1] * c1
        assert free >= 0
        nxt = []
        if p < 4:
            nxt.append(('producer_dec64', (p + 1, c0, c1)))
        if c0 < 4 and free >= acquire[0]:
            nxt.append(('WG0_inc', (p, c0 + 1, c1)))
        if c1 < 4 and free >= acquire[1]:
            nxt.append(('WG1_inc', (p, c0, c1 + 1)))
        if not nxt:
            terminals.append({'state': state, 'free_registers': free})
            if state != (4, 4, 4):
                dead.append(state)
        for action, target in nxt:
            transitions += 1
            if target not in seen:
                parents[target] = (state, action)
                seen.add(target)
                todo.append(target)
    witnesses = []
    for state in dead[:3]:
        actions = []
        cursor = state
        while cursor != start:
            cursor, action = parents[cursor]
            actions.append(action)
        witnesses.append({'blocked_state': state, 'trace': list(reversed(actions))})
    return {'initial': initial, 'role': list(role),
            'status': 'PASS_NO_INCOMPLETE_TERMINAL_UNDER_MODEL' if not dead else 'BLOCKED_TERMINAL_COUNTEREXAMPLE',
            'states': len(seen), 'transitions': transitions,
            'producer_free_registers_per_warp': release,
            'consumer_request_registers_per_warp': list(acquire),
            'terminals': terminals, 'incomplete_terminal_witnesses': witnesses,
            'physical_scheduler_allocator_forward_progress': 'PENDING; abstract model only'}


def cuda12_4_occupancy_arithmetic(raw_regs):
    # cudaOccMaxBlocksPerSMRegsLimit PARTITIONED_GC_OFF with hypothetical SM90
    # warp32 / 4 sub-partitions / 65536 registers-SM / regsPerBlock65536.
    per_warp = ((raw_regs * 32 + 255) // 256) * 256
    cta = per_warp * 12
    regs_only_limit = ((65536 // 4 // per_warp) * 4) // 12 if per_warp else 99999
    if cta > 65536 or raw_regs > 256:
        regs_only_limit = 0
    return {'reported_numRegs_hypothesis': raw_regs,
            'allocation_per_warp_registers': per_warp,
            'allocation_equivalent_per_thread': per_warp // 32,
            'allocation_per_CTA_registers': cta,
            'two_CTA_allocation_registers': 2 * cta,
            'registers_only_CTA_limit_from_public_formula': regs_only_limit,
            'dynamic_effective_initial_quota': 'NOT_DERIVED_OR_CERTIFIED_BY_OCCUPANCY_FORMULA'}


def main():
    actual_frozen = {p: sha(ROOT / p) for p in FROZEN}
    assert actual_frozen == FROZEN, 'old four files must remain byte-frozen'
    old_report = json.loads((ROOT / '.scratch/cooperative_st_once_psplit_review.json').read_text())
    old_bindings = old_report['evidence_bindings']
    mismatch = [p for p, expected in old_bindings.items() if sha(ROOT / p) != expected]
    assert not mismatch, f'inherited original source/header bindings changed: {mismatch}'
    source = source_review()
    asym = [quota_row(q) for q in range(24, 257, 8)]
    sym = [quota_row(q, (64, 96, 96)) for q in range(24, 257, 8)]
    assert [r['effective_initial_per_thread'] for r in asym if r['accepted_by_direction_and_pool']] == [80]
    assert [r['effective_initial_per_thread'] for r in sym if r['accepted_by_direction_and_pool']] == [88, 96]
    models = [transfer_model(80), transfer_model(72), transfer_model(64),
              transfer_model(88), transfer_model(88, (64, 96, 96)),
              transfer_model(96, (64, 96, 96))]
    assert models[0]['status'] == 'PASS_NO_INCOMPLETE_TERMINAL_UNDER_MODEL'
    assert all(models[i]['status'] == 'BLOCKED_TERMINAL_COUNTEREXAMPLE' for i in (1, 2))
    assert models[3]['status'] == 'INVALID_DIRECTION_NO_MODEL'
    assert all(models[i]['status'] == 'PASS_NO_INCOMPLETE_TERMINAL_UNDER_MODEL' for i in (4, 5))
    occupancies = [cuda12_4_occupancy_arithmetic(r) for r in range(1, 257)]
    matching = [r['reported_numRegs_hypothesis'] for r in occupancies
                if r['allocation_equivalent_per_thread'] == 80]
    assert matching == list(range(73, 81))
    assert cuda12_4_occupancy_arithmetic(80)['registers_only_CTA_limit_from_public_formula'] == 2
    assert cuda12_4_occupancy_arithmetic(81)['registers_only_CTA_limit_from_public_formula'] == 1
    paths = [ROOT / '.scratch/psplit_asymmetric_candidate.cu', Path(__file__),
             ROOT / '.scratch/psplit_asymmetric_candidate.diff',
             ROOT / '.scratch/psplit_asymmetric_semantic.diff',
             ROOT / 'third_party/cutlass/include/cutlass/arch/reg_reconfig.h',
             Path('/share/app/cuda/cuda-12.4/include/cuda_occupancy.h')]
    report = {
        'status': 'PASS_CPU_SOURCE_NORMALIZATION_AND_HYPOTHETICAL_REGISTER_POOL_ONLY',
        'no_host_CUDA_compilation_GPU_queries_execution_sanitizer_performance': True,
        'scratch_only_future_candidate': True, 'formal_policy_unassigned': True,
        'frozen_four_file_sha256': actual_frozen,
        'inherited_CPU_evidence_bindings_verified': old_bindings,
        'evidence_bindings': {str(p): sha(p) for p in paths},
        'source_review': source,
        'budget': {'asymmetric': asym, 'symmetric_frozen96_96': sym,
                   'asymmetric_valid_effective_initial_quotas': [80],
                   'symmetric_valid_effective_initial_quotas': [88, 96],
                   'asymmetric_pool': 30720, 'symmetric_pool': 32768,
                   'two_asymmetric_initial80_CTA': 61440,
                   'two_symmetric_min_initial88_CTA': 67584,
                   'producer_total_release_at_initial80': 2048,
                   'WG0_total_acquire_at_initial80': 0,
                   'WG1_total_acquire_at_initial80': 2048},
        'per_warp_pool_transfer_models': models,
        'occupancy': {'assumptions': {'threads': 384, 'warps_per_CTA': 12,
                                     'warp_size': 32, 'SM_registers': 65536,
                                     'CTA_register_limit': 65536,
                                     'SM_subpartitions': 4,
                                     'register_granularity_per_warp': 256,
                                     'shared_granularity_bytes': 128,
                                     'partitioned_gc': 'OFF', 'actual_device_query': 'NONE'},
                      'raw_numRegs_with_allocation_equivalent80': matching,
                      'all_reported_register_rows': occupancies,
                      'estimated_source_shared_bytes': 86144,
                      'two_estimated_shared_bytes_excluding_actual_static_reserved': 172288,
                      'actual_shared_carveout_reserved_static_and_SM_CTA_quota': 'PENDING'},
        'compile_resource_gates': [
            'Compile actual BF16 and FP16 sm90a instantiations; verify .maxntid384/.minnctapersm2 and three uniform setmaxnreg paths',
            'Distinguish reported REG, rounded allocated CTA registers, and actual effective initial ownership/max-reg quota; verify valid directive and initial80 pool',
            'Initial/common pre-deallocation CFG must use only initial quota registers, including all implicit-width reads/writes',
            'Producer including common tail/U after dec64 requires every explicit/implicit register index <=63',
            'WG0 including accumulator-init/descriptors/HGMMA outputs/tail/U after inc80 requires every explicit/implicit register index <=79',
            'WG1 including accumulator-init/descriptors/HGMMA outputs/tail/U after inc96 requires every explicit/implicit register index <=95',
            'No accumulator initialization or speculative high-register fragment access before successful role alloc',
            'Actual PTX/ELF/SASS STACK and LOCAL, LD/ST local or spills, all loop/tail CFG paths and call targets; plain arrays must scalarize',
            'Operand fences/commit/wait0/release and producer async proxy fence retain actual generated ordering',
            'Actual dynamic/static shared sizeof/alignment, reserved shared and selected carveout; query occupancy for actual compiled function and launch'],
        'future_execution_gates': old_report['pending'],
        'PLAN_contract': 'Original no-global-ST/P independent CTA experiment; native helpers await common Plan policy/alias/range/fallback/certification integration, one-node Graph/current AB replay and original numeric gates remain pending',
        'not_claimed': ['Compilation', 'Actual initial80', 'Physical two-CTA residency',
                        'No-spill machine code', 'GPU liveness', 'Numerical qualification',
                        'Single-node Graph certification', 'Any speedup'],
        'documentation': {
            'launch_bounds': 'https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#launch-bounds',
            'setmaxnreg': 'https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg',
            'launch_bounds_setmaxnreg_cap_exemption': 'NOT_DOCUMENTED_IN_REVIEWED_PRIMARY_SECTIONS; actual compiler behavior is pending'},
    }
    target = PREFIX.with_name(PREFIX.name + '_review.json')
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['evidence_bindings'][str(paths[0])],
                      'model': str(Path(__file__)), 'json': str(target),
                      'initial80_states': models[0]['states'],
                      'initial80_transitions': models[0]['transitions'],
                      'only_valid_asymmetric_effective_initial': [80],
                      'source_two_change_comparison': 'PASS',
                      'frozen_four_files': 'UNCHANGED',
                      'compile_GPU_or_performance': 'NONE'}, indent=2))


if __name__ == '__main__':
    main()

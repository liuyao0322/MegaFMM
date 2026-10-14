"""CPU-only abstract scheduling model; never compiles or executes a GPU kernel."""
import hashlib
import json
import pathlib
import random


class Rejected(Exception):
    pass


def require(condition, reason):
    if not condition:
        raise Rejected(reason)


def simulate(stages, seed, fault=None):
    rng = random.Random(seed)
    banks = [dict(generation=-1, ready=False, released=set()) for _ in range(2)]
    wg = [dict(stage=0, pc=0, pending=[], complete=set(), released=set(), done=False)
          for _ in range(2)]
    transformed = {}
    leaf_atoms = [[] for _ in range(7)]
    producer = dict(next=0, active=None, tail=False)
    overlap = 0
    events = []

    def emit(actor, event, stage):
        events.append(dict(actor=actor, event=event, stage=stage, bank=stage % 2,
                           consumer_wait_phase=(stage // 2) % 2,
                           producer_wait_phase=1 ^ ((stage // 2) % 2)))

    def release(role, stage):
        state = wg[role]
        require(stage in state['complete'], 'release before matching WGMMA completion')
        require(stage not in state['released'], 'duplicate release')
        bank = banks[stage % 2]
        require(bank['generation'] == stage, 'release wrong bank generation')
        state['released'].add(stage)
        bank['released'].add(role)
        emit(role, 'release', stage)

    def choices():
        out = []
        if producer['active'] is not None:
            out.append(('producer', 'complete_raw'))
        elif producer['next'] < stages:
            stage = producer['next']
            bank = banks[stage % 2]
            if bank['generation'] == -1 or bank['released'] == {0, 1}:
                out.append(('producer', 'acquire'))
        elif not producer['tail'] and all(b['generation'] == -1 or
                                               b['released'] == {0, 1} for b in banks):
            out.append(('producer', 'tail'))
        for role, state in enumerate(wg):
            if state['done']:
                continue
            stage, pc = state['stage'], state['pc']
            if stage == stages:
                out.append((role, 'drain_last'))
            elif pc == 0:
                bank = banks[stage % 2]
                if bank['generation'] == stage and bank['ready']:
                    out.append((role, 'wait_raw'))
            elif pc == 1:
                out.append((role, 'transform_fence_arrive'))
            elif pc == 2 and (transformed.get(stage, set()) == {0, 1} or
                               fault == 'missing_consumer_barrier'):
                out.append((role, 'issue_commit'))
            elif pc == 3:
                out.append((role, 'wait1_release_prior'))
        return out

    for _ in range(10000):
        if producer['tail'] and all(s['done'] for s in wg):
            break
        eligible = choices()
        require(eligible, 'deadlock')
        actor, event = rng.choice(eligible)
        if actor == 'producer':
            if event == 'acquire':
                stage = producer['next']
                bank = banks[stage % 2]
                for state in wg:
                    require(all(p % 2 != stage % 2 for p in state['pending']),
                            'overwrite while WGMMA still reads bank')
                bank.update(generation=stage, ready=False, released=set())
                producer['active'] = stage
                emit(actor, event, stage)
            elif event == 'complete_raw':
                stage = producer['active']
                banks[stage % 2]['ready'] = True
                producer['active'] = None
                producer['next'] += 1
                emit(actor, 'eight_TMA_boxes_complete_then_raw_publish', stage)
            else:
                require(all(not s['pending'] for s in wg), 'tail with pending WGMMA')
                producer['tail'] = True
                emit(actor, 'both_banks_tail_retired', stages)
            continue

        state = wg[actor]
        stage = state['stage']
        if event == 'wait_raw':
            state['pc'] = 1
        elif event == 'transform_fence_arrive':
            require(fault != 'store_before_four_raw_loads', 'mutating store before four raw loads')
            if state['pending']:
                overlap += 1
                require(fault != 'generic_pending_P_access', 'generic register access overlaps pending P')
            transformed.setdefault(stage, set()).add(actor)
            state['pc'] = 2
        elif event == 'issue_commit':
            require(transformed.get(stage) == {0, 1}, 'WGMMA before all256 transforms/fences/barrier')
            require(fault != 'uninitialized_P', 'P used before initialization')
            for product in (range(3) if actor == 0 else range(3, 7)):
                leaf_atoms[product].extend([stage * 2, stage * 2 + 1])
            state['pending'].append(stage)
            state['pc'] = 3
        elif event == 'wait1_release_prior':
            while len(state['pending']) > 1:
                state['complete'].add(state['pending'].pop(0))
            if fault == 'release_current_after_wait1':
                release(actor, stage)
            if stage > 0:
                release(actor, stage if fault == 'release_wrong_state' else stage - 1)
            state['stage'] += 1
            state['pc'] = 0
        elif event == 'drain_last':
            if fault != 'missing_tail_wait0':
                state['complete'].update(state['pending'])
                state['pending'].clear()
            if stages:
                release(actor, stages - 1)
            state['done'] = True
        emit(actor, event, stage)
    else:
        raise Rejected('step bound exceeded')

    require(all(s['released'] == set(range(stages)) for s in wg), 'missing release')
    require(all(atoms == list(range(2 * stages)) for atoms in leaf_atoms), 'wholeP K16 order changed')
    require(all(not s['pending'] for s in wg), 'pending P at U/tail publication')
    return dict(stages=stages, seed=seed, event_count=len(events),
                transform_with_prior_stage_pending=overlap,
                phase_wraps=max(0, (stages - 1) // 4),
                whole7P_ascending_K16=True, both_roles_exactly_one_release_per_stage=True)


def main():
    counts = [0, 1, 2, 3, 4, 5, 16, 17, 31, 32, 33]
    positives = [simulate(stages, seed) for stages in counts for seed in range(32)]
    faults = ['release_current_after_wait1', 'release_wrong_state',
              'missing_consumer_barrier', 'generic_pending_P_access',
              'missing_tail_wait0', 'store_before_four_raw_loads', 'uninitialized_P']
    negatives = []
    for fault in faults:
        rejected = None
        for seed in range(128):
            try:
                simulate(16, seed, fault=fault)
            except Rejected as error:
                rejected = dict(fault=fault, rejected=True, first_rejected_seed=seed,
                                reason=str(error))
                break
        require(rejected is not None, 'negative self-check failed: ' + fault)
        negatives.append(rejected)
    sources = [
        'results/wide_tile_feasibility_future46_design.md',
        'results/wide_tile_feasibility_future46_geometry.json',
        'solution/megakernel_wgmma_psplit.cu',
        'solution/megakernel_wgmma_psplit_ss_bn16.cu',
        'third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp',
        'third_party/cutlass/include/cute/arch/mma_sm90_gmma.hpp',
        'third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp',
    ]
    bound = {name: hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in sources}
    out = dict(schema='megafmm.ss-overlap-future47.abstract-events.v1', status='PASS',
               CPU_only=True, input_sha256=bound, actual_CuTe_instantiation=False,
               actual_TMA_hardware_validation=False, compiled_register_check=False,
               CUDA_compilation_performed=False, GPU_execution_performed=False,
               numerical_model_performed=False, precision_certified=False,
               performance_claim=False, stage_counts=counts, schedules=len(positives),
               seeds_per_stage_count=32, summaries=[
                   dict(stages=n, schedules=32,
                        max_event_count=max(v['event_count'] for v in positives if v['stages'] == n),
                        min_overlap=min(v['transform_with_prior_stage_pending'] for v in positives if v['stages'] == n),
                        max_overlap=max(v['transform_with_prior_stage_pending'] for v in positives if v['stages'] == n),
                        phase_wraps=max(v['phase_wraps'] for v in positives if v['stages'] == n))
                   for n in counts], negative_self_checks=negatives,
               model_limits=['Whole WG actions abstract128 participants; no actual thread-level lowering.',
                             'No timing, bank conflict, register interference or numerical proof.',
                             'Raw/final byte alias geometry belongs to independent CPU future46 only.',
                             'Same-shape WGMMA accumulation order modeled; generic P accesses forbidden while pending.'])
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

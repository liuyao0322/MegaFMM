#!/usr/bin/env python3
"""Three-bank CPU ownership/phase model, not a CUDA or numerical simulator.

Producer publication and consumer release are abstract 128-thread events.
There are three operand banks and at most two committed unretired groups.
Retirement requires completion of the oldest committed prefix, not FIFO
physical completion. All three producer_tail acquires remain independent.
"""
from collections import deque
from dataclasses import dataclass, replace
import json
import random

BANKS = 3
MAX_UNRETIRED_GROUPS = 2
ISSUE, WAIT1, RELEASE, TAIL_WAIT0, TAIL_RELEASE, DONE = range(6)


@dataclass(frozen=True)
class State:
    produced: int = 0
    issued: int = 0
    retired: int = 0
    mode: int = ISSUE
    owner: tuple = (-1,) * BANKS
    complete: tuple = (False,) * BANKS
    producer_tail_acquired: int = 0


def bank_arrivals(prefix_count, bank):
    """Number of prefix stages mapped to this bank; initialization is count0."""
    assert prefix_count >= 0 and 0 <= bank < BANKS
    return (prefix_count + BANKS - 1 - bank) // BANKS


def barrier_phase(prefix_count, bank):
    return bank_arrivals(prefix_count, bank) & 1


def producer_phase(stage):
    return 1 ^ ((stage // BANKS) & 1)


def consumer_phase(stage):
    return (stage // BANKS) & 1


def invariant(s, stages):
    assert 0 <= s.retired <= s.issued <= s.produced <= stages
    assert s.produced - s.retired <= BANKS
    assert s.issued - s.retired <= MAX_UNRETIRED_GROUPS
    assert 0 <= s.producer_tail_acquired <= BANKS
    assert len(s.owner) == len(s.complete) == BANKS
    if s.producer_tail_acquired:
        assert s.produced == stages
        for stage in range(stages, stages + s.producer_tail_acquired):
            bank = stage % BANKS
            assert s.owner[bank] == -1
            assert barrier_phase(s.retired, bank) != producer_phase(stage)
    assert sorted(group for group in s.owner if group >= 0) == list(
        range(s.retired, s.produced))
    for bank in range(BANKS):
        if s.owner[bank] == -1:
            assert not s.complete[bank]
        else:
            assert s.owner[bank] % BANKS == bank
            if s.complete[bank]:
                assert s.retired <= s.owner[bank] < s.issued
    if s.mode == ISSUE:
        assert s.issued - s.retired <= 1
    if s.mode in (WAIT1, RELEASE):
        assert s.issued - s.retired == 2
    if s.mode in (TAIL_WAIT0, TAIL_RELEASE, DONE):
        assert s.issued == stages
    if s.mode == DONE:
        assert s.retired == s.issued == s.produced == stages
        assert s.owner == (-1,) * BANKS
        for stage in range(stages, stages + BANKS):
            assert barrier_phase(s.retired, stage % BANKS) != producer_phase(stage)


def successors(s, stages):
    invariant(s, stages)
    result = []
    if s.produced < stages:
        bank = s.produced % BANKS
        if s.owner[bank] == -1:
            # All128 real writers acquire this phase before stores and full
            # publication; real writes/fences/arrivals are not simulated here.
            assert barrier_phase(s.retired, bank) != producer_phase(s.produced)
            owner = list(s.owner)
            owner[bank] = s.produced
            result.append(("produce", replace(s, produced=s.produced + 1,
                                             owner=tuple(owner))))

    if s.produced == stages and s.producer_tail_acquired < BANKS:
        stage = stages + s.producer_tail_acquired
        bank = stage % BANKS
        if barrier_phase(s.retired, bank) != producer_phase(stage):
            assert s.owner[bank] == -1
            result.append(("producer_tail_acquire", replace(
                s, producer_tail_acquired=s.producer_tail_acquired + 1)))

    if s.mode == ISSUE and s.issued < stages:
        bank = s.issued % BANKS
        if s.owner[bank] == s.issued:
            assert barrier_phase(s.produced, bank) != consumer_phase(s.issued)
            issued = s.issued + 1
            pending = issued - s.retired
            assert pending <= MAX_UNRETIRED_GROUPS
            mode = WAIT1 if pending == 2 else TAIL_WAIT0 if issued == stages else ISSUE
            result.append(("issue", replace(s, issued=issued, mode=mode)))

    # Permit the latest group to complete before the older group. Only a wait
    # covering the old prefix grants retirement of the old bank.
    for group in range(s.retired, s.issued):
        bank = group % BANKS
        assert s.owner[bank] == group
        if not s.complete[bank]:
            complete = list(s.complete)
            complete[bank] = True
            result.append(("complete", replace(s, complete=tuple(complete))))

    if s.mode == WAIT1 and s.complete[s.retired % BANKS]:
        result.append(("wait1", replace(s, mode=RELEASE)))
    if s.mode == RELEASE:
        bank = s.retired % BANKS
        assert s.owner[bank] == s.retired and s.complete[bank]
        owner, complete = list(s.owner), list(s.complete)
        owner[bank], complete[bank] = -1, False
        retired = s.retired + 1
        mode = TAIL_WAIT0 if s.issued == stages else ISSUE
        result.append(("retire", replace(s, retired=retired, mode=mode,
                                        owner=tuple(owner), complete=tuple(complete))))

    if s.mode == TAIL_WAIT0:
        if all(s.complete[group % BANKS] for group in range(s.retired, s.issued)):
            mode = DONE if s.retired == s.issued else TAIL_RELEASE
            result.append(("wait0", replace(s, mode=mode)))
    if s.mode == TAIL_RELEASE:
        bank = s.retired % BANKS
        assert s.owner[bank] == s.retired and s.complete[bank]
        owner, complete = list(s.owner), list(s.complete)
        owner[bank], complete[bank] = -1, False
        retired = s.retired + 1
        mode = DONE if retired == s.issued else TAIL_RELEASE
        result.append(("tail_retire", replace(s, retired=retired, mode=mode,
                                             owner=tuple(owner), complete=tuple(complete))))
    for _, next_state in result:
        invariant(next_state, stages)
    return result


def state_json(s):
    return {name: getattr(s, name) for name in s.__dataclass_fields__}


def exhaustive(stages):
    initial = State(mode=TAIL_WAIT0 if stages == 0 else ISSUE)
    queue, seen = deque([initial]), {initial}
    transitions = terminal = out_of_order = max_owned = max_pending = 0
    negative_witnesses, positive_witnesses = {}, {}
    while queue:
        s = queue.popleft()
        invariant(s, stages)
        outgoing = successors(s, stages)
        max_owned = max(max_owned, s.produced - s.retired)
        max_pending = max(max_pending, s.issued - s.retired)
        if not outgoing:
            assert s.mode == DONE and s.producer_tail_acquired == BANKS, (
                "deadlock", stages, s)
            terminal += 1
        if s.produced - s.retired == BANKS and s.issued == s.retired:
            positive_witnesses.setdefault("three_full_banks_before_first_issue", s)
        if s.produced - s.retired == BANKS and s.issued - s.retired == 2:
            positive_witnesses.setdefault("third_bank_produced_while_two_commits_unretired", s)
        if s.issued - s.retired == 2:
            old, new = s.retired % BANKS, (s.retired + 1) % BANKS
            if not s.complete[old] and s.complete[new]:
                out_of_order += 1
            if s.complete[old] and not s.complete[new]:
                negative_witnesses.setdefault("release_latest_after_wait1_is_unsafe", s)
                if s.mode == RELEASE:
                    positive_witnesses.setdefault("retire_old_while_latest_still_inflight", s)
        if s.issued - s.retired == 1 and not s.complete[s.retired % BANKS]:
            negative_witnesses.setdefault("release_only_group_after_wait1_is_unsafe", s)
            if s.issued == stages:
                negative_witnesses.setdefault("generic_U_after_tail_wait1_is_unsafe", s)
        if s.produced - s.retired == BANKS and s.produced < stages:
            bank = s.produced % BANKS
            assert s.owner[bank] >= 0
            negative_witnesses.setdefault("overwrite_next_producer_bank_while_owned_is_unsafe", s)
        for _, nxt in outgoing:
            transitions += 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    assert terminal == 1
    if stages >= 2:
        assert out_of_order > 0
        assert max_pending == 2
        assert {"release_latest_after_wait1_is_unsafe",
                "release_only_group_after_wait1_is_unsafe",
                "generic_U_after_tail_wait1_is_unsafe"} <= negative_witnesses.keys()
    if stages >= 3:
        assert max_owned == BANKS
        assert len(positive_witnesses) == 3
    if stages >= 4:
        assert len(negative_witnesses) == 4
    row = {"stages": stages, "states": len(seen), "transitions": transitions,
           "terminal_states": terminal, "out_of_order_completion_states": out_of_order,
           "maximum_owned_banks": max_owned, "maximum_committed_unretired_groups": max_pending}
    return row, negative_witnesses, positive_witnesses


def randomized(stages, seed):
    rng = random.Random(seed)
    s = State(mode=TAIL_WAIT0 if stages == 0 else ISSUE)
    actions = 0
    counts = {label: 0 for label in ("issue", "retire", "tail_retire", "wait1",
                                    "wait0", "producer_tail_acquire")}
    while s.mode != DONE or s.producer_tail_acquired != BANKS:
        outgoing = successors(s, stages)
        assert outgoing, ("randomized deadlock", stages, seed, s)
        label, s = rng.choice(outgoing)
        if label in counts:
            counts[label] += 1
        actions += 1
        assert actions <= 10 * stages + 10
    invariant(s, stages)
    assert counts["issue"] == stages
    assert counts["retire"] + counts["tail_retire"] == stages
    assert counts["wait1"] == max(0, stages - 1)
    assert counts["wait0"] == 1
    assert counts["producer_tail_acquire"] == BANKS
    return actions


def phase_crosscheck(max_stages=512):
    """Independently count each bank's arrivals instead of reusing the formula."""
    checks = 0
    for stages in range(max_stages + 1):
        actual_arrivals = [0] * BANKS
        for stage in range(stages):
            bank = stage % BANKS
            assert (actual_arrivals[bank] & 1) != producer_phase(stage)
            actual_arrivals[bank] += 1
            assert (actual_arrivals[bank] & 1) != consumer_phase(stage)
            for b in range(BANKS):
                assert actual_arrivals[b] == bank_arrivals(stage + 1, b)
                assert (actual_arrivals[b] & 1) == barrier_phase(stage + 1, b)
                checks += 1
        for tail_stage in range(stages, stages + BANKS):
            assert (actual_arrivals[tail_stage % BANKS] & 1) != producer_phase(tail_stage)
            checks += 1
    return checks


if __name__ == "__main__":
    exhaustive_counts = tuple(range(18)) + (32, 33, 64, 512)
    rows = []
    mutation_witnesses = positive_witnesses = None
    for stage_count in exhaustive_counts:
        row, negative, positive = exhaustive(stage_count)
        rows.append(row)
        if stage_count == 4:
            mutation_witnesses = {label: state_json(s) for label, s in negative.items()}
            positive_witnesses = {label: state_json(s) for label, s in positive.items()}
    random_counts = (0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 32, 33, 64, 512)
    random_actions = sum(randomized(stages, 100000 * stages + seed)
                         for stages in random_counts for seed in range(32))
    print(json.dumps({"status": "CPU_THREE_BANK_ABSTRACT_PASS", "banks": BANKS,
                      "max_unretired_groups": MAX_UNRETIRED_GROUPS,
                      "exhaustive": rows,
                      "total_states": sum(row["states"] for row in rows),
                      "total_transitions": sum(row["transitions"] for row in rows),
                      "randomized_stage_counts": random_counts,
                      "randomized_sequences": len(random_counts) * 32,
                      "randomized_actions": random_actions,
                      "phase_crosscheck_stage_counts": [0, 512],
                      "phase_crosschecks": phase_crosscheck(),
                      "mutation_counterexamples": mutation_witnesses,
                      "positive_witnesses": positive_witnesses,
                      "hardware_memory_order_or_thread_convergence_proven": False}, indent=2))

#!/usr/bin/env python3
"""CPU-only abstract bank/group lifetime model; no GPU or numerical simulation."""
from collections import deque
from dataclasses import dataclass, replace
import json
import random

ISSUE, WAIT1, RELEASE, TAIL_WAIT0, TAIL_RELEASE, DONE = range(6)


@dataclass(frozen=True)
class State:
    produced: int = 0
    issued: int = 0
    retired: int = 0
    mode: int = ISSUE
    owner: tuple = (-1, -1)
    complete: tuple = (False, False)
    producer_tail_acquired: int = 0


def barrier_phase(arrivals, bank):
    """Full/empty barrier phase after all collective stage arrivals."""
    return ((arrivals + 1 - bank) // 2) & 1


def invariant(s, stages):
    assert 0 <= s.retired <= s.issued <= s.produced <= stages
    assert s.produced - s.retired <= 2
    assert 0 <= s.issued - s.retired <= 2
    assert 0 <= s.producer_tail_acquired <= 2
    if s.producer_tail_acquired:
        assert s.produced == stages
        for count in range(stages, stages + s.producer_tail_acquired):
            bank = count % 2
            producer_phase = 1 ^ ((count // 2) & 1)
            assert s.owner[bank] < 0
            assert barrier_phase(s.retired, bank) != producer_phase
    assert sorted(x for x in s.owner if x >= 0) == list(range(s.retired, s.produced))
    for bank in range(2):
        if s.owner[bank] < 0:
            assert not s.complete[bank]
        else:
            assert s.owner[bank] % 2 == bank
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
        assert s.owner == (-1, -1)
        # The unchanged producer_tail acquires both empty banks from its final
        # state. Verify both phases are satisfied after complete consumer drain.
        for count in (stages, stages + 1):
            bank = count % 2
            producer_phase = 1 ^ ((count // 2) & 1)
            assert barrier_phase(s.retired, bank) != producer_phase
        # First generic accumulator access is permitted only in this mode,
        # after the modeled final wait0, drain and compiler operand fences.


def successors(s, stages):
    invariant(s, stages)
    result = []
    # Raw loading, transforms and full publication are one abstract producer
    # event. Their real128-thread publication is unchanged and not modeled here.
    if s.produced < stages:
        bank = s.produced % 2
        if s.owner[bank] < 0:
            producer_phase = 1 ^ ((s.produced // 2) & 1)
            assert barrier_phase(s.retired, bank) != producer_phase
            owner = list(s.owner)
            owner[bank] = s.produced
            result.append(("produce", replace(s, produced=s.produced + 1,
                                             owner=tuple(owner))))

    # Explicit independent producer_tail events. The source executes two empty
    # acquires from its final producer state; an unused/freed bank may already
    # satisfy the first acquire before the consumer tail reaches final wait0.
    if s.produced == stages and s.producer_tail_acquired < 2:
        count = stages + s.producer_tail_acquired
        bank = count % 2
        producer_phase = 1 ^ ((count // 2) & 1)
        if barrier_phase(s.retired, bank) != producer_phase:
            assert s.owner[bank] < 0
            result.append(("producer_tail_acquire", replace(
                s, producer_tail_acquired=s.producer_tail_acquired + 1)))

    if s.mode == ISSUE and s.issued < stages:
        bank = s.issued % 2
        if s.owner[bank] == s.issued:
            read_phase = (s.issued // 2) & 1
            assert barrier_phase(s.produced, bank) != read_phase
            issued = s.issued + 1
            pending = issued - s.retired
            mode = WAIT1 if pending == 2 else TAIL_WAIT0 if issued == stages else ISSUE
            # A single commit covers all seven P and their chronological K16
            # substeps from this one read bank. No accumulator generic access.
            result.append(("issue", replace(s, issued=issued, mode=mode)))

    # Deliberately permit out-of-order group completion in the abstraction.
    # Correctness relies only on wait-observed completed submission prefixes,
    # not on a stronger claim about physical execution/completion order.
    for group in range(s.retired, s.issued):
        bank = group % 2
        assert s.owner[bank] == group
        if not s.complete[bank]:
            complete = list(s.complete)
            complete[bank] = True
            result.append(("complete", replace(s, complete=tuple(complete))))

    if s.mode == WAIT1 and s.complete[s.retired % 2]:
        # With exactly two unretired commits, wait1 covers the older group.
        result.append(("wait1", replace(s, mode=RELEASE)))
    if s.mode == RELEASE:
        bank = s.retired % 2
        assert s.complete[bank] and s.owner[bank] == s.retired
        owner, complete = list(s.owner), list(s.complete)
        owner[bank], complete[bank] = -1, False
        retired = s.retired + 1
        mode = TAIL_WAIT0 if s.issued == stages else ISSUE
        result.append(("retire", replace(s, retired=retired, mode=mode,
                                        owner=tuple(owner), complete=tuple(complete))))
    if s.mode == TAIL_WAIT0:
        if all(s.complete[group % 2] for group in range(s.retired, s.issued)):
            mode = DONE if s.retired == s.issued else TAIL_RELEASE
            result.append(("wait0", replace(s, mode=mode)))
    if s.mode == TAIL_RELEASE:
        bank = s.retired % 2
        assert s.complete[bank] and s.owner[bank] == s.retired
        owner, complete = list(s.owner), list(s.complete)
        owner[bank], complete[bank] = -1, False
        retired = s.retired + 1
        mode = DONE if retired == s.issued else TAIL_RELEASE
        result.append(("tail_retire", replace(s, retired=retired, mode=mode,
                                             owner=tuple(owner), complete=tuple(complete))))
    for _, next_state in result:
        invariant(next_state, stages)
    return result


def exhaustive(stages):
    initial = State(mode=TAIL_WAIT0 if stages == 0 else ISSUE)
    queue, seen = deque([initial]), {initial}
    transitions = terminal = out_of_order = 0
    witnesses = {}
    while queue:
        s = queue.popleft()
        invariant(s, stages)
        outgoing = successors(s, stages)
        if not outgoing:
            assert s.mode == DONE and s.producer_tail_acquired == 2, ("deadlock", stages, s)
            terminal += 1
        if s.issued - s.retired == 2:
            old, new = s.retired % 2, (s.retired + 1) % 2
            if not s.complete[old] and s.complete[new]:
                out_of_order += 1
            if s.complete[old] and not s.complete[new]:
                witnesses.setdefault("release_latest_is_unsafe", s)
        if s.issued - s.retired == 1 and not s.complete[s.retired % 2]:
            witnesses.setdefault("release_first_after_wait1_is_unsafe", s)
            if s.issued == stages:
                witnesses.setdefault("generic_U_after_tail_wait1_is_unsafe", s)
        for _, nxt in outgoing:
            transitions += 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    assert terminal == 1
    if stages >= 2:
        assert out_of_order > 0
        assert len(witnesses) == 3
    return {"stages": stages, "states": len(seen), "transitions": transitions,
            "terminal_states": terminal, "out_of_order_completion_states": out_of_order}, witnesses


def randomized(stages, seed):
    rng = random.Random(seed)
    s = State(mode=TAIL_WAIT0 if stages == 0 else ISSUE)
    actions = 0
    counts = {"issue": 0, "retire": 0, "tail_retire": 0, "wait1": 0, "wait0": 0,
              "producer_tail_acquire": 0}
    while s.mode != DONE or s.producer_tail_acquired != 2:
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
    assert counts["producer_tail_acquire"] == 2
    return actions


if __name__ == "__main__":
    exhaustive_rows, mutation_witnesses = [], None
    for stage_count in range(18):
        row, witnesses = exhaustive(stage_count)
        exhaustive_rows.append(row)
        if stage_count == 3:
            mutation_witnesses = sorted(witnesses)
    random_stage_counts = (0, 1, 2, 3, 7, 8, 15, 16, 17, 32, 33, 64, 512)
    random_actions = sum(randomized(stages, 100000 * stages + seed)
                         for stages in random_stage_counts for seed in range(32))
    print(json.dumps({"status": "CPU_ABSTRACT_PASS", "exhaustive": exhaustive_rows,
                      "total_states": sum(x["states"] for x in exhaustive_rows),
                      "total_transitions": sum(x["transitions"] for x in exhaustive_rows),
                      "randomized_sequences": len(random_stage_counts) * 32,
                      "randomized_actions": random_actions,
                      "mutation_counterexamples": mutation_witnesses}, indent=2))

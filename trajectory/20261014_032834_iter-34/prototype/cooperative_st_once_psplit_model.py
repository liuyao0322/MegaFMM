#!/usr/bin/env python3
"""CPU-only P3+4 consumer protocol/ownership model. No CUDA compilation/GPU."""
from collections import deque
from dataclasses import dataclass, replace, asdict
from pathlib import Path
import hashlib
import json
import random
import re

ROOT = Path(__file__).resolve().parents[1]
BM, BN, BK = 64, 32, 32
WG, THREADS, BANKS, PRODUCTS = 128, 384, 2, 7
TILE_ELEMENTS = BM * BN
BANK_BYTES = 2 * PRODUCTS * (BM * BK + BN * BK)
OPERAND_BYTES = BANKS * BANK_BYTES
P_TAIL_BYTES = 4 * PRODUCTS * TILE_ELEMENTS
READY, INFLIGHT, COMPLETE, WAITED = range(4)
PRODUCT_MASKS = (0b0000111, 0b1111000)


def collective_phase(count, bank):
    """Phase after count complete, chronological collective stage arrivals."""
    return ((count + 1 - bank) // BANKS) & 1


@dataclass(frozen=True)
class State:
    produced: int = 0
    consumer: tuple = (0, 0)
    mode: tuple = (READY, READY)
    owner: tuple = (-1, -1)
    released_mask: tuple = (0, 0)
    full_phase: tuple = (0, 0)
    empty_phase: tuple = (0, 0)
    producer_tail: int = 0
    retire_cta_arrivals: int = 0
    stored_products: int = 0
    publish_cta_arrivals: int = 0
    u_read: bool = False


def invariant(s, stages):
    assert 0 <= s.produced <= stages
    assert all(0 <= c <= s.produced for c in s.consumer)
    retired = min(s.consumer)
    assert s.produced - retired <= BANKS
    assert sorted(x for x in s.owner if x >= 0) == list(range(retired, s.produced))
    for bank in range(BANKS):
        assert s.full_phase[bank] == collective_phase(s.produced, bank)
        assert s.empty_phase[bank] == collective_phase(retired, bank)
        stage = s.owner[bank]
        if stage < 0:
            assert s.released_mask[bank] == 0
        else:
            assert stage % BANKS == bank
            assert s.released_mask[bank] in (0, 1, 2)
            for group in range(2):
                assert bool(s.released_mask[bank] & (1 << group)) == (s.consumer[group] > stage)
    for group in range(2):
        if s.mode[group] != READY:
            stage = s.consumer[group]
            assert stage < stages
            assert s.owner[stage % BANKS] == stage
            assert not s.released_mask[stage % BANKS] & (1 << group)
    assert 0 <= s.producer_tail <= BANKS
    if s.producer_tail:
        assert s.produced == stages
        for count in range(stages, stages + s.producer_tail):
            bank = count % BANKS
            assert s.empty_phase[bank] != 1 ^ ((count // BANKS) & 1)
            assert s.owner[bank] < 0
    if s.retire_cta_arrivals & 1:
        assert s.produced == stages and s.producer_tail == BANKS
    for group in range(2):
        if s.retire_cta_arrivals & (2 << group):
            assert s.consumer[group] == stages and s.mode[group] == READY
    if s.stored_products or s.publish_cta_arrivals or s.u_read:
        assert s.retire_cta_arrivals == 7
        assert s.owner == (-1, -1) and s.mode == (READY, READY)
        assert s.consumer == (stages, stages) and s.producer_tail == BANKS
    for group in range(2):
        if s.publish_cta_arrivals & (2 << group):
            assert s.stored_products & PRODUCT_MASKS[group] == PRODUCT_MASKS[group]
    if s.u_read:
        assert s.stored_products == 127 and s.publish_cta_arrivals == 7


def successors(s, stages):
    invariant(s, stages)
    result = []
    if s.produced < stages:
        bank = s.produced % BANKS
        phase = 1 ^ ((s.produced // BANKS) & 1)
        if s.empty_phase[bank] != phase:
            assert s.owner[bank] < 0
            owner, full = list(s.owner), list(s.full_phase)
            owner[bank] = s.produced
            full[bank] ^= 1
            # Abstract all128 vector writers/fences/full arrivals together.
            result.append(("publish_full_128", replace(s, produced=s.produced + 1,
                                                       owner=tuple(owner), full_phase=tuple(full))))
    if s.produced == stages and s.producer_tail < BANKS:
        count = stages + s.producer_tail
        bank, phase = count % BANKS, 1 ^ ((count // BANKS) & 1)
        if s.empty_phase[bank] != phase:
            assert s.owner[bank] < 0
            result.append(("producer_tail_acquire", replace(s, producer_tail=s.producer_tail + 1)))
    for group in range(2):
        stage, mode = s.consumer[group], s.mode[group]
        bank = stage % BANKS
        modes = list(s.mode)
        if mode == READY and stage < stages:
            if s.full_phase[bank] != ((stage // BANKS) & 1):
                assert s.owner[bank] == stage
                modes[group] = INFLIGHT
                # This group owns3 or4 distinct P, all their two ascending K16
                # substeps, and one commit. No generic P access until wait0.
                result.append((f"issue_g{group}", replace(s, mode=tuple(modes))))
        elif mode == INFLIGHT:
            modes[group] = COMPLETE
            # Completions may happen in either WG order; there is no assumed
            # lockstep progress or execution order between distinct P groups.
            result.append((f"complete_g{group}", replace(s, mode=tuple(modes))))
        elif mode == COMPLETE:
            modes[group] = WAITED
            result.append((f"wait0_g{group}", replace(s, mode=tuple(modes))))
        elif mode == WAITED:
            assert s.owner[bank] == stage
            masks, owner, empty = list(s.released_mask), list(s.owner), list(s.empty_phase)
            masks[bank] |= 1 << group
            if masks[bank] == 3:
                # 128+128 actual arrivals are required for the empty phase.
                masks[bank], owner[bank] = 0, -1
                empty[bank] ^= 1
            consumers = list(s.consumer)
            consumers[group] += 1
            modes[group] = READY
            result.append((f"release_128_g{group}", replace(s, consumer=tuple(consumers),
                          mode=tuple(modes), released_mask=tuple(masks), owner=tuple(owner),
                          empty_phase=tuple(empty))))
    eligible = [s.produced == stages and s.producer_tail == BANKS]
    eligible += [s.consumer[g] == stages and s.mode[g] == READY for g in range(2)]
    for role in range(3):
        bit = 1 << role
        if eligible[role] and not s.retire_cta_arrivals & bit:
            result.append((f"retire_cta_arrive_role{role}", replace(
                s, retire_cta_arrivals=s.retire_cta_arrivals | bit)))
    if s.retire_cta_arrivals == 7:
        for product in range(PRODUCTS):
            bit = 1 << product
            if not s.stored_products & bit:
                result.append((f"store_p{product + 1}", replace(s, stored_products=s.stored_products | bit)))
        publish_eligible = [True] + [s.stored_products & PRODUCT_MASKS[g] == PRODUCT_MASKS[g]
                                    for g in range(2)]
        for role in range(3):
            bit = 1 << role
            if publish_eligible[role] and not s.publish_cta_arrivals & bit:
                result.append((f"publish_cta_arrive_role{role}", replace(
                    s, publish_cta_arrivals=s.publish_cta_arrivals | bit)))
    if s.publish_cta_arrivals == 7 and not s.u_read:
        assert s.stored_products == 127
        result.append(("first_generic_u_read", replace(s, u_read=True)))
    for _, nxt in result:
        invariant(nxt, stages)
    return result


def exhaustive(stages):
    queue, seen = deque([State()]), {State()}
    transitions = terminal = 0
    witnesses = {}
    while queue:
        s = queue.popleft()
        outgoing = successors(s, stages)
        if not outgoing:
            assert s.u_read, ("deadlock", stages, s)
            terminal += 1
        if stages >= 3 and s.produced < stages:
            bank = s.produced % BANKS
            if s.owner[bank] >= 0 and s.released_mask[bank] in (1, 2):
                witnesses.setdefault("empty_count128_would_reuse_before_second_wg_release", asdict(s))
        if any(s.consumer[g] == stages for g in range(2)) and any(m == INFLIGHT for m in s.mode):
            witnesses.setdefault("one_consumer_finished_cannot_authorize_tail_overwrite", asdict(s))
        if s.retire_cta_arrivals == 7 and 0 < s.stored_products < 127:
            witnesses.setdefault("omit_second_cta_barrier_can_read_unpublished_p", asdict(s))
        for _, nxt in outgoing:
            transitions += 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    assert terminal == 1
    if stages >= 3:
        assert len(witnesses) == 3
    return {"stages": stages, "states": len(seen), "transitions": transitions,
            "terminal_states": terminal, "deadlocks": 0}, witnesses


def randomized(stages, seed):
    rng, s, actions = random.Random(seed), State(), 0
    labels = {}
    while not s.u_read:
        outgoing = successors(s, stages)
        assert outgoing
        label, s = rng.choice(outgoing)
        labels[label] = labels.get(label, 0) + 1
        actions += 1
        assert actions <= 9 * stages + 16
    assert actions == 9 * stages + 16
    assert labels.get("publish_full_128", 0) == stages
    assert labels["producer_tail_acquire"] == 2
    for group in range(2):
        for name in ("issue", "complete", "wait0", "release_128"):
            assert labels.get(f"{name}_g{group}", 0) == stages
    return actions


def mma_coordinate(thread, value):
    # Literal pinned CLayout_64x32 nested Shape/Stride, decoded in Python.
    # Atom output linear indexing is column-major (M stride1/N stride64).
    linear = (thread % 4) * 128 + ((thread // 4) % 8) + (thread // 32) * 16
    linear += (value % 2) * 64 + ((value // 2) % 2) * 8 + (value // 4) * 512
    return linear % BM, linear // BM


def tail_word(product, row, col, swizzled):
    if swizzled:
        col ^= (row & 1) | ((row & 6) << 2)
    return product * TILE_ELEMENTS + row * BN + col


def bank_patterns():
    """Predicted scalar32-bit warp accesses, not measured/generated SASS."""
    from collections import Counter
    store_max = {"row_major": 0, "column_major": 0, "xor_swizzle": 0}
    store_patterns = 0
    for product in range(PRODUCTS):
        for warp in range(4):
            for value in range(16):
                words = {"row_major": [], "column_major": [], "xor_swizzle": []}
                for lane in range(32):
                    row, col = mma_coordinate(warp * 32 + lane, value)
                    words["row_major"].append(tail_word(product, row, col, False))
                    words["column_major"].append(product * TILE_ELEMENTS + col * BM + row)
                    words["xor_swizzle"].append(tail_word(product, row, col, True))
                for layout, addresses in words.items():
                    assert len(set(addresses)) == 32  # These are not broadcast words.
                    max_depth = max(Counter(address % 32 for address in addresses).values())
                    store_max[layout] = max(store_max[layout], max_depth)
                store_patterns += 1
    read_max = {"row_major": 0, "column_major": 0, "xor_swizzle": 0}
    read_patterns = 0
    for product in range(PRODUCTS):
        for iteration in range(6):
            for warp in range(THREADS // 32):
                start = iteration * THREADS + warp * 32
                if start >= TILE_ELEMENTS:
                    continue
                assert start + 31 < TILE_ELEMENTS
                words = {"row_major": [], "column_major": [], "xor_swizzle": []}
                for lane in range(32):
                    row, col = divmod(start + lane, BN)
                    words["row_major"].append(tail_word(product, row, col, False))
                    words["column_major"].append(product * TILE_ELEMENTS + col * BM + row)
                    words["xor_swizzle"].append(tail_word(product, row, col, True))
                for layout, addresses in words.items():
                    assert len(set(addresses)) == 32
                    max_depth = max(Counter(address % 32 for address in addresses).values())
                    read_max[layout] = max(read_max[layout], max_depth)
                read_patterns += 1
    assert store_max == {"row_major": 8, "column_major": 4, "xor_swizzle": 1}
    assert read_max == {"row_major": 1, "column_major": 32, "xor_swizzle": 1}
    assert store_patterns == read_patterns == 448
    return {"bank_model": "32banks,4-byte words; bank=physical_word%32. Actual SASS issue grouping and hardware conflicts not measured.",
            "primary_source": "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html",
            "scalar_warp_store_patterns": store_patterns, "scalar_warp_read_patterns": read_patterns,
            "predicted_worst_store_bank_multiplicity": store_max,
            "predicted_worst_read_bank_multiplicity": read_max,
            "recommended_word_offset": "p*2048+row*32+(col^((row&1)|((row&6)<<2)))",
            "swizzle_padding_bytes": 0, "actual_compiler_shared_load_store_and_hardware_metrics": "PENDING"}


def ownership():
    traits = ROOT / "third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp"
    source = traits.read_text()
    expected = "using CLayout_64x32 = Layout<Shape <Shape < _4,_8, _4>,Shape < _2,_2, _4>>, Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>>;"
    match = re.search(r"using CLayout_64x32\s*=.*?;", source, re.S)
    assert match and re.sub(r"\s+", " ", match.group()).strip() == expected
    p_counts = bytearray(PRODUCTS * TILE_ELEMENTS)
    swizzled_counts = bytearray(PRODUCTS * TILE_ELEMENTS)
    addresses, coordinate_owner = [], {}
    for group, products in enumerate(((0, 1, 2), (3, 4, 5, 6))):
        coordinates = set()
        for thread in range(WG):
            for value in range(16):
                row, col = mma_coordinate(thread, value)
                assert 0 <= row < BM and 0 <= col < BN
                assert (row, col) not in coordinates
                coordinates.add((row, col))
                for product in products:
                    logical = product * TILE_ELEMENTS + row * BN + col
                    p_counts[logical] += 1
                    swizzled_counts[tail_word(product, row, col, True)] += 1
                    addresses.append(4 * logical)
                    coordinate_owner[(product, row, col)] = (group, thread, value)
        assert len(coordinates) == TILE_ELEMENTS
    assert all(x == 1 for x in p_counts)
    assert all(x == 1 for x in swizzled_counts)
    assert len(coordinate_owner) == PRODUCTS * TILE_ELEMENTS
    assert len(set(addresses)) == PRODUCTS * TILE_ELEMENTS
    assert min(addresses) == 0 and max(addresses) + 4 == P_TAIL_BYTES
    assert P_TAIL_BYTES <= OPERAND_BYTES and P_TAIL_BYTES > BANK_BYTES
    counts = bytearray(TILE_ELEMENTS)
    loads, coords_per_thread = 0, []
    for thread in range(THREADS):
        tasks = list(range(thread, TILE_ELEMENTS, THREADS))
        coords_per_thread.append(len(tasks))
        for index in tasks:
            row, col = divmod(index, BN)
            counts[index] += 1
            for product in range(PRODUCTS):
                offset = 4 * tail_word(product, row, col, True)
                assert 0 <= offset and offset + 4 <= P_TAIL_BYTES
                loads += 1
    assert all(x == 1 for x in counts)
    assert loads == PRODUCTS * TILE_ELEMENTS
    assert sorted(set(coords_per_thread)) == [5, 6]
    return {"status": "PASS_LITERAL_PINNED_C_LAYOUT_DECODE_CPU_ONLY",
            "p_store_float_count": len(addresses), "p_store_bytes_per_cta": 4 * len(addresses),
            "u_p_read_float_count": loads, "u_p_read_bytes_per_cta": 4 * loads,
            "p_address_min_bytes": 0, "p_address_max_exclusive_bytes": P_TAIL_BYTES,
            "p_store_unique": True, "u_owner_unique": True,
            "xor_swizzled_p_store_unique_and_all_slots_covered": True,
            "tail_bank_analysis": bank_patterns(),
            "u_coordinates_per_thread": {"six": coords_per_thread.count(6), "five": coords_per_thread.count(5)},
            "bank0_bytes": BANK_BYTES, "two_bank_bytes": OPERAND_BYTES,
            "bank1_overlap_bytes": P_TAIL_BYTES - BANK_BYTES,
            "per_product_tail_byte_spans": [[p * TILE_ELEMENTS * 4, (p + 1) * TILE_ELEMENTS * 4]
                                             for p in range(PRODUCTS)],
            "actual_CuTe_instantiation_and_descriptors": "NOT_RUN_PENDING_CUDA_OWNER"}


def global_output_coverage(m, n, ldd):
    assert m % (2 * BM) == 0 and n % (2 * BN) == 0 and ldd >= n
    touched = bytearray(m * ldd)
    writes = 0
    for tile_m in range(m // (2 * BM)):
        for tile_n in range(n // (2 * BN)):
            for thread in range(THREADS):
                for index in range(thread, TILE_ELEMENTS, THREADS):
                    row, col = divmod(index, BN)
                    row += tile_m * BM
                    col += tile_n * BN
                    quad = (row * ldd + col, row * ldd + col + n // 2,
                            (row + m // 2) * ldd + col, (row + m // 2) * ldd + col + n // 2)
                    assert len(set(quad)) == 4
                    for address in quad:
                        assert 0 <= address < (m - 1) * ldd + n
                        assert touched[address] == 0
                        touched[address] = 1
                        writes += 1
    assert writes == m * n
    for row in range(m):
        assert all(touched[row * ldd:row * ldd + n])
        assert not any(touched[row * ldd + n:(row + 1) * ldd])
    return {"m": m, "n": n, "ldd": ldd, "unique_active_output_writes": writes,
            "active_holes": 0, "padding_writes": 0}


def alias_trace(seed, beta_active):
    rng = random.Random(seed)
    m = n = 1024
    stride = 1041  # Scalar output/C0 path, no vector alignment assumptions.
    row_base, col_base = 3 * BM, 7 * BN
    memory, addresses_by_owner = {}, []
    for thread in range(THREADS):
        tasks = []
        for index in range(thread, TILE_ELEMENTS, THREADS):
            row, col = divmod(index, BN)
            row, col = row + row_base, col + col_base
            addresses = (row * stride + col, row * stride + col + n // 2,
                         (row + m // 2) * stride + col,
                         (row + m // 2) * stride + col + n // 2)
            tasks.append(addresses)
            for address in addresses:
                assert address not in memory
                memory[address] = ("C0", address)
        addresses_by_owner.append(tasks)
    # Each warp-thread task does all4 loads before any of its4 stores. Random
    # scalar interleaving permits other threads to run between any two events.
    next_task, event = [0] * THREADS, [0] * THREADS
    active = list(range(THREADS))
    reads = stores = 0
    while active:
        active_index = rng.randrange(len(active))
        thread = active[active_index]
        addresses = addresses_by_owner[thread][next_task[thread]]
        stage = event[thread]
        if beta_active and stage < 4:
            address = addresses[stage]
            assert memory[address] == ("C0", address)
            reads += 1
        else:
            quadrant = stage - 4 if beta_active else stage
            address = addresses[quadrant]
            memory[address] = ("D", address)
            stores += 1
        event[thread] += 1
        if event[thread] == (8 if beta_active else 4):
            next_task[thread] += 1
            event[thread] = 0
            if next_task[thread] == len(addresses_by_owner[thread]):
                active.pop(active_index)
    assert stores == 4 * TILE_ELEMENTS
    assert reads == (stores if beta_active else 0)
    assert all(tag == ("D", address) for address, tag in memory.items())
    return reads, stores


def resources_and_bytes():
    grid, stages = (1024 // (2 * BM)) * (1024 // (2 * BN)), 1024 // (2 * BK)
    input_bytes = 2 * 4 * (BM * BK + BN * BK)
    transforms = 4 * (BM * BK + BN * BK)
    total_full_commits = grid * stages
    # 3 products *2 K16 instructions and4 products *2, still14 m64n32.
    mma_instructions = (3 + 4) * (BK // 16)
    initial80_pool = 80 * THREADS
    role_pool = WG * (64 + 80 + 80)
    assert role_pool == 28672 <= initial80_pool == 30720
    assert BANK_BYTES == 43008 and OPERAND_BYTES == 86016 and P_TAIL_BYTES == 57344
    return {"grid_ctas": grid, "k32_stages": stages,
            "shape": {"BM": BM, "BN": BN, "BK": BK, "threads": THREADS},
            "accumulator_floats_per_thread": {"consumer_g0": 48, "consumer_g1": 64},
            "optional_lower_quota_derivations_not_implemented": {"producer": 64, "consumer_g0": 80, "consumer_g1": 80,
                                           "role_pool_registers": role_pool,
                                           "hypothetical_initial80_pool_registers": initial80_pool,
                                           "hypothetical_two_cta_initial80_pool_registers": 2 * initial80_pool,
                                           "asymmetric80_96_option": {"role_pool_registers": WG * (64 + 80 + 96),
                                                                      "initial80_pool_registers": initial80_pool,
                                                                      "directions": "64<=initial80<=consumer80/96",
                                                                      "actual_role_register_paths": "PENDING"},
                                           "symmetric96_option": {"role_pool_registers": WG * (64 + 96 + 96),
                                                                   "minimum_initial8_step_quota": 88,
                                                                   "two_initial88_cta_pool_registers": 2 * THREADS * 88},
                                           "actual_compiler_owned_pool_quota_paths_spills": "PENDING"},
            "frozen_candidate_register_quota_hypothesis": {"producer": 64, "consumer_g0": 96,
                "consumer_g1": 96, "role_pool_registers": WG * (64 + 96 + 96),
                "minimum_initial8_step_quota": 88, "maximum_initial_quota_for_inc96_direction": 96,
                "initial88_cta_pool_registers": THREADS * 88,
                "two_initial88_cta_pool_registers": 2 * THREADS * 88,
                "actual_initial_role_paths_implicit_register_widths_spills": "PENDING"},
            "barriers": {"full_arrivals_per_stage": 128, "empty_arrivals_per_stage": 256,
                         "uniform_cta_barriers": 3, "uniform_cta_barrier_threads": THREADS},
            "predicted_shared": {"operand_bytes": OPERAND_BYTES, "barrier_bytes": 32,
                                 "align128_struct_bytes": ((OPERAND_BYTES + 32 + 127) // 128) * 128,
                                 "tail_buffer_extra_allocation_bytes": 0,
                                 "actual_compiled_dynamic_static_bytes": "PENDING"},
            "per_cta_stage": {"logical_global_A_quadrant_input_bytes": 4 * BM * BK * 2,
                              "logical_global_B_quadrant_input_bytes": 4 * BN * BK * 2,
                              "logical_global_input_bytes": input_bytes,
                              "fp32_transform_add_sub_count": transforms,
                              "logical_final_operand_generic_shared_store_bytes": BANK_BYTES,
                              "wgmma_logical_operand_use_bytes": BANK_BYTES,
                              "wgmma_m64n32k16_instructions": mma_instructions},
            "per_cta_tail": {"generic_shared_P_store_bytes": P_TAIL_BYTES,
                             "generic_shared_U_P_load_bytes": P_TAIL_BYTES,
                             "generic_shared_P_rendezvous_bytes": 2 * P_TAIL_BYTES,
                             "global_D_store_bytes": 4 * TILE_ELEMENTS * 4,
                             "global_active_C0_load_bytes": 4 * TILE_ELEMENTS * 4,
                             "global_beta0_C0_load_bytes": 0},
            "whole1024": {"logical_global_input_bytes": input_bytes * total_full_commits,
                          "fp32_transform_add_sub_count": transforms * total_full_commits,
                          "logical_final_operand_generic_shared_store_bytes": BANK_BYTES * total_full_commits,
                          "wgmma_logical_operand_use_bytes": BANK_BYTES * total_full_commits,
                          "wgmma_m64n32k16_instructions": mma_instructions * total_full_commits,
                          "generic_shared_P_store_bytes": P_TAIL_BYTES * grid,
                          "generic_shared_U_P_load_bytes": P_TAIL_BYTES * grid,
                          "generic_shared_P_rendezvous_bytes": 2 * P_TAIL_BYTES * grid,
                          "global_D_store_bytes": 4 * TILE_ELEMENTS * 4 * grid,
                          "global_active_C0_load_bytes": 4 * TILE_ELEMENTS * 4 * grid,
                          "global_beta0_C0_load_bytes": 0},
            "traffic_scope": "Logical source-level scalar/vector bytes; no HBM/L2/shared hardware traffic, transaction count, bank-conflict or performance measurement."}


def producer_logical_ownership():
    counts = {}
    for operand, rows, cols, contiguous in (("A", BM, BK, "K"), ("B", BK, BN, "N")):
        vectors = rows * cols // 8
        seen = bytearray(rows * cols)
        vector_tasks = []
        for thread in range(WG):
            for task in range(thread, vectors, WG):
                row, col_base = divmod(task, cols // 8)
                col_base *= 8
                vector_tasks.append(task)
                for lane in range(8):
                    index = row * cols + col_base + lane
                    seen[index] += 1
        assert len(vector_tasks) == vectors and len(set(vector_tasks)) == vectors
        assert all(x == 1 for x in seen)
        counts[operand] = {"producer_vector_tasks": vectors, "original_each_quadrant_elements": len(seen),
                           "four_original_quadrant_load_bytes": 4 * len(seen) * 2,
                           "seven_final_slot_store_bytes": 7 * len(seen) * 2,
                           "contiguous_vector_axis": contiguous, "logical_unique_writer": True,
                           "actual_CuTe_swizzled_physical_vector_geometry": "PENDING_COMPILATION"}
    assert counts["A"]["producer_vector_tasks"] == 256
    assert counts["B"]["producer_vector_tasks"] == 128
    return counts


def strip_comments(text):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)


def compact(text):
    return re.sub(r"\s+", "", strip_comments(text))


def block_at(text, opening):
    assert text[opening] == "{"
    depth = 1
    for position in range(opening + 1, len(text)):
        depth += (text[position] == "{") - (text[position] == "}")
        if depth == 0:
            return text[opening + 1:position], position + 1
    raise AssertionError("unclosed source block")


def function_body(text, name):
    match = re.search(r"\b" + re.escape(name) + r"\s*\(", text)
    assert match, name
    return block_at(text, text.index("{", match.end()))[0]


def source_review():
    path = ROOT / ".scratch/cooperative_st_once_psplit_candidate.cu"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == "84e56c44deb1585f6be457461dcc9073cad6631316e159249e77dbbb77c78664"
    source = strip_comments(raw.decode())
    reference = strip_comments((ROOT / "solution/megakernel_wgmma_pipeline.cu").read_text())
    def rename(text):
        return text.replace("pipeline_", "psplit_").replace("PipelinePackedEight", "PSplitPackedEight")
    helpers = ("contiguous_eight_sentinels", "read_eight", "unpack", "store_eight",
               "round_pair", "put_word", "fill_a", "fill_b")
    for helper in helpers:
        assert compact(rename(function_body(reference, "pipeline_" + helper))) == compact(function_body(source, "psplit_" + helper)), helper
    for layout in ("A", "B"):
        pattern = r"using Layout" + layout + r" =.*?;"
        assert compact(re.search(pattern, source, re.S).group()) == compact(re.search(pattern, reference, re.S).group())
    kernel = function_body(source, "winograd_wgmma_psplit")
    old_kernel = function_body(reference, "winograd_wgmma_pipeline")
    producer_anchor = "if (tid < T::ProducerThreads)"
    producer, producer_end = block_at(kernel, kernel.index("{", kernel.index(producer_anchor)))
    old_producer, _ = block_at(old_kernel, old_kernel.index("{", old_kernel.index(producer_anchor)))
    assert compact(rename(old_producer)) == compact(producer)
    consumer_start = kernel.index("{", kernel.index("else", producer_end))
    consumer, consumer_end = block_at(kernel, consumer_start)
    assert "warpgroup_reg_alloc<96>()" in consumer
    assert "warpgroup_reg_dealloc<64>()" in producer
    assert "params.consumer_arv_count = T::AllConsumerThreads;" in kernel
    assert "params.producer_arv_count = T::ProducerThreads;" in kernel
    assert "static constexpr int AllConsumerThreads = 256;" in source
    assert "__launch_bounds__(384,1)" in source
    quota = consumer.index("warpgroup_reg_alloc<96>()")
    for q in range(4):
        assert consumer.index(f"clear(q{q})") > quota
    assert not re.search(r"auto\s+q[0-3]\s*=\s*thr_mma\.make_fragment_C", kernel)
    assert "using QLayout = typename decltype(thr_mma.make_fragment_C(coords))::layout_type;" in kernel
    for q in range(4):
        assert f"float q{q}_data[cosize_v<QLayout>];" in kernel
        assert f"auto q{q} = make_tensor(make_rmem_ptr(q{q}_data),QLayout{{}});" in kernel
        # The arrays have no initializer; all four views initialize only their
        # pointer/layout metadata, not any underlying float element.
        assert len(re.findall(rf"\bfloat\s+q{q}_data\s*\[", kernel)) == 1
    barriers = list(re.finditer(r"__syncthreads\(\);", kernel))
    assert len(barriers) == 3
    for barrier in barriers:
        prefix = kernel[:barrier.start()]
        assert prefix.count("{") == prefix.count("}"), "divergent CTA barrier"
    assert barriers[0].start() < kernel.index(producer_anchor) < consumer_end < barriers[1].start()
    assert not re.search(r"\breturn\b", kernel)
    assert producer.index("producer_tail(state)") > producer.index("producer_commit(state)")
    assert producer.index("fence.proxy.async.shared::cta") < producer.index("producer_commit(state)")
    calls = re.findall(r"gemm\(mma,tCrA\(_,_,_,Int<(\d)>\{\}\),tCrB\(_,_,_,Int<(\d)>\{\}\),q(\d)\)", consumer)
    assert calls == [("0", "0", "0"), ("1", "1", "1"), ("2", "2", "2"),
                     ("3", "3", "0"), ("4", "4", "1"), ("5", "5", "2"), ("6", "6", "3")]
    assert consumer.index("consumer_wait(state)") < consumer.index("warpgroup_arrive()") < consumer.index("gemm(")
    assert consumer.rindex("gemm(") < consumer.index("warpgroup_commit_batch()") < consumer.index("warpgroup_wait<0>()")
    assert consumer.index("warpgroup_wait<0>()") < consumer.rindex("warpgroup_fence_operand(q3)") < consumer.index("consumer_release(state)")
    assert "for (int64_t k_base = 0; k_base < p.k/2; k_base += T::BK)" in producer
    assert "for (int64_t k_base = 0; k_base < p.k/2; k_base += T::BK)" in consumer
    first, second = barriers[1].end(), barriers[2].start()
    tail_store = kernel[first:second]
    assert "float* tail = storage.tail_products;" in tail_store
    for expression, q in (("first_product+0", 0), ("first_product+1", 1), ("first_product+2", 2), ("6", 3)):
        assert f"tail[psplit_tail_word({expression},row,col)] = q{q}(i);" in tail_store
    assert "const int first_product = tid < 256 ? 0 : 3;" in tail_store
    assert "if (tid >= 256) tail[psplit_tail_word(6,row,col)] = q3(i);" in tail_store
    output = kernel[barriers[2].end():]
    assert "for (int linear = tid; linear < T::BM*T::BN; linear += T::Threads)" in output
    for product in range(7):
        assert f"const float p{product+1} = tail[psplit_tail_word({product},local_row,local_col)];" in output
    end_statement = "D[(row+mh)*p.ldd+col+nh] = d22;"
    old_epilogue = old_kernel[old_kernel.index("const float u1"):old_kernel.index(end_statement)+len(end_statement)]
    epilogue = output[output.index("const float u1"):output.index(end_statement)+len(end_statement)]
    for product in range(1, 8):
        old_epilogue = old_epilogue.replace(f"p{product}(i)", f"p{product}")
    assert compact(epilogue) == compact(old_epilogue), "fixed U/alpha RN/beta FMA/output ordering changed"
    assert "const int xor_mask = (row&1) | ((row&6)<<2);" in source
    assert "return product*2048+row*32+(col^xor_mask);" in source
    assert "float tail_products[7*BM*BN];" in source and "union {" in source
    assert source.index("union {") < source.index("float tail_products[") < source.index("typename Pipeline::SharedStorage barriers;")
    assert "reinterpret_cast<float" not in source
    assert source.count("<<<") == 1
    assert "cudaMalloc" not in source and "cudaDeviceSynchronize" not in source and "cudaStreamSynchronize" not in source
    tuple_source = (ROOT / "third_party/cutlass/include/cute/container/tuple.hpp").read_text()
    assert "EBO() : t_{} {}" in tuple_source
    pointer_source = (ROOT / "third_party/cutlass/include/cute/pointer.hpp").read_text()
    assert "struct is_rmem : bool_constant<not (is_gmem<T>::value || is_smem<T>::value)> {};" in pointer_source
    return {"status": "PASS_READONLY_SOURCE_AND_PINNED_API_ONLY_NOT_COMPILED",
            "candidate_sha256": digest, "candidate_lines": len(raw.decode().splitlines()),
            "source30_identical_renamed_helper_bodies": list(helpers), "source30_identical_producer_branch": True,
            "source30_identical_A_B_layout_type_expressions": True,
            "source30_identical_fixed_U_RNalpha_betaFMA_fourC0_before_fourD_epilogue": True,
            "actual_source_P_assignment": {"g0": [1, 2, 3], "g1": [4, 5, 6, 7]},
            "same_per_P_ascending_K32_and_pinned_two_K16_gemm_atom": "SOURCE_AND_PINNED_API_ONLY_ACTUAL_TEMPLATE_AND_GPU_BITS_PENDING",
            "lexical_common_CTA_barriers": 3, "early_kernel_returns": 0,
            "consumer_wait0_and_operand_fences_before_128_release": True,
            "source_initialization_fix": "Former owning Tensor() declarations can value-initialize ArrayEngine via default nonempty EBO:t_{}. Revised decltype layout plus noninitialized plain float arrays and rmem pointer views; clear occurs after consumer inc96. Actual SASS must exclude early float accesses and prove scalarization/no unsafe local spill.",
            "typed_shared_union_operands_and_float_tail": True,
            "tail_XOR_layout_matches_CPU_model": True, "one_kernel_launch_site": 1,
            "actual_source_quota_hypothesis": {"producer": 64, "consumer_g0": 96, "consumer_g1": 96},
            "compiled_resource_and_physical_layout": "NOT_RUN_PENDING",
            "host_safety_scope": "Native-only helper; integrating formal Plan must retain common capacity/overflow/overlap/alignment/range/invalid-policy/fallback/Auto guards."}


def main():
    stage_counts = list(range(18)) + [31, 32, 33, 63, 64, 65, 511, 512]
    rows, witnesses = [], {}
    for stages in stage_counts:
        row, current = exhaustive(stages)
        rows.append(row)
        if stages == 3:
            witnesses = current
    random_counts = (0, 1, 2, 3, 7, 8, 15, 16, 17, 32, 33, 64, 512)
    actions = sum(randomized(stages, 100000 * stages + seed) for stages in random_counts for seed in range(16))
    output_shapes = [(128, 64, 64), (128, 64, 65), (256, 128, 137),
                     (1024, 1024, 1024), (1024, 1024, 1025), (1024, 1024, 1041)]
    coverage = [global_output_coverage(*shape) for shape in output_shapes]
    alias_reads = alias_stores = beta0_stores = 0
    for seed in range(32):
        reads, stores = alias_trace(seed, True)
        alias_reads += reads
        alias_stores += stores
        reads, stores = alias_trace(seed + 10000, False)
        assert reads == 0
        beta0_stores += stores
    bindings = {}
    for path in ("solution/megakernel_wgmma_pipeline.cu",
                 "third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp",
                 "third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp",
                 "third_party/cutlass/include/cute/arch/mma_sm90_gmma.hpp",
                 "third_party/cutlass/include/cute/container/tuple.hpp",
                 "third_party/cutlass/include/cute/container/array_aligned.hpp",
                 "third_party/cutlass/include/cute/container/array.hpp",
                 "third_party/cutlass/include/cute/tensor_impl.hpp",
                 "third_party/cutlass/include/cute/pointer.hpp",
                 ".scratch/cooperative_st_once_psplit_candidate.cu",
                 ".scratch/cooperative_st_once_psplit_model.py"):
        bindings[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    report = {"status": "PASS_CPU_ABSTRACT_PROTOCOL_AND_LITERAL_HEADER_OWNERSHIP_ONLY",
              "no_formal_cuda_candidate": True, "scratch_cuda_candidate_exists": True,
              "no_host_or_cuda_compilation": True,
              "no_gpu_execution": True, "no_sanitizer_or_performance": True,
              "evidence_bindings": bindings,
              "synchronization": {"model": "Three collective128-thread roles, partial128+128 empty arrivals, explicit full/empty phases, independent WG completions and wait0, producer_tail two acquires, retire CTA barrier, 7P stores, publication CTA barrier, first generic U read.",
                                  "exhaustive": rows, "total_states": sum(r["states"] for r in rows),
                                  "total_transitions": sum(r["transitions"] for r in rows),
                                  "randomized_traces": 16 * len(random_counts),
                                  "randomized_actions": actions,
                                  "mutation_counterexamples": witnesses,
                                  "fairness_scope": "No terminal deadlock in the model; assumes valid CTA-owned quotas/acquisition, fair scheduling and eventual issued WGMMA completion, does not establish physical GPU forward progress."},
              "source_review": source_review(),
              "ownership": ownership(), "producer_logical_ownership": producer_logical_ownership(),
              "output_coverage": coverage,
              "alias": {"scope": "Exact C0==D with ldc==ldd, arbitrary interleaving of scalar per-coordinate4 loads then4 stores within one owner tile; whole-grid disjointness checked separately.",
                        "active_beta_traces": 32, "original_C0_reads_verified": alias_reads,
                        "active_beta_D_stores": alias_stores, "beta0_null_C0_traces": 32,
                        "beta0_C0_reads": 0, "beta0_D_stores": beta0_stores,
                        "arbitrary_C0_D_overlap": "Not permitted by common Plan; not proved safe."},
              "resources_and_bytes": resources_and_bytes(),
              "pending": ["Actual CuTe template/descriptor compilation", "Compiled object/linked role CFG and implicit register widths", "CTA-owned initial pool and setmaxnreg direction/quotas", "Generic P spill/reload absence while WGMMA in flight", "Actual dynamic/static shared and alignment", "CUDA execution and exact same-atom seven P/final U bits", "Original numerical gates", "Fresh input/rebind/Graph single-kernel capture", "Targeted memcheck/racecheck/synccheck", "Complete-call latency vs frozen bestMega and tuned cuBLAS"]}
    target = ROOT / ".scratch/cooperative_st_once_psplit_review.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "json": str(target),
                      "states": report["synchronization"]["total_states"],
                      "transitions": report["synchronization"]["total_transitions"],
                      "random_traces": report["synchronization"]["randomized_traces"],
                      "random_actions": actions, "p_store_bytes_per_cta": P_TAIL_BYTES,
                      "p_read_bytes_per_cta": P_TAIL_BYTES,
                      "whole1024_P_rendezvous_bytes": 2 * P_TAIL_BYTES * 128}, indent=2))


if __name__ == "__main__":
    main()

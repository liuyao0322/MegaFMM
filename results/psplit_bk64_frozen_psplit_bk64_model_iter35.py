#!/usr/bin/env python3
"""Scratch-only BK64 geometry plus freshly executed frozen P3+4 CPU protocol.

No CuTe/C++/CUDA instantiation, GPU work, or file mutation is performed.
Literal-header address decoding is an expectation to verify by later compile.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ".scratch/cooperative_st_once_psplit_candidate.cu"
BASE_SHA = "84e56c44deb1585f6be457461dcc9073cad6631316e159249e77dbbb77c78664"
PROTO = ".scratch/cooperative_st_once_psplit_model.py"
PROTO_SHA = "d5a159f926b9608a467d39ca806027f14772ce44fc035f078fc337f5ffa6755f"
CANDIDATE = ".scratch/psplit_bk64_candidate.cu"
BM, BN, BK, PRODUCTS, BANKS = 64, 32, 64, 7, 2
A_ELEMENTS, B_ELEMENTS = PRODUCTS * BM * BK, PRODUCTS * BN * BK
BANK_BYTES = 2 * (A_ELEMENTS + B_ELEMENTS)
OPERAND_BYTES = BANKS * BANK_BYTES
TAIL_BYTES = PRODUCTS * BM * BN * 4


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def load_frozen_protocol():
    assert digest(BASE) == BASE_SHA and digest(PROTO) == PROTO_SHA
    spec = importlib.util.spec_from_file_location("_frozen_bk32_psplit_protocol", ROOT / PROTO)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # Its main is NOT called and no frozen file is written.
    return module


def source_geometry_check():
    base = (ROOT / BASE).read_text()
    source = (ROOT / CANDIDATE).read_text()
    inverse = {"PSplitBK64Traits": "PSplitTraits", "PSplitBK64PackedEight": "PSplitPackedEight"}
    normalized = re.sub(r"\b[A-Za-z_]\w*\b",
                        lambda match: inverse.get(match[0], match[0].replace("psplit_bk64", "psplit")),
                        source.split("\n", 2)[2])
    for before, after in (
        ("static constexpr int BK = 64;", "static constexpr int BK = 32;"),
        ("const int row = task/8, k = (task&7)*8;", "const int row = task/4, k = (task&3)*8;"),
        ("p.k%128 == 0", "p.k%64 == 0"),
        ("// Each task owns eight K elements of one row. All512 BK64 tasks are assigned once",
         "// Each task owns eight K elements of one row. All256 tasks are assigned once"),
        ("// BN32/BK64:256 tasks. All use the unchanged vector8 original reads/stores.",
         "// BN16:64 tasks; BN32:128 tasks. All use vector8 original reads/stores."),
        ("// four ascending K16 steps/stage and a complete ascending whole-K sum.",
         "// two ascending K16 steps/stage and a complete ascending whole-K sum.")):
        assert normalized.count(before) == 1
        normalized = normalized.replace(before, after)
    assert normalized == base
    assert source.count("__syncthreads();") == 3
    assert source.count("warpgroup_wait<0>();") == 1
    assert "params.consumer_arv_count = T::AllConsumerThreads;" in source
    assert "static constexpr int AllConsumerThreads = 256;" in source
    assert "cutlass::arch::warpgroup_reg_dealloc<64>();" in source
    assert "cutlass::arch::warpgroup_reg_alloc<96>();" in source
    # Counterexample to replacing ONLY BK: half of A tasks are out of bounds,
    # and the valid half leaves K32..63 unwritten in every logical product slot.
    naive = [(task // 4, (task & 3) * 8) for task in range(BM * BK // 8)]
    invalid = [(task, row, k) for task, (row, k) in enumerate(naive) if row >= BM]
    assert len(invalid) == 256 and invalid[0] == (256, 64, 0)
    valid = {(row, k) for row, k in naive if row < BM}
    missing = {(row, k) for row in range(BM) for k in range(0, BK, 8)} - valid
    assert len(missing) == 256 and all(k >= 32 for _, k in missing)
    guard_rows = []
    for parent_k in (64, 128, 192, 256, 384, 512, 1024, 2176):
        old = parent_k % 64 == 0
        new = parent_k % (2 * BK) == 0
        partial = (parent_k // 2) % BK != 0
        assert new != partial
        guard_rows.append({"parent_K": parent_k, "old_BK32_geometry_accepts": old,
                           "new_BK64_geometry_accepts": new, "partial_BK64_leaf": partial,
                           "new_full_stages": parent_k // (2 * BK) if new else None})
    chronology = []
    for parent_k in (128, 256, 384, 512, 1024, 2176, 4224):
        old = [base_k + sub for base_k in range(0, parent_k // 2, 32)
               for sub in range(0, 32, 16)]
        new = [base_k + sub for base_k in range(0, parent_k // 2, BK)
               for sub in range(0, BK, 16)]
        assert old == new == list(range(0, parent_k // 2, 16))
        chronology.append({"parent_K": parent_k, "same_ordered_K16_offsets_each_P": True,
                           "K16_steps_each_P": len(new), "BK64_stages": parent_k // 128})
    return {"normalized_exact_frozen_source_match_after_three_geometry_reverts": True,
            "executable_differences": ["BK32 to64", "A task /4,&3 to /8,&7", "parent Kguard %64 to%128"],
            "naive_BK_only_change_is_unsafe": {"invalid_A_vector_tasks": len(invalid),
                                               "first_invalid_task_row_K": invalid[0],
                                               "missing_valid_A_vectors": len(missing)},
            "guard_checks": guard_rows, "per_P_source_chronology": chronology,
            "actual_GPU_bits_or_CuTe_instantiation_proven": False}


def literal_layout_check():
    traits = (ROOT / "third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp").read_text()
    compact = lambda text: "".join(text.split())
    expected_A = "using Layout_K_SW32_Atom_Bits = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape<_8, _256>,Stride<_256,_1>>>;"
    expected_B = "using Layout_MN_SW32_Atom_Bits = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape<_256,_8>,Stride<_1,_256>>>;"
    assert compact(expected_A) in compact(traits) and compact(expected_B) in compact(traits)
    # Verify the pinned source operations used in this Python decoding. These
    # are source checks, not successful C++ template instantiation.
    layout = (ROOT / "third_party/cutlass/include/cute/layout.hpp").read_text()
    composed = (ROOT / "third_party/cutlass/include/cute/layout_composed.hpp").read_text()
    flagged = (ROOT / "third_party/cutlass/include/cute/pointer_flagged.hpp").read_text()
    pointer = (ROOT / "third_party/cutlass/include/cute/pointer_swizzle.hpp").read_text()
    swizzle = (ROOT / "third_party/cutlass/include/cute/swizzle.hpp").read_text()
    selector = (ROOT / "third_party/cutlass/include/cute/arch/mma_sm90.hpp").read_text()
    gemm = (ROOT / "third_party/cutlass/include/cute/algorithm/gemm.hpp").read_text()
    assert "weakly_compatible(block_shape, target_shape)" in layout
    assert "tile_to_shape(layout.layout_b(), trg_shape, ord_shape)" in composed
    assert "make_smem_ptr(ptr.get(), layout.layout_a())" in flagged
    assert "SwizzleFn::apply(reinterpret_cast<uintptr_t>(ptr))" in pointer
    assert "offset ^ shiftr(offset & yyy_msk{}, msk_sft{})" in swizzle
    assert "SM90_64x32x16_F32F16F16_SS<MajorA, MajorB, Args...>" in selector
    assert "SM90_64x32x16_F32BF16BF16_SS<MajorA, MajorB, Args...>" in selector
    assert "for (int k = 0; k < K; ++k)" in gemm
    # Upcast16 converts atom logical bit layouts to A8x16/B16x8 elements.
    # tile_to_shape defaults to LayoutLeft: repeat first mode, then K, then P.
    assert BM % 8 == BK % 16 == BN % 16 == BK % 8 == 0
    def raw_A(row, k, product):
        return (product * 8 * 4 + (k // 16) * 8 + row // 8) * 128 + (row % 8) * 16 + k % 16
    def raw_B(col, k, product):
        return (product * 2 * 8 + (k // 8) * 2 + col // 16) * 128 + col % 16 + (k % 8) * 16
    # smem_ptr_flag keeps the swizzle in the pointer. Decode BYTES at absolute
    # base+element_offset*2, never apply byte swizzle directly to element indices.
    def swizzle_byte(address):
        return address ^ ((address & 128) >> 3)
    raw_sets = [set(), set()]
    for product in range(PRODUCTS):
        for row in range(BM):
            for k in range(BK):
                raw_sets[0].add(raw_A(row, k, product))
        for col in range(BN):
            for k in range(BK):
                raw_sets[1].add(raw_B(col, k, product))
    assert raw_sets[0] == set(range(A_ELEMENTS))
    assert raw_sets[1] == set(range(B_ELEMENTS))
    vector_checks = element_checks = 0
    for base in (0, 128, 256, 384):
        allocation_addresses = set()
        for bank in range(BANKS):
            bank_base = base + bank * BANK_BYTES
            for product in range(PRODUCTS):
                for operand, rows, vector_starts, byte_bias, raw in (
                    ("A", range(BM), range(0, BK, 8), 0, raw_A),
                    ("B", range(BK), range(0, BN, 8), 2 * A_ELEMENTS, raw_B)):
                    for row in rows:
                        for start in vector_starts:
                            coordinates = [(row, start + lane) if operand == "A" else (start + lane, row)
                                           for lane in range(8)]
                            addresses = [swizzle_byte(bank_base + byte_bias + 2 * raw(i, k, product))
                                         for i, k in coordinates]
                            assert addresses[0] % 16 == 0
                            assert addresses == list(range(addresses[0], addresses[0] + 16, 2))
                            assert all(bank_base + byte_bias <= address < bank_base + byte_bias +
                                       (2 * A_ELEMENTS if operand == "A" else 2 * B_ELEMENTS)
                                       for address in addresses)
                            assert not allocation_addresses.intersection(addresses)
                            allocation_addresses.update(addresses)
                            vector_checks += 1
                            element_checks += 8
        assert allocation_addresses == set(range(base, base + OPERAND_BYTES, 2))
    A_tasks = [(task // 8, (task & 7) * 8) for task in range(BM * BK // 8)]
    B_tasks = [(task // (BN // 8), (task % (BN // 8)) * 8) for task in range(BN * BK // 8)]
    assert len(set(A_tasks)) == BM * BK // 8 and all(0 <= m < BM and 0 <= k < BK for m, k in A_tasks)
    assert len(set(B_tasks)) == BN * BK // 8 and all(0 <= k < BK and 0 <= n < BN for k, n in B_tasks)
    return {"status": "PASS_PYTHON_LITERAL_HEADER_LAYOUT_EXPECTATION_ONLY",
            "A_atom_shape_elements": [8, 16], "B_atom_shape_elements": [16, 8],
            "A_target_shape": [BM, BK, PRODUCTS], "B_target_shape": [BN, BK, PRODUCTS],
            "A_repeat_shape": [8, 4, 7], "B_repeat_shape": [2, 8, 7],
            "A_elements": A_ELEMENTS, "B_elements": B_ELEMENTS,
            "A_vector_tasks_each_stage": len(A_tasks), "B_vector_tasks_each_stage": len(B_tasks),
            "bank_bytes": BANK_BYTES, "bank_origins": [0, BANK_BYTES], "B_origin_in_bank": 2 * A_ELEMENTS,
            "bank_and_B_origins_divisible_by512": BANK_BYTES % 512 == 2 * A_ELEMENTS % 512 == 0,
            "absolute_128B_base_phases": [0, 128, 256, 384],
            "contiguous_aligned_vector_checks": vector_checks, "physical_element_checks": element_checks,
            "physical_unique_coverage_and_bank_bounds": True,
            "pinned_plain_composed_pointer_swizzle_and_gemm_source_checks": True,
            "actual_CuTe_template_descriptors_compile": "PENDING"}


def tail_and_output_check(protocol):
    traits = (ROOT / "third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp").read_text()
    expected = "using CLayout_64x32 = Layout<Shape <Shape < _4,_8, _4>,Shape < _2,_2, _4>>, Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>>;"
    match = re.search(r"using CLayout_64x32\s*=.*?;", traits, re.S)
    assert match and re.sub(r"\s+", " ", match.group()).strip() == expected
    counts = bytearray(PRODUCTS * BM * BN)
    for group, products in enumerate(((0, 1, 2), (3, 4, 5, 6))):
        coords = set()
        for thread in range(128):
            for value in range(16):
                row, col = protocol.mma_coordinate(thread, value)
                assert 0 <= row < BM and 0 <= col < BN and (row, col) not in coords
                coords.add((row, col))
                for product in products:
                    counts[protocol.tail_word(product, row, col, True)] += 1
        assert len(coords) == BM * BN
    assert all(count == 1 for count in counts)
    assert TAIL_BYTES < BANK_BYTES and TAIL_BYTES <= OPERAND_BYTES
    shape_rows = [protocol.global_output_coverage(*shape) for shape in
                  ((128, 64, 64), (128, 64, 65), (256, 128, 137),
                   (1024, 1024, 1024), (1024, 1024, 1025), (1024, 1024, 1041))]
    reads = stores = beta0_stores = 0
    for seed in range(32):
        nread, nstore = protocol.alias_trace(seed, True)
        reads += nread
        stores += nstore
        nread, nstore = protocol.alias_trace(seed + 10000, False)
        assert nread == 0
        beta0_stores += nstore
    return {"P_unique_writers_and_U_read_words": len(counts), "tail_bytes": TAIL_BYTES,
            "P_tail_crosses_bank0_end": False, "bank1_overlap_bytes": 0,
            "tail_spans": [[product * BM * BN * 4, (product + 1) * BM * BN * 4]
                           for product in range(PRODUCTS)],
            "both_banks_two_consumers_and_two_producer_tail_acquires_still_drained": True,
            "three_uniform_CTA_barriers_unchanged": True,
            "output_coverage": shape_rows,
            "exact_alias_active_beta_traces": 32, "original_C0_reads_verified": reads,
            "active_beta_D_stores": stores, "beta0_null_C0_traces": 32,
            "beta0_C0_reads": 0, "beta0_D_stores": beta0_stores,
            "arbitrary_overlap_scope": "Not allowed or proven; integrating Plan preserves common guards."}


def resources():
    barriers = 2 * BANKS * 8
    payload = max(OPERAND_BYTES, TAIL_BYTES)
    predicted_shared = ((payload + barriers + 127) // 128) * 128
    assert (BANK_BYTES, OPERAND_BYTES, TAIL_BYTES, predicted_shared) == (86016, 172032, 57344, 172160)
    grid, stages = 128, 8
    return {"shape": [BM, BN, BK], "threads": 384, "producer_consumer_counts": [128, 128, 128],
            "full_arrival_count": 128, "empty_arrival_count": 256, "CTA_barriers": 3,
            "predicted_union_payload_bytes": payload, "predicted_barrier_bytes": barriers,
            "predicted_align128_SharedStorage_bytes": predicted_shared,
            "tail_additional_shared_bytes": 0, "actual_dynamic_static_shared": "PENDING",
            "source_quotas": [64, 96, 96], "hypothetical_role_pool_registers": 128 * (64 + 96 + 96),
            "minimum_8step_initial_uniform_quota_hypothesis": 88, "maximum_initial_for_inc96_direction": 96,
            "actual_initial_owned_pool_role_CFG_implicit_register_widths_spills": "PENDING",
            "per_stage_source_logical": {"A_input_bytes": 4 * BM * BK * 2,
                                         "B_input_bytes": 4 * BN * BK * 2,
                                         "final_operand_shared_store_bytes": BANK_BYTES,
                                         "transform_add_sub_count": 4 * (BM * BK + BN * BK),
                                         "WGMMA_K16_instructions_total": 7 * 4,
                                         "WGMMA_K16_instructions_by_group": [3 * 4, 4 * 4]},
            "whole1024_source_logical": {"grid_CTAs": grid, "BK64_stages": stages,
                                         "input_bytes": grid * stages * 4 * (BM * BK + BN * BK) * 2,
                                         "final_operand_shared_store_bytes": grid * stages * BANK_BYTES,
                                         "transform_add_sub_count": grid * stages * 4 * (BM * BK + BN * BK),
                                         "WGMMA_K16_instructions_total": grid * stages * 7 * 4,
                                         "P_tail_store_and_U_read_bytes": grid * TAIL_BYTES * 2},
            "traffic_scope": "Logical requested bytes/instructions, no measured transactions or occupancy/performance."}


def main():
    protocol = load_frozen_protocol()
    stage_counts = list(range(18)) + [31, 32, 33, 63, 64, 65, 511, 512]
    rows, witnesses = [], {}
    for stages in stage_counts:
        row, current = protocol.exhaustive(stages)
        rows.append(row)
        if stages == 3:
            witnesses = current
    random_counts = (0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 32, 33, 64, 512)
    actions = sum(protocol.randomized(stages, 100000 * stages + seed)
                  for stages in random_counts for seed in range(16))
    bindings = {path: digest(path) for path in
                (BASE, PROTO, CANDIDATE, ".scratch/psplit_bk64_model.py",
                 "third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp",
                 "third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp",
                 "third_party/cutlass/include/cute/layout.hpp",
                 "third_party/cutlass/include/cute/layout_composed.hpp",
                 "third_party/cutlass/include/cute/pointer_flagged.hpp",
                 "third_party/cutlass/include/cute/pointer_swizzle.hpp",
                 "third_party/cutlass/include/cute/swizzle.hpp",
                 "third_party/cutlass/include/cute/arch/mma_sm90.hpp",
                 "third_party/cutlass/include/cute/algorithm/gemm.hpp")}
    print(json.dumps({"status": "PASS_CPU_SOURCE_GEOMETRY_AND_ABSTRACT_PROTOCOL_ONLY",
                      "formal_policy_id": None, "CUDA_or_host_compilation": False,
                      "GPU_Graph_SAN_FP64_bitwise_performance_evidence": None,
                      "evidence_bindings": bindings,
                      "source_review": source_geometry_check(),
                      "layout_expectation": literal_layout_check(),
                      "synchronization": {"exhaustive": rows,
                                          "total_states": sum(row["states"] for row in rows),
                                          "total_transitions": sum(row["transitions"] for row in rows),
                                          "randomized_stage_counts": random_counts,
                                          "randomized_traces": 16 * len(random_counts),
                                          "randomized_actions": actions,
                                          "mutation_counterexamples": witnesses,
                                          "1024_corresponds_to_stage_count": 8,
                                          "model_scope": "Three collective128-thread roles; not physical per-thread PCs, compiler or hardware memory order."},
                      "tail_output_alias": tail_and_output_check(protocol),
                      "resources_prediction_only": resources(),
                      "remaining_blockers": ["Actual CuTe instantiation and canonical descriptors",
                                             "New compiled initial CTA-owned pool/role CFG/implicit R widths/spills/inflight P accesses",
                                             "GPU P/U bits, original FP64 gates, fresh bindings and Graph",
                                             "Serial targeted sanitizers and full complete-call timing versus cuBLAS"]}, indent=2))


if __name__ == "__main__":
    main()

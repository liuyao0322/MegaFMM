#!/usr/bin/env python3
"""CPU-only proposed SW64/SW32 layout ownership enumeration; no CUDA or numerics.

This evaluates an independent numeric model of the pinned CuTe flagged byte
swizzles. It does not instantiate CuTe and does not certify TMA hardware layout,
compiled register resources, residency, performance, or numerical accuracy.
"""
import json
import sys
from pathlib import Path


def byte_swizzle(plain_byte, bits):
    return plain_byte ^ (((plain_byte >> 7) & ((1 << bits) - 1)) << 4)


def verify(bn):
    bm, bk, stages, consumers = 64, 32, 2, 256
    a_plane, b_plane = bm * bk * 2, bn * bk * 2
    a_bytes, b_bytes = 7 * a_plane, 7 * b_plane
    bank_bytes = a_bytes + b_bytes
    b_bits = 2 if bn == 32 else 1
    writes, reads, full_elements = {}, {}, set()
    task_counts = {}
    checks = 0
    for stage in range(stages):
        bank_base = stage * bank_bytes
        for side, rows, cols, side_base, plane_bytes, bits in [
            ("A", bm, bk, bank_base, a_plane, 2),
            ("B", bk, bn, bank_base + a_bytes, b_plane, b_bits),
        ]:
            assert side_base % (128 << bits) == 0
            tasks = rows * cols // 2
            counts = [0] * consumers
            for task in range(tasks):
                owner = task % consumers
                counts[owner] += 1
                row, col = divmod(task * 2, cols)
                identity = (stage, side, task)
                loaded = []
                # All four raw words belong to this coordinate owner. A future
                # helper must actually execute all four loads before stores.
                for product in range(4):
                    plain = product * plane_bytes + 2 * (row * cols + col)
                    address = side_base + byte_swizzle(plain, bits)
                    assert address % 4 == 0
                    assert side_base + byte_swizzle(plain + 2, bits) == address + 2
                    assert address not in reads
                    reads[address] = identity
                    loaded.append(address)
                assert len(set(loaded)) == 4
                for product in range(7):
                    plain = product * plane_bytes + 2 * (row * cols + col)
                    address = side_base + byte_swizzle(plain, bits)
                    assert address not in writes
                    writes[address] = identity
                    full_elements.add(address)
                    full_elements.add(address + 2)
                    checks += 1
            assert min(counts) == max(counts)
            task_counts[f"{side}_stage{stage}"] = {
                "pair_tasks": tasks,
                "tasks_per_consumer": min(counts),
                "swizzle_bits": bits,
            }
            # TMA/final product planes share this model's identical coordinate
            # mapping, and all aligned vector8 subranges remain contiguous.
            for product in range(7):
                assert (side_base + product * plane_bytes) % (128 << bits) == 0
                for row in range(rows):
                    for col in range(0, cols, 8):
                        plain = product * plane_bytes + 2 * (row * cols + col)
                        base = side_base + byte_swizzle(plain, bits)
                        assert base % 16 == 0
                        for lane in range(8):
                            assert side_base + byte_swizzle(plain + 2 * lane, bits) == base + 2 * lane
                            checks += 1
    assert full_elements == set(range(0, stages * bank_bytes, 2))
    assert set(writes) == set(range(0, stages * bank_bytes, 4))
    assert all(writes[address] == identity for address, identity in reads.items())
    p_bytes = 7 * bm * bn * 4
    assert p_bytes <= stages * bank_bytes
    producer_q, c0_q, c1_q, initial = ((40, 88, 104, 80) if bn == 32 else (40, 64, 72, 64))
    owned = 384 * initial
    required = 128 * (producer_q + c0_q + c1_q)
    assert initial >= producer_q and initial <= c0_q and initial <= c1_q
    assert required <= owned
    payload = stages * bank_bytes
    predicted_shared = ((payload + 48 + 1023) // 1024) * 1024
    return {
        "geometry": [bm, bn, bk],
        "proposed_threads": 384,
        "proposed_role_quotas": [producer_q, c0_q, c1_q],
        "assumed_initial_registers": initial,
        "CTA_owned_registers_arithmetic": owned,
        "role_registers_arithmetic": required,
        "unused_CTA_pool_arithmetic": owned - required,
        "two_CTA_owned_registers_arithmetic": 2 * owned,
        "final_A_bytes_per_bank": a_bytes,
        "final_B_bytes_per_bank": b_bytes,
        "operand_bank_bytes": bank_bytes,
        "two_bank_payload_bytes": payload,
        "raw_transaction_bytes_per_bank": 4 * (a_plane + b_plane),
        "tail_P_bytes": p_bytes,
        "shared_size_align1024_plus48_arithmetic_only": predicted_shared,
        "grid_CTAs_at_square1024": (512 // bm) * (512 // bn),
        "pair_task_counts": task_counts,
        "unique_final_32bit_words": len(writes),
        "unique_raw_32bit_words": len(reads),
        "unique_final_16bit_elements": len(full_elements),
        "assertion_checks_including_all_vector8_lanes": checks,
        "all_first4_raw_aliases_owned_by_their_final_writer": True,
        "all_final_elements_dense_without_padding_or_cross_bank_overlap": True,
        "all_pair2_words_aligned_and_contiguous": True,
        "all_aligned_vector8_ranges_aligned_and_contiguous": True,
        "CuTe_instantiation_checked": False,
        "TMA_hardware_layout_checked": False,
        "compiled_role_GPR_checked": False,
        "actual_initial_pool_checked": False,
        "actual_shared_size_checked": False,
        "measured_residency": None,
        "performance_measured": False,
        "numerical_model_run": False,
        "original_precision_certified": False,
    }


if __name__ == "__main__":
    output = Path(sys.argv[1])
    assert output.name.startswith("wide_tile_feasibility_future46_")
    result = {
        "scope": "Independent CPU address/ownership and resource arithmetic only",
        "primary": verify(32),
        "BN16_sibling": verify(16),
        "future_state": "frozen_cpu_only_no_formal_implementation_or_registration",
    }
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "primary_geometry": result["primary"]["geometry"],
                      "primary_final_words": result["primary"]["unique_final_32bit_words"],
                      "primary_raw_words": result["primary"]["unique_raw_32bit_words"],
                      "sibling_geometry": result["BN16_sibling"]["geometry"],
                      "output": str(output)}, indent=2))

#!/usr/bin/env python3
"""CPU-only hybrid-S DAG, final-onecast, and logical-slot accounting model.

No candidate CUDA is produced or executed. No frozen parent module is imported.
Default stdout contains the complete CPU model result and no files are written.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path('/share/home/liuyao/workspace/MegaFMM')
MANIFEST = ROOT / '.scratch/psplit_rsA_future35_manifest.json'
BK32_MANIFEST = ROOT / '.scratch/psplit_rsA_bk32_future36_manifest.json'
MANIFEST_SHA = '78b454ce6247a2da0f08dadb48c2bf1a0a6b378ed01fe66673fa5cbfb4a46108'
BK32_MANIFEST_SHA = '08dec83cafe9920c0b3823cd4343a0cce1972ad9a000904855d116adfa94456d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protect_writes(event, args):
    if event != 'open' or not args:
        return
    mode = args[1] or ''
    flags = args[2] if len(args) > 2 else 0
    writes = isinstance(mode, str) and any(letter in mode for letter in 'wax+')
    writes |= isinstance(flags, int) and bool(flags & (1 | 2 | 64 | 512 | 1024))
    if writes:
        raise RuntimeError('CPU model rejects all file writes: ' + str(args[0]))


def decode(bits, dtype):
    bits = np.asarray(bits, dtype=np.uint16)
    return ((bits.astype(np.uint32) << 16).view(np.float32) if dtype == 'BF16'
            else bits.view(np.float16).astype(np.float32))


def final_cast(values, dtype):
    values = np.asarray(values, dtype=np.float32)
    assert np.isfinite(values).all()
    if dtype == 'FP16':
        return values.astype(np.float16).view(np.uint16)
    words = values.view(np.uint32)
    return ((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype(np.uint16)


def original_ss(raw, dtype):
    x11, x12, x21, x22 = [decode(value, dtype) for value in raw]
    s1 = np.add(x21, x22, dtype=np.float32)
    s2 = np.subtract(s1, x11, dtype=np.float32)
    s3 = np.subtract(x11, x21, dtype=np.float32)
    s4 = np.subtract(x12, s2, dtype=np.float32)
    return [raw[0], raw[1], final_cast(s4, dtype), raw[3],
            final_cast(s1, dtype), final_cast(s2, dtype), final_cast(s3, dtype)]


def original_raw_rs(raw, dtype):
    x11, x12, x21, x22 = [decode(value, dtype) for value in raw]
    wg0_s1 = np.add(x21, x22, dtype=np.float32)
    wg0_s2 = np.subtract(wg0_s1, x11, dtype=np.float32)
    wg0_s4 = np.subtract(x12, wg0_s2, dtype=np.float32)
    wg1_s1 = np.add(x21, x22, dtype=np.float32)
    wg1_s2 = np.subtract(wg1_s1, x11, dtype=np.float32)
    wg1_s3 = np.subtract(x11, x21, dtype=np.float32)
    return [raw[0], raw[1], final_cast(wg0_s4, dtype), raw[3],
            final_cast(wg1_s1, dtype), final_cast(wg1_s2, dtype), final_cast(wg1_s3, dtype)]


def hybrid_rs(raw, dtype):
    # Producer builds S2 from unrounded FP32 s1. The S1 storage cast is P5 only.
    x11, x12, x21, x22 = [decode(value, dtype) for value in raw]
    s1_fp32 = np.add(x21, x22, dtype=np.float32)
    slots = {'A11': raw[0], 'A12': raw[1], 'A21': raw[2], 'A22': raw[3],
             'S1': final_cast(s1_fp32, dtype),
             'S2': np.subtract(s1_fp32, x11, dtype=np.float32)}
    # WG0 reads A11/A12/S2; WG1 reads A22/S1/S2/A11/A21.
    s4 = np.subtract(decode(slots['A12'], dtype), slots['S2'], dtype=np.float32)
    s3 = np.subtract(decode(slots['A11'], dtype), decode(slots['A21'], dtype), dtype=np.float32)
    return [slots['A11'], slots['A12'], final_cast(s4, dtype), slots['A22'],
            slots['S1'], final_cast(slots['S2'], dtype), final_cast(s3, dtype)], slots


def numerical_model(dtype):
    all_bits = np.arange(65536, dtype=np.uint16)
    all_values = decode(all_bits, dtype)
    finite = all_bits[np.isfinite(all_values) & (np.abs(all_values) <= 16)]
    count = len(finite)
    indices = np.arange(count, dtype=np.uint32)
    quads = [finite[indices], finite[(indices * 17 + 31) % count],
             finite[(indices * 37 + 7) % count], finite[(indices * 53 + 101) % count]]
    assert all(np.isfinite(decode(q, dtype)).all() and (np.abs(decode(q, dtype)) <= 16).all()
               for q in quads)
    expected = original_ss(quads, dtype)
    raw_rs = original_raw_rs(quads, dtype)
    hybrid, slots = hybrid_rs(quads, dtype)
    assert slots['S2'].dtype == np.float32 and slots['S1'].dtype == np.uint16
    raw_mismatch = sum(np.count_nonzero(a != b) for a, b in zip(expected, raw_rs))
    hybrid_mismatch = sum(np.count_nonzero(a != b) for a, b in zip(expected, hybrid))
    assert raw_mismatch == hybrid_mismatch == 0
    midpoint = np.array([1, 0, 1, 2 ** (-8 if dtype == 'BF16' else -11)], dtype=np.float32)
    midpoint_raw = [final_cast(np.array([value], dtype=np.float32), dtype) for value in midpoint]
    assert all(float(decode(v, dtype)[0]) == float(w) for v, w in zip(midpoint_raw, midpoint))
    correct, correct_slots = hybrid_rs(midpoint_raw, dtype)
    wrong_s2 = np.subtract(decode(correct_slots['S1'], dtype), decode(midpoint_raw[0], dtype), dtype=np.float32)
    wrong_s4 = np.subtract(decode(midpoint_raw[1], dtype), wrong_s2, dtype=np.float32)
    wrong_p3 = final_cast(wrong_s4, dtype)
    wrong_p6 = final_cast(wrong_s2, dtype)
    assert float(decode(correct_slots['S1'], dtype)[0]) == 1
    assert float(correct_slots['S2'][0]) == float(midpoint[3]) and float(wrong_s2[0]) == 0
    assert correct[2][0] != wrong_p3[0] and correct[5][0] != wrong_p6[0]
    assert all(np.array_equal(a, b) for a, b in zip(correct, original_ss(midpoint_raw, dtype)))
    return {'dtype': dtype, 'finite_input_abs_limit': 16,
            'deterministic_quadruples': count, 'seven_A_final_bit_comparisons': 7 * count,
            'hybrid_vs_original_SS_mismatches': int(hybrid_mismatch),
            'raw_RS_vs_original_SS_mismatches': int(raw_mismatch),
            'FP32_S2_storage_verified': True,
            'rounded_S1_negative_control': {'raw': dict(zip(('A11', 'A12', 'A21', 'A22'), map(float, midpoint))),
                 'S1_cast_bits': int(correct_slots['S1'][0]),
                 'correct_S2_FP32': float(correct_slots['S2'][0]), 'wrong_S2_FP32': float(wrong_s2[0]),
                 'correct_P3_A_bits': int(correct[2][0]), 'wrong_P3_A_bits': int(wrong_p3[0]),
                 'correct_P6_A_bits': int(correct[5][0]), 'wrong_P6_A_bits': int(wrong_p6[0]),
                 'wrong_chain_rejected': True},
            'scope': 'NumPy FP32 RN and dtype RNE input-transform bit-model only; finite bounded deterministic quadruples, not all quadruples. No CUDA converter/FTZ/NaN payload/FP64 full-GEMM gate proof.'}


def logical_counts(bk):
    bm, bn, stages, threads = 64, 32, 512 // bk, 384
    a, b = bm * bk, bn * bk
    tail_bytes = 7 * bm * bn * 4
    records = {}
    for name, payload, reads, producer_ops, wg0_ops, wg1_ops, producer_casts, wg0_casts, wg1_casts in (
        ('SS', 7, [3, 4], 4, 0, 0, 4, 0, 0),
        ('raw_RS', 4, [4, 3], 0, 3, 3, 0, 1, 3),
        ('hybrid_RS', 7, [4, 6], 2, 1, 1, 1, 1, 2)):
        bank = (payload * a + 7 * b) * 2
        ops = producer_ops + wg0_ops + wg1_ops
        casts = producer_casts + wg0_casts + wg1_casts
        assert casts == 4
        records[name] = {
            'A_storage_BF16_equivalents': payload,
            'A_producer_original_input_read_bytes': 4 * a * 2,
            'A_producer_slot_write_bytes': payload * a * 2,
            'A_consumer_logical_distinct_slot_BF16_equivalents_by_WG': reads,
            'A_consumer_logical_distinct_slot_bytes_by_WG': [v * a * 2 for v in reads],
            'A_consumer_logical_distinct_slot_bytes_total': sum(reads) * a * 2,
            'A_FP32_RN_addsub_per_cell_producer_WG0_WG1': [producer_ops, wg0_ops, wg1_ops],
            'A_FP32_RN_addsub_per_stage': ops * a,
            'A_final_casts_per_cell_producer_WG0_WG1': [producer_casts, wg0_casts, wg1_casts],
            'A_final_casts_per_stage': casts * a,
            'B_original_read_bytes': 4 * b * 2, 'B_final_slot_write_bytes': 7 * b * 2,
            'B_FP32_RN_addsub_per_stage': 4 * b, 'B_final_casts_per_stage': 4 * b,
            'nominal_operand_bank_payload_bytes': bank,
            'nominal_two_bank_operand_payload_bytes': 2 * bank,
            'nominal_two_bank_plus_32B_barriers_align128_bytes': ((2 * bank + 32 + 127) // 128) * 128,
            'tail_payload_bytes': tail_bytes, 'tail_exceeds_single_bank_bytes': max(0, tail_bytes - bank),
            'whole_CTA_A_FP32_RN_addsub': ops * a * stages,
            'whole_CTA_A_producer_slot_write_bytes': payload * a * 2 * stages,
            'whole_CTA_A_consumer_logical_distinct_slot_bytes_total': sum(reads) * a * 2 * stages,
            'whole_CTA_A_B_producer_slot_write_bytes': bank * stages,
            'per_CTA_context': {'BM': bm, 'BN': bn, 'BK': bk, 'Khalf': 512,
                                'stages': stages, 'threads': threads},
            'scope': 'Analytic compact-payload and per-WG distinct-slot consumption accounting, not physical unique LD requests, transactions, traffic or compiled shared storage.'}
    assert records['hybrid_RS']['A_FP32_RN_addsub_per_stage'] == records['SS']['A_FP32_RN_addsub_per_stage']
    assert records['hybrid_RS']['nominal_operand_bank_payload_bytes'] == records['SS']['nominal_operand_bank_payload_bytes']
    assert records['hybrid_RS']['A_consumer_logical_distinct_slot_bytes_total'] * 7 == records['raw_RS']['A_consumer_logical_distinct_slot_bytes_total'] * 10
    return {'BK': bk, 'CTA_grid_1024': [8, 16], 'CTA_count_1024': 128, 'routes': records}


def symbolic_contract():
    slots = {'A11': 'input16', 'A12': 'input16', 'A21': 'input16', 'A22': 'input16',
             'S1': 'input16 finalcast(FP32_RN(A21+A22)) for P5 only',
             'S2': 'FP32_RN(FP32_RN(A21+A22)-A11), no intermediate input16 cast'}
    reads = {'WG0': {'A11': 1, 'A12': 1, 'S2_FP32': 2},
             'WG1': {'A22': 1, 'S1_input16': 1, 'S2_FP32': 2, 'A11': 1, 'A21': 1}}
    assert sum(reads['WG0'].values()) == 4 and sum(reads['WG1'].values()) == 6
    return {'A_slots': slots, 'distinct_storage_slots': 6, 'storage_input16_equivalents': 7,
            'P_A_operands': ['raw A11', 'raw A12', 'finalcast(FP32_RN(A12-S2))', 'raw A22',
                             'S1 input16', 'finalcast(S2)', 'finalcast(FP32_RN(A11-A21))'],
            'producer_RN_addsub_per_cell': 2, 'WG0_RN_addsub_per_cell': 1, 'WG1_RN_addsub_per_cell': 1,
            'final_casts_per_cell': 4, 'unrounded_s1_FP32_required_for_S2': True,
            'S1_input16_users': ['P5'], 'S2_FP32_users': ['P3', 'P6'],
            'WG_logical_distinct_slot_input16_equivalents': reads,
            'retained_contract': 'Original ordered FP32 RN S/T DAG, four derived A finalcasts, B transforms, whole-K seven FP32 P, fixed U and alphaRN/betaFMA epilogue are required. This model checks only A input DAG; no kernel is instantiated.',
            'lifetime_scope': 'A11/A12 or A11/A21/A22 may be reused inside a WG. All whole-stage RS fragments still need safe live ranges until their own WGMMA completion; physical lifetime, ownership and fit require fresh source/compiled gate.'}


def main():
    sys.addaudithook(protect_writes)
    assert sha(MANIFEST) == MANIFEST_SHA and sha(BK32_MANIFEST) == BK32_MANIFEST_SHA
    manifest = json.loads(MANIFEST.read_text())
    protected = {**manifest['files'], str(MANIFEST.relative_to(ROOT)): sha(MANIFEST)}
    assert len(protected) == 9
    before = {path: sha(ROOT / path) for path in protected}
    assert before == protected
    bk32_manifest = json.loads(BK32_MANIFEST.read_text())
    protected_bk32 = {**bk32_manifest['files'], str(BK32_MANIFEST.relative_to(ROOT)): sha(BK32_MANIFEST)}
    assert len(bk32_manifest['files']) == 13 and len(protected_bk32) == 14
    before_bk32 = {path: sha(ROOT / path) for path in protected_bk32}
    assert before_bk32 == protected_bk32
    symbolic = symbolic_contract()
    numerical = [numerical_model(dtype) for dtype in ('BF16', 'FP16')]
    counts = [logical_counts(bk) for bk in (32, 64)]
    for route in ('SS', 'raw_RS', 'hybrid_RS'):
        for field in ('whole_CTA_A_FP32_RN_addsub', 'whole_CTA_A_producer_slot_write_bytes',
                      'whole_CTA_A_consumer_logical_distinct_slot_bytes_total', 'whole_CTA_A_B_producer_slot_write_bytes'):
            assert counts[0]['routes'][route][field] == counts[1]['routes'][route][field]
    after = {path: sha(ROOT / path) for path in protected}
    assert after == before
    after_bk32 = {path: sha(ROOT / path) for path in protected_bk32}
    assert after_bk32 == before_bk32
    result = {'schema': 'megafmm.future37.hybrid-S.cpu-model.v1',
              'status': 'PASS_CPU_ONLY_HYBRID_S_DAG_AND_LOGICAL_SLOT_MODEL',
              'model_sha256': sha(Path(__file__)), 'original_RS_nine_files_sha256': protected,
              'BK32_future13_bindings_and_manifest_sha256': protected_bk32,
              'old_model_imported_or_old_main_called': False, 'old_nine_files_verified_unchanged': True,
              'BK32_future13_bindings_and_manifest_verified_unchanged': True,
              'write_open_policy': 'All file writes rejected during model execution; stdout only.',
              'symbolic_contract': symbolic, 'numerical_bit_model': numerical, 'analytic_counts_BK32_BK64': counts,
              'not_proved': ['Candidate CUDA layout/ABI or compilation', 'initial or role register ownership and fit',
                             'physical instruction requests, traffic, occupancy or stall', 'GPU synchronization or whole-stage physical lifetime',
                             'original FP64 full-GEMM numerical gates or changed precision policy', 'GPU benchmark, cuBLAS speedup or goal completion']}
    print(json.dumps(result, indent=2) + '\n', end='')


if __name__ == '__main__':
    main()

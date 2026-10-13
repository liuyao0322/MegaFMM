#!/usr/bin/env python3
"""Literal CPU geometry for hypothetical even/odd FP32 S2 planar storage.

No old model is imported, no C++/CUDA is instantiated, and no GPU is accessed.
All output is stdout. CuTe definitions are fingerprinted and checked as text.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path('/share/home/liuyao/workspace/MegaFMM')
SOURCE = 'solution/megakernel_wgmma_psplit_rsa_bk32.cu'
HEADERS = [
    'third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp',
    'third_party/cutlass/include/cute/layout.hpp',
    'third_party/cutlass/include/cute/layout_composed.hpp',
    'third_party/cutlass/include/cute/pointer_flagged.hpp',
    'third_party/cutlass/include/cute/pointer_swizzle.hpp',
    'third_party/cutlass/include/cute/swizzle.hpp',
]
FROZEN_HYBRID = {
    '.scratch/psplit_rsA_hybridS_future37_model.py': '5bc9d8a360177fdbc84a3fefb7f0e295f985ff17240a952cbb9ed9058a299dba',
    '.scratch/psplit_rsA_hybridS_future37_review.json': '9fa53047988c1a046a5ab0882b6fe5a7c5ca2f29380b444567bfc7541c506932',
    '.scratch/psplit_rsA_hybridS_future37_cpu.exit': '9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa',
    '.scratch/psplit_rsA_hybridS_future37_protocol.md': '2fe2174852c6f1b0e271ac58a0615baf67aa96467aa9581fa3cabbefa2a65ab6',
}


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def refuse_write(event, args):
    if event != 'open' or not args:
        return
    mode = args[1] or ''
    flags = args[2] if len(args) > 2 else 0
    writes = isinstance(mode, str) and any(char in mode for char in 'wax+')
    writes |= isinstance(flags, int) and bool(flags & (1 | 2 | 64 | 512 | 1024))
    if writes:
        raise RuntimeError('CPU geometry model rejects all file write opens: ' + str(args[0]))


def check_local_definitions():
    texts = {path: (ROOT / path).read_text() for path in [SOURCE, *HEADERS]}
    traits, layout, composed, flagged, pointer, swizzle = [texts[path] for path in HEADERS]
    required = {
        HEADERS[0]: [
            'using Layout_K_SW32_Atom_Bits   = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape<_8, _256>,Stride< _256,_1>>>;',
            'using Layout_K_SW32_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW32_Atom_Bits{}));',
            'using CLayout_64x16  = Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,  _2>>,',
            'Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>>;',
            'using ALayout_64x16 = CLayout_64x16;',
        ],
        HEADERS[1]: [
            'Defaults to GenColMajor',
            'return coalesce(blocked_product(padded_block, make_ordered_layout(product_shape, ord_shape)), product_shape);',
        ],
        HEADERS[2]: [
            'class Shape, class ModeOrder = GenColMajor>',
            'return composition(layout.layout_a(), layout.offset(), tile_to_shape(layout.layout_b(), trg_shape, ord_shape));',
        ],
        HEADERS[3]: [
            'return make_tensor(make_smem_ptr(ptr.get(), layout.layout_a()),',
            'return composition(layout.layout_a(), smem_ptr_flag_bits<B*N>{}, upcast<N>(layout.layout_b()));',
        ],
        HEADERS[4]: [
            'return reinterpret_cast<T*>(SwizzleFn::apply(reinterpret_cast<uintptr_t>(ptr)));',
            'return *apply_swizzle(this->get() + i);',
        ],
        HEADERS[5]: [
            'return offset ^ shiftr(offset & yyy_msk{}, msk_sft{});',
            'bit_msk{} << (num_base + max(0,num_shft))',
        ],
        SOURCE: [
            'const int row=task/(BK/8),k=(task%(BK/8))*8;',
            'for (int task = tid; task < T::BM*T::BK/8; task += T::ProducerThreads)',
            'thr_mma.partition_A(make_identity_tensor(make_shape(Int<64>{},Int<BK>{})))',
            'const auto c=coords(2*pair,0,atom);',
            'asm volatile("ld.shared.u32 %0, [%1];"',
            'GMMA::Layout_K_SW32_Atom<Element>{}',
        ],
    }
    for path, snippets in required.items():
        assert all(snippet in texts[path] for snippet in snippets), path
    return {path: [{'line': texts[path][:texts[path].index(snippet)].count('\n') + 1,
                    'snippet': snippet} for snippet in snippets] for path, snippets in required.items()}


def sw32_byte(address):
    # Flagged pointer swizzle is applied to actual byte address, unchanged by upcast.
    return address ^ ((address & 128) >> 3)


def tiled_float_unswizzled(row, col, cols, plane=0):
    # Float atom is Shape8x8/Stride8,1. Default tiling repeats row-block first.
    block = plane * 8 * (cols // 8) + (col // 8) * 8 + row // 8
    return 4 * (block * 64 + (row % 8) * 8 + col % 8)


def planar_address(base, row, k, bk):
    return sw32_byte(base + tiled_float_unswizzled(row, k // 2, bk // 2, k & 1))


def flat_address(base, row, k, bk):
    return sw32_byte(base + tiled_float_unswizzled(row, k, bk))


def original_packed_address(base, row, k):
    assert k % 2 == 0
    # Existing half/BF16 atom8x16, with k-even pair packed into one u32.
    unswizzled = ((k // 16) * 8 + row // 8) * 256 + (row % 8) * 32 + (k % 16) * 2
    return sw32_byte(base + unswizzled)


def rs_coordinate(thread, value, atom):
    # ALayout_64x16=CLayout_64x16: literal nested-shape/stride scalarization.
    offset = (128 * (thread % 4) + (thread // 4 % 8) + 16 * (thread // 32 % 4)
              + 64 * (value % 2) + 8 * (value // 2 % 2) + 512 * (value // 4 % 2))
    return offset % 64, offset // 64 + 16 * atom


def bank_histogram(addresses):
    # Duplicated same-word requests would be counted once; no transaction model.
    distinct_words = {address // 4 for address in addresses}
    counts = Counter(word % 32 for word in distinct_words)
    return max(counts.values()), len(distinct_words), len(counts)


def geometry(bk):
    assert bk in (32, 64)
    bm, bn = 64, 32
    full_bytes = bm * bk * 4
    plane_bytes = full_bytes // 2
    # Same nominal Hybrid bank payload as frozen analytic model, not a CUDA layout.
    nominal_bank = (7 * bm * bk + 7 * bn * bk) * 2
    nominal_s2_offset = 5 * bm * bk * 2
    origins = sorted({0, 128, 256, 384, nominal_s2_offset, nominal_bank + nominal_s2_offset})
    logical_coordinates = {(row, k) for row in range(bm) for k in range(bk)}
    rs_all = [rs_coordinate(thread, value, atom)
              for thread in range(128) for atom in range(bk // 16) for value in range(8)]
    assert len(rs_all) == len(set(rs_all)) == bm * bk and set(rs_all) == logical_coordinates
    producer_vectors = stores = consumer_pairs = scalar_loads = map_equalities = 0
    flat_vector4_checks = 0
    hist = {name: Counter() for name in ('original_packed_u32', 'flat_even_u32', 'flat_odd_u32',
                                        'planar_even_u32', 'planar_odd_u32')}
    word_counts = {name: Counter() for name in hist}
    used_banks = {name: Counter() for name in hist}
    examples = {}
    for base in origins:
        assert base % 128 == 0
        expected_words = set(range(base, base + full_bytes, 4))
        flat = {flat_address(base, row, k, bk) for row, k in logical_coordinates}
        planar = {planar_address(base, row, k, bk) for row, k in logical_coordinates}
        assert flat == planar == expected_words
        assert len(flat) == len(planar) == len(logical_coordinates)
        plane_sets = [{planar_address(base, row, 2 * j + parity, bk)
                       for row in range(bm) for j in range(bk // 2)} for parity in (0, 1)]
        assert plane_sets[0] == set(range(base, base + plane_bytes, 4))
        assert plane_sets[1] == set(range(base + plane_bytes, base + full_bytes, 4))
        assert not plane_sets[0].intersection(plane_sets[1])
        written, task_coordinates = {}, set()
        for tid in range(128):
            for task in range(tid, bm * bk // 8, 128):
                row, start = task // (bk // 8), (task % (bk // 8)) * 8
                assert (row, start) not in task_coordinates
                task_coordinates.add((row, start))
                producer_vectors += 1
                for parity in (0, 1):
                    ks = [start + parity + 2 * lane for lane in range(4)]
                    addresses = [planar_address(base, row, k, bk) for k in ks]
                    assert addresses[0] % 16 == 0
                    assert addresses == list(range(addresses[0], addresses[0] + 16, 4))
                    assert base <= addresses[0] and addresses[-1] + 4 <= base + full_bytes
                    for k, address in zip(ks, addresses):
                        assert address not in written
                        written[address] = (row, k)
                    stores += 1
                # Flat comparator also permits two aligned v4 stores, but split
                # into contiguous halves, rather than planar even/odd packing.
                for half in (0, 4):
                    addresses = [flat_address(base, row, start + half + lane, bk) for lane in range(4)]
                    assert addresses[0] % 16 == 0 and addresses == list(range(addresses[0], addresses[0] + 16, 4))
                    flat_vector4_checks += 1
        assert len(task_coordinates) == bm * bk // 8 and set(written) == expected_words
        consumed = set()
        for thread in range(128):
            for atom in range(bk // 16):
                for pair in range(4):
                    row, k = rs_coordinate(thread, 2 * pair, atom)
                    row_odd, k_odd = rs_coordinate(thread, 2 * pair + 1, atom)
                    assert row_odd == row and k % 2 == 0 and k_odd == k + 1
                    assert planar_address(base, row, k, bk) == original_packed_address(base, row, k)
                    assert planar_address(base, row, k + 1, bk) == original_packed_address(base, row, k) + plane_bytes
                    map_equalities += 2
                    for kk in (k, k + 1):
                        address = planar_address(base, row, kk, bk)
                        assert address % 4 == 0 and base <= address and address + 4 <= base + full_bytes
                        assert written[address] == (row, kk) and (row, kk) not in consumed
                        consumed.add((row, kk))
                        scalar_loads += 1
                    consumer_pairs += 1
        assert consumed == logical_coordinates
        for warp in range(4):
            for atom in range(bk // 16):
                for pair in range(4):
                    addresses = {name: [] for name in hist}
                    for lane in range(32):
                        row, k = rs_coordinate(warp * 32 + lane, pair * 2, atom)
                        addresses['original_packed_u32'].append(original_packed_address(base, row, k))
                        addresses['flat_even_u32'].append(flat_address(base, row, k, bk))
                        addresses['flat_odd_u32'].append(flat_address(base, row, k + 1, bk))
                        addresses['planar_even_u32'].append(planar_address(base, row, k, bk))
                        addresses['planar_odd_u32'].append(planar_address(base, row, k + 1, bk))
                    for name, values in addresses.items():
                        worst, distinct, banks = bank_histogram(values)
                        hist[name][worst] += 1
                        word_counts[name][distinct] += 1
                        used_banks[name][banks] += 1
                        if base == 0 and warp == atom == pair == 0:
                            examples[name] = {'addresses_bytes_lane0_to31': values,
                                              'banks_lane0_to31': [(value // 4) % 32 for value in values]}
    batches = len(origins) * 4 * (bk // 16) * 4
    assert hist['original_packed_u32'] == hist['planar_even_u32'] == hist['planar_odd_u32'] == Counter({1: batches})
    assert hist['flat_even_u32'] == hist['flat_odd_u32'] == Counter({2: batches})
    assert all(count == Counter({32: batches}) for count in word_counts.values())
    assert all(used_banks[name] == Counter({32: batches}) for name in ('original_packed_u32', 'planar_even_u32', 'planar_odd_u32'))
    assert used_banks['flat_even_u32'] == used_banks['flat_odd_u32'] == Counter({16: batches})
    return {'BK': bk, 'row_count': bm, 'plane_shape': [bm, bk // 2], 'plane_count': 2,
            'FP32_payload_bytes': full_bytes, 'plane_payload_bytes': plane_bytes,
            'origins_bytes_tested': origins, 'base_alignment_bytes': 128,
            'per_origin_address_bijection_words': bm * bk,
            'producer_vector8_tasks_checked': producer_vectors,
            'producer_split_even_odd_v4_16B_stores_checked': stores,
            'flat_two_contiguous_v4_16B_stores_checked': flat_vector4_checks,
            'consumer_original_RS_pair_coordinates_checked': consumer_pairs,
            'consumer_S2_scalar_u32_reads_checked': scalar_loads,
            'planar_vs_original_packed_mapping_equalities': map_equalities,
            'scalar_comparison_warp_batches_per_route': batches,
            'per_warp_max_distinct_words_per_bank_histograms': {name: dict(values) for name, values in hist.items()},
            'per_warp_distinct_word_histograms': {name: dict(values) for name, values in word_counts.items()},
            'per_warp_used_bank_histograms': {name: dict(values) for name, values in used_banks.items()},
            'example_first_warp_atom_pair_base0': examples,
            'producer_mapping_scope': 'Existing source task=tid+j*128, row=task/(BK/8), k=(task%(BK/8))*8. Hypothetical planar S2 splits four even and four odd FP32 values into two aligned16B stores; actual CUDA shuffle/register packing is unimplemented.',
            'bank_scope': 'Literal scalar-u32 coordinate comparator: bank=(byte_address//4)%32; count distinct words per bank within one assumed warp scalar instruction. No physical transaction, compiler grouping, conflict-cycle, WGMMA bank, throughput or performance proof.'}


def main():
    sys.addaudithook(refuse_write)
    protected = {path: sha(path) for path in [SOURCE, *HEADERS]}
    assert {path: sha(path) for path in FROZEN_HYBRID} == FROZEN_HYBRID
    snippets = check_local_definitions()
    results = [geometry(bk) for bk in (32, 64)]
    assert {path: sha(path) for path in protected} == protected
    assert {path: sha(path) for path in FROZEN_HYBRID} == FROZEN_HYBRID
    report = {
        'schema': 'megafmm.future37.hybrid-S-planar.cpu-geometry.v1',
        'status': 'PASS_CPU_ONLY_LITERAL_PLANAR_GEOMETRY',
        'model_sha256': sha(str(Path(__file__).relative_to(ROOT)) if Path(__file__).is_absolute() else __file__),
        'source_layout_sha256': protected, 'source_layout_verified_unchanged': True,
        'frozen_HybridS_four_files_sha256': FROZEN_HYBRID, 'frozen_HybridS_four_files_verified_unchanged': True,
        'local_definition_assertions': snippets,
        'literal_derivation': {
            'SW32_pointer_byte_function': 'address XOR ((address AND 128) >> 3)',
            'FP32_SW32_atom': 'Shape(8,8), Stride(8,1); pointer SW32 retained by flagged upcast<32>',
            'tiled_FP32_unswizzled_byte_address': 'base +4*[64*(plane*8*(cols/8)+8*floor(col/8)+floor(row/8))+8*(row%8)+(col%8)]',
            'flat': 'cols=BK, plane=0, col=k', 'planar': 'cols=BK/2, plane=k%2, col=floor(k/2)',
            'plane0_equals_existing_16bit_packed_u32_address': True,
            'plane1_equals_plane0_plus_bytes': '128*BK',
            'tensor_scope': 'Literal interpretation of local CuTe atom, GenColMajor tiled product, actual pointer byte swizzle and original RS coordinates. No C++/CUDA instantiation or physical execution.'},
        'geometry_BK32_BK64': results, 'old_model_imported_or_old_main_called': False,
        'write_open_policy': 'All file write opens rejected during execution; stdout only.',
        'not_proved': ['Candidate CUDA instantiation/layout ABI', 'Producer even/odd shuffle or register packing implementation',
                       'compiled shared/register fit or occupancy', 'physical shared instruction grouping, transactions or conflict cycles',
                       'async synchronization and RS fragment lifetime', 'precision/FP64 full-GEMM gates', 'GPU performance or cuBLAS speedup']}
    print(json.dumps(report, indent=2) + '\n', end='')


if __name__ == '__main__':
    main()

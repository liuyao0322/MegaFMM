"""Uncompiled BN16 future source and independent literal CPU geometry.

This script neither imports CUDA nor invokes compilation, Git, GPU, benchmarks,
or older model top-level code. Every write is restricted to this scratch prefix.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys

ROOT = Path('/share/home/liuyao/workspace/MegaFMM')
PREFIX = ROOT / '.scratch/psplit_bn16_future38_'
BASE = ROOT / 'solution/megakernel_wgmma_psplit_rsa_bk32.cu'
OLD34 = ROOT / 'results/psplit_revision6_megakernel_wgmma_psplit_iter34.cu'
HEADERS = [
    'third_party/cutlass/include/cute/arch/mma_sm90.hpp',
    'third_party/cutlass/include/cute/arch/mma_sm90_gmma.hpp',
    'third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp',
    'third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp',
    'third_party/cutlass/include/cutlass/arch/barrier.h',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restrict_writes(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
        return
    mode, flags = args[1] or '', args[2] if len(args) > 2 else 0
    writes = isinstance(mode, str) and any(c in mode for c in 'wax+')
    writes |= isinstance(flags, int) and bool(flags & (1 | 2 | 64 | 512 | 1024))
    if writes and not str(Path(args[0]).absolute()).startswith(str(PREFIX)):
        raise RuntimeError('write outside own future38 prefix rejected: ' + str(args[0]))


sys.addaudithook(restrict_writes)
protected = [BASE, OLD34, *[ROOT / h for h in HEADERS]]
before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
source = BASE.read_text()
original = source
changes = []


def replace(old, new, count=1):
    global source
    actual = source.count(old)
    assert actual == count, (old, actual, count)
    changes.append({'old': old, 'new': new, 'count': count})
    source = source.replace(old, new)


replace(
    '// Formal iteration37 RS-A BK32 baseline; launch requires a fresh compiled gate.\n'
    '// BM64 BN32 BK32, raw A4/final B7 shared and whole-stage distinct register A.\n'
    '// Source quotas64/192/192; actual initial quota/owned pool must be revalidated.\n'
    '// Original onecast ST, whole seven FP32 P, fixedU/alphaRN/betaFMA retained.\n',
    '// CPU-only future38 BN16 source; uncompiled and not registered in any Plan.\n'
    '// BM64 BN16 BK32, raw A4/final B7 shared and whole-stage distinct register A.\n'
    '// Hypothesis q48/96/96 and minblocks2; actual ELF/host/SASS gate is absent.\n'
    '// Original onecast ST, whole seven FP32 P, fixedU/alphaRN/betaFMA retained.\n')
replace('psplit_rsa_bk32', 'psplit_bn16_future38', source.count('psplit_rsa_bk32'))
replace('PSplitRSABK32Traits', 'PSplitBN16Future38Traits', source.count('PSplitRSABK32Traits'))
replace('static_assert(Cols == 32,"this prototype uses BN32 only");',
        'static_assert(Cols == 16,"this uncompiled candidate uses typed BN16 only");\n'
        '  static_assert(std::is_same_v<Element,cutlass::half_t> ||\n'
        '      std::is_same_v<Element,cutlass::bfloat16_t>,"FP16/BF16 operands only");')
replace('// Identical atom/tiling construction to the existing M1. Each bank has its',
        '// Shared layout construction uses existing SW32 atoms. Each bank has its')
replace('  using Op = decltype(GMMA::rs_op_selector<Element,Element,float,Tile,\n'
        '      GMMA::Major::K,GMMA::Major::MN>());',
        '  using Op = std::conditional_t<std::is_same_v<Element,cutlass::half_t>,\n'
        '      SM90_64x16x16_F32F16F16_RS<GMMA::Major::K,GMMA::Major::MN>,\n'
        '      SM90_64x16x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>>;')
replace('// BK32/BN32 has128 B vector tasks;', '// BK32/BN16 has64 B vector tasks;')
replace('CUTE_DEVICE int psplit_bn16_future38_tail_word(int product,int row,int col) {\n'
        '  const int xor_mask = (row&1) | ((row&6)<<2);\n'
        '  return product*2048+row*32+(col^xor_mask);\n'
        '}',
        'template <int Rows,int Cols>\n'
        'CUTE_DEVICE int psplit_bn16_future38_tail_word(int product,int row,int col) {\n'
        '  static_assert(Cols >= 4 && (Cols&(Cols-1)) == 0,"power-of-two tail columns");\n'
        '  const int xor_mask = row & (Cols/4-1);\n'
        '  return product*Rows*Cols+row*Cols+(col^xor_mask);\n'
        '}')
replace('tail[psplit_bn16_future38_tail_word(',
        'tail[psplit_bn16_future38_tail_word<T::BM,T::BN>(', 11)
replace('__launch_bounds__(384,1)', '__launch_bounds__(384,2)')
replace('warpgroup_reg_dealloc<64>();', 'warpgroup_reg_dealloc<48>();')
replace('static_assert(size(q0) == 16,"each assigned full P has 16 floats/thread");',
        'static_assert(size(q0) == T::BM*T::BN/T::ConsumerThreads,\n'
        '        "each assigned full P has BM*BN/128 floats/thread (8 at BN16)");')
replace('warpgroup_reg_alloc<192>();', 'warpgroup_reg_alloc<96>();')
replace('// Complete all14336 FP32 P words', '// Complete all7168 FP32 P words')
replace('PSplitBN16Future38Traits<Element,32>', 'PSplitBN16Future38Traits<Element,16>', 2)
replace('winograd_wgmma_psplit_bn16_future38<Element,32>',
        'winograd_wgmma_psplit_bn16_future38<Element,16>', 2)
replace('static_assert(shared > 48*1024 && shared < 227*1024,"H100 two-bank shared budget");',
        'static_assert(shared > 0 && shared <= 48*1024,"BN16 two-bank source budget");')

# Source arithmetic, role count, and fulltile domain comparisons are exact text
# comparisons after normalizing only the renamed helpers and tail template args.
def normalize(text):
    return text.replace('psplit_bn16_future38', 'psplit_rsa_bk32').replace('<T::BM,T::BN>', '')


for begin, end in [
    ('// Producer writes four raw original A quadrants', '// Tail FP32 rendezvous'),
    ('    const float p1 =', '\ntemplate <class Element>\nvoid prepare_'),
]:
    a = original[original.index(begin):original.index(end, original.index(begin))]
    b = source[source.index(begin):source.index(end, source.index(begin))]
    # The sole producer comment records the changed number of B vector tasks.
    b = normalize(b).replace('// BK32/BN16 has64 B vector tasks;', '// BK32/BN32 has128 B vector tasks;')
    assert a == b, begin

restored = source
for change in reversed(changes):
    assert restored.count(change['new']) == change['count'], change
    restored = restored.replace(change['new'], change['old'])
assert restored == original
assert source.count('barrier.sync 6,384;') == 2
assert source.count('__syncthreads();') == 2
assert source.count('warpgroup_wait<0>();') == 2
assert source.count('fence.proxy.async.shared::cta;') == 1
assert 'params.producer_arv_count = T::ProducerThreads;' in source
assert 'params.consumer_arv_count = T::AllConsumerThreads;' in source
assert 'p.n%64 == 0' in source and 'p.n%32' not in source
assert 'p.m%128 == 0 && p.n%64 == 0 && p.k%64 == 0 && p.lda%8 == 0 && p.ldb%8 == 0' in source

traits = (ROOT / HEADERS[2]).read_text()
arch = (ROOT / HEADERS[1]).read_text()
assert 'using CLayout_64x16  = Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,  _2>>,' in traits
assert 'Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>>;' in traits
assert 'using ALayout_64x16 = CLayout_64x16;' in traits
for opname, ptx in [
    ('SM90_64x16x16_F32F16F16_RS', 'm64n16k16.f32.f16.f16'),
    ('SM90_64x16x16_F32BF16BF16_RS', 'm64n16k16.f32.bf16.bf16'),
]:
    atom = arch[arch.index('struct ' + opname):]
    atom = atom[:atom.index('\n};')]
    assert 'using ARegisters = uint32_t[4];' in atom
    assert 'using CRegisters = float[8];' in atom
    assert ptx in atom and 'tnspA == GMMA::Major::K' in atom

BM, BN, BK, THREADS, WG = 64, 16, 32, 384, 128
STAGES = 2
A_BYTES, B_BYTES = 4*BM*BK*2, 7*BN*BK*2
BANK = A_BYTES+B_BYTES
TAIL = 7*BM*BN*4
SHARED_PREDICTED = ((STAGES*BANK+32+127)//128)*128
assert (A_BYTES, B_BYTES, BANK, TAIL, SHARED_PREDICTED) == (16384, 7168, 23552, 28672, 47232)


def swizzle_byte(a):
    # Literal 16-bit upcast of pinned SW32 atom in byte addresses.
    return a ^ ((a & 128) >> 3)


def raw_a(row, k, slot):
    return (slot*8*(BK//16)+(k//16)*8+row//8)*128+(row%8)*16+k%16


def raw_b(col, k, slot):
    # BN16 has one N atom; K repeats four 8-row blocks per product slot.
    return (slot*(BN//16)*(BK//8)+(k//8)*(BN//16)+col//16)*128+col%16+(k%8)*16


def fragment_coord(tid, value, atom=0):
    linear = (128*(tid%4)+(tid//4%8)+16*(tid//32%4)
              +64*(value%2)+8*(value//2%2)+512*(value//4%2))
    return linear%64, linear//64+16*atom


def tail_word(product, row, col):
    return product*BM*BN+row*BN+(col ^ (row & (BN//4-1)))


assert {raw_a(r,k,q) for q in range(4) for r in range(BM) for k in range(BK)} == set(range(4*BM*BK))
assert {raw_b(c,k,q) for q in range(7) for c in range(BN) for k in range(BK)} == set(range(7*BN*BK))
vectors = pairs = 0
for base in (0,128,256,384):
    physical = set()
    for stage in range(STAGES):
        origin = base+stage*BANK
        for q in range(4):
            for row in range(BM):
                for k in range(0,BK,8):
                    addresses = [swizzle_byte(origin+2*raw_a(row,k+j,q)) for j in range(8)]
                    assert addresses[0]%16 == 0
                    assert addresses == list(range(addresses[0],addresses[0]+16,2))
                    assert not physical.intersection(addresses)
                    physical.update(addresses)
                    vectors += 1
        for q in range(7):
            for k in range(BK):
                for col in range(0,BN,8):
                    addresses = [swizzle_byte(origin+A_BYTES+2*raw_b(col+j,k,q)) for j in range(8)]
                    assert addresses[0]%16 == 0
                    assert addresses == list(range(addresses[0],addresses[0]+16,2))
                    assert not physical.intersection(addresses)
                    physical.update(addresses)
                    vectors += 1
        for tid in range(WG):
            for atom in range(BK//16):
                for pair in range(4):
                    row,k = fragment_coord(tid,2*pair,atom)
                    row1,k1 = fragment_coord(tid,2*pair+1,atom)
                    assert row1 == row and k1 == k+1
                    for q in range(4):
                        ad = swizzle_byte(origin+2*raw_a(row,k,q))
                        ad1 = swizzle_byte(origin+2*raw_a(row1,k1,q))
                        assert ad%4 == 0 and ad1 == ad+2
                        pairs += 1
    assert physical == set(range(base,base+STAGES*BANK,2))

a_coords = [fragment_coord(t,v,k) for t in range(WG) for k in range(BK//16) for v in range(8)]
assert len(a_coords) == len(set(a_coords)) == BM*BK
c_coords = [fragment_coord(t,v) for t in range(WG) for v in range(BM*BN//WG)]
assert len(c_coords) == len(set(c_coords)) == BM*BN
pwords = {tail_word(q,r,c) for q in range(7) for r,c in c_coords}
assert pwords == set(range(7*BM*BN))
owners = [(t,l//BN,l%BN) for t in range(THREADS) for l in range(t,BM*BN,THREADS)]
assert len(owners) == len({(r,c) for t,r,c in owners}) == BM*BN
reads = Counter(tail_word(q,r,c) for t,r,c in owners for q in range(7))
assert set(reads) == pwords and set(reads.values()) == {1}
owner_work = Counter(t for t,r,c in owners)
assert Counter(owner_work.values()) == Counter({3:256,2:128})
out = Counter((BM*x+r+512*qr)*1024+BN*y+c+512*qc
              for x in range(8) for y in range(32) for t,r,c in owners
              for qr in range(2) for qc in range(2))
assert set(out) == set(range(1024*1024)) and set(out.values()) == {1}
bad_old_tail = [q*2048+r*32+(c ^ ((r&1)|((r&6)<<2)))
                for q in range(7) for r,c in c_coords]
assert max(bad_old_tail) == 14335
assert max(bad_old_tail)*4 >= STAGES*BANK
assert any((c ^ ((r&1)|((r&6)<<2))) >= BN for r,c in c_coords)


def lifetime(products, stages):
    labels = {(p,k,w) for p in range(products) for k in range(BK//16) for w in range(4)}
    pending = set()
    issues = 0
    for stage in range(stages):
        assert not pending
        for label in labels:
            assert label not in pending
        for p in range(products):
            for k in range(BK//16):
                args = {(p,k,w) for w in range(4)}
                assert not args.intersection(pending)
                pending.update(args)
                issues += 1
        assert pending == labels
        pending.clear()  # conservative source wait0 boundary, not hardware evidence
        assert not pending.intersection(labels)
    bad = {(0,0,w) for w in range(4)}
    pending.update(bad)
    assert pending.intersection(bad)  # negative control: rewriting/fencing pending A
    return {'products': products, 'stage_count': stages,
            'stage_A_packed_words_per_thread': len(labels),
            'P_FP32_words_per_thread': products*BM*BN//WG,
            'payload_word_lower_bound': len(labels)+products*BM*BN//WG,
            'virtual_MMA_issues': issues,
            'physical_allocation_or_GPU_lifetime_proved': False}


initial = 80
pool = WG*(48+96+96)
assert pool == 30720 == THREADS*initial
assert WG*(initial-48) == 2*WG*(96-initial) == 4096
quota_initials = [i for i in range(48,97,8) if THREADS*i >= pool]
assert quota_initials == [80,88,96]
base_ctas, new_ctas, stages1024 = 128, 256, 16
a_read_stage, b_read_stage = 4*BM*BK*2, 4*BN*BK*2
new_traffic = {'A':new_ctas*stages1024*a_read_stage,
               'B':new_ctas*stages1024*b_read_stage}
old_traffic = {'A':base_ctas*stages1024*a_read_stage,
               'B':base_ctas*stages1024*(4*32*BK*2)}
assert new_traffic == {'A':64*1024**2, 'B':16*1024**2}
assert old_traffic == {'A':32*1024**2, 'B':16*1024**2}

candidate = Path(str(PREFIX)+'candidate.cu')
delta_path = Path(str(PREFIX)+'source_delta.json')
report_path = Path(str(PREFIX)+'review.json')
for path in [candidate,delta_path,report_path]:
    assert not path.exists(), 'refuse to overwrite frozen future artifact: '+str(path)
candidate.write_text(source)
delta = {'scope':'CPU_ONLY_UNCOMPILED_SOURCE_PREPARATION',
         'base_path':str(BASE), 'base_sha256':before[str(BASE.relative_to(ROOT))],
         'candidate_path':str(candidate), 'candidate_sha256':sha(candidate),
         'inverse_restores_base_byte_for_byte':True, 'replacements':changes}
delta_path.write_text(json.dumps(delta,indent=2)+'\n')
report = {
    'status':'PASS_CPU_LITERAL_GEOMETRY_ONLY_UNCOMPILED_UNREGISTERED',
    'CUTLASS_HEAD':(ROOT/'third_party/cutlass/.git/HEAD').read_text().strip(),
    'source_identity':delta,
    'evidence_sha256':before,
    'script_sha256':sha(Path(__file__)),
    'geometry_scope':'Independent literal interpretation of pinned CuTe source; no C++/CUDA template instantiation',
    'tile':{'BM':BM,'BN':BN,'BK':BK,'threads':THREADS,'warpgroup_threads':WG,'operand_banks':STAGES},
    'geometry':{'aligned_contiguous_vector8_checks':vectors,'aligned_RS_pair_checks':pairs,
                'A_unique_cells':len(set(a_coords)),'P_unique_words':len(pwords),
                'P_words_per_thread_per_product':8,'U_unique_owners':len(owners),
                'U_owner_task_count_histogram':dict(Counter(owner_work.values())),
                'each_P_word_read_count':1,'full1024_unique_output_count':len(out),
                'each_output_store_count':1,'old_tail_negative_control_max_word':max(bad_old_tail),
                'old_tail_wrong_stride_and_out_of_row_xor_rejected':True,
                'tail_layout':'product*BM*BN+row*BN+(col XOR (row AND (BN/4-1)))'},
    'predicted_shared_bytes':{'raw_A4':A_BYTES,'final_B7':B_BYTES,'one_bank':BANK,
                              'two_banks':STAGES*BANK,'P_tail':TAIL,'barriers':32,
                              'sizeof_align128':SHARED_PREDICTED,
                              'tail_bank0_overlap':BANK,'tail_bank1_overlap':TAIL-BANK,
                              'barrier_union_overlap':0,'actual_host_ABI_proved':False},
    'barrier_source':{'pipeline_full_arrivals':128,'pipeline_empty_arrivals':256,
                      'CTA_initial_bar0_participants':384,'CTA_tail_drain_bar6_participants':384,
                      'CTA_tail_publish_bar0_participants':384,'hardware_proved':False},
    'virtual_A_lifetimes':[lifetime(p,s) for p in (3,4) for s in (1,2,3,4,6,7,8,9,10,16,17,18,33,34)],
    'quota_hypothesis':{'producer':48,'WG0':96,'WG1':96,'required_source_pool_words':pool,
                        'uniform_initial_algebra_candidates':quota_initials,
                        'initial80_pool':THREADS*initial,'initial80_producer_release':4096,
                        'initial80_each_consumer_request':2048,'source_launch_bounds_minblocks':2,
                        'actual_ELF_SASS_host_resource_gate':None,'actual_two_CTA_residency':None},
    'logical_source_reads_1024_bytes':{'BN16':new_traffic,'BN32_parent':old_traffic,
                                      'actual_L2_DRAM_bytes':None},
    'grid_1024':{'BN16':[8,32],'BN16_CTAs':256,'BN32_parent':[8,16],'BN32_parent_CTAs':128},
    'native_host_domain':{'candidate_n_modulus':64,'preserves_old_metadata_domain':True,
                          'natural_BN16_fulltile_modulus_deferred':32,
                          'current_A_B_base_alignment_bytes':16,'lda_ldb_modulus_elements':8,
                          'original_AB32_boundaries_still_safe':True},
    'source_DAG_U_alphaRN_betaFMA_normalized_byte_equal_to_base':True,
    'source_change_is_single_controlled_axis':False,
    'changed_axes':['BN32 to BN16 atom/grid/layout/tail','q64/192/192 to q48/96/96','launch_bounds minblocks1 to2'],
    'reference_warning':'Variant36 has N32 atom and is not same-atom to BN16; later validation needs private SS N16 or explicitly independent host mathematics.',
    'not_proved':['C++/CUDA instantiation or actual layouts/sizeof','ptxas initial/role allocation, qfit, no spills',
                  'physical async A/P lifetimes and synchronization','two CTA physical resources/residency/occupancy',
                  'GPU precision/FP64 original numerical gate','GPU/Sanitizer/Graph forward progress',
                  'performance, actual L2/DRAM traffic, cuBLASLt speedup'],
    'writes_restricted_to_own_future38_prefix':True,
}
assert {str(p.relative_to(ROOT)):sha(p) for p in protected} == before
report['protected_base_old34_headers_unchanged'] = True
report_path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'candidate_sha256':sha(candidate),
                  'delta_sha256':sha(delta_path),'report_sha256':sha(report_path),
                  'geometry':report['geometry'],'predicted_shared_bytes':report['predicted_shared_bytes'],
                  'quota_hypothesis':report['quota_hypothesis'],
                  'logical_source_reads_1024_bytes':report['logical_source_reads_1024_bytes']},indent=2))

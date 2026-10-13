"""Independent uncompiled SS BN16 companion and literal CPU-only geometry.

Writes are restricted to this companion prefix. Frozen raw BN16 artifacts are
checked, not imported or executed. No compiler, GPU, Git, or benchmark is used.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys

ROOT = Path('/share/home/liuyao/workspace/MegaFMM')
PREFIX = ROOT / '.scratch/psplit_bn16_ss_future38_'
BASE = ROOT / 'solution/megakernel_wgmma_psplit.cu'
OLD34 = ROOT / 'results/psplit_revision6_megakernel_wgmma_psplit_iter34.cu'
RAW_MANIFEST = ROOT / '.scratch/psplit_bn16_future38_manifest.json'
BASE_SHA = 'fdaae75a80893ee161a9db202d4746e86daf3d38135332759098bd9f7faa7f91'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restrict_writes(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
        return
    mode, flags = args[1] or '', args[2] if len(args) > 2 else 0
    writes = isinstance(mode, str) and any(c in mode for c in 'wax+')
    writes |= isinstance(flags, int) and bool(flags & (1 | 2 | 64 | 512 | 1024))
    if writes and not str(Path(args[0]).absolute()).startswith(str(PREFIX)):
        raise RuntimeError('write outside SS companion prefix rejected: '+str(args[0]))


sys.addaudithook(restrict_writes)
assert sha(BASE) == sha(OLD34) == BASE_SHA
raw_manifest = json.loads(RAW_MANIFEST.read_text())
raw_protected = {p:sha(Path(p)) for p in raw_manifest['files']}
assert raw_protected == raw_manifest['files']
headers = [ROOT/'third_party/cutlass/include'/p for p in (
    'cute/arch/mma_sm90_gmma.hpp','cute/atom/mma_traits_sm90_gmma.hpp',
    'cutlass/pipeline/sm90_pipeline.hpp','cutlass/arch/barrier.h')]
protected = [BASE,OLD34,RAW_MANIFEST,*headers]
before = {str(p):sha(p) for p in protected}
original = source = BASE.read_text()
changes = []


def replace(old,new,count=1):
    global source
    assert source.count(old) == count, (old,source.count(old),count)
    changes.append({'old':old,'new':new,'count':count})
    source = source.replace(old,new)


replace('// Iteration34 P3+4 consumer split, consumer budget96 restored after scope repair.\n'
        '// Original prototype and intermediate128-budget evidence are archived separately.\n'
        '// One ordinary launch, zero global ST/P workspace, full seven P then fixed U.\n'
        '// Preserves source30 vector producer and source-level per-P ascending K order.\n',
        '// CPU-only future38 SS BN16 companion; uncompiled and not registered.\n'
        '// BM64 BN16 BK32, final shared A7/B7; wholeP3+4 and two banks retained.\n'
        '// Hypothesis q64/80/80 and minblocks2; no inherited compiler/resource gate.\n'
        '// Original ordered producer final onecast ST and fixedU/alphaRN/betaFMA.\n')
replace('psplit','psplit_bn16_ss_future38',source.count('psplit'))
replace('PSplit','PSplitBN16SSFuture38',source.count('PSplit'))
replace('static_assert(Cols == 32,"this prototype uses BN32 only");',
        'static_assert(Cols == 16,"this uncompiled SS companion uses typed BN16 only");\n'
        '  static_assert(std::is_same_v<Element,cutlass::half_t> ||\n'
        '      std::is_same_v<Element,cutlass::bfloat16_t>,"FP16/BF16 operands only");')
replace('// Identical atom/tiling construction to the existing M1. Each bank has its',
        '// Shared layout construction uses existing SW32 atoms. Each bank has its')
replace('  using Op = decltype(GMMA::ss_op_selector<Element,Element,float,Tile,\n'
        '      GMMA::Major::K,GMMA::Major::MN>());',
        '  using Op = std::conditional_t<std::is_same_v<Element,cutlass::half_t>,\n'
        '      SM90_64x16x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::MN>,\n'
        '      SM90_64x16x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>>;')
replace('CUTE_DEVICE int psplit_bn16_ss_future38_tail_word(int product,int row,int col) {\n'
        '  const int xor_mask = (row&1) | ((row&6)<<2);\n'
        '  return product*2048+row*32+(col^xor_mask);\n'
        '}',
        'template <int Rows,int Cols>\n'
        'CUTE_DEVICE int psplit_bn16_ss_future38_tail_word(int product,int row,int col) {\n'
        '  static_assert(Cols >= 4 && (Cols&(Cols-1)) == 0,"power-of-two tail columns");\n'
        '  const int xor_mask = row & (Cols/4-1);\n'
        '  return product*Rows*Cols+row*Cols+(col^xor_mask);\n'
        '}')
replace('tail[psplit_bn16_ss_future38_tail_word(',
        'tail[psplit_bn16_ss_future38_tail_word<T::BM,T::BN>(',11)
replace('__launch_bounds__(384,1)','__launch_bounds__(384,2)')
replace('static_assert(size(q0) == 16,"each assigned full P has 16 floats/thread");',
        'static_assert(size(q0) == T::BM*T::BN/T::ConsumerThreads,\n'
        '        "each assigned full P has BM*BN/128 floats/thread (8 at BN16)");')
replace('// Final pool need=128*(64+96+96)=32768; legal uniform initial quota88..96,',
        '// Source pool need=128*(64+80+80)=28672; hypothetical I80 owns30720,')
replace('warpgroup_reg_alloc<96>();','warpgroup_reg_alloc<80>();')
replace('// Complete all14336 FP32 P words','// Complete all7168 FP32 P words')
replace('PSplitBN16SSFuture38Traits<Element,32>','PSplitBN16SSFuture38Traits<Element,16>',2)
replace('winograd_wgmma_psplit_bn16_ss_future38<Element,32>',
        'winograd_wgmma_psplit_bn16_ss_future38<Element,16>',2)


def normalize(text):
    return text.replace('psplit_bn16_ss_future38','psplit').replace('PSplitBN16SSFuture38','PSplit').replace('<T::BM,T::BN>','')


for begin,end in [('// Each task owns eight K elements','// Tail FP32 rendezvous'),
                  ('    const float p1 =','\ntemplate <class Element>\nvoid prepare_')]:
    a = original[original.index(begin):original.index(end,original.index(begin))]
    b = source[source.index(begin):source.index(end,source.index(begin))]
    assert normalize(b) == a
restored = source
for edit in reversed(changes):
    assert restored.count(edit['new']) == edit['count'],edit
    restored = restored.replace(edit['new'],edit['old'])
assert restored == original
assert source.count('barrier.sync 6,384;') == 2 and source.count('__syncthreads();') == 2
assert source.count('warpgroup_wait<0>();') == 2
assert source.count('warpgroup_reg_dealloc<64>();') == 1
assert source.count('warpgroup_reg_alloc<80>();') == 1
assert 'a0_data' not in source and 'build_stage' not in source
assert source.count('auto tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA));') == 2
assert 'p.m%128 == 0 && p.n%64 == 0 && p.k%64 == 0 && p.lda%8 == 0 && p.ldb%8 == 0' in source

arch = headers[0].read_text()
traits = headers[1].read_text()
assert 'using CLayout_64x16  = Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,  _2>>,' in traits
for opname,ptx in [('SM90_64x16x16_F32F16F16_SS','m64n16k16.f32.f16.f16'),
                   ('SM90_64x16x16_F32BF16BF16_SS','m64n16k16.f32.bf16.bf16')]:
    atom = arch[arch.index('struct '+opname):]
    atom = atom[:atom.index('\n};')]
    assert 'using ARegisters = uint64_t[1];' in atom and 'using CRegisters = float[8];' in atom
    assert ptx in atom

BM,BN,BK,THREADS,WG,BANKS = 64,16,32,384,128,2
A_BYTES,B_BYTES = 7*BM*BK*2,7*BN*BK*2
BANK,TAIL = A_BYTES+B_BYTES,7*BM*BN*4
SHARED = ((BANKS*BANK+32+127)//128)*128
assert (A_BYTES,B_BYTES,BANK,TAIL,SHARED) == (28672,7168,35840,28672,71808)
assert 48*1024 < SHARED < 227*1024


def swizzle_byte(a):
    return a ^ ((a&128)>>3)


def a_index(r,k,q):
    return (q*8*(BK//16)+(k//16)*8+r//8)*128+(r%8)*16+k%16


def b_index(c,k,q):
    return (q*(BK//8)+(k//8))*128+c+(k%8)*16


def q_coord(t,v):
    x = 128*(t%4)+(t//4%8)+16*(t//32%4)+64*(v%2)+8*(v//2%2)+512*(v//4%2)
    return x%64,x//64


def tail_word(q,r,c):
    return q*BM*BN+r*BN+(c^(r&(BN//4-1)))


assert {a_index(r,k,q) for q in range(7) for r in range(BM) for k in range(BK)} == set(range(7*BM*BK))
assert {b_index(c,k,q) for q in range(7) for c in range(BN) for k in range(BK)} == set(range(7*BN*BK))
vectors = 0
for base in (0,128,256,384):
    physical = set()
    for bank in range(BANKS):
        origin = base+bank*BANK
        for q in range(7):
            for r in range(BM):
                for k in range(0,BK,8):
                    ad = [swizzle_byte(origin+2*a_index(r,k+j,q)) for j in range(8)]
                    assert ad[0]%16 == 0 and ad == list(range(ad[0],ad[0]+16,2))
                    assert not physical.intersection(ad)
                    physical.update(ad)
                    vectors += 1
            for k in range(BK):
                for c in range(0,BN,8):
                    ad = [swizzle_byte(origin+A_BYTES+2*b_index(c+j,k,q)) for j in range(8)]
                    assert ad[0]%16 == 0 and ad == list(range(ad[0],ad[0]+16,2))
                    assert not physical.intersection(ad)
                    physical.update(ad)
                    vectors += 1
    assert physical == set(range(base,base+BANKS*BANK,2))
q_coords = [q_coord(t,v) for t in range(WG) for v in range(8)]
assert len(q_coords) == len(set(q_coords)) == BM*BN
pwords = {tail_word(q,r,c) for q in range(7) for r,c in q_coords}
assert pwords == set(range(7*BM*BN))
owners = [(t,l//BN,l%BN) for t in range(THREADS) for l in range(t,BM*BN,THREADS)]
assert len(owners) == len({(r,c) for t,r,c in owners}) == BM*BN
reads = Counter(tail_word(q,r,c) for t,r,c in owners for q in range(7))
assert set(reads) == pwords and set(reads.values()) == {1}
outputs = Counter((BM*x+r+512*qr)*1024+BN*y+c+512*qc
                  for x in range(8) for y in range(32) for t,r,c in owners
                  for qr in range(2) for qc in range(2))
assert set(outputs) == set(range(1024**2)) and set(outputs.values()) == {1}
# SS old tail remains within the complete union but exceeds the typed tail and
# first bank; preserve a negative control even though it cannot hit the barriers.
old_tail = {q*2048+r*32+(c^((r&1)|((r&6)<<2))) for q in range(7) for r,c in q_coords}
assert max(old_tail) == 14335 and max(old_tail)*4 < BANKS*BANK
assert max(old_tail)*4 >= BANK and max(old_tail)*4 >= TAIL
initial,requested,owned = 80,WG*(64+80+80),THREADS*80
assert (requested,owned) == (28672,30720)
assert WG*(initial-64) == owned-requested == 2048

candidate,delta_path,report_path = [Path(str(PREFIX)+n) for n in ('candidate.cu','source_delta.json','review.json')]
for p in (candidate,delta_path,report_path):
    assert not p.exists(),'refuse to overwrite frozen SS artifact '+str(p)
candidate.write_text(source)
delta = {'scope':'CPU_ONLY_UNCOMPILED_SS_N16_COMPANION','base_path':str(BASE),
         'base_sha256':BASE_SHA,'candidate_path':str(candidate),'candidate_sha256':sha(candidate),
         'inverse_restores_base_byte_for_byte':True,'replacements':changes}
delta_path.write_text(json.dumps(delta,indent=2)+'\n')
report = {
    'status':'PASS_CPU_LITERAL_SS_BN16_GEOMETRY_ONLY_UNCOMPILED_UNREGISTERED',
    'CUTLASS_HEAD':(ROOT/'third_party/cutlass/.git/HEAD').read_text().strip(),
    'script_sha256':sha(Path(__file__)),'source_identity':delta,'evidence_sha256':before,
    'raw_frozen_artifact_sha256':raw_protected,
    'geometry_scope':'Independent literal CPU interpretation; no CuTe or CUDA instantiation',
    'geometry':{'aligned_contiguous_vector8_checks':vectors,'P_coords_per_WG':len(set(q_coords)),
                'P_unique_words':len(pwords),'P_words_per_thread_per_product':8,
                'WG0_P_payload_words':24,'WG1_P_payload_words':32,'RS_A_register_payload_words':0,
                'U_unique_owners':len(owners),'each_P_word_read_count':1,
                'full1024_unique_outputs':len(outputs),'each_output_store_count':1,
                'old_tail_negative_control_max_word':max(old_tail),
                'old_tail_exceeds_typed_tail_and_bank0_rejected':True},
    'predicted_shared_bytes':{'A7':A_BYTES,'B7':B_BYTES,'one_bank':BANK,'two_banks':BANKS*BANK,
                              'tail':TAIL,'barriers':32,'sizeof_align128':SHARED,
                              'tail_bank0_overlap':TAIL,'tail_bank1_overlap':0,
                              'barrier_union_overlap':0,'actual_host_ABI_proved':False},
    'quota_hypothesis':{'producer':64,'WG0':80,'WG1':80,'requested_pool_words':requested,
                        'hypothetical_initial':80,'hypothetical_initial_owned_pool_words':owned,
                        'producer_dec_release_words':2048,'consumer_each_inc_request_words':0,
                        'pool_surplus_words':2048,'source_minblocks':2,
                        'ptxas_initial_role_width_no_spill_qfit':None,'two_CTA_residency':None},
    'barriers_source':{'full_arrivals':128,'empty_arrivals':256,'CTA_bar0_and_bar6':384,
                       'both_banks_drained_before_tail':True,'hardware_proved':False},
    'native_host_domain':{'m_modulus':128,'n_modulus':64,'k_modulus':64,'lda_ldb_modulus':8,
                          'A_B_alignment_bytes':16,'n_modulus32_expansion_deferred':True},
    'source_final_ST_U_alphaRN_betaFMA_normalized_byte_equal_to_old34':True,
    'producer_A_final_RN_operations_per_cell':4,'producer_A_derived_final_onecasts_per_cell':4,
    'consumer_has_no_RS_A_stage_array_or_builder':True,
    'N16_pair_scope':'SS/RS same m64n16k16 dtype/accumulator shape and K-atom order, distinct A source forms; fresh compiled/numerical paired evidence still required.',
    'reference36_N32_is_not_same_atom':True,
    'changed_axes':['BN32 to16 with typed SS atoms/layouts/tail','consumer source q96 to80','minblocks1 to2'],
    'logical_source_reads_1024_bytes':{'A':64*1024**2,'B':16*1024**2,'actual_L2_DRAM':None},
    'grid_1024':[8,32],'CTAs_1024':256,
    'not_proved':['CuTe/CUDA actual instantiation and layouts/sizeof','new initial/role allocation and complete SASS widths',
                  'qfit/zero local or spill/physical async synchronization','physical two CTA residency/occupancy',
                  'GPU precision and original FP64 gate','Graph/sanitizer/forward progress','performance or actual traffic'],
    'writes_restricted_to_own_SS_prefix':True,
}
assert {str(p):sha(p) for p in protected} == before
assert {p:sha(Path(p)) for p in raw_protected} == raw_protected
report['old34_and_frozen_raw_future_unchanged'] = True
report_path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'candidate_sha256':sha(candidate),
                  'delta_sha256':sha(delta_path),'review_sha256':sha(report_path),
                  'geometry':report['geometry'],'predicted_shared_bytes':report['predicted_shared_bytes'],
                  'quota_hypothesis':report['quota_hypothesis'],
                  'old34_and_frozen_raw_future_unchanged':True},indent=2))

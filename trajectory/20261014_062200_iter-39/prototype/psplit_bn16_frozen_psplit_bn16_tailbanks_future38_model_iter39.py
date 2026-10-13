"""Scalar-word BN16 tail bank geometry; not hardware transactions/performance.

Every output is restricted to this prefix. Frozen raw/SS candidates and all
their eight artifacts plus manifests are read and verified, never rewritten.
"""
from pathlib import Path
from collections import Counter
from itertools import product
import hashlib
import json
import sys

ROOT = Path('/share/home/liuyao/workspace/MegaFMM')
PREFIX = ROOT / '.scratch/psplit_bn16_tailbanks_future38_'
FROZEN = [ROOT/'.scratch'/n for n in ('psplit_bn16_future38_manifest.json',
                                    'psplit_bn16_ss_future38_manifest.json')]
HEADER = ROOT/'third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp'
RAW = ROOT/'.scratch/psplit_bn16_future38_candidate.cu'
SS = ROOT/'.scratch/psplit_bn16_ss_future38_candidate.cu'
EXPECTED_RAW = 'ab97837744a071341f7c4b7f21f8009171e4bfc471180eca7c07c37b408c3b90'
EXPECTED_SS = '94c482bc73a4e5dfa32436b9931f300242afd89058d817522c898a1ccd009ab6'
MANIFEST_SHAS = ['a3c5f94b8a454c8ae84748eda9bb40096fa102592806964df8cf26d877127a95',
                 '257b0e37a6ea2b98a51f3ad2fb04693b2f26a55bf1cdba79d8f2267340837bce']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restrict_writes(event,args):
    if event != 'open' or not args or not isinstance(args[0],(str,bytes)):
        return
    mode,flags = args[1] or '',args[2] if len(args)>2 else 0
    writes = isinstance(mode,str) and any(c in mode for c in 'wax+')
    writes |= isinstance(flags,int) and bool(flags & (1|2|64|512|1024))
    if writes and not str(Path(args[0]).absolute()).startswith(str(PREFIX)):
        raise RuntimeError('write outside tailbanks prefix rejected: '+str(args[0]))


sys.addaudithook(restrict_writes)
protected = {str(HEADER):sha(HEADER)}
for p,expected in zip(FROZEN,MANIFEST_SHAS):
    assert sha(p) == expected
    protected[str(p)] = expected
    for name,expected_file in json.loads(p.read_text())['files'].items():
        assert sha(Path(name)) == expected_file
        protected[name] = expected_file
assert sha(RAW) == EXPECTED_RAW and sha(SS) == EXPECTED_SS
text = HEADER.read_text()
assert 'using CLayout_64x16  = Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,  _2>>,' in text
assert 'Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>>;' in text
for name in ('SM90_64x16x16_F32F16F16_SS','SM90_64x16x16_F32F16F16_RS',
             'SM90_64x16x16_F32BF16BF16_SS','SM90_64x16x16_F32BF16BF16_RS'):
    trait = text[text.index('struct MMA_Traits<'+name):]
    trait = trait[:trait.index('\n};')]
    assert 'using ThrID   = Layout<_128>;' in trait
    assert 'using CLayout = GMMA::CLayout_64x16;' in trait

BM,BN,WG,THREADS,WORDS_PER_P,NUM_BANKS = 64,16,128,384,1024,32
OLD_LINE = '  const int xor_mask = row & (Cols/4-1);\n'
NEW_LINE = '  const int xor_mask = ((row&2)>>1) | ((row&4)<<1);\n'
for p in (RAW,SS):
    src = p.read_text()
    assert src.count(OLD_LINE) == 1
    assert 'for (int linear = tid; linear < T::BM*T::BN; linear += T::Threads)' in src
    assert 'for (int i = 0; i < size(q0); ++i)' in src
    assert 'static_assert(Cols == 16,' in src


def coord(thread,value):
    # Pinned thread mode (4,8,4), value mode (2,2,2), strides as above.
    linear = (128*(thread%4)+(thread//4%8)+16*(thread//32%4)
              +64*(value%2)+8*(value//2%2)+512*(value//4%2))
    return linear%BM,linear//BM


MASKS = [
    ('current_row_and3','row&3',lambda r:r&3),
    ('zero','0',lambda r:0),
    ('row0_col0_row1_col3','(row&1)|((row&2)<<2)',lambda r:(r&1)|((r&2)<<2)),
    ('old_mask_and15','[(row&1)|((row&6)<<2)]&15',lambda r:((r&1)|((r&6)<<2))&15),
    ('selected_row1_col0_row2_col3','((row&2)>>1)|((row&4)<<1)',lambda r:((r&2)>>1)|((r&4)<<1)),
]
assert all(MASKS[2][2](r) == MASKS[3][2](r) for r in range(BM))


def word_address(p,r,c,mask):
    return p*WORDS_PER_P+r*BN+(c^mask(r))


def scalar_instance(words):
    unique = set(words)
    by_bank = [sorted(w for w in unique if w%NUM_BANKS == b) for b in range(NUM_BANKS)]
    multiplicity = [len(items) for items in by_bank]
    return {'word_addresses':words,'bank_indices':[w%NUM_BANKS for w in words],
            'unique_words_by_bank':by_bank,'bank_unique_word_multiplicity':multiplicity,
            'active_lane_count':len(words),'distinct_word_count':len(unique),
            'same_word_duplicate_requests':len(words)-len(unique),
            'nonempty_bank_count':sum(bool(n) for n in multiplicity),
            'worst_distinct_words_per_bank':max(multiplicity,default=0)}


coords = [coord(t,v) for t in range(WG) for v in range(8)]
assert len(coords) == len(set(coords)) == BM*BN
full = []
summaries = []
for mask_name,expression,mask in MASKS:
    assert all(0 <= mask(r) < BN for r in range(BM))
    assert all({c^mask(r) for c in range(BN)} == set(range(BN)) for r in range(BM))
    stores,loads = [],[]
    total_writes = Counter()
    total_reads = Counter()
    for p in range(7):
        group = 0 if p<3 else 1
        first_tid = 128 if group == 0 else 256
        for warp in range(4):
            for slot in range(8):
                coordinates = [coord(warp*32+lane,slot) for lane in range(32)]
                words = [word_address(p,r,c,mask) for r,c in coordinates]
                total_writes.update(words)
                entry = scalar_instance(words)
                assert entry['active_lane_count'] == entry['distinct_word_count'] == 32
                entry.update({'product_index':p,'product_label':'P'+str(p+1),
                              'consumer_group':group,'local_warp':warp,
                              'CTA_warp':first_tid//32+warp,'fragment_slot':slot,
                              'CTA_thread_indices':[first_tid+warp*32+l for l in range(32)],
                              'coordinates':coordinates})
                stores.append(entry)
        for loop in range(3):
            for warp in range(12):
                active = [tid for tid in range(warp*32,(warp+1)*32) if tid+THREADS*loop<WORDS_PER_P]
                linear = [tid+THREADS*loop for tid in active]
                coordinates = [(x//BN,x%BN) for x in linear]
                words = [word_address(p,r,c,mask) for r,c in coordinates]
                total_reads.update(words)
                entry = scalar_instance(words)
                assert len(active) in (0,32)  # no partial warp in this fixed domain
                if active:
                    assert entry['nonempty_bank_count'] == 32
                    assert entry['worst_distinct_words_per_bank'] == 1
                entry.update({'product_index':p,'product_label':'P'+str(p+1),
                              'U_loop_round':loop,'CTA_warp':warp,
                              'CTA_thread_indices':active,'active_lanes':[t%32 for t in active],
                              'linear_coordinates':linear,'coordinates':coordinates})
                loads.append(entry)
    assert set(total_writes) == set(total_reads) == set(range(7*WORDS_PER_P))
    assert set(total_writes.values()) == set(total_reads.values()) == {1}
    assert len(stores) == 224 and len(loads) == 252
    active_loads = [r for r in loads if r['active_lane_count']]
    assert len(active_loads) == 224
    per_product = []
    for p in range(7):
        st = [r for r in stores if r['product_index']==p]
        ld = [r for r in loads if r['product_index']==p]
        per_product.append({'product_index':p,'store_warp_slot_instances':len(st),
                            'store_worst_multiplicity_histogram':dict(Counter(r['worst_distinct_words_per_bank'] for r in st)),
                            'store_nonempty_bank_histogram':dict(Counter(r['nonempty_bank_count'] for r in st)),
                            'U_round_active_warp_counts':[sum(r['U_loop_round']==j and bool(r['active_lane_count']) for r in ld) for j in range(3)],
                            'U_active_scalar_instances':sum(bool(r['active_lane_count']) for r in ld),
                            'U_inactive_warp_rounds':sum(not r['active_lane_count'] for r in ld),
                            'U_partial_warp_instances':sum(0<r['active_lane_count']<32 for r in ld)})
    summary = {'name':mask_name,'expression':expression,'row0_to7_masks':[mask(r) for r in range(8)],
               'STORE_worst_distinct_words_per_bank':max(r['worst_distinct_words_per_bank'] for r in stores),
               'STORE_nonempty_bank_counts':sorted({r['nonempty_bank_count'] for r in stores}),
               'STORE_scalar_instances_all_products':224,
               'U_LOAD_worst_distinct_words_per_bank':max(r['worst_distinct_words_per_bank'] for r in active_loads),
               'U_LOAD_nonempty_bank_counts':sorted({r['nonempty_bank_count'] for r in active_loads}),
               'U_LOAD_active_scalar_instances_all_products':224,'U_LOAD_zero_active_instances_all_products':28,
               'same_word_duplicate_requests_all_instances':sum(r['same_word_duplicate_requests'] for r in stores+loads),
               'per_product':per_product}
    summaries.append(summary)
    full.append({'summary':summary,'STORE':stores,'U_LOAD':loads})
assert [r['STORE_worst_distinct_words_per_bank'] for r in summaries] == [4,4,2,2,1]

# Exhaustive periodic8 linear row-XOR family: each of three row bits selects
# an arbitrary four-bit column mask. Search scope is explicitly bounded.
search_hist = Counter()
best = []
for coefficients in product(range(16),repeat=3):
    def linear_mask(r,coeff=coefficients):
        return ((coeff[0] if r&1 else 0) ^ (coeff[1] if r&2 else 0) ^ (coeff[2] if r&4 else 0))
    words = [word_address(0,r,c,linear_mask) for r,c in [coord(l,0) for l in range(32)]]
    worst = scalar_instance(words)['worst_distinct_words_per_bank']
    search_hist[worst] += 1
    if worst == 1:
        best.append(list(coefficients))
assert sum(search_hist.values()) == 4096 and [0,1,8] in best
assert min(search_hist) == 1  # lower bound for 32 unique words across 32 banks
selected = MASKS[-1][2]
assert summaries[-1]['STORE_nonempty_bank_counts'] == [32]
assert all(selected(r) == ((1 if r&2 else 0)^(8 if r&4 else 0)) for r in range(BM))

derivatives = []
for label,parent in (('raw',RAW),('ss',SS)):
    p = Path(str(PREFIX)+label+'_candidate.cu')
    assert not p.exists()
    base = parent.read_bytes()
    assert base.count(OLD_LINE.encode()) == 1
    derived = base.replace(OLD_LINE.encode(),NEW_LINE.encode())
    assert derived.count(NEW_LINE.encode()) == 1
    assert derived.replace(NEW_LINE.encode(),OLD_LINE.encode()) == base
    p.write_bytes(derived)
    derivatives.append({'label':label,'parent_path':str(parent),'parent_sha256':sha(parent),
                        'derivative_path':str(p),'derivative_sha256':sha(p),
                        'only_changed_bytes':{'old':OLD_LINE,'new':NEW_LINE,'count':1},
                        'inverse_restores_parent_byte_for_byte':True})
report = {'status':'PASS_CPU_LITERAL_SCALAR_TAIL_BANK_GEOMETRY_ONLY',
          'script_sha256':sha(Path(__file__)),'pinned_source_sha256':protected,
          'CUTLASS_HEAD':(ROOT/'third_party/cutlass/.git/HEAD').read_text().strip(),
          'word_bytes':4,'bank_count':32,'shared_base_alignment_bytes':128,
          'N16_CLayout_literal':'(t=(4,8,4),v=(2,2,2)); strides ((128,1,16),(64,8,512))',
          'thread_mode':'t0=tid%4,t1=(tid//4)%8,t2=(tid//32)%4',
          'value_mode':'v0=slot%2,v1=(slot//2)%2,v2=(slot//4)%2',
          'coordinates':'row=t1+16*t2+8*v1; col=2*t0+v0+8*v2',
          'product_stride_words':1024,'product_stride_mod32':0,
          'STORE_groups':'P1/P2/P3 CTA tids128..255; P4/P5/P6/P7 tids256..383, local tid modulo128',
          'U_thread_indices':'fixed CTA tid0..383, linear=tid+384*round<1024',
          'U_round_active_warps':[12,12,8],'U_round_inactive_warps':[0,0,4],
          'U_partial_warp_instances':0,'summaries':summaries,
          'linear_period8_mask_search':{'scope':'three row bits independently XOR four-bit column coefficients; representative warp0/slot0',
                                       'candidates':4096,'worst_distinct_word_histogram':dict(search_hist),
                                       'one_word_per_bank_coefficient_count':len(best),
                                       'one_word_per_bank_coefficients':best,
                                       'selected_coefficients':[0,1,8],
                                       'selected_all_warps_slots_products_verified':True},
          'derived_candidate_identity':derivatives,
          'selected_future_hypothesis_only':True,
          'actual_CUDA_template_or_SASS_instruction_grouping_proved':False,
          'actual_transactions_stalls_GPU_performance_qfit_residency_proved':False,
          'not_claimed':['CUDA actual transaction grouping/broadcast rules/stalls','GPU speedup or causal performance effect',
                         'new compiled initial/role fit/no-spill/residency','GPU numerical precision'],
          'all_old_frozen_bytes_verified_unchanged':True}
assert {p:sha(Path(p)) for p in protected} == protected
for suffix,value in (('lane_maps.json',{'scope':'literal scalar word maps only','masks':full}),
                     ('source_delta.json',{'scope':'one line only derivative hypothesis','derivatives':derivatives}),
                     ('review.json',report)):
    p = Path(str(PREFIX)+suffix)
    assert not p.exists()
    p.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'status':report['status'],'summaries':[{k:v for k,v in s.items() if k!='per_product'} for s in summaries],
                  'search_histogram':dict(search_hist),'best_coefficients_count':len(best),
                  'derivatives':derivatives,'frozen_sources_headers_manifests_unchanged':True,
                  'review_sha256':sha(Path(str(PREFIX)+'review.json')),
                  'lane_maps_sha256':sha(Path(str(PREFIX)+'lane_maps.json'))},indent=2))

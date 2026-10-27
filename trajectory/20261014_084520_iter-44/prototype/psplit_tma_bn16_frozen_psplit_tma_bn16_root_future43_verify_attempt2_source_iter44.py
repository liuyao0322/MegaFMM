#!/usr/bin/env python3
"""Independent CPU byte/source review; no compiler/GPU, predictions kept separate."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent.parent;P=R/'.scratch/psplit_tma_bn16_root_future43_'
D=lambda s:hashlib.sha256(s.encode()).hexdigest();H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=Path(str(P)+'base45.cu').read_text();s=Path(str(P)+'candidate.cu').read_text()
d=json.loads(Path(str(P)+'source_delta.json').read_text())
assert D(base)==d['parent_sha256']=='3c84b3104bf65fa90b6b3373343aa12d10b16bbe4d8c1859cc3e1d6c87958197'
assert D(s)==d['candidate_sha256']
f=base
for e in d['ordered_exact_replacements']:
 assert D(f)==e['before_sha256'] and f.count(e['old'])==e['count']
 for pos in e['old_positions']:assert f[pos:pos+len(e['old'])]==e['old']
 f=f.replace(e['old'],e['new']);assert D(f)==e['after_sha256']
assert f==s
f=s
for e in reversed(d['ordered_exact_replacements']):
 assert D(f)==e['after_sha256']
 for pos in reversed(e['new_positions']):
  assert f[pos:pos+len(e['new'])]==e['new'];f=f[:pos]+e['old']+f[pos+len(e['new']):]
 assert D(f)==e['before_sha256']
assert f==base
n=s.replace('psplit_tma_bn16_root_future43','psplit_selective_tma_a').replace('PSplitTmaBN16Traits','PSplitSelectiveTmaATraits').replace('TmaBN16Descriptor','SelectiveTmaADescriptor').replace('make_tma_bn16_descriptor','make_selective_tmaA_descriptor')
def block(text,a,b):
 i=text.index(a);return text[i:text.index(b,i)]
equal={}
for name,a,b in [
 ('A_fullstage_onecast_RN_math','// First0:','// BK64/BN'),
 ('B_final7_onecast_SIMT_math','template <class Element,int Cols,class SmemB>','// Tail FP32 rendezvous'),
 ('A_current_map_encoder','SelectiveTmaADescriptor make_selective_tmaA_descriptor','template <class Element>\nvoid launch_wgmma_'),
 ('producer_A4_TMA_raw_wait_and_128writer_publication','      // A raw transaction is distinct','      ++state;'),
 ('raw_barrier_init_ABI','  Pipeline pipeline(storage.barriers,params);','  typename T::Mma mma;'),
 ('bothbank_drain','    // Wait for BOTH consumer WGs','  } else {'),
 ('WG0_wholeK_pending_lifetime','      typename Pipeline::PipelineState state;','    } else {\n      clear(q0);'),
 ('fixed_U_RN_alpha_beta_FMA_outputs','  float* tail = storage.tail_products;\n  for (int linear','template <class Element>\nvoid prepare_wgmma_')]:
 left=block(base,a,b)
 if name=='WG0_wholeK_pending_lifetime':b='    } else {\n      clear(q0);'
 right=block(n,a,b)
 assert left==right,name;equal[name]={'bytes':len(left.encode()),'sha256':D(left),'exact_after_private_name_normalization':True}
# The complete WG1 loop remains byte-exact after private-name normalization.
a='      clear(q3);\n      typename Pipeline::PipelineState state;';b='    // All384 threads rendezvous'
assert block(base,a,b)==block(n,a,b)
equal['WG1_wholeK_pending_lifetime']={'exact_after_private_name_normalization':True,'sha256':D(block(base,a,b))}
# Private guard stays exactly parent domain except the declared N32 tile requirement.
a='bool wgmma_psplit_selective_tma_a_can_implement';b='} // namespace megafmm\n'
left=block(base,a,b);right=block(n,a,'} // namespace tma_bn16_future43\n').replace('p.n%32 == 0','p.n%64 == 0')
assert left==right
for token,count in [('SM90_TMA_LOAD_2D::copy(',4),('cuTensorMapEncodeTiled(',1),('ClusterTransactionBarrier::wait(',1),('ClusterTransactionBarrier::init(',1),('arrive_and_expect_tx(',1),('warpgroup_wait<0>();',2),('pipeline.consumer_release(state);',2),('pipeline.producer_tail(state);',1),('__syncthreads();',2),('barrier.sync 6,384;',2)]:assert n.count(token)==base.count(token)==count,(token,count)
assert 'const uint32_t RawABytes = 4*BM*BK*sizeof(Element)' not in s # Actual declaration is constexpr; checked below.
assert 'static constexpr uint32_t RawABytes = 4*BM*BK*sizeof(Element);' in s
assert 'GMMA::Layout_K_SW128_Atom<Element>{}' in s and 'GMMA::Layout_MN_SW32_Atom<Element>{}' in s
assert 'return product*1024+row*16+(col^xor_mask);' in s
assert 'static_assert(size(q0) == 8' in s and 'struct alignas(1024) SharedStorage' in s
assert 'cutlass::arch::warpgroup_reg_dealloc<40>();' in s
token='cutlass::arch::warpgroup_reg_alloc<128>();';assert s.count(token)==1
assert s.index(token)<s.index('    if (tid < 256) {\n      clear(q0); clear(q1); clear(q2);')
assert s.index(token)<s.index('    } else {\n      clear(q0); clear(q1); clear(q2);')
assert 'warpgroup_reg_alloc<192>' not in s
assert 'namespace tma_bn16_future43 {' in s and 'p.n%32 == 0' in s
assert s.count('PSplitTmaBN16Traits<Element,16>')==2 and s.count('winograd_wgmma_psplit_tma_bn16_root_future43<Element,16>')==2
traits=(R/'third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp').read_text()
assert 'using CLayout_64x16  = Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,  _2>>,' in traits
assert 'Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>>' in traits
assert 'LayoutType::B32          : Swizzle<1,4,3> o smem_ptr o ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO))' in traits
assert 'desc.bitfield.leading_byte_offset_' in traits and 'desc.bitfield.stride_byte_offset_' in traits
arch=(R/'third_party/cutlass/include/cute/arch/mma_sm90_gmma.hpp').read_text()
selector=(R/'third_party/cutlass/include/cute/arch/mma_sm90.hpp').read_text()
for dtype in ['F16F16','BF16BF16']:
 tag='SM90_64x16x16_F32'+dtype+'_RS'
 body=arch[arch.index('struct '+tag):];body=body[:body.index('static_assert')]
 assert 'using CRegisters = float[8];' in body and 'using ARegisters = uint32_t[4];' in body and 'using BRegisters = uint64_t[1];' in body
 assert 'return '+tag+'<MajorA, MajorB, Args...>{};' in selector
 pos=traits.index('struct MMA_Traits<'+tag);body=traits[pos:traits.index('GMMA::ScaleOut accumulate_',pos)]
 assert 'using ALayout = GMMA::ALayout_64x16;' in body and 'using CLayout = GMMA::CLayout_64x16;' in body

pipeline=(R/'third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp').read_text()
assert 'struct PipelineAsyncSharedStorage' in pipeline
assert 'FullBarrier full_barrier_[Stages];' in pipeline and 'EmptyBarrier empty_barrier_[Stages];' in pipeline
barrier=(R/'third_party/cutlass/include/cutlass/arch/barrier.h').read_text()
assert 'struct ClusterBarrier {' in barrier and 'using ValueType = uint64_t;' in barrier
bindings=json.loads(Path(str(P)+'bindings.json').read_text())
for key in ['protected_old_future_sha256','immutable_library_source_sha256','pinned_header_sha256']:
 for name,want in bindings[key].items():assert H(R/name)==want,(key,name)
# CPU arithmetic checks only. They establish no compiler or hardware allocation.
required=128*(40+128+128);assert required==37888 and required<=65536 and 2*required>65536
minimum_initial=(required+383)//384;assert minimum_initial==99
layout=(R/'third_party/cutlass/include/cute/layout.hpp').read_text()
assert 'return Layout<_1,_0>{};' in layout and 'complement(tiler, shape(layout))' in layout
assert 'const int xor_mask = ((row&1) | ((row&6)<<2)) & 15;' in s
result={'status':'PASS_CPU_SOURCE_SCOPE','candidate_sha256':D(s),'api_sha256':H(str(P)+'api.h'),
 'forward_inverse_replacement_count':len(d['ordered_exact_replacements']),'unchanged_parent45_scopes':equal,
 'N32_native_guard_only_declared_guard_change':True,'common_consumer_INC128_before_either_first_clear':True,
 'raw_A4_TX32768_preserved':True,'current_A_encoder_consumed_by_TMA':True,'original_three_CTA_epochs':True,
 'source_quota_hypothesis':{'producer':40,'WG0':128,'WG1':128,'requested_pool_words':required,'initial_per_thread_minimum_required':minimum_initial,'physical_pool_upper_bound_words':65536,'source_minimum_P_A_words':[72,96],'other_word_slack_only_lower_bound':[56,32], 'two_CTA_quota_request_exceeds_65536':True},
 'future_required':{'actual_initial_owned_pool_at_least_words':37888,'actual_initial_per_thread_at_least':minimum_initial,'highest_GPR_strictly_below':[40,128,128],'fresh_initialization_scalarization_pending_lifetime_and_zero_stack_local_spill':True},
 'protected_old_future_files':142,'immutable_library_sources':41,'pinned_headers':10,'protected_drift':0,
 'actual_CUDA_compiled':None,'actual_CAN':None,'actual_GPR':None,'actual_initial_pool':None,'actual_shared_size':None,'actual_Driver_resources':None,'actual_SAN':None,'actual_precision':None,'actual_performance':None}
print(json.dumps(result,indent=2))

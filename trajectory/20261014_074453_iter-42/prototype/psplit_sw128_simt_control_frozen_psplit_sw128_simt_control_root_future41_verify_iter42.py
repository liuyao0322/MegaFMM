#!/usr/bin/env python3
"""CPU source review independent of generator and transport-order model."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
P=ROOT/'.scratch/psplit_sw128_simt_control_root_future41_'
OLD='psplit_selective_tma_a';NEW='psplit_sw128_simt_control_root_future41'
digest=lambda text:hashlib.sha256(text.encode()).hexdigest()
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
base=Path(str(P)+'base45.cu').read_text();candidate=Path(str(P)+'candidate.cu').read_text()
delta=json.loads(Path(str(P)+'source_delta.json').read_text())
assert digest(base)==delta['parent_sha256']=='3c84b3104bf65fa90b6b3373343aa12d10b16bbe4d8c1859cc3e1d6c87958197'
assert digest(candidate)==delta['candidate_sha256']
forward=base
for e in delta['ordered_exact_replacements']:
    assert digest(forward)==e['before_sha256'] and forward.count(e['old'])==e['count']
    assert len(e['old_positions'])==e['count']
    for position in e['old_positions']:assert forward[position:position+len(e['old'])]==e['old']
    forward=forward.replace(e['old'],e['new']);assert digest(forward)==e['after_sha256']
assert forward==candidate
inverse=candidate
for e in reversed(delta['ordered_exact_replacements']):
    assert digest(inverse)==e['after_sha256']
    for position in reversed(e['new_positions']):
        assert inverse[position:position+len(e['new'])]==e['new']
        inverse=inverse[:position]+e['old']+inverse[position+len(e['new']):]
    assert digest(inverse)==e['before_sha256']
assert inverse==base

normalized=candidate.replace(NEW,OLD).replace('PSplitSW128SimtControlTraits','PSplitSelectiveTmaATraits').replace('SW128SimtControlDescriptor','SelectiveTmaADescriptor').replace('make_sw128_simt_control_descriptor','make_selective_tmaA_descriptor')
def block(text,start,end):
    begin=text.index(start);return text[begin:text.index(end,begin)]
equal_scopes={}
scopes=[('A_SW128_traits_shared_storage', 'template <class Element, int Cols>\nstruct PSplitSelectiveTmaATraits','struct PSplitRSAPackedEight'),
 ('whole_A_build_math','// First0:','// BK64/BN32'),
 ('B_SIMT_math','// BK64/BN32','// Tail FP32 rendezvous'),
 ('tail_coordinate_map','// Tail FP32 rendezvous','// One current-A descriptor'),
 ('raw_done_init','  Pipeline pipeline(storage.barriers,params);','  typename T::Mma mma;'),
 ('publication_consumers_both_banks_tail_fixed_U_epilogue','      pipeline.producer_commit(state);','template <class Element>\nvoid prepare_wgmma_'),
 ('descriptor_encode_function','SelectiveTmaADescriptor make_selective_tmaA_descriptor','template <class Element>\nvoid launch_wgmma_')]
for name,start,end in scopes:
    # The descriptor comment alone is intentionally changed for truthfulness.
    if name=='tail_coordinate_map':end='struct alignas(64) SelectiveTmaADescriptor'
    left=block(base,start,end);right=block(normalized,start,end)
    if name=='tail_coordinate_map':
        left=left[:left.index('// One current-A descriptor')]
        right=right[:right.index('// CONTROL: one current-A descriptor')]
    assert left==right,name
    equal_scopes[name]={'byte_exact_after_name_normalization':True,'sha256':digest(left),'bytes':len(left.encode())}

old39=(ROOT/'solution/megakernel_wgmma_psplit_rsa.cu').read_text()
normalized39=candidate.replace(NEW,'psplit_rsa')
borrowed={}
for name,start,end in [
 ('fill_a','// Producer writes four raw original A quadrants','CUTE_DEVICE uint32_t psplit_rsa_load_pair'),
 ('read_eight','template <class Element>\nCUTE_DEVICE PSplitRSAPackedEight psplit_rsa_read_eight','template <class Element>\nCUTE_DEVICE float psplit_rsa_unpack'),
 ('store_eight','CUTE_DEVICE void psplit_rsa_store_eight','template <class Element>\nCUTE_DEVICE uint32_t psplit_rsa_round_pair')]:
    left=block(old39,start,end);right=block(normalized39,start,end);assert left==right,name
    borrowed[name]={'byte_exact_name_only_old39_copy':True,'sha256':digest(left),'bytes':len(left.encode())}

start='bool wgmma_'+OLD+'_can_implement'
base_guards=base[base.index(start):base.index('} // namespace megafmm\n')]
candidate_guards=normalized[normalized.index(start):normalized.index('} // namespace sw128_simt_control_future41\n')]
assert base_guards==candidate_guards
equal_scopes['all_private_metadata_binding_guards']={'byte_exact_after_name_normalization':True,'sha256':digest(base_guards)}
assert normalized.count('cuTensorMapEncodeTiled(')==base.count('cuTensorMapEncodeTiled(')==1
assert normalized.count('const auto maps=make_selective_tmaA_descriptor(p,A);')==1
assert normalized.count('CUTE_GRID_CONSTANT const SelectiveTmaADescriptor maps')==1
assert '(void)maps;' in normalized
assert '      p,maps,static_cast<const Element*>(A),static_cast<const Element*>(B),C0,D);' in normalized
assert 'const Element* __restrict__ A,' in normalized
assert 'SM90_TMA_LOAD_2D::copy' not in normalized and 'arrive_and_expect_tx' not in normalized and 'ClusterTransactionBarrier::wait(' not in normalized
assert normalized.count('&maps.a')==1 # Host encode only; no descriptor use in kernel.
for token,count in [('__syncthreads();',2),('barrier.sync 6,384;',2),('pipeline.consumer_release(state);',2),('pipeline.producer_tail(state);',1),('warpgroup_wait<0>();',2),('ClusterTransactionBarrier::init(',1)]:
    assert normalized.count(token)==base.count(token)==count,(token,count)
source_header=(ROOT/'third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp').read_text()
assert 'Layout<Shape<_8,_1024>,Stride<_1024,_1>>>' in source_header
assert 'Layout<Shape<_8, _256>,Stride< _256,_1>>>' in source_header
assert 'return offset ^ shiftr(offset & yyy_msk{}, msk_sft{});' in (ROOT/'third_party/cutlass/include/cute/swizzle.hpp').read_text()
assert 'make_tensor(make_smem_ptr(ptr.get(), layout.layout_a()),' in (ROOT/'third_party/cutlass/include/cute/pointer_flagged.hpp').read_text()
pipeline=(ROOT/'third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp').read_text()
assert 'constexpr uint32_t InitialProducerPhase = 1;' in pipeline
assert 'PipelineState(): index_{}, phase_{}, count_{} {}' in pipeline
bindings=json.loads(Path(str(P)+'bindings.json').read_text())
for key in ('protected_future_before_sha256','library_source_before_sha256','pinned_header_before_sha256'):
    assert all(sha(ROOT/name)==want for name,want in bindings[key].items()),key
print(json.dumps({'status':'PASS_CPU_SOURCE_SCOPE','candidate_sha256':digest(candidate),
 'exact_forward_inverse_replacements':len(delta['ordered_exact_replacements']),
 'equal_parent45_scopes':equal_scopes,'borrowed_old39_functions':borrowed,
 'retained_unused_map_encode_per_ordinary_call':1,'encoded_map_consumed_by_control_kernel':False,
 'raw_done_storage_and_init_retained':True,'new_TMA_copy_expect_wait_count':0,
 'three_original_CTA_epochs_retained':True,'private_namespace_only':True,
 'old_future_files_rechecked':71,'library_source_files_rechecked':len(bindings['library_source_before_sha256']),
 'old_protected_drift':0,'CUDA_compiled':False,'GPU_launched':False,'new_actual_shared_ABI':None,
 'independent_CAN':None,'new_actual_precision':None,'new_actual_performance':None},indent=2))

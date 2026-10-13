#!/usr/bin/env python3
"""CPU source generation only. Derived from exact frozen measured45 bytes."""
import difflib
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
P=ROOT/'.scratch/psplit_sw128_simt_control_root_future41_'
BASE=ROOT/'solution/megakernel_wgmma_psplit_selective_tma_a.cu'
BASE_SHA='3c84b3104bf65fa90b6b3373343aa12d10b16bbe4d8c1859cc3e1d6c87958197'
OLD39=ROOT/'solution/megakernel_wgmma_psplit_rsa.cu'
OLD39_SHA='1f964339034c3e800455704b6d33482d072ea99a2b6eb9f08773df57f705f054'
OLD='psplit_selective_tma_a'
NEW='psplit_sw128_simt_control_root_future41'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        mode=args[1] or '';flags=args[2] if len(args)>2 else 0
        writes=isinstance(mode,str) and any(c in mode for c in 'wax+')
        writes|=isinstance(flags,int) and bool(flags&(1|2|64|512|1024))
        if writes and not str(Path(args[0]).absolute()).startswith(str(P)):
            raise RuntimeError('write outside new SW128 SIMT control prefix')
sys.addaudithook(guard)
assert sha(BASE)==BASE_SHA and sha(OLD39)==OLD39_SHA
original=BASE.read_text();old39=OLD39.read_text();source=original;edits=[]
def replace(old,new,count=1):
    global source
    assert source.count(old)==count,(old[:120],source.count(old),count)
    before=source;positions=[];cursor=0
    for _ in range(count):
        position=before.index(old,cursor);positions.append(position);cursor=position+len(old)
    source=source.replace(old,new)
    edits.append({'old':old,'new':new,'count':count,'old_positions':positions,
      'new_positions':[position+i*(len(new)-len(old)) for i,position in enumerate(positions)],
      'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(source.encode()).hexdigest()})

replace('// Iteration41 selective rawA4 TMA, B transforms remain SIMT; explicit variant45.\n',
 '// CPU-only SW128/SIMT rawA control derived from frozen measured45; private and uncompiled.\n')
replace(OLD,NEW,source.count(OLD))
replace('PSplitSelectiveTmaATraits','PSplitSW128SimtControlTraits',source.count('PSplitSelectiveTmaATraits'))
replace('SelectiveTmaADescriptor','SW128SimtControlDescriptor',source.count('SelectiveTmaADescriptor'))
replace('make_selective_tmaA_descriptor','make_sw128_simt_control_descriptor',source.count('make_selective_tmaA_descriptor'))
replace('#include "megafmm/kernels.h"\n','#include "megafmm/kernels.h"\n#include "psplit_sw128_simt_control_root_future41_api.h"\n')
replace('namespace megafmm {\n','namespace megafmm {\nnamespace sw128_simt_control_future41 {\n')
replace('} // namespace megafmm\n','} // namespace sw128_simt_control_future41\n} // namespace megafmm\n')

# The old39 raw-copy function is copied verbatim, with names normalized only.
first=old39.index('// Producer writes four raw original A quadrants')
last=old39.index('CUTE_DEVICE uint32_t psplit_rsa_load_pair',first)
fill_a=old39[first:last].replace('psplit_rsa',NEW)
replace('// Four raw A quadrants are copied by TMA, without transform or cast.\n',fill_a)
replace('void winograd_wgmma_'+NEW+'(Problem p,CUTE_GRID_CONSTANT const SW128SimtControlDescriptor maps,\n',
 'void winograd_wgmma_'+NEW+'(Problem p,CUTE_GRID_CONSTANT const SW128SimtControlDescriptor maps,\n    const Element* __restrict__ A,\n')
replace('  using T = PSplitSW128SimtControlTraits<Element,Cols>;\n',
 '  // CONTROL: maps is encoded/passed for host-cost/ABI control, never consumed.\n  (void)maps;\n  using T = PSplitSW128SimtControlTraits<Element,Cols>;\n')
first=source.index('      // A raw transaction is distinct from final-ready PipelineAsync.')
last=source.index('      CUTE_UNROLL\n      for (int task = tid; task < T::BN*T::BK/8;',first)
old_transport=source[first:last]
new_transport='''      // CONTROL: all128 producer writers copy rawA4 with original39 vector8 SIMT.
      // SW128/1024 alignment and unused raw_done init/storage remain from45.
      CUTE_UNROLL
      for (int task = tid; task < T::BM*T::BK/8; task += T::ProducerThreads)
        '''+NEW+'''_fill_a<Element,T::BK>(p,A,row_base,k_base,task,sA);
'''
replace(old_transport,new_transport)
old_wait='''      // raw_done starts phase0, opposite the producer start phase1.
      // Every writer waits for all32768 A bytes before final publication.
      cutlass::arch::ClusterTransactionBarrier::wait(
          &storage.raw_done[state.index()],state.phase()^1u);
      asm volatile("" ::: "memory"); // pinned wait lacks compiler memory clobber
'''
replace(old_wait,'')
replace('      p,maps,static_cast<const Element*>(B),C0,D);',
 '      p,maps,static_cast<const Element*>(A),static_cast<const Element*>(B),C0,D);')
replace('// One current-A descriptor, no B descriptor and no pointer-bound cache.\n',
 '// CONTROL: one current-A descriptor is still encoded/passed but is NOT consumed.\n// No B descriptor or pointer-bound cache; retain original45 host encode cost.\n')
replace('// This encode remains inside each ordinary timed host invocation/capture.\n',
 '// CONTROL: deliberately retain one unused current-A encode in each ordinary call/capture.\n// Do not report the encoded map as consumed by this SIMT kernel.\n')

api='''#pragma once
// Private CPU-prepared future control. Not a Plan route or public variant.
#include "megafmm/gemm.h"
namespace megafmm { namespace sw128_simt_control_future41 {
bool wgmma_'''+NEW+'''_can_implement(const Problem&,int configuration=0);
bool wgmma_'''+NEW+'''_full_tiles(const Problem&,const void*,const void*,int configuration=0);
void prepare_wgmma_'''+NEW+'''(const Problem&,int configuration=0);
// One current-A map encode is retained per call solely for the host-cost control.
// The kernel consumes explicit current A via SIMT and never consumes that map.
// Caller owns native compatible A/B/C0/D; no allocation/sync/fallback here.
void launch_wgmma_'''+NEW+'''(const Problem&,const void*,const void*,const float*,float*,
    cudaStream_t,int configuration=0);
} }
'''
Path(str(P)+'base45.cu').write_text(original)
Path(str(P)+'candidate.cu').write_text(source)
Path(str(P)+'api.h').write_text(api)
source_sha=sha(str(P)+'candidate.cu')
delta={'scope':'Distinct private names/API and explicit current-A kernel parameter; replace only rawA TMA/expect/wait transport by original39 vector8 SIMT. SW128/storage/init/maps/guards/B/consumers/wholeP/U unchanged.',
 'parent_path':str(BASE),'parent_sha256':BASE_SHA,'old39_raw_copy_sha256':OLD39_SHA,
 'candidate_path':str(P)+'candidate.cu','candidate_sha256':source_sha,
 'ordered_exact_replacements':edits,'replacement_count':len(edits),
 'new_GPU_evidence':None,'formal_or_public_registration':False}
Path(str(P)+'source_delta.json').write_text(json.dumps(delta,indent=2)+'\n')
Path(str(P)+'diff.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile=str(BASE),tofile=str(P)+'candidate.cu')))
assert 'SM90_TMA_LOAD_2D::copy' not in source and 'arrive_and_expect_tx' not in source
assert 'ClusterTransactionBarrier::wait(' not in source
assert source.count('ClusterTransactionBarrier::init')==original.count('ClusterTransactionBarrier::init')==1
assert source.count('cuTensorMapEncodeTiled(')==original.count('cuTensorMapEncodeTiled(')==1
assert source.count('CUTE_GRID_CONSTANT const SW128SimtControlDescriptor maps')==1
assert 'GMMA::Layout_K_SW128_Atom<Element>{}' in source and 'struct alignas(1024) SharedStorage' in source
assert 'uint64_t(p.lda)<(uint64_t(1)<<39)' in source
print(json.dumps({'status':'GENERATED_CPU_ONLY','candidate_sha256':source_sha,'exact_ordered_replacements':len(edits),'old39_fill_a_name_only_copy':True,'actual_CUDA_build':False,'actual_GPU_launch':False},indent=2))

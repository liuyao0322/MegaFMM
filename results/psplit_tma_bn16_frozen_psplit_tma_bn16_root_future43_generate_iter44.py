#!/usr/bin/env python3
"""CPU-only exact source derivation. This script never compiles or runs CUDA."""
from pathlib import Path
import difflib,hashlib,json,sys
R=Path(__file__).resolve().parent.parent;P=R/'.scratch/psplit_tma_bn16_root_future43_'
BASE=R/'solution/megakernel_wgmma_psplit_selective_tma_a.cu'
BASE_SHA='3c84b3104bf65fa90b6b3373343aa12d10b16bbe4d8c1859cc3e1d6c87958197'
OLD='psplit_selective_tma_a';NEW='psplit_tma_bn16_root_future43'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)):
  mode=args[1] or '';flags=args[2] if len(args)>2 else 0
  writes=isinstance(mode,str) and any(c in mode for c in 'wax+')
  writes|=isinstance(flags,int) and bool(flags&(1|2|64|512|1024))
  if writes and not str(Path(args[0]).absolute()).startswith(str(P)):raise RuntimeError('write outside new BN16 prefix')
sys.addaudithook(guard)
assert H(BASE)==BASE_SHA
original=BASE.read_text();source=original;edits=[]
def replace(old,new,count=1):
 global source
 assert source.count(old)==count,(old[:100],source.count(old),count)
 before=source;positions=[];cursor=0
 for _ in range(count):
  position=before.index(old,cursor);positions.append(position);cursor=position+len(old)
 source=source.replace(old,new)
 edits.append({'old':old,'new':new,'count':count,'old_positions':positions,
  'new_positions':[position+i*(len(new)-len(old)) for i,position in enumerate(positions)],
  'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(source.encode()).hexdigest()})
replace('// Iteration41 selective rawA4 TMA, B transforms remain SIMT; explicit variant45.\n',
 '// Private CPU-prepared BN16 TMA hypothesis derived from frozen45; no public variant.\n')
replace('// BM64/BN32/BK64, producer64/consumer192/192; actual allocation must be proven.\n',
 '// BM64/BN16/BK64, producer40/WG0 128/WG1 128; fresh actual allocation proof required.\n')
replace(OLD,NEW,source.count(OLD))
replace('PSplitSelectiveTmaATraits','PSplitTmaBN16Traits',source.count('PSplitSelectiveTmaATraits'))
replace('SelectiveTmaADescriptor','TmaBN16Descriptor',source.count('SelectiveTmaADescriptor'))
replace('make_selective_tmaA_descriptor','make_tma_bn16_descriptor',source.count('make_selective_tmaA_descriptor'))
replace('#include "megafmm/kernels.h"\n','#include "megafmm/kernels.h"\n#include "psplit_tma_bn16_root_future43_api.h"\n')
replace('namespace megafmm {\n','namespace megafmm {\nnamespace tma_bn16_future43 {\n')
replace('} // namespace megafmm\n','} // namespace tma_bn16_future43\n} // namespace megafmm\n')
replace('static_assert(Cols == 32,"this prototype uses BN32 only");',
 'static_assert(Cols == 16,"this private hypothesis uses BN16 only");')
replace('// BK64/BN32 has256 B vector tasks; the finalB7 onecast DAG is unchanged.\n',
 '// BK64/BN16 has128 B vector tasks; the finalB7 onecast DAG is unchanged.\n')
replace('const int xor_mask = (row&1) | ((row&6)<<2);','const int xor_mask = ((row&1) | ((row&6)<<2)) & 15;')
replace('return product*2048+row*32+(col^xor_mask);','return product*1024+row*16+(col^xor_mask);')
replace('cutlass::arch::warpgroup_reg_dealloc<64>();','cutlass::arch::warpgroup_reg_dealloc<40>();')
replace('static_assert(size(q0) == 16,"each assigned full P has 16 floats/thread");',
 'static_assert(size(q0) == 8,"each assigned full P has 8 floats/thread");')
replace('cutlass::arch::warpgroup_reg_alloc<192>();','cutlass::arch::warpgroup_reg_alloc<128>();')
replace('// Complete all14336 FP32 P words before any generic U load across WGs.',
 '// Complete all7168 FP32 P words before any generic U load across WGs.')
replace('PSplitTmaBN16Traits<Element,32>','PSplitTmaBN16Traits<Element,16>',source.count('PSplitTmaBN16Traits<Element,32>'))
replace('winograd_wgmma_'+NEW+'<Element,32>','winograd_wgmma_'+NEW+'<Element,16>',source.count('winograd_wgmma_'+NEW+'<Element,32>'))
replace('p.m%128 == 0 && p.n%64 == 0 && p.k%128 == 0',
 'p.m%128 == 0 && p.n%32 == 0 && p.k%128 == 0')
api='''#pragma once
// Private CPU-prepared BM64 BN16 BK64 TMA hypothesis; never a public Plan route.
#include "megafmm/gemm.h"
namespace megafmm { namespace tma_bn16_future43 {
bool wgmma_psplit_tma_bn16_root_future43_can_implement(const Problem&,int configuration=0);
bool wgmma_psplit_tma_bn16_root_future43_full_tiles(const Problem&,const void*,const void*,int configuration=0);
void prepare_wgmma_psplit_tma_bn16_root_future43(const Problem&,int configuration=0);
// Native only: M/K multiples128, N multiple32, AB16B, lda/ldb multiples8, A lda<2^39.
// Calls the current-A encoder every invocation/capture, consumes four rawA TMA boxes.
// Caller supplies current AB/C0/D and the common metadata/span/overlap/alignment gate.
// No allocation, synchronization, padding, fallback, or implicit Plan registration.
void launch_wgmma_psplit_tma_bn16_root_future43(const Problem&,const void*,const void*,const float*,float*,cudaStream_t,int configuration=0);
} }
'''
Path(str(P)+'base45.cu').write_text(original)
Path(str(P)+'candidate.cu').write_text(source)
Path(str(P)+'api.h').write_text(api)
delta={'scope':'Private BM64 BN16 BK64 source derivation: widen N tile, B/P/tail coordinates and native N guard; source quotas40/128/128 with common consumer INC before either branch clear; preserve rawA4 TMA/math/order/phase/publication/host encode.',
 'parent_path':str(BASE),'parent_sha256':BASE_SHA,'candidate_path':str(P)+'candidate.cu','candidate_sha256':H(str(P)+'candidate.cu'),
 'replacement_count':len(edits),'ordered_exact_replacements':edits,'actual_CUDA_compilation':None,'actual_GPU':None,'actual_performance':None,'formal_public_registration':False}
Path(str(P)+'source_delta.json').write_text(json.dumps(delta,indent=2)+'\n')
Path(str(P)+'diff.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile=str(BASE),tofile=str(P)+'candidate.cu')))
assert source.count('SM90_TMA_LOAD_2D::copy(')==4
assert source.count('cuTensorMapEncodeTiled(')==1
assert source.count('warpgroup_reg_alloc<128>')==1
assert 'warpgroup_reg_alloc<192>' not in source
print(json.dumps({'status':'CPU_GENERATED_ONLY','candidate_sha256':H(str(P)+'candidate.cu'),'api_sha256':H(str(P)+'api.h'),'ordered_exact_replacements':len(edits),'actual_CUDA_compilation':None,'actual_GPU':None},indent=2))

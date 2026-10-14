#pragma once
// Private CPU-prepared BM64 BN64 BK64 TMA hypothesis; never a public Plan route.
#include "megafmm/gemm.h"
namespace megafmm { namespace tma_bn64_future42 {
bool wgmma_psplit_tma_bn64_root_future42_can_implement(const Problem&,int configuration=0);
bool wgmma_psplit_tma_bn64_root_future42_full_tiles(const Problem&,const void*,const void*,int configuration=0);
void prepare_wgmma_psplit_tma_bn64_root_future42(const Problem&,int configuration=0);
// Native only: M/N/K multiples128, AB16B, lda/ldb multiples8, A lda<2^39.
// Calls the current-A encoder every invocation/capture, consumes four rawA TMA boxes.
// Caller supplies current AB/C0/D and the common metadata/span/overlap/alignment gate.
// No allocation, synchronization, padding, fallback, or implicit Plan registration.
void launch_wgmma_psplit_tma_bn64_root_future42(const Problem&,const void*,const void*,const float*,float*,cudaStream_t,int configuration=0);
} }

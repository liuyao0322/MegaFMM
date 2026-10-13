#pragma once
// Private CPU-prepared future control. Not a Plan route or public variant.
#include "megafmm/gemm.h"
namespace megafmm { namespace sw128_simt_control_future41 {
bool wgmma_psplit_sw128_simt_control_root_future41_can_implement(const Problem&,int configuration=0);
bool wgmma_psplit_sw128_simt_control_root_future41_full_tiles(const Problem&,const void*,const void*,int configuration=0);
void prepare_wgmma_psplit_sw128_simt_control_root_future41(const Problem&,int configuration=0);
// One current-A map encode is retained per call solely for the host-cost control.
// The kernel consumes explicit current A via SIMT and never consumes that map.
// Caller owns native compatible A/B/C0/D; no allocation/sync/fallback here.
void launch_wgmma_psplit_sw128_simt_control_root_future41(const Problem&,const void*,const void*,const float*,float*,
    cudaStream_t,int configuration=0);
} }

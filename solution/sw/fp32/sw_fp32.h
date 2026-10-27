#pragma once
#include "internal.h"

namespace megafmm::detail {
bool sw_fp32_tma_problem(const Problem& p);
bool sw_fp32_tma_binding(const Problem& p,const void* A,const void* B);
SWResources setup_sw_fp32_tma(int bk);
void run_sw_fp32_config(const Problem& p,const float* A,const float* B,const float* C,
                       float* D,cudaStream_t stream,int bk,bool tma,int max_blocks=65535);
SWResources setup_sw_fp32_guarded(int bk);
bool sw_fp32_guarded_tail(const Problem& p,int bk);
SWResources setup_sw_fp32(const Problem& p,int bk);
void run_sw_fp32_guarded_config(const Problem& p,const float* A,const float* B,const float* C,
                       float* D,cudaStream_t stream,int bk,bool tma,int max_blocks,bool guarded);
}

#pragma once
#include "megafmm/gemm.h"
namespace megafmm::detail {
void scale_output(const Problem& p, const float* C0, float* D, cudaStream_t stream);
void run_sw_multisize(const Problem& p,const void* A,const void* B,const float* C0,float* D,cudaStream_t stream);
struct SWState;
std::shared_ptr<SWState> make_sw_pipeline(const Problem& p);
void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream);
}

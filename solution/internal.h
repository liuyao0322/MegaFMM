#pragma once
#include "megafmm/gemm.h"
namespace megafmm::detail {
void scale_output(const Problem& p, const float* C0, float* D, cudaStream_t stream);
void run_sw_multisize(const Problem& p,const void* A,const void* B,const float* C0,float* D,cudaStream_t stream);
bool dense_agreement(const Problem& p,const float* D,const float* reference,cudaStream_t stream);
const char* hopper_candidate_name(int id);
bool hopper_dense(int id,const Problem& p,const void* A,const void* B,const float* C0,float* D,
                  void* workspace,int sm_count,cudaStream_t stream,bool launch);
struct SWState;
std::shared_ptr<SWState> make_sw_pipeline(const Problem& p);
void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream);
}

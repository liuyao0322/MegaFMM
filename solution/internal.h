#pragma once
#include "megafmm/gemm.h"
namespace megafmm::detail {
void scale_output(const Problem& p, const float* C0, float* D, cudaStream_t stream);
void run_sw_multisize(const Problem& p,const void* A,const void* B,const float* C0,float* D,cudaStream_t stream);
constexpr int HopperCandidateCount=85;
bool wmma_dense(int bk,const Problem& p,const void* A,const void* B,const float* C,float* D,
                 cudaStream_t stream,int operation);
bool mma_even_dense(int bk,const Problem& p,const void* A,const void* B,const float* C,float* D,
                    cudaStream_t stream,int operation);
bool ampere_dense(int id,const Problem& p,const void* A,const void* B,const float* C,float* D,
                  cudaStream_t stream,int operation);
size_t hopper_workspace(const Problem& p,int id,int sm_count=0);
size_t hopper_streamk_workspace(const Problem& p,int id,int sm_count);
bool padded_hopper(int base,const Problem& p,const void* A,const void* B,const float* C,float* D,
                    void* workspace,int sm_count,cudaStream_t stream,int operation);
bool dense_agreement(const Problem& p,const float* D,const float* reference,cudaStream_t stream);
const char* hopper_candidate_name(int id);
bool hopper_dense(int id,const Problem& p,const void* A,const void* B,const float* C0,float* D,
                  void* workspace,int sm_count,cudaStream_t stream,int operation);
struct SWState;
std::shared_ptr<SWState> make_sw_pipeline(const Problem& p);
void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream);
}

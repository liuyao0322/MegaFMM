#pragma once
#include "megafmm/gemm.h"
namespace megafmm::detail {
struct SWSevenState;
void execute_seven_leaf(Plan& p,const void* A,const void* B,const float* C,float* D,void* workspace,cudaStream_t stream,bool tuning);
bool seven_eligible(const Problem& p,size_t limit);
std::shared_ptr<SWSevenState> make_seven(const Problem& p);
size_t seven_workspace(const SWSevenState& state);
void run_seven(const SWSevenState& s,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream);
void tune_seven(SWSevenState& s,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream);
}

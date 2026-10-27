#pragma once
#include "internal.h"

namespace megafmm::detail::sw_fp32_tiny {
inline bool eligible(const Problem& p) {
  return p.dtype==DType::FP32 && p.m==1 && p.n==1 && p.k==1;
}
SWResources setup(int id);
std::string name(int id);
void run(const Problem& p,int id,const float* A,const float* B,const float* C,
         float* D,cudaStream_t stream);
}

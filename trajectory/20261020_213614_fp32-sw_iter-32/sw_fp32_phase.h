#pragma once
#include "internal.h"
#include <array>

namespace megafmm::detail::sw_fp32_phase {
struct Config {int id,bm,bn,rm,rn;};
inline constexpr std::array<Config,5> configurations{{
  {0,32,64,4,16},{1,64,32,8,8},{2,32,32,4,8},
  {3,32,16,4,4},{4,16,32,2,8}
}};
bool problem(const Problem& p);
bool binding(const Problem& p,const void* A,const void* B);
SWResources setup(int id);
std::string name(int id);
void run(const Problem& p,int id,const float* A,const float* B,const float* C,
         float* D,cudaStream_t stream,int max_blocks=65535);
}

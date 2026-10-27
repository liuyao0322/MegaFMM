#pragma once
#include "internal.h"

// Compile-time parameters of the restored SW specialization. The generator
// emits each configured tuple; this interface selects the emitted instance.
namespace megafmm::detail::sw_parameterized {
struct Config {
  int m,n,k,bn,bk,stages,pack_threads,pack_blocks;
};
std::vector<int> configs(const Problem& p);
Config describe(int id);
size_t workspace(const Problem& p,int id);
SWResources setup(const Problem& p,int id);
void run(const Problem& p,int id,const void* A,const void* B,float* D,
         void* workspace,cudaStream_t stream);
}

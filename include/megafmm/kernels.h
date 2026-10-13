#pragma once
#include "megafmm/gemm.h"
namespace megafmm {
size_t hopper_workspace_bytes(const Problem& p, int variant=0);
bool hopper_can_implement(const Problem& p, int variant=0);
void launch_hopper_gemm(const Problem& p, const void* A, const void* B,
                       const float* C0, float* D, void* workspace,
                       cudaStream_t stream, int variant=0);
// Seven compact product operands and FP32 outputs; tile 0=64x64x32, 1=32x64x32.
bool small_batched_can_implement(const Problem& leaf, int tile_variant=0);
size_t small_batched_workspace_bytes(const Problem& leaf, int tile_variant=0);
void launch_small_batched_gemm(const Problem& leaf, const void* A7,
                              const void* B7, float* P, cudaStream_t stream,
                              int tile_variant=0);
bool megakernel_can_implement(const Problem& p, int variant=0);
// One-time launch configuration; called during planning, never in steady-state run.
void prepare_megakernel(const Problem& p, int variant=0);
void launch_megakernel(const Problem& p, const void* A, const void* B,
                      const float* C0, float* D, cudaStream_t stream,
                      int variant=0);
}

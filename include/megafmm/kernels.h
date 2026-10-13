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
// Fused warp mma.sync path: variants 4/5/6 use one/two/four owner warps per CTA.
bool small_mma_can_implement(const Problem& p, int variant=4);
void prepare_small_mma(const Problem& p, int variant=4);
void launch_small_mma(const Problem& p, const void* A, const void* B,
                      const float* C0, float* D, cudaStream_t stream,
                      int variant=4);
// Paired final operand conversions, one owner warp: 7=general, 8=full tiles
// when parent M/N/K are divisible by 32, with general paired fallback for tails.
bool packed_mma_can_implement(const Problem& p, int variant=7);
void prepare_packed_mma(const Problem& p, int variant=7);
void launch_packed_mma(const Problem& p, const void* A, const void* B,
                       const float* C0, float* D, cudaStream_t stream,
                       int variant=7);
// Product-parallel M1: seven product warps retain complete P tiles, then fixed U.
bool small_product_can_implement(const Problem& p, int variant=9);
void prepare_small_product(const Problem& p, int variant=9);
void launch_small_product(const Problem& p, const void* A, const void* B,
                          const float* C0, float* D, cudaStream_t stream,
                          int variant=9);
// Complete-tile product warps, with variant 9 as the generic tail fallback.
bool small_product_fast_can_implement(const Problem& p, int variant=10);
void prepare_small_product_fast(const Problem& p, int variant=10);
void launch_small_product_fast(const Problem& p, const void* A, const void* B,
                               const float* C0, float* D, cudaStream_t stream,
                               int variant=10);
}

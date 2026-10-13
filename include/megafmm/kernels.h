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
// One warp owns a complete 16x8 tile; Plan uses packed variant 7 for tails.
bool mma16x8_can_implement(const Problem& p, int variant=11);
bool mma16x8_full_tiles(const Problem& p, const void* A, const void* B);
void prepare_mma16x8(const Problem& p, int variant=11);
void launch_mma16x8(const Problem& p, const void* A, const void* B,
                    const float* C0, float* D, cudaStream_t stream,
                    int variant=11);
// Same complete 16x8 tiles with four-element B vectors from all 32 lanes.
bool mma16x8_vec4_can_implement(const Problem& p, int variant=12);
bool mma16x8_vec4_full_tiles(const Problem& p, const void* A, const void* B);
void prepare_mma16x8_vec4(const Problem& p, int variant=12);
void launch_mma16x8_vec4(const Problem& p, const void* A, const void* B,
                        const float* C0, float* D, cudaStream_t stream,
                        int variant=12);
// Eight-CTA product cluster, with variant 9 for generic tails and strides.
bool small_product_cluster_can_implement(const Problem& p, int variant=13);
void prepare_small_product_cluster(const Problem& p, int variant=13);
void launch_small_product_cluster(const Problem& p, const void* A, const void* B,
                                  const float* C0, float* D, cudaStream_t stream,
                                  int variant=13);
// Float2 epilogue; Plan rechecks actual D and active C0 alignment on every run.
bool mma16x8_epilogue_can_implement(const Problem& p, int variant=14);
bool mma16x8_epilogue_full_tiles(const Problem& p, const void* A, const void* B,
                                const float* C0, const float* D);
void prepare_mma16x8_epilogue(const Problem& p, int variant=14);
void launch_mma16x8_epilogue(const Problem& p, const void* A, const void* B,
                            const float* C0, float* D, cudaStream_t stream,
                            int variant=14);
// Identity alpha/beta float2 output; Plan uses 14, 12, then 7 for other bindings.
bool mma16x8_identity_can_implement(const Problem& p, int variant=15);
bool mma16x8_identity_full_tiles(const Problem& p, const void* A, const void* B,
                                const float* C0, const float* D);
void prepare_mma16x8_identity(const Problem& p, int variant=15);
void launch_mma16x8_identity(const Problem& p, const void* A, const void* B,
                            const float* C0, float* D, cudaStream_t stream,
                            int variant=15);
// Identity output with a 24-word raw input bank loaded one K tile ahead.
bool mma16x8_prefetch_can_implement(const Problem& p, int variant=16);
bool mma16x8_prefetch_full_tiles(const Problem& p, const void* A, const void* B,
                                const float* C0, const float* D);
void prepare_mma16x8_prefetch(const Problem& p, int variant=16);
void launch_mma16x8_prefetch(const Problem& p, const void* A, const void* B,
                            const float* C0, float* D, cudaStream_t stream,
                            int variant=16);
}

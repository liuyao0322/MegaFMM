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
// CTA32x64x64: 7=three-stage Multistage, 8=two-stage Pipelined.
// The two schedules also use different global-load paths in pinned CUTLASS.
bool small_batched_k64_can_implement(const Problem& leaf, int variant=7);
size_t small_batched_k64_workspace_bytes(const Problem& leaf, int variant=7);
void launch_small_batched_k64(const Problem& leaf, const void* A7,
                             const void* B7, float* P, cudaStream_t stream,
                             int variant=7);
// Original A/B transforms fused into uniform seven compact full-K P slots.
// Internal config0/1 maps to Hopper9/10; existing combine is a second launch.
// Plan reserves P only and uses ordinary GEMM for incompatible metadata/bindings.
bool fused_leaf_can_implement(const Problem& p, int config=0);
bool fused_leaf_full_tiles(const Problem& p, const void* A, const void* B,
                           const float* P, int config=0);
size_t fused_leaf_product_workspace_bytes(const Problem& p, int config=0);
void prepare_fused_leaf(const Problem& p, int config=0);
void launch_fused_leaf(const Problem& p, const void* A, const void* B, float* P,
                       cudaStream_t stream, int config=0);
// Fresh uniform A7/B7 input slots, one matched complete-P leaf, then combine.
// Internal configs0/1 map to Hopper11/12; complete run uses three launches.
struct PackedLeafMetadata {
  int64_t mh, nh, kh;
  int64_t a_row_stride, b_row_stride, p_row_stride;
  int64_t a_product_stride, b_product_stride, p_product_stride;
  size_t a_bytes, b_bytes, p_bytes, compact_workspace_bytes;
};
bool packed_leaf_can_implement(const Problem& p, int config=0);
PackedLeafMetadata packed_leaf_metadata(const Problem& p, int config=0);
bool packed_leaf_original_input_guard(const Problem& p, const void* A, const void* B, int config=0);
bool packed_leaf_full_tiles(const Problem& p, const void* A7, const void* B7,
                            const float* P, int config=0);
size_t packed_leaf_workspace_bytes(const Problem& p, int config=0);
void prepare_packed_leaf(const Problem& p, int config=0);
void launch_packed_leaf(const Problem& p, const void* A7, const void* B7, float* P,
                        cudaStream_t stream, int config=0);
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
// Raw quadrant cp.async double buffering; noncomplete inputs use packed 7.
// General alpha/beta and scalar FP32 C0/D retain four-byte alignment support.
bool small_cpasync_can_implement(const Problem& p, int variant=17);
bool small_cpasync_full_tiles(const Problem& p, const void* A, const void* B);
void prepare_small_cpasync(const Problem& p, int variant=17);
void launch_small_cpasync(const Problem& p, const void* A, const void* B,
                          const float* C0, float* D, cudaStream_t stream,
                          int variant=17);
// PLAN M2: independently complete each K16 product and apply fourteen RN
// contributions to four long C accumulators; noncomplete inputs use packed 7.
// This FP32 stage order differs from M1's complete-P accumulation and U DAG.
bool small_stage_can_implement(const Problem& p, int variant=18);
bool small_stage_full_tiles(const Problem& p, const void* A, const void* B);
void prepare_small_stage(const Problem& p, int variant=18);
void launch_small_stage(const Problem& p, const void* A, const void* B,
                        const float* C0, float* D, cudaStream_t stream,
                        int variant=18);
// Identity 8x8 logical output owner using native 16x8 MMA and raw prefetch.
// Plan rechecks current pointers, then falls back through 16/15/14/12/7.
bool mma8x8_prefetch_can_implement(const Problem& p, int variant=19);
bool mma8x8_prefetch_full_tiles(const Problem& p, const void* A, const void* B,
                              const float* C0, const float* D);
void prepare_mma8x8_prefetch(const Problem& p, int variant=19);
void launch_mma8x8_prefetch(const Problem& p, const void* A, const void* B,
                          const float* C0, float* D, cudaStream_t stream,
                          int variant=19);
// Direct native fragments: 20 owns 16x8, 21 owns 8x8 with lower A words zero.
// Identity output only; each current binding falls back through 16/15/14/12/7.
bool direct_mma_can_implement(const Problem& p, int variant=20);
bool direct_mma_full_tiles(const Problem& p, const void* A, const void* B,
                           const float* C0, const float* D, int variant=20);
void prepare_direct_mma(const Problem& p, int variant=20);
void launch_direct_mma(const Problem& p, const void* A, const void* B,
                       const float* C0, float* D, cudaStream_t stream,
                       int variant=20);
// Original vector64 inputs routed by full-warp butterfly/PRMT to native MMA.
// 22 owns 16x8; 23 owns 8x8 with lower A zero. Each falls back through 16/15/14/12/7.
bool coalesced_mma_can_implement(const Problem& p, int variant=22);
bool coalesced_mma_full_tiles(const Problem& p, const void* A, const void* B,
                            const float* C0, const float* D, int variant=22);
void prepare_coalesced_mma(const Problem& p, int variant=22);
void launch_coalesced_mma(const Problem& p, const void* A, const void* B,
                         const float* C0, float* D, cudaStream_t stream,
                         int variant=22);
// Original19 identity owners with an exact parent-K128, four-stage leaf bound.
// Policy24/25 choose only outer-loop no-unroll/full-unroll; other bindings use
// the unchanged19 ->16/15/14/12/7 Plan fallback chain.
enum class K128OuterLoop { FixedBoundNoUnroll, FixedBoundFullUnroll };
bool mma8x8_k128_specialized_can_implement(const Problem& p);
bool mma8x8_k128_specialized_full_tiles(const Problem& p, const void* A,
    const void* B, const float* C0, const float* D);
void prepare_mma8x8_k128_specialized(const Problem& p);
void launch_mma8x8_k128_specialized(const Problem& p, const void* A,
    const void* B, const float* C0, float* D, cudaStream_t stream,
    K128OuterLoop control);

// Cooperative complete-P M1:26 has32x16/64threads,27 has32x32/128threads.
// Shared transformed A/B operands are produced once per CTA/K16 stage.
// Native metadata/current AB16 guards fail to packed7; scalar D/C0 supported.
bool cooperative_mma_can_implement(const Problem& p, int variant=26);
bool cooperative_mma_full_tiles(const Problem& p, const void* A, const void* B,
                                int variant=26);
void prepare_cooperative_mma(const Problem& p, int variant=26);
void launch_cooperative_mma(const Problem& p, const void* A, const void* B,
                            const float* C0, float* D, cudaStream_t stream,
                            int variant=26);

// Dedicated producer/consumer WGMMA M1, two final operand banks, one launch.
// Internal configs0/1 map to policies28/29 with BM64,BN16/32,BK32.
// Raw APIs accept native metadata/bindings only; Plan chooses packed7/Dense.
bool wgmma_pipeline_can_implement(const Problem& p, int configuration=0);
bool wgmma_pipeline_full_tiles(const Problem& p, const void* A, const void* B,
                               int configuration=0);
void prepare_wgmma_pipeline(const Problem& p, int configuration=0);
void launch_wgmma_pipeline(const Problem& p, const void* A, const void* B,
                           const float* C0, float* D, cudaStream_t stream,
                           int configuration=0);

// Three-bank vector producer wait1 consumer, native configs0/1 map to34/35.
// Full/empty128 participation preserved; modulo3 phases and three tail acquires.
// P3+4 independent consumer WGs; both native variants use BN32.
// Native P3+4 BK64; parent K must be a multiple of128, same N32 atom.
bool wgmma_psplit_bk64_can_implement(const Problem& p, int configuration=0);
bool wgmma_psplit_bk64_full_tiles(const Problem& p, const void* A, const void* B, int configuration=0);
void prepare_wgmma_psplit_bk64(const Problem& p, int configuration=0);
void launch_wgmma_psplit_bk64(const Problem& p, const void* A, const void* B,
    const float* C0, float* D, cudaStream_t stream, int configuration=0);
bool wgmma_psplit_can_implement(const Problem& p, int configuration=0);
bool wgmma_psplit_full_tiles(const Problem& p, const void* A, const void* B, int configuration=0);
void prepare_wgmma_psplit(const Problem& p, int configuration=0);
void launch_wgmma_psplit(const Problem& p, const void* A, const void* B,
    const float* C0, float* D, cudaStream_t stream, int configuration=0);
bool wgmma_psplit_asymmetric_can_implement(const Problem& p, int configuration=0);
bool wgmma_psplit_asymmetric_full_tiles(const Problem& p, const void* A, const void* B, int configuration=0);
void prepare_wgmma_psplit_asymmetric(const Problem& p, int configuration=0);
void launch_wgmma_psplit_asymmetric(const Problem& p, const void* A, const void* B,
    const float* C0, float* D, cudaStream_t stream, int configuration=0);

bool wgmma_wait1_three_stage_can_implement(const Problem& p, int configuration=0);
bool wgmma_wait1_three_stage_full_tiles(const Problem& p, const void* A, const void* B,
                                       int configuration=0);
void prepare_wgmma_wait1_three_stage(const Problem& p, int configuration=0);
void launch_wgmma_wait1_three_stage(const Problem& p, const void* A, const void* B,
                                   const float* C0, float* D, cudaStream_t stream,
                                   int configuration=0);

// Vector producer wait1 consumer, native configs0/1 map to32/33.
// Read/retire states delay bank reuse; final wait0 before any generic P access.
bool wgmma_wait1_can_implement(const Problem& p, int configuration=0);
bool wgmma_wait1_full_tiles(const Problem& p, const void* A, const void* B,
                           int configuration=0);
void prepare_wgmma_wait1(const Problem& p, int configuration=0);
void launch_wgmma_wait1(const Problem& p, const void* A, const void* B,
                       const float* C0, float* D, cudaStream_t stream,
                       int configuration=0);

// TMA raw aliases into two final banks, native-only configs0/1 map to30/31.
// Every native host run encodes current A/B then launches one kernel.
bool wgmma_tma_alias_can_implement(const Problem& p, int configuration=0);
bool wgmma_tma_alias_full_tiles(const Problem& p, const void* A, const void* B,
                                int configuration=0);
void prepare_wgmma_tma_alias(const Problem& p, int configuration=0);
void launch_wgmma_tma_alias(const Problem& p, const void* A, const void* B,
                            const float* C0, float* D, cudaStream_t stream,
                            int configuration=0);

}

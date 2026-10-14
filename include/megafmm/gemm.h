#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace megafmm {
enum class DType { BF16, FP16 };
enum class Algorithm { Auto, Dense, WinogradLt, WinogradHopper, Megakernel };
struct Problem {
  int64_t m=0, n=0, k=0;
  int64_t lda=0, ldb=0, ldc=0, ldd=0;
  bool trans_a=false, trans_b=false;
  DType dtype=DType::BF16;
  float alpha=1.0f, beta=0.0f;
};
struct Policy {
  Algorithm algorithm=Algorithm::Auto;
  // Caller declaration; run() does not scan input contents or synchronize.
  bool certified_range=false;
  int levels=1;
  size_t workspace_limit=size_t(8)<<30;
  int heuristic_candidates=32;
  // Lt: 0=seven separate products using views, 1=uniform slots and one batch.
  int lt_variant=0;
  // Hopper: 0=128x128, 1=128x256, 2=64x128 warp-specialized,
  // 3=source-free 128x256, 4=source-free with B multicast,
  // 5/6=uniform seven-batch SM80 leaves with CTA64x64/CTA32x64,
  // 7/8=CTA32x64x64 with three-stage Multistage/two-stage Pipelined.
  // 9/10=fused original-input 32x32/32x64 products + required combine (two launches).
  // Native uses P-only workspace; incompatible metadata/input bindings use Dense.
  // 11/12=matched compact-input 32x32/32x64: fresh uniform pack + leaf + combine.
  int hopper_variant=0;
  // Mega: 0/1=seven live P tiles, 2=three live P/U tiles,
  // 3=full-tile beta-zero two-stage pipeline (otherwise variant 2),
  // 4/5/6=warp mma.sync with one/two/four owners per 32/64/128-thread CTA,
  // 7=paired operand conversions, 8=paired full tiles (otherwise variant 7),
  // 9=product-parallel M1 with seven product warps and final fixed U merge,
  // 10=complete-tile product warps (otherwise variant 9),
  // 11=complete 16x8 warp MMA tiles (otherwise packed variant 7),
  // 12=same complete tiles with four-element B vectors (otherwise variant 7),
  // 13=eight-CTA product cluster (otherwise variant 9),
  // 14=float2 epilogue (otherwise variant 12 or packed variant 7),
  // 15=alpha-one beta-zero float2 output (otherwise variant 14, 12, or 7),
  // 16=identity output with raw register prefetch (otherwise variant 15/14/12/7),
  // 17=raw input cp.async double buffering (otherwise packed variant 7),
  // 18=PLAN M2 per-K16 product contributions into four long C accumulators
  //    with distinct FP32 stage order (otherwise packed M1 variant 7),
  // 19=identity 8x8 output owners with raw register prefetch
  //    (otherwise variants 16/15/14/12/7 using current pointer guards),
  // 20/21=direct native register operands with 16x8/8x8 identity owners
  //    (each otherwise uses 16/15/14/12/7; no mutual fallback),
  // 22/23=vector64 original inputs with butterfly/PRMT native register routing
  //    and 16x8/8x8 identity owners (each otherwise 16/15/14/12/7; no mutual fallback).
  // 24/25=original19 parent-K128 fixed four-stage no-unroll/full-unroll
  //    (otherwise original19 then16/15/14/12/7; no mutual fallback).
  // 26/27=cooperative32x16/32x32 transformed operand tiles, complete seven P
  //    then fixed U (otherwise packed7; common input binding guard uses Dense).
  // 28/29=dedicated producer/consumer M1 WGMMA64x16/64x32 K32, two banks,
  // full seven P and fixed U in one launch (otherwise packed7; common AB16
  // current binding failure uses Dense). No global S/T/P or caller workspace.
  // 30/31=TMA raw aliases in final banks64x16/64x32, one launch; two
  // current-pointer tensor-map encodes occur in each run/capture, not setup.
  // Same packed7/Dense fallbacks; no global S/T/P or native workspace.
  // 32/33=vector producer64x16/64x32 with two WGMMA groups in flight,
  // wait1 retires only the old bank; tail wait0 precedes fixed U. One launch.
  // Original precision and packed7/Dense guards; no native matrix workspace.
  // 34/35=three-bank vector64x16/64x32 wait1; old bank retirement
  // and tail wait0 remain, original precision and packed7/Dense guards.
  // 36/37=P3+4 two-consumer WGs, both64x32; quotas64/96/96 or64/80/96.
  // Original two-bank transforms and full7P fixedU, one launch, no globalSTP.
  // 38=P3+4 BK64 two-bank,64x32; quotas64/96/96, parent K%128==0.
  // 39=RS-A P3+4 BK64,64x32: raw A4 shared/final B7 shared, quotas64/192/192.
  // Complete onecast A transform in registers, whole7P then fixedU, no globalSTP.
  // 40=RS-A P3+4 BK32,64x32; rawA4/finalB7 shared, quotas64/192/192, parentK%64.
  // 41=hybrid S RS-A P3+4 BK32,64x32; shared finalS1/FP32S2 planes, rawA4/finalB7.
  // 42=RS-A BN16 BK32 P3+4, rawA4/finalB7; initial quotas48/96/96 and minCTAs2.
  // 43=SS BN16 BK32 P3+4, finalA7/finalB7; initial quotas64/80/80 and minCTAs2.
  // WholeK/fixedU/onecast and currentAB16/parentK%64; actual fresh gate required.
  // 44=BK64 RS-A single owning stageA; perP fourK16 commit/wait0 before A reuse.
  // 45=BK64 RS-A rawA4 TMA/finalB7 SIMT; one currentA map per run, A lda<2^39.
  // 46=SW128 SIMT rawA control; same one unused currentA map encode and native domain.
  // 47=BN64 BK64 RS-A rawA4 TMA/finalB7 SIMT; native parentM/N/K multiples128.
  // 48=BN16 BK64 RS-A rawA4 TMA/finalB7 SIMT; parentM/K%128, parentN%32.
  // 49=BN16 BK64 selectiveA TMA minCTAs2 hypothesis; fresh role quotas40/88/112.
  // 50=BN16 BK64 two consecutive K16 inputs/window, two wait0 windows; fresh q40/96/96/min2.
  int mega_variant=0;
};
class Plan {
 public:
  struct Impl;
  ~Plan();
  Plan(Plan&&) noexcept;
  Plan& operator=(Plan&&) noexcept;
  Plan(const Plan&)=delete;
  Plan& operator=(const Plan&)=delete;
  size_t workspace_bytes() const;
  std::string algorithm_name() const;
  // CPU-only metadata for the actual bound Dense/WinogradLt operators. No GPU
  // work or synchronization; call outside capture and steady-state timing.
  std::string lt_algorithm_config(const void* A, const void* B, const float* C0,
                                 float* D, void* workspace) const;
  const Problem& problem() const;
  explicit Plan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
std::unique_ptr<Plan> create_plan(Problem problem, Policy policy={});
// Tuning is explicit and outside steady-state timing. It may synchronize.
void tune(Plan& plan, const void* A, const void* B, const float* C0,
          float* D, void* workspace, cudaStream_t stream=nullptr);
// run() performs no allocation or host synchronization. Workspace is caller-owned.
void run(const Plan& plan, const void* A, const void* B, const float* C0,
         float* D, void* workspace, cudaStream_t stream=nullptr);
const char* dtype_name(DType dtype);
}

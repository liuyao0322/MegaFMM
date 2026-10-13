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
  // 5/6=uniform seven-batch SM80 leaves with CTA64x64/CTA32x64.
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
  // 16=identity output with raw register prefetch (otherwise variant 15/14/12/7).
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

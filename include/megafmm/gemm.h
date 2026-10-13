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
  // Hopper: 0=128x128, 1=128x256, 2=64x128 pingpong,
  // 3=source-free 128x256, 4=source-free with B multicast.
  int hopper_variant=0;
  // Mega: 0/1=seven live P tiles, 2=three live P/U tiles,
  // 3=full-tile beta-zero two-stage pipeline (otherwise variant 2).
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

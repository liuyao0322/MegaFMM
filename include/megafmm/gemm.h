#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace megafmm {
enum class DType { BF16, FP16 };
// Experimental forced SW: specialized contiguous 1024^3 pipeline, otherwise
// one-level Winograd with logical even padding, transpose/stride/alpha/beta.
// Auto remains independently tuned Dense; SW precision is not certified.
enum class Algorithm { Auto, Dense, SWPipeline=3, Tuned=4 };
struct Problem {
  int64_t m=0, n=0, k=0;
  int64_t lda=0, ldb=0, ldc=0, ldd=0;
  bool trans_a=false, trans_b=false;
  DType dtype=DType::BF16;
  float alpha=1.0f, beta=0.0f;
};
struct Policy {
  Algorithm algorithm=Algorithm::Auto;
  size_t workspace_limit=size_t(8)<<30;
  int heuristic_candidates=32;
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
  // CPU-only metadata for Dense or the experimental SW configuration. No GPU
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

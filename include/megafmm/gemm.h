#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace megafmm {
enum class DType { BF16, FP16, FP32 };
constexpr size_t dtype_size(DType dtype) { return dtype == DType::FP32 ? 4 : 2; }
// SWPipeline uses shared seven-product kernels for all dimensions: a zero-workspace
// fallback, plus one padded Hopper BF16/FP16 path when bindings and budget allow.
// FP32 remains full FP32 throughout. SWTuned measures bounded compile-time
// parameter registries for the packed FP32 and native BF16/FP16 Hopper templates.
// SWTuned measures configurations for each problem; no size-specific kernels.
// Auto/Dense use cuBLASLt. Tuned is retained as a compatibility alias for that
// same explicitly tuned cuBLASLt implementation.
enum class Algorithm { Auto, Dense, SWPipeline=3, Tuned=4, SWTuned=5 };
struct Problem {
  int64_t m=0, n=0, k=0;
  int64_t lda=0, ldb=0, ldc=0, ldd=0;
  bool trans_a=false, trans_b=false;
  DType dtype=DType::BF16;
  float alpha=1.0f, beta=0.0f;
};
// SWTuned searches only seven-product SW. Explicit tune() is required before run().
// Default selection uses measured latency; original-input FP64 error is reported.
// Opt into the historical precision gate with sw_tuning_allow_inaccurate=false.
struct SWCandidateResult {
  int id=-1;
  std::string config, status;
  size_t workspace_bytes=0, shared_bytes=0;
  int registers_per_thread=0, resident_ctas=0;
  double physical_work_ratio=0, model_score=0;
  double relative_error=0, gate_limit=0;
  bool numerical_pass=false;
  std::vector<double> precision_errors, precision_limits;
  std::vector<float> ranking_ms, confirmation_ms;
};
struct Policy {
  Algorithm algorithm=Algorithm::Auto;
  size_t workspace_limit=size_t(8)<<30;
  int heuristic_candidates=32;
  // Explicit Tuned setup may duplicate inputs to model a rotating working set.
  // Scratch copies are freed before tune returns; run retains no input values.
  int tuning_input_sets=1;
  // SW only: 0 searches all legal configurations; otherwise model shortlist size.
  int sw_tuning_candidates=0;
  // Kept for source compatibility; true is rejected (ordinary launch only).
  bool sw_tuning_graph=false;
  // Rank all finite SW candidates regardless of the historical precision threshold.
  // Set false to require that threshold; numerical_pass always reports its verdict.
  bool sw_tuning_allow_inaccurate=true;
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
  const std::vector<SWCandidateResult>& sw_tuning_results() const;
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

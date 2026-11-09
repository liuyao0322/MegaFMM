#include "megafmm/gemm.h"
#include "internal.h"
#include "sw_fp32.h"
#include "sw_fp32_batch.h"
#include <cublasLt.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace megafmm {
namespace {
void cuda_check(cudaError_t status, const char* action) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}
void lt_check(cublasStatus_t status, const char* action) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(action) + ": cuBLAS status " + std::to_string(int(status)));
}
size_t checked_mul(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b)
    throw std::invalid_argument("matrix size or workspace size overflows size_t");
  return a * b;
}
size_t checked_add(size_t a, size_t b) {
  if (a > std::numeric_limits<size_t>::max() - b)
    throw std::invalid_argument("matrix size or workspace size overflows size_t");
  return a + b;
}
size_t matrix_bytes(int64_t rows, int64_t cols, int64_t ld, size_t item) {
  if (!rows || !cols) return 0;
  return checked_mul(checked_add(checked_mul(size_t(rows - 1), size_t(ld)), size_t(cols)), item);
}
bool overlaps(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
  if (!a_bytes || !b_bytes) return false;
  const uintptr_t pa = reinterpret_cast<uintptr_t>(a), pb = reinterpret_cast<uintptr_t>(b);
  if (pa > std::numeric_limits<uintptr_t>::max() - a_bytes ||
      pb > std::numeric_limits<uintptr_t>::max() - b_bytes)
    throw std::invalid_argument("address range overflows uintptr_t");
  return pa < pb + b_bytes && pb < pa + a_bytes;
}
void require_alignment(const void* p, uintptr_t alignment, const char* name) {
  if (reinterpret_cast<uintptr_t>(p) % alignment)
    throw std::invalid_argument(std::string(name) + " has invalid element alignment");
}
void validate_problem(Problem& p) {
  if (p.m < 0 || p.n < 0 || p.k < 0)
    throw std::invalid_argument("M, N and K must be nonnegative");
  if (p.lda < 0 || p.ldb < 0 || p.ldc < 0 || p.ldd < 0)
    throw std::invalid_argument("leading dimensions must be nonnegative");
  if (p.dtype != DType::BF16 && p.dtype != DType::FP16 && p.dtype != DType::FP32)
    throw std::invalid_argument("unsupported input dtype");
  const int64_t ac = p.trans_a ? p.m : p.k, bc = p.trans_b ? p.k : p.n;
  if (!p.lda) p.lda = ac;
  if (!p.ldb) p.ldb = bc;
  if (!p.ldc) p.ldc = p.n;
  if (!p.ldd) p.ldd = p.n;
  if (p.lda < ac || p.ldb < bc || p.ldc < p.n || p.ldd < p.n)
    throw std::invalid_argument("leading dimension is smaller than the physical row width");
  matrix_bytes(p.trans_a ? p.k : p.m, ac, p.lda, dtype_size(p.dtype));
  matrix_bytes(p.trans_b ? p.n : p.k, bc, p.ldb, dtype_size(p.dtype));
  matrix_bytes(p.m, p.n, p.ldc, 4);
  matrix_bytes(p.m, p.n, p.ldd, 4);
  if (p.m > std::numeric_limits<int>::max() || p.n > std::numeric_limits<int>::max() ||
      p.k > std::numeric_limits<int>::max())
    throw std::invalid_argument("dimensions exceed the supported GEMM backend range");
}
struct LtOp {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> candidates;
  std::vector<std::array<uint32_t, 4>> alignments;
  bool robust_small_tuning = false;
  bool strategy_reranked = false;
  int strategy_input_sets = 1;
  std::vector<float> candidate_tuning_ms;
  bool tuned = false;
  float selected_tuning_ms = std::numeric_limits<float>::quiet_NaN();
  size_t selected = 0;
  float alpha = 1.0f, beta = 0.0f;
  LtOp() = default;
  LtOp(const LtOp&) = delete;
  LtOp& operator=(const LtOp&) = delete;
  ~LtOp() {
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (d) cublasLtMatrixLayoutDestroy(d);
    if (operation) cublasLtMatmulDescDestroy(operation);
  }
  size_t scratch_bytes() const {
    size_t result = 0;
    for (const auto& candidate : candidates) result = std::max(result, candidate.workspaceSize);
    return result;
  }
};
void make_layout(cublasLtMatrixLayout_t* layout, cudaDataType_t dtype,
                 uint64_t rows, uint64_t cols, int64_t ld) {
  lt_check(cublasLtMatrixLayoutCreate(layout, dtype, rows, cols, ld), "create matrix layout");
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  lt_check(cublasLtMatrixLayoutSetAttribute(*layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
      &order, sizeof(order)), "set row-major matrix layout");
}
std::unique_ptr<LtOp> make_lt_op(cublasLtHandle_t handle, const Problem& p,
                               size_t scratch_limit, int requested) {
  auto op = std::make_unique<LtOp>();
  op->robust_small_tuning = std::max({p.m, p.n, p.k}) <= 4096;
  op->alpha = p.alpha;
  op->beta = p.beta;
  lt_check(cublasLtMatmulDescCreate(&op->operation, p.dtype==DType::FP32 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F, CUDA_R_32F), "create matmul descriptor");
  const cublasOperation_t ta = p.trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t tb = p.trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  lt_check(cublasLtMatmulDescSetAttribute(op->operation, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "set transpose A");
  lt_check(cublasLtMatmulDescSetAttribute(op->operation, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "set transpose B");
  const cudaDataType_t input_type = p.dtype == DType::FP32 ? CUDA_R_32F : p.dtype == DType::BF16 ? CUDA_R_16BF : CUDA_R_16F;
  make_layout(&op->a, input_type, p.trans_a ? p.k : p.m, p.trans_a ? p.m : p.k, p.lda);
  make_layout(&op->b, input_type, p.trans_b ? p.n : p.k, p.trans_b ? p.k : p.n, p.ldb);
  make_layout(&op->c, CUDA_R_32F, p.m, p.n, p.beta == 0.0f ? p.ldd : p.ldc);
  make_layout(&op->d, CUDA_R_32F, p.m, p.n, p.ldd);
  cublasLtMatmulPreference_t preference = nullptr;
  lt_check(cublasLtMatmulPreferenceCreate(&preference), "create matmul preference");
  try {
    lt_check(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &scratch_limit, sizeof(scratch_limit)), "set Lt workspace budget");
    // Query the full candidate budget for aligned cudaMalloc inputs, then retain
    // pools for element-aligned rebindings. Restricting every
    // query to two-byte alignment can silently exclude the fastest Tensor Cores.
    for (uint32_t input_alignment : {uint32_t(256), uint32_t(16), uint32_t(dtype_size(p.dtype))}) {
      const uint32_t output_alignment = std::max(input_alignment, uint32_t(4));
      lt_check(cublasLtMatmulPreferenceSetAttribute(preference,
          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &input_alignment, sizeof(input_alignment)), "set A alignment");
      lt_check(cublasLtMatmulPreferenceSetAttribute(preference,
          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &input_alignment, sizeof(input_alignment)), "set B alignment");
      lt_check(cublasLtMatmulPreferenceSetAttribute(preference,
          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &output_alignment, sizeof(output_alignment)), "set C alignment");
      lt_check(cublasLtMatmulPreferenceSetAttribute(preference,
          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &output_alignment, sizeof(output_alignment)), "set D alignment");
      std::vector<cublasLtMatmulHeuristicResult_t> pool(static_cast<size_t>(requested));
      int returned = 0;
      const auto query_status = cublasLtMatmulAlgoGetHeuristic(handle, op->operation, op->a, op->b, op->c, op->d,
          preference, requested, pool.data(), &returned);
      lt_check(query_status, "query Lt heuristics");
      for (int i = 0; i < returned; ++i) {
        const auto& candidate = pool[size_t(i)];
        if (candidate.state != CUBLAS_STATUS_SUCCESS || candidate.workspaceSize > scratch_limit) continue;
        const bool duplicate = std::any_of(op->candidates.begin(), op->candidates.end(), [&](const auto& existing) {
          return std::memcmp(&candidate.algo, &existing.algo, sizeof(candidate.algo)) == 0;
        });
        if (duplicate) continue;
        std::array<uint32_t, 4> requirements{input_alignment, input_alignment, output_alignment, output_alignment};
        const cublasLtMatmulAlgoCapAttributes_t attributes[4] = {
            CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
            CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES};
        for (int operand = 0; operand < 4; ++operand) {
          uint32_t requirement = 0;
          size_t written = 0;
          if (cublasLtMatmulAlgoCapGetAttribute(&candidate.algo, attributes[operand], &requirement,
              sizeof(requirement), &written) == CUBLAS_STATUS_SUCCESS && requirement)
            requirements[operand] = requirement;
        }
        op->candidates.push_back(candidate);
        op->alignments.push_back(requirements);
      }
    }
  } catch (...) {
    cublasLtMatmulPreferenceDestroy(preference);
    throw;
  }
  cublasLtMatmulPreferenceDestroy(preference);
  if (op->candidates.empty()) {
    throw std::runtime_error("cuBLASLt has no valid heuristic for this problem and workspace budget");
  }
  return op;
}
bool compatible(const LtOp& op, size_t candidate, const void* A, const void* B,
                const float* C0, const float* D) {
  const void* pointers[4] = {A, B, op.beta == 0.0f ? D : C0, D};
  for (int i = 0; i < 4; ++i)
    if (reinterpret_cast<uintptr_t>(pointers[i]) % op.alignments[candidate][i]) return false;
  return true;
}
size_t bound_candidate(const LtOp& op, const void* A, const void* B,
                       const float* C0, const float* D) {
  if (compatible(op, op.selected, A, B, C0, D)) return op.selected;
  for (size_t i = 0; i < op.candidates.size(); ++i)
    if (compatible(op, i, A, B, C0, D)) return i;
  throw std::invalid_argument("bound pointers satisfy no available cuBLASLt algorithm alignment");
}
cublasStatus_t execute_lt(cublasLtHandle_t handle, const LtOp& op, size_t candidate,
                        const void* A, const void* B, const float* C0, float* D,
                        void* scratch, cudaStream_t stream) {
  const auto& algorithm = op.candidates[candidate];
  if (!compatible(op, candidate, A, B, C0, D)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return cublasLtMatmul(handle, op.operation, &op.alpha, A, op.a, B, op.b,
      &op.beta, op.beta == 0.0f ? D : C0, op.c, D, op.d, &algorithm.algo,
      scratch, algorithm.workspaceSize, stream);
}
void tune_lt(cublasLtHandle_t handle, LtOp& op, const void* A, const void* B,
             const float* C0, float* D, void* scratch, cudaStream_t stream) {
  op.tuned = false;op.strategy_reranked=false;
  op.candidate_tuning_ms.assign(op.candidates.size(),std::numeric_limits<float>::infinity());
  op.selected_tuning_ms = std::numeric_limits<float>::quiet_NaN();
  cudaEvent_t start = nullptr, stop = nullptr;
  cuda_check(cudaEventCreate(&start), "create tuning start event");
  try {
    cuda_check(cudaEventCreate(&stop), "create tuning stop event");
    // Reuse the same candidate pool, alignment checks and workspace budget.
    // Small operators use a median of three batches rather than one tiny batch.
    const int warmup = op.robust_small_tuning ? 4 : 1;
    const int groups = op.robust_small_tuning ? 3 : 1;
    const int repetitions = op.robust_small_tuning ? 10 : 3;
    float best = std::numeric_limits<float>::infinity();
    size_t chosen = 0;
    bool found = false;
    for (size_t i = 0; i < op.candidates.size(); ++i) {
      bool valid = true;
      for (int rep = 0; rep < warmup; ++rep) {
        if (execute_lt(handle, op, i, A, B, C0, D, scratch, stream) != CUBLAS_STATUS_SUCCESS) {
          valid = false;
          break;
        }
      }
      if (!valid) continue;
      std::array<float, 3> samples{};
      for (int group = 0; group < groups; ++group) {
        cuda_check(cudaEventRecord(start, stream), "record tuning start");
        for (int rep = 0; rep < repetitions; ++rep)
          if (execute_lt(handle, op, i, A, B, C0, D, scratch, stream) != CUBLAS_STATUS_SUCCESS) valid = false;
        cuda_check(cudaEventRecord(stop, stream), "record tuning stop");
        cuda_check(cudaEventSynchronize(stop), "wait for tuning candidate");
        cuda_check(cudaEventElapsedTime(&samples[size_t(group)], start, stop), "measure tuning candidate");
      }
      std::sort(samples.begin(), samples.begin() + groups);
      const float score = samples[size_t(groups / 2)];
      if(valid) op.candidate_tuning_ms[i]=score/repetitions;
      if (valid && score < best) { best = score; chosen = i; found = true; }
    }
    if (!found) throw std::runtime_error("all cuBLASLt tuning candidates rejected their bound pointers");
    op.selected = chosen;
    op.selected_tuning_ms = best / repetitions;
    op.tuned = true;
  } catch (...) {
    if (stop) cudaEventDestroy(stop);
    cudaEventDestroy(start);
    throw;
  }
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
}
template <class T>
std::string lt_config_value(const cublasLtMatmulAlgo_t& algorithm,
                            cublasLtMatmulAlgoConfigAttributes_t attribute) {
  T value{};
  size_t written = 0;
  const auto status = cublasLtMatmulAlgoConfigGetAttribute(&algorithm, attribute,
                                                        &value, sizeof(value), &written);
  if (status != CUBLAS_STATUS_SUCCESS || written != sizeof(value))
    return "NA(status=" + std::to_string(int(status)) + ")";
  return std::to_string(value);
}
std::string describe_lt(const LtOp& op, size_t bound) {
  const auto& candidate = op.candidates[bound];
  const auto& algorithm = candidate.algo;
  // Types match the installed CUDA 12.4 cublasLt.h attribute definitions.
  std::string result = "id=" + lt_config_value<int32_t>(algorithm, CUBLASLT_ALGO_CONFIG_ID);
  result += ";tile=" + lt_config_value<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID);
  result += ";split_k=" + lt_config_value<int32_t>(algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
  result += ";reduction=" + lt_config_value<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
  result += ";swizzle=" + lt_config_value<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
  result += ";stages=" + lt_config_value<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID);
  result += ";custom=" + lt_config_value<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
  result += ";inner=" + lt_config_value<uint16_t>(algorithm, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID);
  result += ";cluster=" + lt_config_value<uint16_t>(algorithm, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID);
  result += ";bound_candidate=" + std::to_string(bound) + "/" + std::to_string(op.candidates.size());
  result += ";tuned_candidate=" + std::to_string(op.selected);
  result += ";bound_workspace=" + std::to_string(candidate.workspaceSize);
  result += ";tuned=" + std::string(op.tuned ? "1" : "0");
  result += ";tune=" + (op.strategy_reranked ? "4w/5x20median_sets"+std::to_string(op.strategy_input_sets) :
      std::string(op.robust_small_tuning ? "4w/3x10median" : "1w/1x3"));
  // A rebound pointer can legitimately use another compatible candidate. Its
  // identity is reported above, but the selected candidate's score is not its score.
  if (!op.tuned) result += ";tuning_ms=NA(untuned)";
  else if (bound != op.selected) result += ";tuning_ms=NA(rebound)";
  else result += ";tuning_ms=" + std::to_string(op.selected_tuning_ms);
  for (int operand = 0; operand < 4; ++operand)
    result += ";align" + std::to_string(operand) + "=" + std::to_string(op.alignments[bound][operand]);
  return result;
}

}  // namespace

struct Plan::Impl {
  Problem problem;
  bool empty = false, scale = false, sw = false, sw_hopper = false;
  size_t bytes = 0;
  bool sw_tuned=false;
  int sw_selected=-1,sw_device=-1;
  Policy sw_policy;
  std::string sw_tune_status="untuned",sw_gpu;
  std::vector<detail::SWConfig> sw_configs;
  std::vector<SWCandidateResult> sw_results;
  std::vector<detail::sw_fp32_batch::Maps> sw_batch_maps;
  bool strategy = false;  // Legacy Tuned API now uses the same cuBLASLt pool.
  int strategy_input_sets = 1;
  cublasLtHandle_t handle = nullptr;
  std::unique_ptr<LtOp> dense;
  ~Impl() {
    dense.reset();
    if (handle) cublasLtDestroy(handle);
  }
};

namespace {
void validate_binding(const Plan::Impl& plan, const void* A, const void* B,
                      const float* C0, float* D, void* workspace) {
  const auto& p = plan.problem;
  if (plan.empty) return;
  if (!D) throw std::invalid_argument("D is null for a nonempty output");
  require_alignment(D, 4, "D");
  const size_t dbytes = matrix_bytes(p.m, p.n, p.ldd, 4);
  overlaps(D, dbytes, D, dbytes);  // Also validates the address end.
  if (p.beta != 0.0f) {
    if (!C0) throw std::invalid_argument("C0 is required when beta is nonzero");
    require_alignment(C0, 4, "C0");
    const size_t cbytes = matrix_bytes(p.m, p.n, p.ldc, 4);
    if (overlaps(C0, cbytes, D, dbytes) && (C0 != D || p.ldc != p.ldd))
      throw std::invalid_argument("C0 and D overlap without an exact compatible in-place view");
    if (plan.bytes && overlaps(C0, cbytes, workspace, plan.bytes))
      throw std::invalid_argument("workspace overlaps C0");
  }
  if (plan.scale) return;
  if (!A || !B) throw std::invalid_argument("A and B are required for a nondegenerate GEMM");
  require_alignment(A, dtype_size(p.dtype), "A");
  require_alignment(B, dtype_size(p.dtype), "B");
  const size_t abytes = matrix_bytes(p.trans_a ? p.k : p.m, p.trans_a ? p.m : p.k, p.lda, dtype_size(p.dtype));
  const size_t bbytes = matrix_bytes(p.trans_b ? p.n : p.k, p.trans_b ? p.k : p.n, p.ldb, dtype_size(p.dtype));
  if (overlaps(A, abytes, D, dbytes) || overlaps(B, bbytes, D, dbytes))
    throw std::invalid_argument("D overlaps A or B");
  if (plan.bytes) {
    if (!workspace) throw std::invalid_argument("caller workspace is required");
    require_alignment(workspace, 256, "workspace");
    if (overlaps(A, abytes, workspace, plan.bytes) || overlaps(B, bbytes, workspace, plan.bytes) ||
        overlaps(D, dbytes, workspace, plan.bytes))
      throw std::invalid_argument("workspace overlaps A, B or D");
  }
}
void execute_sw_candidate(const Plan::Impl&,int,const void*,const void*,float*,void*,cudaStream_t);
void run_internal(const Plan::Impl& plan, const void* A, const void* B,
                  const float* C0, float* D, void* workspace, cudaStream_t stream) {
  if (plan.empty) return;
  if (plan.scale) {
    detail::scale_output(plan.problem, C0, D, stream);
    cuda_check(cudaPeekAtLastError(), "launch degenerate output kernel");
    return;
  }
  if(plan.sw_tuned) {
    if(plan.sw_selected<0)throw std::runtime_error("SWTuned requires successful explicit tune() before run()");
    int device=-1;cuda_check(cudaGetDevice(&device),"SW run device");
    if(device!=plan.sw_device)throw std::invalid_argument("SW tuned plan belongs to another CUDA device");
    if(reinterpret_cast<uintptr_t>(A)%16 || reinterpret_cast<uintptr_t>(B)%16)throw std::invalid_argument("SW tuned binding requires 16-byte alignment");
    execute_sw_candidate(plan,plan.sw_selected,A,B,D,workspace,stream);return;
  }
  if (plan.sw) {
    if (plan.sw_hopper && reinterpret_cast<uintptr_t>(A)%16==0 &&
        reinterpret_cast<uintptr_t>(B)%16==0)
      detail::run_sw_hopper(plan.problem,A,B,D,workspace,stream);
    else
      detail::run_sw_pipeline(plan.problem,A,B,C0,D,stream);
    return;
  }
  lt_check(execute_lt(plan.handle, *plan.dense,
      bound_candidate(*plan.dense, A, B, C0, D), A, B, C0, D,
      plan.bytes ? workspace : nullptr, stream), "run dense GEMM");
}
}  // namespace

#include "sw_tuning.inc"

Plan::Plan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Plan::~Plan() = default;
Plan::Plan(Plan&&) noexcept = default;
Plan& Plan::operator=(Plan&&) noexcept = default;
size_t Plan::workspace_bytes() const { return impl_->bytes; }
const Problem& Plan::problem() const { return impl_->problem; }
const std::vector<SWCandidateResult>& Plan::sw_tuning_results() const {return impl_->sw_results;}
std::string Plan::algorithm_name() const {
  if(impl_->sw_tuned)return "sw_tuned";
  if(impl_->strategy && !impl_->empty && !impl_->scale)return "tuned_dense_cublaslt";
  return impl_->empty ? "empty" : impl_->scale ? "degenerate_scale" : impl_->sw_hopper ? "sw_hopper" : impl_->sw ? "sw_pipeline" : "dense_cublaslt";
}
std::string Plan::lt_algorithm_config(const void* A, const void* B, const float* C0,
                                      float* D, void* workspace) const {
  validate_binding(*impl_, A, B, C0, D, workspace);
  if (impl_->empty || impl_->scale) return "none";
  if(impl_->sw_tuned) {
    std::string info="sw_tuned;leaf_terms=7;status="+impl_->sw_tune_status+";gpu="+impl_->sw_gpu+
      ";mode=launch;input_sets="+std::to_string(impl_->strategy_input_sets)+
      ";candidate_count="+std::to_string(impl_->sw_results.size())+";rank=5x20;confirmation=5x100";
    info+=";timing=launch";
    if(impl_->sw_selected>=0){const auto& r=impl_->sw_results[impl_->sw_selected];info+=";selected_id="+std::to_string(r.id)+";selected="+r.config+";median_ms="+std::to_string(sw_median(r.confirmation_ms))+";ef="+std::to_string(r.relative_error)+";gate="+std::to_string(r.gate_limit);}
    return info;
  }
  if (impl_->sw_hopper && reinterpret_cast<uintptr_t>(A)%16==0 &&
      reinterpret_cast<uintptr_t>(B)%16==0)
    return std::string(detail::sw_hopper_config(impl_->problem))+";workspace="+std::to_string(impl_->bytes);
  if (impl_->sw && impl_->problem.dtype==DType::FP32)
    return std::string("sw_pipeline;tile=16x16;bk=16;threads=256;leaf_terms=7;levels=1;padding=logical_even;input=fp32;transforms=fp32;leaf_math=fp32_simt;accum=fp32;workspace=0;residual_correction=off;load=")+
      (detail::sw_fp32_tma_binding(impl_->problem,A,B)?"tma;raw_stages=2;issuer=thread0":"synchronous");
  if (impl_->sw) return "sw_pipeline;tile=16x16;warps=4;leaf_terms=7;levels=1;padding=logical_even;transforms=fp32_to_input_rn;accum=fp32;workspace=0;residual_correction=off";
  return "dense:" + describe_lt(*impl_->dense, bound_candidate(*impl_->dense, A, B, C0, D));
}
const char* dtype_name(DType dtype) {
  switch(dtype) {
    case DType::BF16:return "bf16";
    case DType::FP16:return "fp16";
    case DType::FP32:return "fp32";
  }
  return "unknown";
}
std::unique_ptr<Plan> create_plan(Problem p, Policy policy) {
  validate_problem(p);
  if (policy.algorithm != Algorithm::Auto && policy.algorithm != Algorithm::Dense && policy.algorithm != Algorithm::SWPipeline && policy.algorithm != Algorithm::Tuned && policy.algorithm != Algorithm::SWTuned)
    throw std::invalid_argument("unsupported algorithm: old SW implementations have been removed");
  if (policy.heuristic_candidates < 1 || policy.heuristic_candidates > 128)
    throw std::invalid_argument("heuristic_candidates must be between 1 and 128");
  if(policy.tuning_input_sets<1 || policy.tuning_input_sets>4)
    throw std::invalid_argument("tuning_input_sets must be between 1 and 4");
  if(policy.sw_tuning_graph)throw std::invalid_argument("SW tuning supports ordinary launch only");
  if(policy.sw_tuning_candidates<0)throw std::invalid_argument("negative SW candidate budget");
  auto impl = std::make_unique<Plan::Impl>();
  impl->sw_policy=policy;
  impl->strategy_input_sets=policy.tuning_input_sets;
  impl->problem = p;
  impl->empty = !p.m || !p.n;
  impl->scale = !p.k || p.alpha == 0.0f;
  if (policy.algorithm==Algorithm::SWTuned && !impl->empty && !impl->scale) {
    impl->sw=impl->sw_tuned=true;
    cuda_check(cudaGetDevice(&impl->sw_device),"SW tuner device");cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop,impl->sw_device),"SW tuner properties");impl->sw_gpu=prop.name;
    if(prop.major!=9 || prop.minor!=0)throw std::invalid_argument("SW tuner requires Hopper sm90");
    if(p.dtype!=DType::FP32 && !detail::sw_hopper_workspace(p))throw std::invalid_argument("SW tuner currently requires positive dimensions <=32768, contiguous non-transposed alpha=1 beta=0");
    auto add=[&](detail::SWConfig c,size_t bytes,const std::string& name,detail::SWResources resource) {
      if(bytes>policy.workspace_limit || resource.resident<1)return;
      SWCandidateResult r;r.id=c.id;r.config=name;r.workspace_bytes=bytes;r.shared_bytes=resource.shared;
      r.registers_per_thread=resource.registers;r.resident_ctas=resource.resident;
      const auto align=[](int64_t value,int multiple){return (value+multiple-1)/multiple*multiple;};
      int64_t mh=c.padded?align((p.m+1)/2,c.n):p.m/2,nh=c.padded?align((p.n+1)/2,64*c.groups):p.n/2,kh=c.padded?align((p.k+1)/2,c.bk):p.k/2;
      r.physical_work_ratio=7.0*mh*nh*kh/(double(p.m)*p.n*p.k);
      const double ctas=c.id==-1?double(mh/c.n)*((nh+15)/16):double((mh+c.n-1)/c.n)*((nh+64*c.groups-1)/(64*c.groups));
      const double capacity=prop.multiProcessorCount*resource.resident;
      const double wave_eff=ctas/(std::ceil(ctas/capacity)*capacity);
      // Ranking heuristic, not a latency claim: actual padded work, wave fill,
      // tile reuse and synchronization count. Measurements determine the winner.
      r.model_score=r.physical_work_ratio/std::max(wave_eff,0.001)*(1.0+32.0/c.n+16.0/c.bk);
      r.status="unmeasured";impl->sw_configs.push_back(c);impl->sw_results.push_back(r);impl->bytes=std::max(impl->bytes,bytes);
    };
    if(p.dtype==DType::FP32) {
      if(p.trans_a || p.trans_b || p.lda!=p.k || p.ldb!=p.n || p.ldd!=p.n ||
         p.alpha!=1 || p.beta!=0 || std::max({p.m,p.n,p.k})>32768)
        throw std::invalid_argument("FP32 SW tuner requires dimensions <=32768, contiguous non-transposed alpha=1 beta=0");
      for(bool tma:{false,true})for(int bk:{16,32}) {
        if(tma && !detail::sw_fp32_tma_problem(p))continue;
        const bool guarded=!tma && detail::sw_fp32_guarded_tail(p,bk);
        const auto resource=tma?detail::setup_sw_fp32_tma(bk):detail::setup_sw_fp32(p,bk);
        SWCandidateResult r;r.id=(tma?9100000:9000000)+bk;r.workspace_bytes=0;
        r.shared_bytes=resource.shared;r.registers_per_thread=resource.registers;r.resident_ctas=resource.resident;
        r.config="sw_fp32;tile=16x16;bk="+std::to_string(bk)+
          ";threads=256;leaf_terms=7;transforms=fp32;leaf_math=fp32_simt;accum=fp32;workspace=0;residual_correction=off;load="+
          (tma?"tma;raw_stages=2;issuer=thread0":"synchronous")+
          (guarded?";tail_policy=guarded_final_k;outputs=active":";tail_policy=padded;outputs=tile");
        const int64_t mh=guarded?(p.m+1)/2:((p.m+1)/2+15)/16*16;
        const int64_t nh=guarded?(p.n+1)/2:((p.n+1)/2+15)/16*16;
        const int64_t kh=guarded?(p.k+1)/2+int(((p.k+1)/2)%bk!=0):((p.k+1)/2+bk-1)/bk*bk;
        r.physical_work_ratio=7.0*mh*nh*kh/(double(p.m)*p.n*p.k);
        r.model_score=r.physical_work_ratio;r.status="unmeasured";
        impl->sw_configs.push_back({r.id,16,bk,tma?2:1,0,0,true});impl->sw_results.push_back(r);
      }
      if(detail::sw_fp32_batch::problem(p)) {
        const auto z=detail::sw_fp32_batch::shape(p);
        for(const auto& c:detail::sw_fp32_batch::configurations) {
          if(!detail::sw_fp32_batch::tuning_eligible(c,p.m,p.n,prop.multiProcessorCount))continue;
          const size_t bytes=detail::sw_fp32_batch::workspace_size(p,c.id);
          if(bytes>policy.workspace_limit)continue;
          const auto resource=detail::sw_fp32_batch::setup(c.id);
          if(resource.resident<1)continue;
          SWCandidateResult r;r.id=9500000+c.id;r.workspace_bytes=bytes;
          r.shared_bytes=resource.shared;r.registers_per_thread=resource.registers;r.resident_ctas=resource.resident;
          r.config=detail::sw_fp32_batch::name(c.id)+";workspace="+std::to_string(bytes)+
            ";packed="+std::to_string(z.mp)+"x"+std::to_string(z.np)+"x"+std::to_string(z.kp);
          const int64_t mh=((p.m+1)/2+c.bm-1)/c.bm*c.bm;
          const int64_t nh=((p.n+1)/2+c.bn-1)/c.bn*c.bn;
          const bool padded_fused=c.strategy==detail::sw_fp32_batch::ExecutionStrategy::CooperativeFused &&
            c.tail_policy==detail::sw_fp32_batch::TailPolicy::Padded;
          const int64_t active_k=(p.k+1)/2;
          const int64_t kh=padded_fused?z.kp:active_k+
            (c.strategy==detail::sw_fp32_batch::ExecutionStrategy::CooperativeFused?active_k<z.kp:active_k%16!=0);
          r.physical_work_ratio=7.0*mh*nh*kh/(double(p.m)*p.n*p.k);
          r.model_score=r.physical_work_ratio;r.status="unmeasured";
          impl->sw_configs.push_back({r.id,c.bm,c.bk,2,c.id,0,true});impl->sw_results.push_back(r);
          impl->bytes=std::max(impl->bytes,bytes);
        }
      }
    } else {
      // The zero-workspace path covers every shape, including 1x1x1.
      add({-1,16,16,1,0,0,true},0,"sw_pipeline;leaf_terms=7;workspace=0",{0,-1,1});
      for(const auto& c:detail::sw_configs(p))if(detail::sw_config_workspace(p,c)<=policy.workspace_limit)
        add(c,detail::sw_config_workspace(p,c),detail::sw_config_name(c),detail::setup_sw_config(p,c));
    }
    if(impl->sw_configs.empty())throw std::invalid_argument("SW tuner: no legal configuration within workspace budget");
  } else if (policy.algorithm==Algorithm::SWPipeline && !impl->empty && !impl->scale) {
    int device; cudaDeviceProp prop{};
    cuda_check(cudaGetDevice(&device),"SW device");
    cuda_check(cudaGetDeviceProperties(&prop,device),"SW properties");
    if (prop.major!=9 || prop.minor!=0)
      throw std::invalid_argument("MegaFMM SW requires Hopper sm90");
    impl->sw=true;
    if(p.dtype==DType::FP32 && detail::sw_fp32_tma_problem(p))
      detail::setup_sw_fp32_tma(16);
    const size_t bytes=detail::sw_hopper_workspace(p);
    if(bytes && bytes<=policy.workspace_limit) {
      detail::setup_sw_hopper(p);impl->sw_hopper=true;impl->bytes=bytes;
    }
  } else if (!impl->empty && !impl->scale) {
    lt_check(cublasLtCreate(&impl->handle), "create cuBLASLt handle");
    impl->dense = make_lt_op(impl->handle, p, policy.workspace_limit, policy.heuristic_candidates);
    impl->bytes = impl->dense->scratch_bytes();
    impl->strategy=policy.algorithm==Algorithm::Tuned;
  }

  return std::make_unique<Plan>(std::move(impl));
}
void run(const Plan& plan, const void* A, const void* B, const float* C0,
         float* D, void* workspace, cudaStream_t stream) {
  validate_binding(*plan.impl_, A, B, C0, D, workspace);
  run_internal(*plan.impl_, A, B, C0, D, workspace, stream);
}
void tune(Plan& plan, const void* A, const void* B, const float* C0,
          float* D, void* workspace, cudaStream_t stream) {
  auto& impl = *plan.impl_;
  cudaStreamCaptureStatus capture;
  cuda_check(cudaStreamIsCapturing(stream,&capture),"tuning capture check");
  if(capture!=cudaStreamCaptureStatusNone)throw std::invalid_argument("tune() must run outside CUDA Graph capture");
  validate_binding(impl, A, B, C0, D, workspace);
  if (impl.empty) return;
  const float* original_c = C0;
  float* snapshot = nullptr;
  if (impl.problem.beta != 0.0f && C0 == D) {
    const size_t bytes = matrix_bytes(impl.problem.m, impl.problem.n, impl.problem.ldc, 4);
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&snapshot), bytes), "allocate in-place C0 tuning snapshot");
    try { cuda_check(cudaMemcpyAsync(snapshot, C0, bytes, cudaMemcpyDeviceToDevice, stream), "copy in-place C0 tuning snapshot"); }
    catch (...) { cudaFree(snapshot); throw; }
    original_c = snapshot;
  }
  try {
    if (!impl.scale && !impl.sw)
      tune_lt(impl.handle, *impl.dense, A, B, original_c, D,
              impl.bytes ? workspace : nullptr, stream);
    if(impl.sw_tuned)tune_sw(impl,A,B,D,workspace,stream);
    run_internal(impl, A, B, original_c, D, workspace, stream);
    cuda_check(cudaStreamSynchronize(stream), "finish explicit plan tuning");
  } catch (...) {
    if(impl.sw_tuned && impl.sw_selected<0 && impl.sw_tune_status=="tuning")impl.sw_tune_status="failed";
    if (snapshot) cudaFree(snapshot);
    throw;
  }
  if (snapshot) cuda_check(cudaFree(snapshot), "free in-place C0 tuning snapshot");
}
}  // namespace megafmm

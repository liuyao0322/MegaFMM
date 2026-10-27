#include "megafmm/gemm.h"
#include "megafmm/kernels.h"
#include "internal.h"
#include <cublasLt.h>
#include <algorithm>
#include <array>
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
size_t aligned(size_t a) { return checked_add(a, 255) & ~size_t(255); }
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
  if (p.dtype != DType::BF16 && p.dtype != DType::FP16)
    throw std::invalid_argument("unsupported input dtype");
  const int64_t ac = p.trans_a ? p.m : p.k, bc = p.trans_b ? p.k : p.n;
  if (!p.lda) p.lda = ac;
  if (!p.ldb) p.ldb = bc;
  if (!p.ldc) p.ldc = p.n;
  if (!p.ldd) p.ldd = p.n;
  if (p.lda < ac || p.ldb < bc || p.ldc < p.n || p.ldd < p.n)
    throw std::invalid_argument("leading dimension is smaller than the physical row width");
  matrix_bytes(p.trans_a ? p.k : p.m, ac, p.lda, 2);
  matrix_bytes(p.trans_b ? p.n : p.k, bc, p.ldb, 2);
  matrix_bytes(p.m, p.n, p.ldc, 4);
  matrix_bytes(p.m, p.n, p.ldd, 4);
  if (p.m > std::numeric_limits<int>::max() || p.n > std::numeric_limits<int>::max() ||
      p.k > std::numeric_limits<int>::max())
    throw std::invalid_argument("dimensions exceed the supported GEMM backend range");
}
bool winograd_shape(const Problem& p, const Policy& policy) {
  return policy.certified_range && !p.trans_a && !p.trans_b &&
      p.m <= 32768 && p.n <= 32768 && p.k <= 32768 &&
      p.m % 16 == 0 && p.n % 16 == 0 && p.k % 16 == 0 &&
      p.lda % 8 == 0 && p.ldb % 8 == 0 &&
      std::isfinite(p.alpha) && std::isfinite(p.beta) &&
      std::abs(p.alpha) <= 1.0f && std::abs(p.beta) <= 1.0f;
}

struct LtOp {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> candidates;
  std::vector<std::array<uint32_t, 4>> alignments;
  int32_t batch_count = 1;
  bool robust_small_tuning = false;
  bool tuned = false;
  float selected_tuning_ms = std::numeric_limits<float>::quiet_NaN();
  std::array<size_t, 4> batch_stride_bytes{};
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
                               size_t scratch_limit, int requested, int32_t batch_count=1,
                               std::array<int64_t, 4> batch_strides={}) {
  auto op = std::make_unique<LtOp>();
  op->batch_count = batch_count;
  op->robust_small_tuning = std::max({p.m, p.n, p.k}) <= 4096;
  op->alpha = p.alpha;
  op->beta = p.beta;
  lt_check(cublasLtMatmulDescCreate(&op->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F), "create matmul descriptor");
  const cublasOperation_t ta = p.trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t tb = p.trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  lt_check(cublasLtMatmulDescSetAttribute(op->operation, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "set transpose A");
  lt_check(cublasLtMatmulDescSetAttribute(op->operation, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "set transpose B");
  const cudaDataType_t input_type = p.dtype == DType::BF16 ? CUDA_R_16BF : CUDA_R_16F;
  make_layout(&op->a, input_type, p.trans_a ? p.k : p.m, p.trans_a ? p.m : p.k, p.lda);
  make_layout(&op->b, input_type, p.trans_b ? p.n : p.k, p.trans_b ? p.k : p.n, p.ldb);
  make_layout(&op->c, CUDA_R_32F, p.m, p.n, p.beta == 0.0f ? p.ldd : p.ldc);
  make_layout(&op->d, CUDA_R_32F, p.m, p.n, p.ldd);
  if (batch_count != 1) {
    if (batch_count < 1) throw std::invalid_argument("Lt batch count must be positive");
    const cublasLtMatrixLayout_t layouts[4] = {op->a, op->b, op->c, op->d};
    for (int i = 0; i < 4; ++i) {
      if (batch_strides[i] <= 0) throw std::invalid_argument("uniform Lt batch strides must be positive");
      op->batch_stride_bytes[i] = checked_mul(size_t(batch_strides[i]), i < 2 ? 2 : 4);
      lt_check(cublasLtMatrixLayoutSetAttribute(layouts[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
          &batch_count, sizeof(batch_count)), "set Lt batch count");
      lt_check(cublasLtMatrixLayoutSetAttribute(layouts[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
          &batch_strides[i], sizeof(batch_strides[i])), "set Lt batch stride");
    }
  }
  cublasLtMatmulPreference_t preference = nullptr;
  lt_check(cublasLtMatmulPreferenceCreate(&preference), "create matmul preference");
  try {
    lt_check(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &scratch_limit, sizeof(scratch_limit)), "set Lt workspace budget");
    // Query the full candidate budget for aligned cudaMalloc inputs, then retain
    // pools for quadrant views and element-aligned rebindings. Restricting every
    // query to two-byte alignment can silently exclude the fastest Tensor Cores.
    for (uint32_t input_alignment : {uint32_t(256), uint32_t(16), uint32_t(2)}) {
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
      if (batch_count > 1 && query_status == CUBLAS_STATUS_NOT_SUPPORTED) continue;
      lt_check(query_status, "query Lt heuristics");
      for (int i = 0; i < returned; ++i) {
        const auto& candidate = pool[size_t(i)];
        if (candidate.state != CUBLAS_STATUS_SUCCESS || candidate.workspaceSize > scratch_limit) continue;
        if (batch_count > 1) {
          int32_t supported = 0;
          size_t written = 0;
          if (cublasLtMatmulAlgoCapGetAttribute(&candidate.algo, CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT,
              &supported, sizeof(supported), &written) != CUBLAS_STATUS_SUCCESS || !supported) continue;
        }
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
        if (batch_count > 1) {
          bool aligned_batches = true;
          for (int operand = 0; operand < 4; ++operand)
            aligned_batches = aligned_batches && op->batch_stride_bytes[operand] % requirements[operand] == 0;
          if (!aligned_batches) continue;
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
    if (batch_count > 1) return nullptr;  // The planner can use the ordinary GEMM fallback.
    throw std::runtime_error("cuBLASLt has no valid heuristic for this problem and workspace budget");
  }
  return op;
}
bool compatible(const LtOp& op, size_t candidate, const void* A, const void* B,
                const float* C0, const float* D) {
  const void* pointers[4] = {A, B, op.beta == 0.0f ? D : C0, D};
  for (int i = 0; i < 4; ++i)
    if (reinterpret_cast<uintptr_t>(pointers[i]) % op.alignments[candidate][i] ||
        (op.batch_count > 1 && op.batch_stride_bytes[i] % op.alignments[candidate][i])) return false;
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
  op.tuned = false;
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
  result += ";tune=" + std::string(op.robust_small_tuning ? "4w/3x10median" : "1w/1x3");
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
  Policy policy;
  Algorithm algorithm = Algorithm::Dense;
  bool empty = false, scale = false, small_batched = false, fused_input = false, packed_input = false;
  size_t bytes = 0, s_offset = 0, t_offset = 0, p_offset = 0, scratch_offset = 0;
  cublasLtHandle_t handle = nullptr;
  std::unique_ptr<LtOp> dense;
  std::unique_ptr<LtOp> batched;
  std::array<std::unique_ptr<LtOp>, 7> leaves;
  std::array<Problem, 7> leaf_problems;
  ~Impl() {
    dense.reset();
    batched.reset();
    for (auto& leaf : leaves) leaf.reset();
    if (handle) cublasLtDestroy(handle);
  }
};

namespace {
bool use_dense(const Plan::Impl& plan, const void* A, const void* B, void* workspace) {
  if (plan.algorithm == Algorithm::Dense || reinterpret_cast<uintptr_t>(A) % 16 ||
      reinterpret_cast<uintptr_t>(B) % 16) return true;
  if (plan.packed_input) {
    const auto* base = static_cast<const unsigned char*>(workspace);
    const auto* products = reinterpret_cast<const float*>(base + plan.p_offset);
    return !packed_leaf_original_input_guard(plan.problem, A, B, plan.policy.hopper_variant - 11) ||
        !packed_leaf_full_tiles(plan.problem, base + plan.s_offset, base + plan.t_offset,
                               products, plan.policy.hopper_variant - 11);
  }
  if (plan.fused_input) {
    const auto* products = reinterpret_cast<const float*>(
        static_cast<const unsigned char*>(workspace) + plan.p_offset);
    return !fused_leaf_full_tiles(plan.problem, A, B, products,
                                 plan.policy.hopper_variant - 9);
  }
  return false;
}
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
  require_alignment(A, 2, "A");
  require_alignment(B, 2, "B");
  const size_t abytes = matrix_bytes(p.trans_a ? p.k : p.m, p.trans_a ? p.m : p.k, p.lda, 2);
  const size_t bbytes = matrix_bytes(p.trans_b ? p.n : p.k, p.trans_b ? p.k : p.n, p.ldb, 2);
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
void operands(const Plan::Impl& plan, const void* A, const void* B, void* workspace,
              std::array<const void*, 7>& left, std::array<const void*, 7>& right) {
  const auto& p = plan.problem;
  const size_t m = size_t(p.m / 2), n = size_t(p.n / 2), k = size_t(p.k / 2);
  const auto* a = static_cast<const unsigned char*>(A);
  const auto* b = static_cast<const unsigned char*>(B);
  const auto* s = static_cast<const unsigned char*>(workspace) + plan.s_offset;
  const auto* t = static_cast<const unsigned char*>(workspace) + plan.t_offset;
  const size_t sa = m * k * 2, tb = k * n * 2;
  left = {a, a + k * 2, s + 3 * sa, a + (m * p.lda + k) * 2,
          s, s + sa, s + 2 * sa};
  right = {b, b + k * p.ldb * 2, b + (k * p.ldb + n) * 2, t + 3 * tb,
           t, t + tb, t + 2 * tb};
}
void run_internal(const Plan::Impl& plan, const void* A, const void* B,
                  const float* C0, float* D, void* workspace, cudaStream_t stream) {
  const auto& p = plan.problem;
  if (plan.empty) return;
  if (plan.scale) {
    detail::scale_output(p, C0, D, stream);
    cuda_check(cudaPeekAtLastError(), "launch degenerate output kernel");
    return;
  }
  void* scratch = plan.bytes ? static_cast<unsigned char*>(workspace) + plan.scratch_offset : nullptr;
  if (use_dense(plan, A, B, workspace)) {
    lt_check(execute_lt(plan.handle, *plan.dense, bound_candidate(*plan.dense, A, B, C0, D),
        A, B, C0, D, scratch, stream), "run dense GEMM");
    return;
  }
  if (plan.algorithm == Algorithm::Megakernel) {
    if (plan.policy.mega_variant == 32 || plan.policy.mega_variant == 33) {
      const int configuration = plan.policy.mega_variant - 32;
      if (wgmma_wait1_full_tiles(p, A, B, configuration))
        launch_wgmma_wait1(p, A, B, C0, D, stream, configuration);
      else launch_packed_mma(p, A, B, C0, D, stream, 7);
    } else if (plan.policy.mega_variant == 30 || plan.policy.mega_variant == 31) {
      const int configuration = plan.policy.mega_variant - 30;
      if (wgmma_tma_alias_full_tiles(p, A, B, configuration))
        launch_wgmma_tma_alias(p, A, B, C0, D, stream, configuration);
      else launch_packed_mma(p, A, B, C0, D, stream, 7);
    } else if (plan.policy.mega_variant == 28 || plan.policy.mega_variant == 29) {
      const int configuration = plan.policy.mega_variant - 28;
      if (wgmma_pipeline_full_tiles(p, A, B, configuration))
        launch_wgmma_pipeline(p, A, B, C0, D, stream, configuration);
      else launch_packed_mma(p, A, B, C0, D, stream, 7);
    } else if (plan.policy.mega_variant == 26 || plan.policy.mega_variant == 27) {
      if (cooperative_mma_full_tiles(p, A, B, plan.policy.mega_variant))
        launch_cooperative_mma(p, A, B, C0, D, stream, plan.policy.mega_variant);
      else launch_packed_mma(p, A, B, C0, D, stream, 7);
    } else if (plan.policy.mega_variant == 24 || plan.policy.mega_variant == 25) {
      if (mma8x8_k128_specialized_full_tiles(p, A, B, C0, D))
        launch_mma8x8_k128_specialized(p, A, B, C0, D, stream,
            plan.policy.mega_variant == 24 ? K128OuterLoop::FixedBoundNoUnroll
                                           : K128OuterLoop::FixedBoundFullUnroll);
      else if (mma8x8_prefetch_full_tiles(p, A, B, C0, D))
        launch_mma8x8_prefetch(p, A, B, C0, D, stream, 19);
      else if (mma16x8_prefetch_full_tiles(p, A, B, C0, D))
        launch_mma16x8_prefetch(p, A, B, C0, D, stream, 16);
      else if (mma16x8_identity_full_tiles(p, A, B, C0, D))
        launch_mma16x8_identity(p, A, B, C0, D, stream, 15);
      else if (mma16x8_epilogue_full_tiles(p, A, B, C0, D))
        launch_mma16x8_epilogue(p, A, B, C0, D, stream, 14);
      else if (mma16x8_vec4_full_tiles(p, A, B))
        launch_mma16x8_vec4(p, A, B, C0, D, stream, 12);
      else
        launch_packed_mma(p, A, B, C0, D, stream, 7);
    } else if (plan.policy.mega_variant == 22 || plan.policy.mega_variant == 23) {
      if (coalesced_mma_full_tiles(p, A, B, C0, D, plan.policy.mega_variant))
        launch_coalesced_mma(p, A, B, C0, D, stream, plan.policy.mega_variant);
      else if (mma16x8_prefetch_full_tiles(p, A, B, C0, D))
        launch_mma16x8_prefetch(p, A, B, C0, D, stream, 16);
      else if (mma16x8_identity_full_tiles(p, A, B, C0, D))
        launch_mma16x8_identity(p, A, B, C0, D, stream, 15);
      else if (mma16x8_epilogue_full_tiles(p, A, B, C0, D))
        launch_mma16x8_epilogue(p, A, B, C0, D, stream, 14);
      else if (mma16x8_vec4_full_tiles(p, A, B))
        launch_mma16x8_vec4(p, A, B, C0, D, stream, 12);
      else
        launch_packed_mma(p, A, B, C0, D, stream, 7);
    } else if (plan.policy.mega_variant == 20 || plan.policy.mega_variant == 21) {
      if (direct_mma_full_tiles(p, A, B, C0, D, plan.policy.mega_variant))
        launch_direct_mma(p, A, B, C0, D, stream, plan.policy.mega_variant);
      else if (mma16x8_prefetch_full_tiles(p, A, B, C0, D))
        launch_mma16x8_prefetch(p, A, B, C0, D, stream, 16);
      else if (mma16x8_identity_full_tiles(p, A, B, C0, D))
        launch_mma16x8_identity(p, A, B, C0, D, stream, 15);
      else if (mma16x8_epilogue_full_tiles(p, A, B, C0, D))
        launch_mma16x8_epilogue(p, A, B, C0, D, stream, 14);
      else if (mma16x8_vec4_full_tiles(p, A, B))
        launch_mma16x8_vec4(p, A, B, C0, D, stream, 12);
      else
        launch_packed_mma(p, A, B, C0, D, stream, 7);
    } else if (plan.policy.mega_variant == 19) {
      if (mma8x8_prefetch_full_tiles(p, A, B, C0, D))
        launch_mma8x8_prefetch(p, A, B, C0, D, stream, 19);
      else if (mma16x8_prefetch_full_tiles(p, A, B, C0, D))
        launch_mma16x8_prefetch(p, A, B, C0, D, stream, 16);
      else if (mma16x8_identity_full_tiles(p, A, B, C0, D))
        launch_mma16x8_identity(p, A, B, C0, D, stream, 15);
      else if (mma16x8_epilogue_full_tiles(p, A, B, C0, D))
        launch_mma16x8_epilogue(p, A, B, C0, D, stream, 14);
      else if (mma16x8_vec4_full_tiles(p, A, B))
        launch_mma16x8_vec4(p, A, B, C0, D, stream, 12);
      else
        launch_packed_mma(p, A, B, C0, D, stream, 7);
    } else if (plan.policy.mega_variant == 18)
      launch_small_stage(p, A, B, C0, D, stream, 18);
    else if (plan.policy.mega_variant == 17)
      launch_small_cpasync(p, A, B, C0, D, stream, 17);
    else if (plan.policy.mega_variant == 16) {
      if (mma16x8_prefetch_full_tiles(p, A, B, C0, D))
        launch_mma16x8_prefetch(p, A, B, C0, D, stream, 16);
      else if (mma16x8_identity_full_tiles(p, A, B, C0, D))
        launch_mma16x8_identity(p, A, B, C0, D, stream, 15);
      else if (mma16x8_epilogue_full_tiles(p, A, B, C0, D))
        launch_mma16x8_epilogue(p, A, B, C0, D, stream, 14);
      else if (mma16x8_vec4_full_tiles(p, A, B))
        launch_mma16x8_vec4(p, A, B, C0, D, stream, 12);
      else
        launch_packed_mma(p, A, B, C0, D, stream, 7);
    } else if (plan.policy.mega_variant == 15) {
      if (mma16x8_identity_full_tiles(p, A, B, C0, D))
        launch_mma16x8_identity(p, A, B, C0, D, stream, 15);
      else if (mma16x8_epilogue_full_tiles(p, A, B, C0, D))
        launch_mma16x8_epilogue(p, A, B, C0, D, stream, 14);
      else if (mma16x8_vec4_full_tiles(p, A, B))
        launch_mma16x8_vec4(p, A, B, C0, D, stream, 12);
      else
        launch_packed_mma(p, A, B, C0, D, stream, 7);
    } else if (plan.policy.mega_variant == 14) {
      if (mma16x8_epilogue_full_tiles(p, A, B, C0, D))
        launch_mma16x8_epilogue(p, A, B, C0, D, stream, 14);
      else if (mma16x8_vec4_full_tiles(p, A, B))
        launch_mma16x8_vec4(p, A, B, C0, D, stream, 12);
      else
        launch_packed_mma(p, A, B, C0, D, stream, 7);
    } else if (plan.policy.mega_variant == 13)
      launch_small_product_cluster(p, A, B, C0, D, stream, plan.policy.mega_variant);
    else if (plan.policy.mega_variant == 12) {
      if (mma16x8_vec4_full_tiles(p, A, B))
        launch_mma16x8_vec4(p, A, B, C0, D, stream, plan.policy.mega_variant);
      else
        launch_packed_mma(p, A, B, C0, D, stream, 7);
    } else if (plan.policy.mega_variant == 11) {
      if (mma16x8_full_tiles(p, A, B))
        launch_mma16x8(p, A, B, C0, D, stream, plan.policy.mega_variant);
      else
        launch_packed_mma(p, A, B, C0, D, stream, 7);
    } else if (plan.policy.mega_variant == 10)
      launch_small_product_fast(p, A, B, C0, D, stream, plan.policy.mega_variant);
    else if (plan.policy.mega_variant == 9)
      launch_small_product(p, A, B, C0, D, stream, plan.policy.mega_variant);
    else if (plan.policy.mega_variant >= 7)
      launch_packed_mma(p, A, B, C0, D, stream, plan.policy.mega_variant);
    else if (plan.policy.mega_variant >= 4)
      launch_small_mma(p, A, B, C0, D, stream, plan.policy.mega_variant);
    else
      launch_megakernel(p, A, B, C0, D, stream, plan.policy.mega_variant);
    cuda_check(cudaPeekAtLastError(), "launch Winograd megakernel");
    return;
  }
  auto* base = static_cast<unsigned char*>(workspace);
  float* products = reinterpret_cast<float*>(base + plan.p_offset);
  if (plan.packed_input) {
    detail::pack_winograd_batched_inputs(p, A, B, base + plan.s_offset,
                                       base + plan.t_offset, stream);
    cuda_check(cudaPeekAtLastError(), "launch matched Winograd operand packing");
    launch_packed_leaf(p, base + plan.s_offset, base + plan.t_offset, products,
                       stream, plan.policy.hopper_variant - 11);
    cuda_check(cudaPeekAtLastError(), "launch matched packed-input Winograd products");
    detail::combine_winograd(p, products, C0, D, stream);
    cuda_check(cudaPeekAtLastError(), "launch matched packed-input Winograd combine");
    return;
  }
  if (plan.fused_input) {
    launch_fused_leaf(p, A, B, products, stream, plan.policy.hopper_variant - 9);
    cuda_check(cudaPeekAtLastError(), "launch fused-input Winograd products");
    detail::combine_winograd(p, products, C0, D, stream);
    cuda_check(cudaPeekAtLastError(), "launch fused-input Winograd combine");
    return;
  }
  if (plan.batched || plan.small_batched) {
    const void* left = base + plan.s_offset;
    const void* right = base + plan.t_offset;
    detail::pack_winograd_batched_inputs(p, A, B, base + plan.s_offset,
                                       base + plan.t_offset, stream);
    cuda_check(cudaPeekAtLastError(), "launch uniform Winograd operand packing");
    if (plan.small_batched) {
      if (plan.policy.hopper_variant >= 7)
        launch_small_batched_k64(plan.leaf_problems[0], left, right, products,
                                 stream, plan.policy.hopper_variant);
      else
        launch_small_batched_gemm(plan.leaf_problems[0], left, right, products,
                                  stream, plan.policy.hopper_variant - 5);
    } else
      lt_check(execute_lt(plan.handle, *plan.batched,
          bound_candidate(*plan.batched, left, right, nullptr, products),
          left, right, nullptr, products, scratch, stream), "run batched Winograd products");
    detail::combine_winograd(p, products, C0, D, stream);
    cuda_check(cudaPeekAtLastError(), "launch batched Winograd combine");
    return;
  }
  detail::pack_winograd_inputs(p, A, B, base + plan.s_offset,
                             base + plan.t_offset, stream);
  cuda_check(cudaPeekAtLastError(), "launch Winograd packing kernels");
  std::array<const void*, 7> left, right;
  operands(plan, A, B, workspace, left, right);
  const size_t product_elements = size_t(p.m / 2) * size_t(p.n / 2);
  for (int i = 0; i < 7; ++i) {
    float* output = products + i * product_elements;
    if (plan.algorithm == Algorithm::WinogradHopper)
      launch_hopper_gemm(plan.leaf_problems[i], left[i], right[i], nullptr, output,
                         scratch, stream, plan.policy.hopper_variant);
    else
      lt_check(execute_lt(plan.handle, *plan.leaves[i], bound_candidate(*plan.leaves[i], left[i], right[i], nullptr, output),
          left[i], right[i], nullptr, output, scratch, stream), "run Winograd leaf GEMM");
  }
  detail::combine_winograd(p, products, C0, D, stream);
  cuda_check(cudaPeekAtLastError(), "launch Winograd combine kernel");
}
}  // namespace

Plan::Plan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Plan::~Plan() = default;
Plan::Plan(Plan&&) noexcept = default;
Plan& Plan::operator=(Plan&&) noexcept = default;
size_t Plan::workspace_bytes() const { return impl_->bytes; }
const Problem& Plan::problem() const { return impl_->problem; }
std::string Plan::lt_algorithm_config(const void* A, const void* B, const float* C0,
                                      float* D, void* workspace) const {
  const auto& impl = *impl_;
  validate_binding(impl, A, B, C0, D, workspace);
  if (impl.empty || impl.scale) return "none";
  if (use_dense(impl, A, B, workspace))
    return "dense:" + describe_lt(*impl.dense, bound_candidate(*impl.dense, A, B, C0, D));
  if (impl.algorithm != Algorithm::WinogradLt) return "none";
  auto* base = static_cast<unsigned char*>(workspace);
  float* products = reinterpret_cast<float*>(base + impl.p_offset);
  if (impl.batched) {
    const void* left = base + impl.s_offset;
    const void* right = base + impl.t_offset;
    return "batch7:" + describe_lt(*impl.batched, bound_candidate(*impl.batched, left, right, nullptr, products));
  }
  std::array<const void*, 7> left{}, right{};
  operands(impl, A, B, workspace, left, right);
  const size_t stride = size_t(impl.problem.m / 2) * size_t(impl.problem.n / 2);
  std::string result;
  for (int i = 0; i < 7; ++i) {
    if (i) result += "|";
    float* output = products + size_t(i) * stride;
    result += "leaf" + std::to_string(i) + ":" + describe_lt(*impl.leaves[i],
        bound_candidate(*impl.leaves[i], left[i], right[i], nullptr, output));
  }
  return result;
}
std::string Plan::algorithm_name() const {
  if (impl_->empty) return "empty";
  if (impl_->scale) return "degenerate_scale";
  switch (impl_->algorithm) {
    case Algorithm::WinogradLt: return impl_->batched ? "winograd_lt_batched" : "winograd_lt_v1";
    case Algorithm::WinogradHopper:
      if (impl_->packed_input) return impl_->policy.hopper_variant == 11
        ? "winograd_packed_input_32x32" : "winograd_packed_input_32x64";
      if (impl_->fused_input) return impl_->policy.hopper_variant == 9
        ? "winograd_fused_input_32x32" : "winograd_fused_input_32x64";
      return impl_->small_batched ? (impl_->policy.hopper_variant >= 7
        ? "winograd_sm80_batched_k64" : "winograd_sm80_batched") : "winograd_hopper_v2";
    case Algorithm::Megakernel:
      if (impl_->policy.mega_variant == 32 || impl_->policy.mega_variant == 33) {
        if (wgmma_wait1_can_implement(impl_->problem, impl_->policy.mega_variant - 32))
          return impl_->policy.mega_variant == 32
            ? "megakernel_wgmma_wait1_64x16" : "megakernel_wgmma_wait1_64x32";
        return "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant == 30 || impl_->policy.mega_variant == 31) {
        if (wgmma_tma_alias_can_implement(impl_->problem, impl_->policy.mega_variant - 30))
          return impl_->policy.mega_variant == 30
            ? "megakernel_wgmma_tma_alias_64x16" : "megakernel_wgmma_tma_alias_64x32";
        return "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant == 28 || impl_->policy.mega_variant == 29) {
        if (wgmma_pipeline_can_implement(impl_->problem, impl_->policy.mega_variant - 28))
          return impl_->policy.mega_variant == 28
            ? "megakernel_wgmma_pipeline_64x16" : "megakernel_wgmma_pipeline_64x32";
        return "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant == 26 || impl_->policy.mega_variant == 27) {
        if (cooperative_mma_can_implement(impl_->problem, impl_->policy.mega_variant))
          return impl_->policy.mega_variant == 26
            ? "megakernel_cooperative_32x16" : "megakernel_cooperative_32x32";
        return "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant == 24 || impl_->policy.mega_variant == 25) {
        if (mma8x8_k128_specialized_can_implement(impl_->problem))
          return impl_->policy.mega_variant == 24
            ? "megakernel_warp_mma_8x8_k128_no_unroll" : "megakernel_warp_mma_8x8_k128_unroll";
        if (mma8x8_prefetch_can_implement(impl_->problem, 19))
          return "megakernel_warp_mma_8x8_prefetch";
        if (mma16x8_prefetch_can_implement(impl_->problem, 16))
          return "megakernel_warp_mma_16x8_prefetch";
        if (mma16x8_identity_can_implement(impl_->problem, 15))
          return "megakernel_warp_mma_16x8_identity";
        if (mma16x8_epilogue_can_implement(impl_->problem, 14))
          return "megakernel_warp_mma_16x8_epilogue";
        return mma16x8_vec4_can_implement(impl_->problem, 12)
          ? "megakernel_warp_mma_16x8_vec4" : "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant == 22 || impl_->policy.mega_variant == 23) {
        if (coalesced_mma_can_implement(impl_->problem, impl_->policy.mega_variant))
          return impl_->policy.mega_variant == 22
            ? "megakernel_warp_mma_coalesced_16x8" : "megakernel_warp_mma_coalesced_8x8";
        if (mma16x8_prefetch_can_implement(impl_->problem, 16))
          return "megakernel_warp_mma_16x8_prefetch";
        if (mma16x8_identity_can_implement(impl_->problem, 15))
          return "megakernel_warp_mma_16x8_identity";
        if (mma16x8_epilogue_can_implement(impl_->problem, 14))
          return "megakernel_warp_mma_16x8_epilogue";
        return mma16x8_vec4_can_implement(impl_->problem, 12)
          ? "megakernel_warp_mma_16x8_vec4" : "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant == 20 || impl_->policy.mega_variant == 21) {
        if (direct_mma_can_implement(impl_->problem, impl_->policy.mega_variant))
          return impl_->policy.mega_variant == 20
            ? "megakernel_warp_mma_direct_16x8" : "megakernel_warp_mma_direct_8x8";
        if (mma16x8_prefetch_can_implement(impl_->problem, 16))
          return "megakernel_warp_mma_16x8_prefetch";
        if (mma16x8_identity_can_implement(impl_->problem, 15))
          return "megakernel_warp_mma_16x8_identity";
        if (mma16x8_epilogue_can_implement(impl_->problem, 14))
          return "megakernel_warp_mma_16x8_epilogue";
        return mma16x8_vec4_can_implement(impl_->problem, 12)
          ? "megakernel_warp_mma_16x8_vec4" : "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant == 19) {
        if (mma8x8_prefetch_can_implement(impl_->problem, 19))
          return "megakernel_warp_mma_8x8_prefetch";
        if (mma16x8_prefetch_can_implement(impl_->problem, 16))
          return "megakernel_warp_mma_16x8_prefetch";
        if (mma16x8_identity_can_implement(impl_->problem, 15))
          return "megakernel_warp_mma_16x8_identity";
        if (mma16x8_epilogue_can_implement(impl_->problem, 14))
          return "megakernel_warp_mma_16x8_epilogue";
        return mma16x8_vec4_can_implement(impl_->problem, 12)
          ? "megakernel_warp_mma_16x8_vec4" : "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant == 18) {
        const auto& p = impl_->problem;
        return p.m % 32 == 0 && p.n % 32 == 0 && p.k % 32 == 0 &&
            p.lda % 8 == 0 && p.ldb % 8 == 0
          ? "megakernel_stage_m2" : "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant == 17) {
        const auto& p = impl_->problem;
        return p.m % 32 == 0 && p.n % 32 == 0 && p.k % 32 == 0 &&
            p.lda % 8 == 0 && p.ldb % 8 == 0
          ? "megakernel_warp_mma_cpasync" : "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant == 16) {
        if (mma16x8_prefetch_can_implement(impl_->problem, 16))
          return "megakernel_warp_mma_16x8_prefetch";
        if (mma16x8_identity_can_implement(impl_->problem, 15))
          return "megakernel_warp_mma_16x8_identity";
        if (mma16x8_epilogue_can_implement(impl_->problem, 14))
          return "megakernel_warp_mma_16x8_epilogue";
        return mma16x8_vec4_can_implement(impl_->problem, 12)
          ? "megakernel_warp_mma_16x8_vec4" : "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant == 15) {
        if (mma16x8_identity_can_implement(impl_->problem, 15))
          return "megakernel_warp_mma_16x8_identity";
        if (mma16x8_epilogue_can_implement(impl_->problem, 14))
          return "megakernel_warp_mma_16x8_epilogue";
        return mma16x8_vec4_can_implement(impl_->problem, 12)
          ? "megakernel_warp_mma_16x8_vec4" : "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant == 14) {
        if (mma16x8_epilogue_can_implement(impl_->problem, 14))
          return "megakernel_warp_mma_16x8_epilogue";
        return mma16x8_vec4_can_implement(impl_->problem, 12)
          ? "megakernel_warp_mma_16x8_vec4" : "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant == 13) {
        const auto& p = impl_->problem;
        return p.m % 32 == 0 && p.n % 32 == 0 && p.k % 32 == 0 &&
            p.lda % 8 == 0 && p.ldb % 8 == 0 && p.ldc % 8 == 0 && p.ldd % 8 == 0
          ? "megakernel_product_cluster" : "megakernel_product_warps";
      }
      if (impl_->policy.mega_variant == 12)
        return mma16x8_vec4_can_implement(impl_->problem, 12)
          ? "megakernel_warp_mma_16x8_vec4" : "megakernel_warp_mma_packed";
      if (impl_->policy.mega_variant == 11)
        return mma16x8_can_implement(impl_->problem, 11)
          ? "megakernel_warp_mma_16x8" : "megakernel_warp_mma_packed";
      if (impl_->policy.mega_variant == 10) {
        const auto& p = impl_->problem;
        return p.m % 32 == 0 && p.n % 32 == 0 && p.k % 32 == 0 &&
            p.lda % 8 == 0 && p.ldb % 8 == 0 && p.ldc % 8 == 0 && p.ldd % 8 == 0
          ? "megakernel_product_warps_fast" : "megakernel_product_warps";
      }
      if (impl_->policy.mega_variant == 9) return "megakernel_product_warps";
      if (impl_->policy.mega_variant >= 7) {
        const auto& p = impl_->problem;
        return impl_->policy.mega_variant == 8 && p.m % 32 == 0 && p.n % 32 == 0 && p.k % 32 == 0
          ? "megakernel_warp_mma_fast" : "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant >= 4) return "megakernel_warp_mma";
      if (impl_->policy.mega_variant == 3) {
        const auto& p = impl_->problem;
        return p.m % 128 == 0 && p.n % 256 == 0 && p.k % 64 == 0 && p.beta == 0.0f
          ? "megakernel_m3_pipeline" : "megakernel_m3";
      }
      return impl_->policy.mega_variant == 2 ? "megakernel_m3" : "megakernel_m1";
    default: return "dense_cublaslt";
  }
}
const char* dtype_name(DType dtype) { return dtype == DType::BF16 ? "bf16" : "fp16"; }

std::unique_ptr<Plan> create_plan(Problem p, Policy policy) {
  validate_problem(p);
  if (policy.levels != 1 && policy.levels != 0)
    throw std::invalid_argument("only levels=0 (dense) and levels=1 are implemented; levels=2 requires separate validation");
  if (policy.heuristic_candidates < 1 || policy.heuristic_candidates > 128)
    throw std::invalid_argument("heuristic_candidates must be between 1 and 128");
  if (policy.lt_variant < 0 || policy.lt_variant > 1)
    throw std::invalid_argument("lt_variant must be 0 (views) or 1 (uniform batched operands)");
  if (policy.hopper_variant < 0 || policy.hopper_variant > 12)
    throw std::invalid_argument("hopper_variant must be between 0 and 12");
  if (policy.mega_variant < 0 || policy.mega_variant > 33)
    throw std::invalid_argument("mega_variant must be between 0 and 33");
  if (policy.algorithm < Algorithm::Auto || policy.algorithm > Algorithm::Megakernel)
    throw std::invalid_argument("unsupported algorithm policy");
  auto impl = std::make_unique<Plan::Impl>();
  impl->problem = p;
  impl->policy = policy;
  impl->empty = !p.m || !p.n;
  impl->scale = !p.k || p.alpha == 0.0f;
  if (impl->empty || impl->scale) return std::make_unique<Plan>(std::move(impl));
  // Auto stays on the tuned baseline until an independently validated policy table exists.
  Algorithm desired = policy.algorithm;
  if (desired == Algorithm::Auto || policy.levels == 0 || !winograd_shape(p, policy)) desired = Algorithm::Dense;
  if (desired == Algorithm::Megakernel) {
    const bool supported = (policy.mega_variant == 32 || policy.mega_variant == 33)
                             ? (wgmma_wait1_can_implement(p, policy.mega_variant - 32) || packed_mma_can_implement(p, 7))
                           : (policy.mega_variant == 30 || policy.mega_variant == 31)
                             ? (wgmma_tma_alias_can_implement(p, policy.mega_variant - 30) || packed_mma_can_implement(p, 7))
                           : (policy.mega_variant == 28 || policy.mega_variant == 29)
                             ? (wgmma_pipeline_can_implement(p, policy.mega_variant - 28) || packed_mma_can_implement(p, 7))
                           : (policy.mega_variant == 26 || policy.mega_variant == 27)
                             ? (cooperative_mma_can_implement(p, policy.mega_variant) || packed_mma_can_implement(p, 7))
                           : (policy.mega_variant == 24 || policy.mega_variant == 25)
                             ? (mma8x8_k128_specialized_can_implement(p) ||
                                mma8x8_prefetch_can_implement(p, 19) ||
                                mma16x8_prefetch_can_implement(p, 16) ||
                                mma16x8_identity_can_implement(p, 15) ||
                                mma16x8_epilogue_can_implement(p, 14) ||
                                mma16x8_vec4_can_implement(p, 12) || packed_mma_can_implement(p, 7))
                           : (policy.mega_variant == 22 || policy.mega_variant == 23)
                             ? (coalesced_mma_can_implement(p, policy.mega_variant) ||
                                mma16x8_prefetch_can_implement(p, 16) ||
                                mma16x8_identity_can_implement(p, 15) ||
                                mma16x8_epilogue_can_implement(p, 14) ||
                                mma16x8_vec4_can_implement(p, 12) || packed_mma_can_implement(p, 7))
                           : (policy.mega_variant == 20 || policy.mega_variant == 21)
                             ? (direct_mma_can_implement(p, policy.mega_variant) ||
                                mma16x8_prefetch_can_implement(p, 16) ||
                                mma16x8_identity_can_implement(p, 15) ||
                                mma16x8_epilogue_can_implement(p, 14) ||
                                mma16x8_vec4_can_implement(p, 12) || packed_mma_can_implement(p, 7))
                           : policy.mega_variant == 19 ? (mma8x8_prefetch_can_implement(p, 19) ||
                               mma16x8_prefetch_can_implement(p, 16) ||
                               mma16x8_identity_can_implement(p, 15) ||
                               mma16x8_epilogue_can_implement(p, 14) ||
                               mma16x8_vec4_can_implement(p, 12) || packed_mma_can_implement(p, 7))
                           : policy.mega_variant == 18 ? small_stage_can_implement(p, 18)
                           : policy.mega_variant == 17 ? small_cpasync_can_implement(p, 17)
                           : policy.mega_variant == 16 ? (mma16x8_prefetch_can_implement(p, 16) ||
                               mma16x8_identity_can_implement(p, 15) ||
                               mma16x8_epilogue_can_implement(p, 14) ||
                               mma16x8_vec4_can_implement(p, 12) || packed_mma_can_implement(p, 7))
                           : policy.mega_variant == 15 ? (mma16x8_identity_can_implement(p, 15) ||
                               mma16x8_epilogue_can_implement(p, 14) ||
                               mma16x8_vec4_can_implement(p, 12) || packed_mma_can_implement(p, 7))
                           : policy.mega_variant == 14 ? (mma16x8_epilogue_can_implement(p, 14) ||
                               mma16x8_vec4_can_implement(p, 12) || packed_mma_can_implement(p, 7))
                           : policy.mega_variant == 13 ? small_product_cluster_can_implement(p, policy.mega_variant)
                           : policy.mega_variant == 12 ? (mma16x8_vec4_can_implement(p, 12) || packed_mma_can_implement(p, 7))
                           : policy.mega_variant == 11 ? (mma16x8_can_implement(p, 11) || packed_mma_can_implement(p, 7))
                           : policy.mega_variant == 10 ? small_product_fast_can_implement(p, policy.mega_variant)
                           : policy.mega_variant == 9 ? small_product_can_implement(p, policy.mega_variant)
                           : policy.mega_variant >= 7 ? packed_mma_can_implement(p, policy.mega_variant)
                           : policy.mega_variant >= 4 ? small_mma_can_implement(p, policy.mega_variant)
                                                     : megakernel_can_implement(p, policy.mega_variant);
    if (!supported) desired = Algorithm::Dense;
  }
  if (desired == Algorithm::WinogradHopper && policy.hopper_variant >= 11) {
    // Compact A7/B7/P capacity is independent of the older P-only route.
    // Metadata or budget failures use ordinary GEMM; current pointer guards are
    // rechecked for every run before original inputs are freshly packed.
    if (!packed_leaf_can_implement(p, policy.hopper_variant - 11)) desired = Algorithm::Dense;
    else {
      const auto meta = packed_leaf_metadata(p, policy.hopper_variant - 11);
      impl->s_offset = 0;
      impl->t_offset = aligned(meta.a_bytes);
      impl->p_offset = aligned(checked_add(impl->t_offset, meta.b_bytes));
      impl->scratch_offset = aligned(checked_add(impl->p_offset, meta.p_bytes));
      if (impl->scratch_offset > policy.workspace_limit) desired = Algorithm::Dense;
      else impl->packed_input = true;
    }
  }
  if (desired == Algorithm::WinogradHopper && policy.hopper_variant >= 9 && policy.hopper_variant <= 10) {
    // P-only native allocation. Unsupported metadata uses ordinary GEMM; a
    // later incompatible binding uses the zero-workspace Dense operator below.
    if (!fused_leaf_can_implement(p, policy.hopper_variant - 9)) desired = Algorithm::Dense;
    else {
      const size_t product_bytes = fused_leaf_product_workspace_bytes(p, policy.hopper_variant - 9);
      impl->p_offset = 0;
      impl->scratch_offset = aligned(product_bytes);
      if (impl->scratch_offset > policy.workspace_limit) desired = Algorithm::Dense;
      else impl->fused_input = true;
    }
  }
  if (desired != Algorithm::Dense && desired != Algorithm::Megakernel && !impl->fused_input && !impl->packed_input) {
    const bool small_batched = desired == Algorithm::WinogradHopper && policy.hopper_variant >= 5;
    const bool batched = (desired == Algorithm::WinogradLt && policy.lt_variant == 1) || small_batched;
    const size_t input_slots = batched ? 7 : 4;
    const size_t a_block = checked_mul(checked_mul(size_t(p.m / 2), size_t(p.k / 2)), 2);
    const size_t b_block = checked_mul(checked_mul(size_t(p.k / 2), size_t(p.n / 2)), 2);
    const size_t p_block = checked_mul(checked_mul(size_t(p.m / 2), size_t(p.n / 2)), 4);
    impl->s_offset = 0;
    impl->t_offset = aligned(checked_mul(a_block, input_slots));
    impl->p_offset = aligned(checked_add(impl->t_offset, checked_mul(b_block, input_slots)));
    impl->scratch_offset = aligned(checked_add(impl->p_offset, checked_mul(p_block, 7)));
    if (impl->scratch_offset > policy.workspace_limit) desired = Algorithm::Dense;
    for (int i = 0; i < 7; ++i) {
      Problem leaf;
      leaf.m = p.m / 2; leaf.n = p.n / 2; leaf.k = p.k / 2;
      leaf.dtype = p.dtype;
      leaf.lda = !batched && (i == 0 || i == 1 || i == 3) ? p.lda : leaf.k;
      leaf.ldb = !batched && (i == 0 || i == 1 || i == 2) ? p.ldb : leaf.n;
      leaf.ldc = leaf.ldd = leaf.n;
      impl->leaf_problems[i] = leaf;
      if (desired == Algorithm::WinogradHopper) {
        const bool supported = !small_batched ? hopper_can_implement(leaf, policy.hopper_variant)
          : policy.hopper_variant >= 7 ? small_batched_k64_can_implement(leaf, policy.hopper_variant)
                                     : small_batched_can_implement(leaf, policy.hopper_variant - 5);
        if (!supported) desired = Algorithm::Dense;
      }
    }
  }
  impl->algorithm = desired;
  lt_check(cublasLtCreate(&impl->handle), "create cuBLASLt handle");
  if (desired == Algorithm::Dense) {
    impl->scratch_offset = 0;
    impl->dense = make_lt_op(impl->handle, p, policy.workspace_limit, policy.heuristic_candidates);
    impl->bytes = impl->dense->scratch_bytes();
  } else if (desired == Algorithm::Megakernel) {
    if (policy.mega_variant == 32 || policy.mega_variant == 33) {
      if (wgmma_wait1_can_implement(p, policy.mega_variant - 32))
        prepare_wgmma_wait1(p, policy.mega_variant - 32);
      else prepare_packed_mma(p, 7);
    } else if (policy.mega_variant == 30 || policy.mega_variant == 31) {
      if (wgmma_tma_alias_can_implement(p, policy.mega_variant - 30))
        prepare_wgmma_tma_alias(p, policy.mega_variant - 30);
      else prepare_packed_mma(p, 7);
    } else if (policy.mega_variant == 28 || policy.mega_variant == 29) {
      if (wgmma_pipeline_can_implement(p, policy.mega_variant - 28))
        prepare_wgmma_pipeline(p, policy.mega_variant - 28);
      else prepare_packed_mma(p, 7);
    } else if (policy.mega_variant == 26 || policy.mega_variant == 27) {
      if (cooperative_mma_can_implement(p, policy.mega_variant)) prepare_cooperative_mma(p, policy.mega_variant);
      else prepare_packed_mma(p, 7);
    } else if (policy.mega_variant == 24 || policy.mega_variant == 25) {
      if (mma8x8_k128_specialized_can_implement(p)) prepare_mma8x8_k128_specialized(p);
      else if (mma8x8_prefetch_can_implement(p, 19)) prepare_mma8x8_prefetch(p, 19);
      else if (mma16x8_prefetch_can_implement(p, 16)) prepare_mma16x8_prefetch(p, 16);
      else if (mma16x8_identity_can_implement(p, 15)) prepare_mma16x8_identity(p, 15);
      else if (mma16x8_epilogue_can_implement(p, 14)) prepare_mma16x8_epilogue(p, 14);
      else if (mma16x8_vec4_can_implement(p, 12)) prepare_mma16x8_vec4(p, 12);
      else prepare_packed_mma(p, 7);
    } else if (policy.mega_variant == 22 || policy.mega_variant == 23) {
      if (coalesced_mma_can_implement(p, policy.mega_variant)) prepare_coalesced_mma(p, policy.mega_variant);
      else if (mma16x8_prefetch_can_implement(p, 16)) prepare_mma16x8_prefetch(p, 16);
      else if (mma16x8_identity_can_implement(p, 15)) prepare_mma16x8_identity(p, 15);
      else if (mma16x8_epilogue_can_implement(p, 14)) prepare_mma16x8_epilogue(p, 14);
      else if (mma16x8_vec4_can_implement(p, 12)) prepare_mma16x8_vec4(p, 12);
      else prepare_packed_mma(p, 7);
    } else if (policy.mega_variant == 20 || policy.mega_variant == 21) {
      if (direct_mma_can_implement(p, policy.mega_variant)) prepare_direct_mma(p, policy.mega_variant);
      else if (mma16x8_prefetch_can_implement(p, 16)) prepare_mma16x8_prefetch(p, 16);
      else if (mma16x8_identity_can_implement(p, 15)) prepare_mma16x8_identity(p, 15);
      else if (mma16x8_epilogue_can_implement(p, 14)) prepare_mma16x8_epilogue(p, 14);
      else if (mma16x8_vec4_can_implement(p, 12)) prepare_mma16x8_vec4(p, 12);
      else prepare_packed_mma(p, 7);
    } else if (policy.mega_variant == 19) {
      if (mma8x8_prefetch_can_implement(p, 19)) prepare_mma8x8_prefetch(p, 19);
      else if (mma16x8_prefetch_can_implement(p, 16)) prepare_mma16x8_prefetch(p, 16);
      else if (mma16x8_identity_can_implement(p, 15)) prepare_mma16x8_identity(p, 15);
      else if (mma16x8_epilogue_can_implement(p, 14)) prepare_mma16x8_epilogue(p, 14);
      else if (mma16x8_vec4_can_implement(p, 12)) prepare_mma16x8_vec4(p, 12);
      else prepare_packed_mma(p, 7);
    } else if (policy.mega_variant == 18) prepare_small_stage(p, 18);
    else if (policy.mega_variant == 17) prepare_small_cpasync(p, 17);
    else if (policy.mega_variant == 16) {
      if (mma16x8_prefetch_can_implement(p, 16)) prepare_mma16x8_prefetch(p, 16);
      else if (mma16x8_identity_can_implement(p, 15)) prepare_mma16x8_identity(p, 15);
      else if (mma16x8_epilogue_can_implement(p, 14)) prepare_mma16x8_epilogue(p, 14);
      else if (mma16x8_vec4_can_implement(p, 12)) prepare_mma16x8_vec4(p, 12);
      else prepare_packed_mma(p, 7);
    } else if (policy.mega_variant == 15) {
      if (mma16x8_identity_can_implement(p, 15)) prepare_mma16x8_identity(p, 15);
      else if (mma16x8_epilogue_can_implement(p, 14)) prepare_mma16x8_epilogue(p, 14);
      else if (mma16x8_vec4_can_implement(p, 12)) prepare_mma16x8_vec4(p, 12);
      else prepare_packed_mma(p, 7);
    } else if (policy.mega_variant == 14) {
      if (mma16x8_epilogue_can_implement(p, 14)) prepare_mma16x8_epilogue(p, 14);
      else if (mma16x8_vec4_can_implement(p, 12)) prepare_mma16x8_vec4(p, 12);
      else prepare_packed_mma(p, 7);
    } else if (policy.mega_variant == 13) prepare_small_product_cluster(p, policy.mega_variant);
    else if (policy.mega_variant == 12) {
      if (mma16x8_vec4_can_implement(p, 12)) prepare_mma16x8_vec4(p, 12);
      else prepare_packed_mma(p, 7);
    } else if (policy.mega_variant == 11) {
      if (mma16x8_can_implement(p, 11)) prepare_mma16x8(p, 11);
      else prepare_packed_mma(p, 7);
    } else if (policy.mega_variant == 10) prepare_small_product_fast(p, policy.mega_variant);
    else if (policy.mega_variant == 9) prepare_small_product(p, policy.mega_variant);
    else if (policy.mega_variant >= 7) prepare_packed_mma(p, policy.mega_variant);
    else if (policy.mega_variant >= 4) prepare_small_mma(p, policy.mega_variant);
    else prepare_megakernel(p, policy.mega_variant);
    // Keep the fused route free of matrix workspace, including its unaligned-view fallback.
    impl->scratch_offset = 0;
    impl->dense = make_lt_op(impl->handle, p, 0, policy.heuristic_candidates);
    impl->bytes = 0;
  } else if (impl->packed_input) {
    prepare_packed_leaf(p, policy.hopper_variant - 11);
    // Reserve the complete compact operands/products plus the Dense fallback
    // candidate pool. The custom leaf itself requires no backend scratch.
    const size_t available = policy.workspace_limit - impl->scratch_offset;
    impl->dense = make_lt_op(impl->handle, p, available, policy.heuristic_candidates);
    impl->bytes = checked_add(impl->scratch_offset, impl->dense->scratch_bytes());
  } else if (impl->fused_input) {
    prepare_fused_leaf(p, policy.hopper_variant - 9);
    // Rebindings retain caller-owned P-only capacity. Dense fallback is
    // deliberately limited to zero scratch; no transformed-buffer fallback.
    impl->dense = make_lt_op(impl->handle, p, 0, policy.heuristic_candidates);
    impl->bytes = impl->scratch_offset;
  } else {
    const size_t available = policy.workspace_limit - impl->scratch_offset;
    impl->dense = make_lt_op(impl->handle, p, available, policy.heuristic_candidates);
    size_t scratch = impl->dense->scratch_bytes();
    bool backend_available = true;
    if (desired == Algorithm::WinogradHopper && policy.hopper_variant >= 5) {
      impl->small_batched = true;
      const size_t leaf_scratch = policy.hopper_variant >= 7
        ? small_batched_k64_workspace_bytes(impl->leaf_problems[0], policy.hopper_variant)
        : small_batched_workspace_bytes(impl->leaf_problems[0], policy.hopper_variant - 5);
      scratch = std::max(scratch, leaf_scratch);
    } else if (desired == Algorithm::WinogradLt && policy.lt_variant == 1) {
      const auto& leaf = impl->leaf_problems[0];
      const int64_t a_stride = leaf.m * leaf.k;
      const int64_t b_stride = leaf.k * leaf.n;
      const int64_t p_stride = leaf.m * leaf.n;
      impl->batched = make_lt_op(impl->handle, leaf, available, policy.heuristic_candidates,
                                7, {a_stride, b_stride, p_stride, p_stride});
      backend_available = bool(impl->batched);
      if (impl->batched) scratch = std::max(scratch, impl->batched->scratch_bytes());
    } else {
      for (int i = 0; i < 7; ++i) {
        if (desired == Algorithm::WinogradHopper)
          scratch = std::max(scratch, hopper_workspace_bytes(impl->leaf_problems[i], policy.hopper_variant));
        else {
          impl->leaves[i] = make_lt_op(impl->handle, impl->leaf_problems[i], available, policy.heuristic_candidates);
          scratch = std::max(scratch, impl->leaves[i]->scratch_bytes());
        }
      }
    }
    if (!backend_available || scratch > available) {
      // A backend's metadata budget can exceed the nominal matrix-buffer budget.
      impl->algorithm = Algorithm::Dense;
      impl->scratch_offset = 0;
      for (auto& leaf : impl->leaves) leaf.reset();
      impl->batched.reset();
      impl->small_batched = false;
      impl->dense = make_lt_op(impl->handle, p, policy.workspace_limit, policy.heuristic_candidates);
      impl->bytes = impl->dense->scratch_bytes();
    } else impl->bytes = checked_add(impl->scratch_offset, scratch);
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
    if (!impl.scale) {
      void* scratch = impl.bytes ? static_cast<unsigned char*>(workspace) + impl.scratch_offset : nullptr;
      if (use_dense(impl, A, B, workspace))
        tune_lt(impl.handle, *impl.dense, A, B, original_c, D, scratch, stream);
      else if (impl.algorithm == Algorithm::WinogradLt) {
        auto* base = static_cast<unsigned char*>(workspace);
        float* products = reinterpret_cast<float*>(base + impl.p_offset);
        if (impl.batched) {
          detail::pack_winograd_batched_inputs(impl.problem, A, B, base + impl.s_offset,
                                             base + impl.t_offset, stream);
          cuda_check(cudaPeekAtLastError(), "launch uniform tuning operands");
          tune_lt(impl.handle, *impl.batched, base + impl.s_offset, base + impl.t_offset,
                  nullptr, products, scratch, stream);
        } else {
          detail::pack_winograd_inputs(impl.problem, A, B, base + impl.s_offset,
                                     base + impl.t_offset, stream);
          cuda_check(cudaPeekAtLastError(), "launch tuning transforms");
          std::array<const void*, 7> left, right;
          operands(impl, A, B, workspace, left, right);
          const size_t elements = size_t(impl.problem.m / 2) * size_t(impl.problem.n / 2);
          for (int i = 0; i < 7; ++i)
            tune_lt(impl.handle, *impl.leaves[i], left[i], right[i], nullptr,
                    products + i * elements, scratch, stream);
        }
      }
    }
    run_internal(impl, A, B, original_c, D, workspace, stream);
    cuda_check(cudaStreamSynchronize(stream), "finish explicit plan tuning");
  } catch (...) {
    if (snapshot) cudaFree(snapshot);
    throw;
  }
  if (snapshot) cuda_check(cudaFree(snapshot), "free in-place C0 tuning snapshot");
}
}  // namespace megafmm

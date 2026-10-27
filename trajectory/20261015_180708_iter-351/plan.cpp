#include "megafmm/gemm.h"
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
  cublasLtMatmulPreference_t preference = nullptr;
  lt_check(cublasLtMatmulPreferenceCreate(&preference), "create matmul preference");
  try {
    lt_check(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &scratch_limit, sizeof(scratch_limit)), "set Lt workspace budget");
    // Query the full candidate budget for aligned cudaMalloc inputs, then retain
    // pools for element-aligned rebindings. Restricting every
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
  bool empty = false, scale = false, sw_multisize = false;
  size_t bytes = 0;
  bool strategy = false, strategy_confirmed = false;
  int confirm_candidate = -1;
  float confirm_dense_ms = NAN, confirm_custom_ms = NAN;
  int hopper_selected = -1, sm_count = 0, strategy_input_sets = 1;
  std::array<float,detail::HopperCandidateCount+1> strategy_ms{};
  cublasLtHandle_t handle = nullptr;
  std::unique_ptr<LtOp> dense;
  std::shared_ptr<detail::SWState> sw;
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
void run_internal(const Plan::Impl& plan, const void* A, const void* B,
                  const float* C0, float* D, void* workspace, cudaStream_t stream) {
  if (plan.empty) return;
  if (plan.scale) {
    detail::scale_output(plan.problem, C0, D, stream);
    cuda_check(cudaPeekAtLastError(), "launch degenerate output kernel");
    return;
  }
  if (plan.sw_multisize) {
    detail::run_sw_multisize(plan.problem,A,B,C0,D,stream);
    return;
  }
  if (plan.sw) {
    detail::run_sw_pipeline(*plan.sw,A,B,D,workspace,stream);
    return;
  }
  if (plan.hopper_selected>=0 && detail::hopper_dense(plan.hopper_selected,plan.problem,
      A,B,C0,D,workspace,plan.sm_count,stream,true)) return;
  lt_check(execute_lt(plan.handle, *plan.dense,
      bound_candidate(*plan.dense, A, B, C0, D), A, B, C0, D,
      plan.bytes ? workspace : nullptr, stream), "run dense GEMM");
}
// Rank complete invocations rather than kernel-only timings. Candidates compete
// against the same independently tuned Lt pool. Alternate sweep direction to
// reduce clock/order bias; require a 3% median gain AND all five paired wins.
void tune_strategy(Plan::Impl& plan,const void* A,const void* B,const float* C,
                   float* D,void* workspace,cudaStream_t stream) {
  plan.hopper_selected=-1;plan.strategy_confirmed=false;plan.confirm_candidate=-1;
  plan.confirm_dense_ms=plan.confirm_custom_ms=NAN;
  plan.strategy_ms.fill(NAN);
  if(!plan.sm_count) return;
  std::array<bool,detail::HopperCandidateCount> eligible{};
  for(int id=0;id<detail::HopperCandidateCount;++id)
    eligible[id]=detail::hopper_workspace(plan.problem,id)<=plan.bytes &&
        detail::hopper_dense(id,plan.problem,A,B,C,D,workspace,plan.sm_count,stream,0);
  if(std::none_of(eligible.begin(),eligible.end(),[](bool x){return x;})) return;
  auto& lt=*plan.dense;
  std::vector<size_t> lt_pool;
  for(size_t i=0;i<lt.candidates.size();++i)
    if(compatible(lt,i,A,B,C,D) && std::isfinite(lt.candidate_tuning_ms[i])) lt_pool.push_back(i);
  std::stable_sort(lt_pool.begin(),lt_pool.end(),[&](size_t x,size_t y) {
    return lt.candidate_tuning_ms[x]<lt.candidate_tuning_ms[y];
  });
  if(lt_pool.size()>8) lt_pool.resize(8);
  if(lt_pool.empty()) lt_pool.push_back(lt.selected);
  std::vector<void*> allocations;
  std::vector<const void*> input_a{A},input_b{B};
  float* reference=nullptr;
  cudaEvent_t start=nullptr,stop=nullptr;
  auto cleanup=[&]() {
    if(start) cudaEventDestroy(start);if(stop) cudaEventDestroy(stop);
    if(reference) cudaFree(reference);
    for(void* p:allocations) cudaFree(p);
  };
  auto dense_launch=[&](size_t index,int set) {
    lt_check(execute_lt(plan.handle,lt,index,input_a[set],input_b[set],C,D,workspace,stream),"strategy dense");
  };
  auto custom_launch=[&](int id,int set) {
    if(!detail::hopper_dense(id,plan.problem,input_a[set],input_b[set],C,D,workspace,plan.sm_count,stream,1))
      throw std::runtime_error("strategy candidate binding changed during tuning");
  };
  try {
    const auto& p=plan.problem;
    const size_t abytes=matrix_bytes(p.trans_a?p.k:p.m,p.trans_a?p.m:p.k,p.lda,2);
    const size_t bbytes=matrix_bytes(p.trans_b?p.n:p.k,p.trans_b?p.k:p.n,p.ldb,2);
    // Explicit tuning may allocate. These temporary copies model the caller's
    // rotating working set; they are released before tune returns. No input
    // values or output values are retained by run(). Preserve pointer alignment.
    for(int set=1;set<plan.strategy_input_sets;++set) {
      void* ar=nullptr;void* br=nullptr;
      cuda_check(cudaMalloc(&ar,checked_add(abytes,256)),"strategy A working set");allocations.push_back(ar);
      cuda_check(cudaMalloc(&br,checked_add(bbytes,256)),"strategy B working set");allocations.push_back(br);
      char* ap=static_cast<char*>(ar)+reinterpret_cast<uintptr_t>(A)%256;
      char* bp=static_cast<char*>(br)+reinterpret_cast<uintptr_t>(B)%256;
      cuda_check(cudaMemcpyAsync(ap,A,abytes,cudaMemcpyDeviceToDevice,stream),"strategy copy A");
      cuda_check(cudaMemcpyAsync(bp,B,bbytes,cudaMemcpyDeviceToDevice,stream),"strategy copy B");
      input_a.push_back(ap);input_b.push_back(bp);
    }
    const size_t bytes=matrix_bytes(p.m,p.n,p.ldd,4);
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&reference),bytes),"strategy reference allocation");
    dense_launch(lt.selected,0);
    cuda_check(cudaMemcpyAsync(reference,D,bytes,cudaMemcpyDeviceToDevice,stream),"strategy reference copy");
    for(int id=0;id<detail::HopperCandidateCount;++id) if(eligible[id]) {
      custom_launch(id,0);
      eligible[id]=detail::dense_agreement(p,D,reference,stream);
    }
    // Negative route IDs encode Lt indices. Re-rank the strongest eight Lt
    // candidates under the SAME working set and timing protocol as custom ones.
    std::vector<int> routes;
    for(size_t index:lt_pool) routes.push_back(-int(index)-1);
    for(int id=0;id<detail::HopperCandidateCount;++id) if(eligible[id]) routes.push_back(id);
    std::vector<std::array<float,5>> times(routes.size());
    auto launch=[&](size_t route,int set) {
      if(routes[route]<0) dense_launch(size_t(-routes[route]-1),set);
      else custom_launch(routes[route],set);
    };
    cuda_check(cudaEventCreate(&start),"strategy start event");
    cuda_check(cudaEventCreate(&stop),"strategy stop event");
    constexpr int repetitions=20;
    for(int round=0;round<5;++round) for(size_t index=0;index<routes.size();++index) {
      const size_t route=round%2 ? routes.size()-1-index:index;
      for(int w=0;w<4;++w) launch(route,w%plan.strategy_input_sets);
      cuda_check(cudaEventRecord(start,stream),"strategy record start");
      for(int rep=0;rep<repetitions;++rep) launch(route,rep%plan.strategy_input_sets);
      cuda_check(cudaEventRecord(stop,stream),"strategy record stop");
      cuda_check(cudaEventSynchronize(stop),"strategy wait");
      cuda_check(cudaEventElapsedTime(&times[route][round],start,stop),"strategy elapsed");
      times[route][round]/=repetitions;
    }
    std::vector<float> medians(routes.size());
    size_t baseline=0;
    for(size_t r=0;r<routes.size();++r) {
      auto sorted=times[r];std::sort(sorted.begin(),sorted.end());medians[r]=sorted[2];
      if(routes[r]<0) {if(medians[r]<medians[baseline]) baseline=r;}
      else plan.strategy_ms[routes[r]+1]=medians[r];
    }
    lt.selected=size_t(-routes[baseline]-1);
    lt.selected_tuning_ms=medians[baseline];lt.strategy_reranked=true;
    lt.strategy_input_sets=plan.strategy_input_sets;
    plan.strategy_ms[0]=medians[baseline];
    std::vector<size_t> shortlist;
    for(size_t r=lt_pool.size();r<routes.size();++r) if(medians[r]<medians[baseline]*0.97f) {
      bool stable=true;
      for(int round=0;round<5;++round) stable=stable && times[r][round]<times[baseline][round];
      if(stable) shortlist.push_back(r);
    }
    std::sort(shortlist.begin(),shortlist.end(),[&](size_t x,size_t y){return medians[x]<medians[y];});
    if(shortlist.size()>2) shortlist.resize(2);
    // Independently confirm short-burst winners using full100-call batches.
    // This exposes sustained-power/cache behavior missed by twenty-call ranking.
    for(size_t candidate:shortlist) {
      std::array<float,5> dense_holdout{},custom_holdout{};
      for(int warm=0;warm<20;++warm) {
        launch(baseline,warm%plan.strategy_input_sets);
        launch(candidate,warm%plan.strategy_input_sets);
      }
      cuda_check(cudaStreamSynchronize(stream),"strategy holdout warmup");
      for(int round=0;round<5;++round) for(int order=0;order<2;++order) {
        const bool custom=round%2 ? order==0:order==1;
        const size_t route=custom?candidate:baseline;
        cuda_check(cudaEventRecord(start,stream),"strategy holdout start");
        for(int rep=0;rep<100;++rep) launch(route,rep%plan.strategy_input_sets);
        cuda_check(cudaEventRecord(stop,stream),"strategy holdout stop");
        cuda_check(cudaEventSynchronize(stop),"strategy holdout wait");
        float elapsed=0;
        cuda_check(cudaEventElapsedTime(&elapsed,start,stop),"strategy holdout elapsed");
        (custom?custom_holdout:dense_holdout)[round]=elapsed/100;
      }
      bool stable=true;
      for(int round=0;round<5;++round) stable=stable && custom_holdout[round]<dense_holdout[round];
      auto d=dense_holdout,c=custom_holdout;std::sort(d.begin(),d.end());std::sort(c.begin(),c.end());
      plan.confirm_candidate=routes[candidate];plan.confirm_dense_ms=d[2];plan.confirm_custom_ms=c[2];
      if(stable && c[2]<d[2]*0.97f) {
        plan.hopper_selected=routes[candidate];plan.strategy_confirmed=true;break;
      }
    }
  } catch(...) {cleanup();throw;}
  cleanup();
}
}  // namespace

Plan::Plan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Plan::~Plan() = default;
Plan::Plan(Plan&&) noexcept = default;
Plan& Plan::operator=(Plan&&) noexcept = default;
size_t Plan::workspace_bytes() const { return impl_->bytes; }
const Problem& Plan::problem() const { return impl_->problem; }
std::string Plan::algorithm_name() const {
  if(impl_->strategy && !impl_->empty && !impl_->scale)
    return impl_->hopper_selected>=50 ? "tuned_ampere_dense" : (impl_->hopper_selected==24 || impl_->hopper_selected==25 || ((impl_->hopper_selected>=33 && impl_->hopper_selected<=38) || impl_->hopper_selected>=43)) ? "tuned_warp_mma_dense" : impl_->hopper_selected==4 ? "tuned_simt_small" : impl_->hopper_selected>=5 && impl_->hopper_selected<=8 ? "tuned_padded_hopper" : impl_->hopper_selected>=0 ? "tuned_hopper_dense" : "tuned_dense_cublaslt";
  return impl_->empty ? "empty" : impl_->scale ? "degenerate_scale" : impl_->sw_multisize ? "sw_multisize_wmma" : impl_->sw ? "sw_cooperative_hilo_64x32x256" : "dense_cublaslt";
}
std::string Plan::lt_algorithm_config(const void* A, const void* B, const float* C0,
                                      float* D, void* workspace) const {
  validate_binding(*impl_, A, B, C0, D, workspace);
  if (impl_->empty || impl_->scale) return "none";
  if (impl_->sw_multisize) return "sw_multisize_wmma;tile=16x16;warps=4;leaf_terms=7;levels=1;padding=logical_even;transforms=fp32_to_input_rn;accum=fp32;workspace=0;residual_correction=off";
  if (impl_->sw) return "sw_two_kernel_pdl;warp_roles=1tma+4wgmma;transform_kernel=128x512;shared_carveout=100percent_both;smem=196736;operand_stages=8;bk=128;workspace=4194304;st=compact4planes_balanced_async_p1p2;pack_conversion=fp32_to_pair_rn;FP16_S3T3=direct_packed_rn;publication=programmatic_dependent_launch_grid_dependency_wait;terminal_cta_barrier=off;recombine=private_static_ptx;accum_init=first_mma_overwrite;product_order=1_2_6_7_5_3_4;product_slots=0;product_layout=registers;leaf=Btrans_Atrans;lhs=MN128;rhs=K128;output=transposed_fragment;operand_release=warp_leader;leaf_terms=7;residual_correction=off;current_pointer_maps=4;descriptor_prefetch=warp1_before_init_barrier;alpha=1;beta=0";
  if(impl_->strategy) {
    std::string info="mnk_strategy;selection=";
    const bool bound=impl_->hopper_selected>=0 && detail::hopper_dense(impl_->hopper_selected,
        impl_->problem,A,B,C0,D,workspace,impl_->sm_count,nullptr,false);
    info+=bound ? detail::hopper_candidate_name(impl_->hopper_selected):"dense_cublaslt";
    info+=";rank=5x20_alternating_median;minimum_gain=3percent;require=5paired_wins;validation=dense_relative_1e-5";
    info+=";input_sets="+std::to_string(impl_->strategy_input_sets)+";dense_rerank=top8";
    info+=";confirmation=5x100;confirmation_status="+std::string(impl_->confirm_candidate<0?"skipped":impl_->strategy_confirmed?"passed":"failed");
    if(impl_->confirm_candidate>=0) info+=";confirmed_candidate="+std::string(detail::hopper_candidate_name(impl_->confirm_candidate))+";confirmed_dense_ms="+std::to_string(impl_->confirm_dense_ms)+";confirmed_custom_ms="+std::to_string(impl_->confirm_custom_ms);
    info+=";dense_ms="+std::to_string(impl_->strategy_ms[0]);
    for(int id=0;id<detail::HopperCandidateCount;++id) info+=";"+std::string(detail::hopper_candidate_name(id))+"_ms="+std::to_string(impl_->strategy_ms[id+1]);
    return info+";fallback="+describe_lt(*impl_->dense,bound_candidate(*impl_->dense,A,B,C0,D));
  }
  return "dense:" + describe_lt(*impl_->dense, bound_candidate(*impl_->dense, A, B, C0, D));
}
const char* dtype_name(DType dtype) { return dtype == DType::BF16 ? "bf16" : "fp16"; }
std::unique_ptr<Plan> create_plan(Problem p, Policy policy) {
  validate_problem(p);
  if (policy.algorithm != Algorithm::Auto && policy.algorithm != Algorithm::Dense && policy.algorithm != Algorithm::SWPipeline && policy.algorithm != Algorithm::Tuned)
    throw std::invalid_argument("unsupported algorithm: old SW implementations have been removed");
  if (policy.heuristic_candidates < 1 || policy.heuristic_candidates > 128)
    throw std::invalid_argument("heuristic_candidates must be between 1 and 128");
  if(policy.tuning_input_sets<1 || policy.tuning_input_sets>4)
    throw std::invalid_argument("tuning_input_sets must be between 1 and 4");
  auto impl = std::make_unique<Plan::Impl>();
  impl->strategy_input_sets=policy.tuning_input_sets;
  impl->problem = p;
  impl->strategy_ms.fill(NAN);
  impl->empty = !p.m || !p.n;
  impl->scale = !p.k || p.alpha == 0.0f;
  if (policy.algorithm==Algorithm::SWPipeline && !impl->empty && !impl->scale) {
    const bool specialized=p.m==1024 && p.n==1024 && p.k==1024 &&
        !p.trans_a && !p.trans_b && p.lda==1024 && p.ldb==1024 &&
        p.ldd==1024 && p.alpha==1.0f && p.beta==0.0f;
    if (specialized) {
      impl->bytes=size_t(4)<<20;
      if (policy.workspace_limit<impl->bytes)
        throw std::invalid_argument("SW 1024 pipeline needs 4 MiB workspace");
      impl->sw=detail::make_sw_pipeline(p);
    } else {
      int device; cudaDeviceProp prop{};
      cuda_check(cudaGetDevice(&device),"SW device");
      cuda_check(cudaGetDeviceProperties(&prop,device),"SW properties");
      if (prop.major!=9 || prop.minor!=0)
        throw std::invalid_argument("MegaFMM SW requires Hopper sm90");
      impl->sw_multisize=true;
    }
  } else if (!impl->empty && !impl->scale) {
    lt_check(cublasLtCreate(&impl->handle), "create cuBLASLt handle");
    impl->dense = make_lt_op(impl->handle, p, policy.workspace_limit, policy.heuristic_candidates);
    impl->bytes = impl->dense->scratch_bytes();
    impl->strategy=policy.algorithm==Algorithm::Tuned;
    if(impl->strategy) {
      int device;cudaDeviceProp prop{};
      cuda_check(cudaGetDevice(&device),"strategy device");
      cuda_check(cudaGetDeviceProperties(&prop,device),"strategy properties");
      if(prop.major==9 && prop.minor==0) {
        impl->sm_count=prop.multiProcessorCount;
        for(int id=5;id<detail::HopperCandidateCount;++id) {
          const size_t required=detail::hopper_workspace(p,id);
          if(required<=policy.workspace_limit) impl->bytes=std::max(impl->bytes,required);
        }
        for(int id=0;id<detail::HopperCandidateCount;++id) if(id!=4 && !(id>=5 && id<=8))
          detail::hopper_dense(id,p,nullptr,nullptr,nullptr,nullptr,nullptr,impl->sm_count,nullptr,2);
      }
    }
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
    if (!impl.scale && !impl.sw && !impl.sw_multisize)
      tune_lt(impl.handle, *impl.dense, A, B, original_c, D,
              impl.bytes ? workspace : nullptr, stream);
    if(impl.strategy) tune_strategy(impl,A,B,original_c,D,workspace,stream);
    run_internal(impl, A, B, original_c, D, workspace, stream);
    cuda_check(cudaStreamSynchronize(stream), "finish explicit plan tuning");
  } catch (...) {
    if (snapshot) cudaFree(snapshot);
    throw;
  }
  if (snapshot) cuda_check(cudaFree(snapshot), "free in-place C0 tuning snapshot");
}
}  // namespace megafmm

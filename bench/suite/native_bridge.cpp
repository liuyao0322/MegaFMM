// C ABI for the Python application suite.  All CUDA allocations remain owned
// by PyTorch; pointers and the caller's current CUDA stream are passed through.
#include "megafmm/gemm.h"
#include <cublas_v2.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
thread_local char last_error[4096] = {};
thread_local std::string runtime_metadata;
using Clock = std::chrono::steady_clock;

struct Handle {
  std::unique_ptr<megafmm::Plan> plan;
  const void* a = nullptr;
  const void* b = nullptr;
  float* out = nullptr;
  void* workspace = nullptr;
  std::string metadata;
  double create_ms = 0;
  double tune_ms = 0;
  bool tuned = false;
};

template <class F> int boundary(F&& fn) noexcept {
  last_error[0] = '\0';
  try { fn(); return 0; }
  catch (const std::exception& e) {
    std::snprintf(last_error, sizeof(last_error), "%s", e.what());
  } catch (...) {
    std::snprintf(last_error, sizeof(last_error), "unknown native exception");
  }
  return -1;
}

Handle& checked(void* p) {
  if (!p) throw std::invalid_argument("null suite plan handle");
  return *static_cast<Handle*>(p);
}

std::string quote(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) out << "\\u" << std::hex << std::setw(4)
                          << std::setfill('0') << unsigned(c) << std::dec;
        else out << char(c);
    }
  }
  out << '"';
  return out.str();
}

// JSON has no NaN or infinity.  Preserve nonfinite diagnostics as strings
// instead of silently dropping failed candidates or emitting invalid JSON.
template <class T> void number(std::ostream& out, T value) {
  if (std::isnan(double(value))) out << "\"nan\"";
  else if (std::isinf(double(value))) out << (value > 0 ? "\"+inf\"" : "\"-inf\"");
  else out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
}

template <class T> void numbers(std::ostream& out, const std::vector<T>& values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    number(out, values[i]);
  }
  out << ']';
}

std::string describe(Handle& h) {
  const auto& p = h.plan->problem();
  std::ostringstream out;
  out << "{\"actual_route\":" << quote(h.plan->algorithm_name())
      << ",\"dtype\":" << quote(megafmm::dtype_name(p.dtype))
      << ",\"m\":" << p.m << ",\"n\":" << p.n << ",\"k\":" << p.k
      << ",\"workspace_bytes\":" << h.plan->workspace_bytes()
      << ",\"tuned\":" << (h.tuned ? "true" : "false")
      << ",\"plan_create_ms\":";
  number(out, h.create_ms);
  out << ",\"tune_ms\":";
  number(out, h.tune_ms);
  if (h.out) out << ",\"configuration\":" << quote(h.plan->lt_algorithm_config(
      h.a, h.b, nullptr, h.out, h.workspace));
  out << ",\"sw_candidates\":[";
  const auto& candidates = h.plan->sw_tuning_results();
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (i) out << ',';
    const auto& c = candidates[i];
    out << "{\"id\":" << c.id << ",\"config\":" << quote(c.config)
        << ",\"status\":" << quote(c.status)
        << ",\"workspace_bytes\":" << c.workspace_bytes
        << ",\"shared_bytes\":" << c.shared_bytes
        << ",\"registers_per_thread\":" << c.registers_per_thread
        << ",\"resident_ctas\":" << c.resident_ctas
        << ",\"numerical_pass\":" << (c.numerical_pass ? "true" : "false")
        << ",\"physical_work_ratio\":";
    number(out, c.physical_work_ratio);
    out << ",\"model_score\":"; number(out, c.model_score);
    out << ",\"relative_error\":"; number(out, c.relative_error);
    out << ",\"gate_limit\":"; number(out, c.gate_limit);
    out << ",\"precision_errors\":"; numbers(out, c.precision_errors);
    out << ",\"precision_limits\":"; numbers(out, c.precision_limits);
    out << ",\"ranking_ms\":"; numbers(out, c.ranking_ms);
    out << ",\"confirmation_ms\":"; numbers(out, c.confirmation_ms);
    out << '}';
  }
  out << "]}";
  return out.str();
}
}  // namespace

extern "C" {
int megafmm_suite_abi_version() noexcept { return 1; }
const char* megafmm_suite_last_error() noexcept { return last_error; }
const char* megafmm_suite_runtime_metadata() noexcept {
  const char* result = nullptr;
  boundary([&] {
    int runtime = 0, driver = 0, cublas = 0;
    auto status = cudaRuntimeGetVersion(&runtime);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status = cudaDriverGetVersion(&driver);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    cublasHandle_t handle = nullptr;
    auto bs = cublasCreate(&handle);
    if (bs != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed during runtime metadata");
    bs = cublasGetVersion(handle, &cublas);
    cublasDestroy(handle);
    if (bs != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGetVersion failed during runtime metadata");
    runtime_metadata = "{\"cuda_runtime_version\":" + std::to_string(runtime) +
      ",\"cuda_driver_version\":" + std::to_string(driver) +
      ",\"cublas_version\":" + std::to_string(cublas) + "}";
    result = runtime_metadata.c_str();
  });
  return result;
}

// dtype: 0 BF16, 1 FP16, 2 FP32.  algorithm: 0 Dense, 1 SWTuned,
// 2 fixed SWPipeline.  The Dense plan independently tunes cuBLASLt.
void* megafmm_suite_create(int64_t m, int64_t n, int64_t k, int dtype,
                          int algorithm, uint64_t workspace_limit,
                          int heuristic_candidates, int tuning_input_sets,
                          int sw_candidates, int graph, int allow_inaccurate) noexcept {
  Handle* result = nullptr;
  boundary([&] {
    if (dtype < 0 || dtype > 2) throw std::invalid_argument("invalid suite dtype");
    if (algorithm < 0 || algorithm > 2) throw std::invalid_argument("invalid suite algorithm");
    megafmm::Problem p;
    p.m = m; p.n = n; p.k = k;
    p.dtype = static_cast<megafmm::DType>(dtype);
    megafmm::Policy policy;
    policy.algorithm = algorithm == 0 ? megafmm::Algorithm::Dense :
                       algorithm == 1 ? megafmm::Algorithm::SWTuned :
                                        megafmm::Algorithm::SWPipeline;
    policy.workspace_limit = size_t(workspace_limit);
    policy.heuristic_candidates = heuristic_candidates;
    policy.tuning_input_sets = tuning_input_sets;
    policy.sw_tuning_candidates = sw_candidates;
    policy.sw_tuning_graph = graph != 0;
    policy.sw_tuning_allow_inaccurate = allow_inaccurate != 0;
    auto h = std::make_unique<Handle>();
    const auto start = Clock::now();
    h->plan = megafmm::create_plan(p, policy);
    h->create_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    result = h.release();
  });
  return result;
}

int megafmm_suite_workspace(void* handle, uint64_t* bytes) noexcept {
  return boundary([&] {
    if (!bytes) throw std::invalid_argument("null workspace size output");
    *bytes = checked(handle).plan->workspace_bytes();
  });
}

int megafmm_suite_tune(void* handle, const void* a, const void* b, float* out,
                       void* workspace, void* stream) noexcept {
  return boundary([&] {
    auto& h = checked(handle);
    h.a = a; h.b = b; h.out = out; h.workspace = workspace;
    const auto start = Clock::now();
    try {
      megafmm::tune(*h.plan, a, b, nullptr, out, workspace,
                    reinterpret_cast<cudaStream_t>(stream));
    } catch (...) {
      h.tune_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
      throw;
    }
    h.tune_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    h.tuned = true;
  });
}

int megafmm_suite_run(void* handle, const void* a, const void* b, float* out,
                      void* workspace, void* stream) noexcept {
  return boundary([&] {
    auto& h = checked(handle);
    if (!h.tuned) throw std::logic_error("suite plan must be tuned before run");
    megafmm::run(*h.plan, a, b, nullptr, out, workspace,
                 reinterpret_cast<cudaStream_t>(stream));
    h.a = a; h.b = b; h.out = out; h.workspace = workspace;
  });
}

const char* megafmm_suite_metadata(void* handle) noexcept {
  const char* result = nullptr;
  boundary([&] { auto& h = checked(handle); h.metadata = describe(h); result = h.metadata.c_str(); });
  return result;
}

int megafmm_suite_destroy(void* handle) noexcept {
  return boundary([&] { delete static_cast<Handle*>(handle); });
}
}  // extern "C"

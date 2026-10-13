#include "megafmm/gemm.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS status " + std::to_string(int(status)));
}
struct Buffer {
  void* data = nullptr;
  explicit Buffer(size_t bytes = 0) { if (bytes) check(cudaMalloc(&data, bytes)); }
  ~Buffer() { if (data) cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  template <class T> T* as() const { return static_cast<T*>(data); }
};
struct Stream {
  cudaStream_t value;
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
struct Event {
  cudaEvent_t value;
  Event() { check(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
};
struct Blas {
  cublasHandle_t value;
  explicit Blas(cudaStream_t stream) { check(cublasCreate(&value)); check(cublasSetStream(value, stream)); }
  ~Blas() { cublasDestroy(value); }
};
template <class T> __device__ float as_float(T value) { return float(value); }
template <> __device__ float as_float(__nv_bfloat16 value) { return __bfloat162float(value); }
template <class T> __device__ T from_float(float value);
template <> __device__ __half from_float(float value) { return __float2half_rn(value); }
template <> __device__ __nv_bfloat16 from_float(float value) { return __float2bfloat16_rn(value); }
__device__ uint64_t hash64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
template <class T> __global__ void fill_input(T* output, int64_t count, uint64_t seed) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += int64_t(gridDim.x) * blockDim.x) {
    float value = float(hash64(uint64_t(i) ^ seed) >> 40) * (2.0f / 16777216.0f) - 1.0f;
    output[i] = from_float<T>(value);
  }
}
template <class T> __global__ void to_double(const T* input, double* output, int64_t count) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += int64_t(gridDim.x) * blockDim.x)
    output[i] = double(as_float(input[i]));
}
struct Partial {
  double err2, ref2, maxabs, scaled, nearzero, nonfinite;
};
__global__ void error_partials(const float* actual, const double* reference,
                               int64_t count, double atol, double rtol, Partial* output) {
  __shared__ Partial scratch[256];
  Partial p{};
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += int64_t(gridDim.x) * blockDim.x) {
    double v = actual[i], ref = reference[i];
    if (!isfinite(v) || !isfinite(ref)) { p.nonfinite += 1; continue; }
    double err = fabs(v - ref);
    p.err2 += err * err;
    p.ref2 += ref * ref;
    p.maxabs = fmax(p.maxabs, err);
    p.scaled = fmax(p.scaled, err / (atol + rtol * fabs(ref)));
    if (fabs(ref) <= atol) p.nearzero = fmax(p.nearzero, err);
  }
  scratch[threadIdx.x] = p;
  __syncthreads();
  for (int stride = 128; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      Partial a = scratch[threadIdx.x], b = scratch[threadIdx.x + stride];
      a.err2 += b.err2; a.ref2 += b.ref2; a.nonfinite += b.nonfinite;
      a.maxabs = fmax(a.maxabs, b.maxabs); a.scaled = fmax(a.scaled, b.scaled);
      a.nearzero = fmax(a.nearzero, b.nearzero); scratch[threadIdx.x] = a;
    }
    __syncthreads();
  }
  if (!threadIdx.x) output[blockIdx.x] = scratch[0];
}
__global__ void norm_partials(const double* input, int64_t count, double* output) {
  __shared__ double scratch[256];
  double sum = 0;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += int64_t(gridDim.x) * blockDim.x) sum += input[i] * input[i];
  scratch[threadIdx.x] = sum; __syncthreads();
  for (int stride = 128; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (!threadIdx.x) output[blockIdx.x] = scratch[0];
}
constexpr int reduction_blocks = 256;
struct Error {
  double maxabs = 0, frobenius = 0, scaled = 0, normalized = 0, nearzero = 0;
  uint64_t nonfinite = 0;
};
double norm2(const double* input, int64_t count, cudaStream_t stream, Buffer& scratch) {
  norm_partials<<<reduction_blocks, 256, 0, stream>>>(input, count, scratch.as<double>());
  check(cudaGetLastError());
  std::vector<double> host(reduction_blocks);
  check(cudaMemcpyAsync(host.data(), scratch.data, host.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  return std::accumulate(host.begin(), host.end(), 0.0);
}
Error error(const float* actual, const double* reference, int64_t count,
            double atol, double rtol, double norm_ab, cudaStream_t stream, Buffer& scratch) {
  error_partials<<<reduction_blocks, 256, 0, stream>>>(actual, reference, count, atol, rtol, scratch.as<Partial>());
  check(cudaGetLastError());
  std::vector<Partial> host(reduction_blocks);
  check(cudaMemcpyAsync(host.data(), scratch.data, host.size() * sizeof(Partial), cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  Partial p{};
  for (const auto& x : host) {
    p.err2 += x.err2; p.ref2 += x.ref2; p.nonfinite += x.nonfinite;
    p.maxabs = std::max(p.maxabs, x.maxabs); p.scaled = std::max(p.scaled, x.scaled);
    p.nearzero = std::max(p.nearzero, x.nearzero);
  }
  Error e;
  e.maxabs = p.maxabs; e.scaled = p.scaled; e.nearzero = p.nearzero;
  e.frobenius = std::sqrt(p.err2 / std::max(p.ref2, std::numeric_limits<double>::min()));
  e.normalized = std::sqrt(p.err2) / std::max(norm_ab, std::numeric_limits<double>::min());
  e.nonfinite = uint64_t(p.nonfinite);
  return e;
}
void keep_worst(Error& a, const Error& b) {
  a.maxabs = std::max(a.maxabs, b.maxabs); a.frobenius = std::max(a.frobenius, b.frobenius);
  a.scaled = std::max(a.scaled, b.scaled); a.normalized = std::max(a.normalized, b.normalized);
  a.nearzero = std::max(a.nearzero, b.nearzero); a.nonfinite += b.nonfinite;
}
struct Options {
  std::vector<std::tuple<int, int, int>> shapes = {
    {512,512,512}, {1024,1024,1024}, {2048,2048,2048}, {4096,4096,4096},
    {8192,8192,8192}, {16384,16384,16384}, {8192,4096,8192},
    {4096,8192,8192}, {8192,8192,4096}, {8192,1024,512}, {1024,8192,512},
    {1025,1537,769}, {514,770,258}, {1,1,1}};
  std::string dtype = "both", algorithm = "lt", output;
  int trials = 100, warmup = 20, groups = 5, correct_trials = 3;
  int lt_variant = 0, mega_variant = 0, hopper_variant = 0;
  int rotate_inputs = 1;
  bool graph = false, profile = false;
};
std::vector<std::tuple<int,int,int>> parse_shapes(std::string text) {
  std::vector<std::tuple<int,int,int>> result;
  std::istringstream all(text); std::string item;
  while (std::getline(all, item, ';')) {
    std::replace(item.begin(), item.end(), ',', ' ');
    std::replace(item.begin(), item.end(), 'x', ' ');
    std::istringstream line(item); int m,n,k; std::string extra;
    if (!(line >> m >> n >> k) || (line >> extra) || m <= 0 || n <= 0 || k <= 0)
      throw std::runtime_error("invalid shape: " + item);
    result.emplace_back(m,n,k);
  }
  if (result.empty()) throw std::runtime_error("empty --shapes");
  return result;
}
Options options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i], value;
    size_t equal = arg.find('=');
    if (equal != std::string::npos) { value = arg.substr(equal + 1); arg.resize(equal); }
    auto next = [&]() { if (!value.empty()) return value; if (++i == argc) throw std::runtime_error("missing value for " + arg); return std::string(argv[i]); };
    if (arg == "--help") {
      std::cout << "megafmm_bench --shapes 'M,N,K;...' --dtype bf16|fp16|both "
        "--algorithm lt|hopper|mega --lt-variant 0|1 --mega-variant N --hopper-variant 0..6 "
        "--trials N --warmup N --groups N --correct-trials N --rotate-inputs N --graph --output CSV --profile\n";
      std::exit(0);
    } else if (arg == "--shapes") o.shapes = parse_shapes(next());
    else if (arg == "--dtype") o.dtype = next();
    else if (arg == "--algorithm") o.algorithm = next();
    else if (arg == "--output") o.output = next();
    else if (arg == "--trials") o.trials = std::stoi(next());
    else if (arg == "--warmup") o.warmup = std::stoi(next());
    else if (arg == "--groups") o.groups = std::stoi(next());
    else if (arg == "--correct-trials") o.correct_trials = std::stoi(next());
    else if (arg == "--lt-variant") o.lt_variant = std::stoi(next());
    else if (arg == "--mega-variant") o.mega_variant = std::stoi(next());
    else if (arg == "--hopper-variant") o.hopper_variant = std::stoi(next());
    else if (arg == "--rotate-inputs") o.rotate_inputs = std::stoi(next());
    else if (arg == "--graph") o.graph = true;
    else if (arg == "--profile") o.profile = true;
    else throw std::runtime_error("unknown option: " + arg);
  }
  if (o.dtype != "bf16" && o.dtype != "fp16" && o.dtype != "both") throw std::runtime_error("invalid dtype");
  if (o.algorithm != "lt" && o.algorithm != "hopper" && o.algorithm != "mega") throw std::runtime_error("invalid algorithm");
  if (o.lt_variant < 0 || o.lt_variant > 1) throw std::runtime_error("lt-variant must be 0 or 1");
  if (o.hopper_variant < 0 || o.hopper_variant > 6) throw std::runtime_error("hopper-variant must be between 0 and 6");
  if (o.trials <= 0 || o.groups <= 0 || o.warmup < 0 || o.correct_trials < 0 || o.rotate_inputs <= 0) throw std::runtime_error("invalid trial count");
  return o;
}
struct Invocation {
  const megafmm::Plan& plan;
  std::vector<const void*> a,b; float* d; void* workspace; cudaStream_t stream;
  std::vector<cudaGraph_t> graphs;
  std::vector<cudaGraphExec_t> executables;
  Invocation(const megafmm::Plan& p, const std::vector<const void*>& aa, const std::vector<const void*>& bb, float* dd,
             void* w, cudaStream_t s, bool use_graph) : plan(p), a(aa), b(bb), d(dd), workspace(w), stream(s) {
    if (use_graph) {
      for (size_t i=0;i<a.size();++i) {
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        megafmm::run(plan, a[i], b[i], nullptr, d, workspace, stream);
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, 0));
        graphs.push_back(graph);executables.push_back(executable);
      }
    }
  }
  ~Invocation() {
    for(auto executable:executables) cudaGraphExecDestroy(executable);
    for(auto graph:graphs) cudaGraphDestroy(graph);
  }
  void run(size_t iteration=0) const {
    size_t index=iteration%a.size();
    if (!executables.empty()) check(cudaGraphLaunch(executables[index], stream));
    else megafmm::run(plan, a[index], b[index], nullptr, d, workspace, stream);
  }
};
struct Measurement { double event_ms, wall_ms; };
Measurement measure(const Invocation& call, int trials) {
  Event start, stop;
  check(cudaStreamSynchronize(call.stream));
  auto wall_start = Clock::now();
  check(cudaEventRecord(start.value, call.stream));
  for (int i = 0; i < trials; ++i) call.run(size_t(i));
  check(cudaEventRecord(stop.value, call.stream));
  check(cudaEventSynchronize(stop.value));
  auto wall_stop = Clock::now();
  float ms; check(cudaEventElapsedTime(&ms, start.value, stop.value));
  return {double(ms) / trials, std::chrono::duration<double,std::milli>(wall_stop-wall_start).count() / trials};
}
struct Stats { double median, p10, p90, cv; };
Stats stats(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  auto quantile = [&](double q) {
    double pos = q * (samples.size()-1); size_t low = size_t(pos), high = std::min(low+1,samples.size()-1);
    return samples[low] + (samples[high]-samples[low]) * (pos-low);
  };
  double mean = std::accumulate(samples.begin(),samples.end(),0.0) / samples.size(), variance = 0;
  for (double x : samples) variance += (x-mean)*(x-mean);
  return {quantile(.5), quantile(.1), quantile(.9), std::sqrt(variance / samples.size()) / mean};
}
std::string csv_string(const std::string& input) {
  std::string output = "\"";
  for (char c : input) { output += c; if (c == '"') output += c; }
  return output + "\"";
}
constexpr const char* csv_header = "m,n,k,dtype,requested,actual,variant,mode,cache,rotate_inputs,input_working_set_bytes,quantile_unit,trials,groups,correct_trials,workspace_limit,dense_workspace,solution_workspace,setup_ms,tune_ms,graph_setup_ms,dense_event_ms,dense_p10_ms,dense_p90_ms,dense_cv,solution_event_ms,solution_p10_ms,solution_p90_ms,solution_cv,dense_wall_ms,solution_wall_ms,speedup,equivalent_tflops,atol,rtol,dense_maxabs,dense_ef,dense_scaled,dense_normalized,dense_nearzero_abs,solution_maxabs,solution_ef,solution_scaled,solution_normalized,solution_nearzero_abs,nonfinite,gate_floor,gate_limit,numerical_pass\n";
template <class T> bool benchmark(const Options& o, int m, int n, int k, megafmm::DType dtype, std::ostream& csv) {
  Stream stream;
  megafmm::Problem problem;
  problem.m=m; problem.n=n; problem.k=k; problem.lda=k; problem.ldb=n; problem.ldc=n; problem.ldd=n; problem.dtype=dtype;
  megafmm::Policy policy;
  policy.certified_range=true; policy.workspace_limit=size_t(8)<<30;
  policy.lt_variant=o.lt_variant; policy.hopper_variant=o.hopper_variant; policy.mega_variant=o.mega_variant;
  policy.algorithm=o.algorithm=="lt" ? megafmm::Algorithm::WinogradLt :
    o.algorithm=="hopper" ? megafmm::Algorithm::WinogradHopper : megafmm::Algorithm::Megakernel;
  auto setup_start=Clock::now();
  auto solution=megafmm::create_plan(problem,policy);
  policy.algorithm=megafmm::Algorithm::Dense;
  std::unique_ptr<megafmm::Plan> dense;
  if (!o.profile) dense=megafmm::create_plan(problem,policy);
  size_t dense_workspace=dense ? dense->workspace_bytes() : 0;
  size_t solution_workspace=solution->workspace_bytes();
  if (dense_workspace>policy.workspace_limit || solution_workspace>policy.workspace_limit) throw std::runtime_error("plan exceeds equal workspace limit");
  int64_t ac=int64_t(m)*k, bc=int64_t(k)*n, dc=int64_t(m)*n;
  Buffer a(size_t(ac)*sizeof(T)), b(size_t(bc)*sizeof(T)), sd(size_t(dc)*sizeof(float));
  Buffer dd(o.profile ? 0 : size_t(dc)*sizeof(float));
  Buffer workspace(std::max(dense_workspace,solution_workspace));
  std::vector<std::unique_ptr<Buffer>> rotation_a,rotation_b;
  std::vector<const void*> input_a{a.data},input_b{b.data};
  if (!o.profile) for(int i=1;i<o.rotate_inputs;++i) {
    rotation_a.push_back(std::make_unique<Buffer>(size_t(ac)*sizeof(T)));
    rotation_b.push_back(std::make_unique<Buffer>(size_t(bc)*sizeof(T)));
    uint64_t seed=1234567+uint64_t(i)*0xbf58476d1ce4e5b9ULL;
    fill_input<<<1024,256,0,stream.value>>>(rotation_a.back()->as<T>(),ac,seed);
    fill_input<<<1024,256,0,stream.value>>>(rotation_b.back()->as<T>(),bc,seed^0x632be59bd9b4e019ULL);
    check(cudaGetLastError());
    input_a.push_back(rotation_a.back()->data);input_b.push_back(rotation_b.back()->data);
  }
  auto fill=[&](uint64_t seed) {
    fill_input<<<1024,256,0,stream.value>>>(a.as<T>(),ac,seed);
    fill_input<<<1024,256,0,stream.value>>>(b.as<T>(),bc,seed ^ 0x632be59bd9b4e019ULL);
    check(cudaGetLastError());
  };
  fill(1234567);
  check(cudaStreamSynchronize(stream.value));
  double setup_ms=std::chrono::duration<double,std::milli>(Clock::now()-setup_start).count();
  auto tune_start=Clock::now();
  if (dense) megafmm::tune(*dense,a.data,b.data,nullptr,dd.as<float>(),workspace.data,stream.value);
  megafmm::tune(*solution,a.data,b.data,nullptr,sd.as<float>(),workspace.data,stream.value);
  check(cudaStreamSynchronize(stream.value));
  double tune_ms=std::chrono::duration<double,std::milli>(Clock::now()-tune_start).count();
  if (o.profile) {
    // Exactly one complete measured invocation; tune kernels precede this invocation.
    check(cudaProfilerStart());
    megafmm::run(*solution,a.data,b.data,nullptr,sd.as<float>(),workspace.data,stream.value);
    check(cudaStreamSynchronize(stream.value));
    check(cudaProfilerStop());
    std::cerr << "PROFILE " << m << ',' << n << ',' << k << ' ' << megafmm::dtype_name(dtype)
              << ' ' << solution->algorithm_name() << " workspace=" << solution_workspace << '\n';
    return true;
  }
  Error de, se;
  const double floor=dtype==megafmm::DType::BF16 ? 1e-3 : 1e-4;
  const double atol=floor, rtol=dtype==megafmm::DType::BF16 ? 1e-2 : 1e-3;
  bool pass=true; double gate_limit=0;
  if (o.correct_trials) {
    // The reference sees original rounded target-dtype inputs, without any Winograd transforms.
    Buffer a64(size_t(ac)*sizeof(double)), b64(size_t(bc)*sizeof(double)), reference(size_t(dc)*sizeof(double));
    Buffer scratch(reduction_blocks*sizeof(Partial));
    Blas reference_blas(stream.value);
    for (int trial=0; trial<o.correct_trials; ++trial) {
      fill(1234567+uint64_t(trial+1)*0x9e3779b97f4a7c15ULL);
      to_double<<<1024,256,0,stream.value>>>(a.as<T>(),a64.as<double>(),ac);
      to_double<<<1024,256,0,stream.value>>>(b.as<T>(),b64.as<double>(),bc);
      check(cudaGetLastError());
      double one=1,zero=0;
      // Row-major AB equals the column-major view B^T A^T, requiring no transpose copies.
      check(cublasDgemm(reference_blas.value,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&one,
                       b64.as<double>(),n,a64.as<double>(),k,&zero,reference.as<double>(),n));
      megafmm::run(*dense,a.data,b.data,nullptr,dd.as<float>(),workspace.data,stream.value);
      megafmm::run(*solution,a.data,b.data,nullptr,sd.as<float>(),workspace.data,stream.value);
      double norm_ab=std::sqrt(norm2(a64.as<double>(),ac,stream.value,scratch)*norm2(b64.as<double>(),bc,stream.value,scratch));
      Error dt=error(dd.as<float>(),reference.as<double>(),dc,atol,rtol,norm_ab,stream.value,scratch);
      Error st=error(sd.as<float>(),reference.as<double>(),dc,atol,rtol,norm_ab,stream.value,scratch);
      double limit=3*dt.frobenius+floor;
      bool trial_pass=dt.nonfinite==0 && st.nonfinite==0 && st.frobenius<=limit;
      pass=pass && trial_pass; gate_limit=std::max(gate_limit,limit);
      keep_worst(de,dt); keep_worst(se,st);
      std::cerr << "CHECK " << m << ',' << n << ',' << k << ' ' << megafmm::dtype_name(dtype)
                << " trial=" << trial << " dense_eF=" << dt.frobenius << " solution_eF=" << st.frobenius
                << " gate=" << limit << " pass=" << trial_pass << '\n';
    }
  }
  // Both calls recompute all input-dependent transforms on every launch/captured launch.
  auto graph_start=Clock::now();
  Invocation dcalls(*dense,input_a,input_b,dd.as<float>(),workspace.data,stream.value,o.graph);
  Invocation scalls(*solution,input_a,input_b,sd.as<float>(),workspace.data,stream.value,o.graph);
  double graph_setup_ms=std::chrono::duration<double,std::milli>(Clock::now()-graph_start).count();
  for (int i=0;i<o.warmup;++i) { dcalls.run(size_t(i)); scalls.run(size_t(i)); }
  check(cudaStreamSynchronize(stream.value));
  std::vector<double> dev,sev,dwall,swall;
  std::mt19937 rng(uint32_t(m)^uint32_t(n<<1)^uint32_t(k<<2)^1234567U);
  for (int group=0;group<o.groups;++group) {
    int order[2]={0,1}; std::shuffle(order,order+2,rng);
    for (int which : order) {
      auto t=measure(which ? scalls : dcalls,o.trials);
      (which ? sev : dev).push_back(t.event_ms);
      (which ? swall : dwall).push_back(t.wall_ms);
      std::cerr << "TIME shape=" << m << ',' << n << ',' << k << " dtype=" << megafmm::dtype_name(dtype)
                << " group=" << group << " case=" << (which ? "solution" : "dense")
                << " event_ms=" << t.event_ms << " wall_ms=" << t.wall_ms << '\n';
    }
  }
  auto ds=stats(dev), ss=stats(sev), dw=stats(dwall), sw=stats(swall);
  int variant=o.algorithm=="mega" ? o.mega_variant : o.algorithm=="hopper" ? o.hopper_variant : o.lt_variant;
  csv << std::setprecision(12) << m << ',' << n << ',' << k << ',' << megafmm::dtype_name(dtype) << ','
      << o.algorithm << ',' << csv_string(solution->algorithm_name()) << ',' << variant << ','
      << (o.graph ? "graph" : "launch") << ',' << (o.rotate_inputs==1 ? "warm_cache":"rotating_inputs") << ','
      << o.rotate_inputs << ',' << (size_t(ac)+size_t(bc))*sizeof(T)*size_t(o.rotate_inputs) << ",group_mean," << o.trials << ',' << o.groups << ',' << o.correct_trials << ','
      << policy.workspace_limit << ',' << dense_workspace << ',' << solution_workspace << ',' << setup_ms << ',' << tune_ms << ',' << graph_setup_ms << ','
      << ds.median << ',' << ds.p10 << ',' << ds.p90 << ',' << ds.cv << ','
      << ss.median << ',' << ss.p10 << ',' << ss.p90 << ',' << ss.cv << ','
      << dw.median << ',' << sw.median << ',' << ds.median/ss.median << ',' << 2.0*m*n*k/(ss.median*1e9) << ','
      << atol << ',' << rtol << ',' << de.maxabs << ',' << de.frobenius << ',' << de.scaled << ',' << de.normalized << ',' << de.nearzero << ','
      << se.maxabs << ',' << se.frobenius << ',' << se.scaled << ',' << se.normalized << ',' << se.nearzero << ','
      << de.nonfinite+se.nonfinite << ',' << floor << ',' << gate_limit << ',';
  if (o.correct_trials) csv << int(pass); else csv << "unchecked";
  csv << '\n'; csv.flush();
  return pass;
}
} // namespace

int main(int argc,char** argv) {
  try {
    Options o=options(argc,argv);
    cudaDeviceProp device{}; check(cudaGetDeviceProperties(&device,0));
    int driver=0,runtime=0; check(cudaDriverGetVersion(&driver)); check(cudaRuntimeGetVersion(&runtime));
    std::cerr << "DEVICE " << device.name << " sm=" << device.multiProcessorCount
              << " cc=" << device.major << '.' << device.minor << " driver=" << driver << " runtime=" << runtime
              << " precision=FP32-output seed=1234567 workspace_limit=" << (size_t(8)<<30) << '\n';
    std::ofstream file;
    bool header=true;
    if (!o.output.empty()) {
      std::ifstream previous(o.output,std::ios::binary|std::ios::ate);
      header=!previous || previous.tellg()==0;
      file.open(o.output,std::ios::app);
      if (!file) throw std::runtime_error("cannot open output: "+o.output);
    }
    std::ostream& csv=o.output.empty() ? std::cout : file;
    if (header && !o.profile) csv << csv_header;
    bool pass=true;
    for (auto [m,n,k] : o.shapes) {
      if (o.dtype=="both" || o.dtype=="bf16") pass=benchmark<__nv_bfloat16>(o,m,n,k,megafmm::DType::BF16,csv) && pass;
      if (o.dtype=="both" || o.dtype=="fp16") pass=benchmark<__half>(o,m,n,k,megafmm::DType::FP16,csv) && pass;
    }
    return pass ? 0 : 2;
  } catch (const std::exception& e) { std::cerr << "ERROR " << e.what() << '\n'; return 1; }
}

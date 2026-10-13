#include "internal.h"
#include <algorithm>
#include <cstdint>

namespace megafmm::detail {
namespace {
template<bool ReadC>
__device__ inline void store(float value, const float* C0, float* D,
                            uint64_t row, uint64_t col, int64_t ldc,
                            int64_t ldd, float alpha, float beta) {
  float result = alpha * value;
  if constexpr (ReadC) result = fmaf(beta, C0[row * ldc + col], result);
  D[row * ldd + col] = result;
}
template<bool ReadC>
__global__ void combine_kernel(const float* __restrict__ P, const float* C0,
                               float* D, int64_t rows, int64_t cols,
                               int64_t ldc, int64_t ldd, float alpha, float beta) {
  const uint64_t count = uint64_t(rows) * cols;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(blockDim.x) * gridDim.x) {
    const float p1 = P[i], p2 = P[count + i], p3 = P[2 * count + i];
    const float p4 = P[3 * count + i], p5 = P[4 * count + i];
    const float p6 = P[5 * count + i], p7 = P[6 * count + i];
    const float u1 = __fadd_rn(p1, p2);
    const float u2 = __fadd_rn(p1, p6);
    const float u3 = __fadd_rn(u2, p7);
    const float u4 = __fadd_rn(u2, p5);
    const float u5 = __fadd_rn(u4, p3);
    const float u6 = __fsub_rn(u3, p4);
    const float u7 = __fadd_rn(u3, p5);
    const uint64_t row = i / cols, col = i % cols;
    store<ReadC>(u1, C0, D, row, col, ldc, ldd, alpha, beta);
    store<ReadC>(u5, C0, D, row, col + cols, ldc, ldd, alpha, beta);
    store<ReadC>(u6, C0, D, row + rows, col, ldc, ldd, alpha, beta);
    store<ReadC>(u7, C0, D, row + rows, col + cols, ldc, ldd, alpha, beta);
  }
}
template<bool ReadC>
__global__ void scale_kernel(const float* C0, float* D, int64_t rows,
                             int64_t cols, int64_t ldc, int64_t ldd, float beta) {
  const uint64_t count = uint64_t(rows) * cols;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(blockDim.x) * gridDim.x) {
    const uint64_t row = i / cols, col = i % cols;
    float value = 0.0f;
    if constexpr (ReadC) value = beta * C0[row * ldc + col];
    D[row * ldd + col] = value;
  }
}
}  // namespace
void combine_winograd(const Problem& p, const float* P, const float* C0,
                      float* D, cudaStream_t stream) {
  const uint64_t count = uint64_t(p.m / 2) * (p.n / 2);
  const unsigned blocks = static_cast<unsigned>(std::min<uint64_t>((count + 255) / 256, 65535));
  if (p.beta == 0.0f)
    combine_kernel<false><<<blocks, 256, 0, stream>>>(P, C0, D, p.m / 2, p.n / 2, p.ldc, p.ldd, p.alpha, p.beta);
  else
    combine_kernel<true><<<blocks, 256, 0, stream>>>(P, C0, D, p.m / 2, p.n / 2, p.ldc, p.ldd, p.alpha, p.beta);
}
void scale_output(const Problem& p, const float* C0, float* D, cudaStream_t stream) {
  const uint64_t count = uint64_t(p.m) * p.n;
  const unsigned blocks = static_cast<unsigned>(std::min<uint64_t>((count + 255) / 256, 65535));
  if (p.beta == 0.0f)
    scale_kernel<false><<<blocks, 256, 0, stream>>>(C0, D, p.m, p.n, p.ldc, p.ldd, p.beta);
  else
    scale_kernel<true><<<blocks, 256, 0, stream>>>(C0, D, p.m, p.n, p.ldc, p.ldd, p.beta);
}
}  // namespace megafmm::detail

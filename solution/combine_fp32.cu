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
__global__ void combine_vector_kernel(const float* __restrict__ P, const float* C0,
                                      float* D, int64_t rows, int64_t cols,
                                      int64_t ldc, int64_t ldd, float alpha, float beta) {
  const uint64_t count = uint64_t(rows) * cols;
  const uint64_t vectors = count / 4;
  for (uint64_t v = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       v < vectors; v += uint64_t(blockDim.x) * gridDim.x) {
    float4 values[7];
    #pragma unroll
    for (int product = 0; product < 7; ++product)
      values[product] = *reinterpret_cast<const float4*>(P + product * count + v * 4);
    const uint64_t row = v / (cols / 4), col = (v % (cols / 4)) * 4;
    const uint64_t ro[4] = {row, row, row + rows, row + rows};
    const uint64_t co[4] = {col, col + cols, col, col + cols};
    float4 output[4], input_c[4];
    if constexpr (ReadC) {
      #pragma unroll
      for (int quadrant = 0; quadrant < 4; ++quadrant)
        input_c[quadrant] = *reinterpret_cast<const float4*>(C0 + ro[quadrant] * ldc + co[quadrant]);
    }
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      const float p1 = reinterpret_cast<float*>(&values[0])[lane];
      const float p2 = reinterpret_cast<float*>(&values[1])[lane];
      const float p3 = reinterpret_cast<float*>(&values[2])[lane];
      const float p4 = reinterpret_cast<float*>(&values[3])[lane];
      const float p5 = reinterpret_cast<float*>(&values[4])[lane];
      const float p6 = reinterpret_cast<float*>(&values[5])[lane];
      const float p7 = reinterpret_cast<float*>(&values[6])[lane];
      const float u1 = __fadd_rn(p1, p2), u2 = __fadd_rn(p1, p6);
      const float u3 = __fadd_rn(u2, p7), u4 = __fadd_rn(u2, p5);
      const float out[4] = {u1, __fadd_rn(u4, p3), __fsub_rn(u3, p4), __fadd_rn(u3, p5)};
      #pragma unroll
      for (int quadrant = 0; quadrant < 4; ++quadrant) {
        float result = alpha * out[quadrant];
        if constexpr (ReadC) result = fmaf(beta, reinterpret_cast<float*>(&input_c[quadrant])[lane], result);
        reinterpret_cast<float*>(&output[quadrant])[lane] = result;
      }
    }
    #pragma unroll
    for (int quadrant = 0; quadrant < 4; ++quadrant)
      *reinterpret_cast<float4*>(D + ro[quadrant] * ldd + co[quadrant]) = output[quadrant];
  }
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
  const bool vector_ok = p.n % 8 == 0 && p.ldd % 4 == 0 &&
    reinterpret_cast<uintptr_t>(D) % 16 == 0 &&
    (p.beta == 0.0f || (p.ldc % 4 == 0 && reinterpret_cast<uintptr_t>(C0) % 16 == 0));
  if (vector_ok) {
    const unsigned blocks = static_cast<unsigned>(std::min<uint64_t>((count / 4 + 255) / 256, 65535));
    if (p.beta == 0.0f)
      combine_vector_kernel<false><<<blocks, 256, 0, stream>>>(P,C0,D,p.m/2,p.n/2,p.ldc,p.ldd,p.alpha,p.beta);
    else
      combine_vector_kernel<true><<<blocks, 256, 0, stream>>>(P,C0,D,p.m/2,p.n/2,p.ldc,p.ldd,p.alpha,p.beta);
    return;
  }
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

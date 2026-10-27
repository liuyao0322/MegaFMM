#include "internal.h"
#include <algorithm>

namespace megafmm::detail {
namespace {
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
}
void scale_output(const Problem& p, const float* C0, float* D, cudaStream_t stream) {
  const uint64_t count = uint64_t(p.m) * p.n;
  if (!count) return;
  const unsigned blocks = static_cast<unsigned>(std::min<uint64_t>((count + 255) / 256, 65535));
  if (p.beta == 0.0f)
    scale_kernel<false><<<blocks, 256, 0, stream>>>(C0,D,p.m,p.n,p.ldc,p.ldd,p.beta);
  else
    scale_kernel<true><<<blocks, 256, 0, stream>>>(C0,D,p.m,p.n,p.ldc,p.ldd,p.beta);
}
}

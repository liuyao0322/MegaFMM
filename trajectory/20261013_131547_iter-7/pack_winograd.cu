#include "internal.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>

namespace megafmm::detail {
namespace {
template<class T> __device__ inline float to_float(T x);
template<> __device__ inline float to_float(__half x) { return __half2float(x); }
template<> __device__ inline float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
template<class T> __device__ inline T from_float(float x);
template<> __device__ inline __half from_float(float x) { return __float2half_rn(x); }
template<> __device__ inline __nv_bfloat16 from_float(float x) { return __float2bfloat16_rn(x); }

template<class T, bool IsA, bool UniformSlots=false>
__device__ __forceinline__ void pack_vectors(const T* __restrict__ input, T* __restrict__ packed,
                                  int64_t rows, int64_t cols, int64_t ld,
                                  uint64_t first, uint64_t step) {
  const uint64_t vectors = uint64_t(rows) * (cols / 8);
  const uint64_t count = uint64_t(rows) * cols;
  for (uint64_t v = first; v < vectors; v += step) {
    const uint64_t row = v / (cols / 8), col = (v % (cols / 8)) * 8;
    const uint4 q11 = *reinterpret_cast<const uint4*>(input + row * ld + col);
    const uint4 q12 = *reinterpret_cast<const uint4*>(input + row * ld + col + cols);
    const uint4 q21 = *reinterpret_cast<const uint4*>(input + (row + rows) * ld + col);
    const uint4 q22 = *reinterpret_cast<const uint4*>(input + (row + rows) * ld + col + cols);
    uint4 z1{}, z2{}, z3{}, z4{};
    #pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      const float x11 = to_float(reinterpret_cast<const T*>(&q11)[lane]);
      const float x12 = to_float(reinterpret_cast<const T*>(&q12)[lane]);
      const float x21 = to_float(reinterpret_cast<const T*>(&q21)[lane]);
      const float x22 = to_float(reinterpret_cast<const T*>(&q22)[lane]);
      float y1, y2, y3, y4;
      if constexpr (IsA) {
        y1 = __fadd_rn(x21, x22); y2 = __fsub_rn(y1, x11);
        y3 = __fsub_rn(x11, x21); y4 = __fsub_rn(x12, y2);
      } else {
        y1 = __fsub_rn(x12, x11); y2 = __fsub_rn(x22, y1);
        y3 = __fsub_rn(x22, x12); y4 = __fsub_rn(y2, x21);
      }
      reinterpret_cast<T*>(&z1)[lane] = from_float<T>(y1);
      reinterpret_cast<T*>(&z2)[lane] = from_float<T>(y2);
      reinterpret_cast<T*>(&z3)[lane] = from_float<T>(y3);
      reinterpret_cast<T*>(&z4)[lane] = from_float<T>(y4);
    }
    const uint64_t i = row * cols + col;
    if constexpr (UniformSlots) {
      // Original quadrants are copied bit-for-bit, with no conversion roundtrip.
      // The four transformed operands still use the shared FP32 DAG above.
      *reinterpret_cast<uint4*>(packed + i) = q11;
      *reinterpret_cast<uint4*>(packed + count + i) = IsA ? q12 : q21;
      *reinterpret_cast<uint4*>(packed + 2 * count + i) = IsA ? z4 : q22;
      *reinterpret_cast<uint4*>(packed + 3 * count + i) = IsA ? q22 : z4;
      *reinterpret_cast<uint4*>(packed + 4 * count + i) = z1;
      *reinterpret_cast<uint4*>(packed + 5 * count + i) = z2;
      *reinterpret_cast<uint4*>(packed + 6 * count + i) = z3;
    } else {
      *reinterpret_cast<uint4*>(packed + i) = z1;
      *reinterpret_cast<uint4*>(packed + count + i) = z2;
      *reinterpret_cast<uint4*>(packed + 2 * count + i) = z3;
      *reinterpret_cast<uint4*>(packed + 3 * count + i) = z4;
    }
  }
}

template<class T, bool IsA>
__global__ void pack_vector_kernel(const T* __restrict__ input, T* __restrict__ packed,
                                  int64_t rows, int64_t cols, int64_t ld) {
  pack_vectors<T, IsA>(input, packed, rows, cols, ld,
      uint64_t(blockIdx.x) * blockDim.x + threadIdx.x,
      uint64_t(blockDim.x) * gridDim.x);
}

// Each CTA belongs to exactly one input. Both independent DAGs share one
// launch without requiring any cross-CTA communication or extra storage.
template<class T, bool UniformSlots=false>
__global__ void pack_inputs_kernel(Problem p, const T* __restrict__ A,
    const T* __restrict__ B, T* __restrict__ S, T* __restrict__ Tpacked,
    unsigned a_blocks, unsigned b_blocks) {
  if (blockIdx.x < a_blocks) {
    pack_vectors<T, true, UniformSlots>(A, S, p.m / 2, p.k / 2, p.lda,
        uint64_t(blockIdx.x) * blockDim.x + threadIdx.x,
        uint64_t(blockDim.x) * a_blocks);
  } else {
    pack_vectors<T, false, UniformSlots>(B, Tpacked, p.k / 2, p.n / 2, p.ldb,
        uint64_t(blockIdx.x - a_blocks) * blockDim.x + threadIdx.x,
        uint64_t(blockDim.x) * b_blocks);
  }
}

template<class T, bool IsA>
__global__ void pack_kernel(const T* __restrict__ input, T* __restrict__ packed,
                           int64_t rows, int64_t cols, int64_t ld) {
  const uint64_t count = uint64_t(rows) * cols;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(blockDim.x) * gridDim.x) {
    const uint64_t row = i / cols, col = i % cols;
    const float x11 = to_float(input[row * ld + col]);
    const float x12 = to_float(input[row * ld + col + cols]);
    const float x21 = to_float(input[(row + rows) * ld + col]);
    const float x22 = to_float(input[(row + rows) * ld + col + cols]);
    float y1, y2, y3, y4;
    if constexpr (IsA) {
      y1 = __fadd_rn(x21, x22);       // S1 = A21 + A22
      y2 = __fsub_rn(y1, x11);        // S2 = S1 - A11
      y3 = __fsub_rn(x11, x21);       // S3 = A11 - A21
      y4 = __fsub_rn(x12, y2);        // S4 = A12 - S2
    } else {
      y1 = __fsub_rn(x12, x11);       // T1 = B12 - B11
      y2 = __fsub_rn(x22, y1);        // T2 = B22 - T1
      y3 = __fsub_rn(x22, x12);       // T3 = B22 - B12
      y4 = __fsub_rn(y2, x21);        // T4 = T2 - B21
    }
    // Only final operands are quantized; dependencies never read rounded S/T.
    packed[i] = from_float<T>(y1);
    packed[count + i] = from_float<T>(y2);
    packed[2 * count + i] = from_float<T>(y3);
    packed[3 * count + i] = from_float<T>(y4);
  }
}

template<bool IsA>
void launch(const Problem& p, const void* input, void* packed, cudaStream_t stream) {
  const int64_t rows = IsA ? p.m / 2 : p.k / 2;
  const int64_t cols = IsA ? p.k / 2 : p.n / 2;
  const int64_t ld = IsA ? p.lda : p.ldb;
  const uint64_t count = uint64_t(rows) * cols;
  const unsigned blocks = static_cast<unsigned>(std::min<uint64_t>((count / 8 + 255) / 256, 65535));
  if (p.dtype == DType::BF16)
    pack_vector_kernel<__nv_bfloat16, IsA><<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input), static_cast<__nv_bfloat16*>(packed), rows, cols, ld);
  else
    pack_vector_kernel<__half, IsA><<<blocks, 256, 0, stream>>>(
        static_cast<const __half*>(input), static_cast<__half*>(packed), rows, cols, ld);
}
}  // namespace
void pack_winograd_a(const Problem& p, const void* A, void* S, cudaStream_t stream) {
  launch<true>(p, A, S, stream);
}
void pack_winograd_b(const Problem& p, const void* B, void* T, cudaStream_t stream) {
  launch<false>(p, B, T, stream);
}
void pack_winograd_inputs(const Problem& p, const void* A, const void* B,
                          void* S, void* T, cudaStream_t stream) {
  // Retain the measured large-shape path; a saved launch matters most where
  // the transforms' own work is short. The branch is solely shape-based.
  if (p.m > 4096 || p.n > 4096 || p.k > 4096) {
    launch<true>(p, A, S, stream);
    launch<false>(p, B, T, stream);
    return;
  }
  const uint64_t av = uint64_t(p.m / 2) * uint64_t(p.k / 16);
  const uint64_t bv = uint64_t(p.k / 2) * uint64_t(p.n / 16);
  const unsigned ab = static_cast<unsigned>((av + 255) / 256);
  const unsigned bb = static_cast<unsigned>((bv + 255) / 256);
  if (p.dtype == DType::BF16)
    pack_inputs_kernel<__nv_bfloat16><<<ab + bb, 256, 0, stream>>>(p,
        static_cast<const __nv_bfloat16*>(A), static_cast<const __nv_bfloat16*>(B),
        static_cast<__nv_bfloat16*>(S), static_cast<__nv_bfloat16*>(T), ab, bb);
  else
    pack_inputs_kernel<__half><<<ab + bb, 256, 0, stream>>>(p,
        static_cast<const __half*>(A), static_cast<const __half*>(B),
        static_cast<__half*>(S), static_cast<__half*>(T), ab, bb);
}
void pack_winograd_batched_inputs(const Problem& p, const void* A, const void* B,
                                  void* left, void* right, cudaStream_t stream) {
  const uint64_t av = uint64_t(p.m / 2) * uint64_t(p.k / 16);
  const uint64_t bv = uint64_t(p.k / 2) * uint64_t(p.n / 16);
  const unsigned ab = static_cast<unsigned>(std::min<uint64_t>((av + 255) / 256, 65535));
  const unsigned bb = static_cast<unsigned>(std::min<uint64_t>((bv + 255) / 256, 65535));
  if (p.dtype == DType::BF16)
    pack_inputs_kernel<__nv_bfloat16, true><<<ab + bb, 256, 0, stream>>>(p,
        static_cast<const __nv_bfloat16*>(A), static_cast<const __nv_bfloat16*>(B),
        static_cast<__nv_bfloat16*>(left), static_cast<__nv_bfloat16*>(right), ab, bb);
  else
    pack_inputs_kernel<__half, true><<<ab + bb, 256, 0, stream>>>(p,
        static_cast<const __half*>(A), static_cast<const __half*>(B),
        static_cast<__half*>(left), static_cast<__half*>(right), ab, bb);
}
}  // namespace megafmm::detail

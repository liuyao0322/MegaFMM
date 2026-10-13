#pragma once
#include "megafmm/gemm.h"
#include <cstddef>

namespace megafmm::detail {
void pack_winograd_a(const Problem& p, const void* A, void* S, cudaStream_t stream);
void pack_winograd_b(const Problem& p, const void* B, void* T, cudaStream_t stream);
void pack_winograd_inputs(const Problem& p, const void* A, const void* B,
                          void* S, void* T, cudaStream_t stream);
// Uniform product slots: A11,A12,S4,A22,S1,S2,S3 and B11,B21,B22,T4,T1,T2,T3.
void pack_winograd_batched_inputs(const Problem& p, const void* A, const void* B,
                                  void* left, void* right, cudaStream_t stream);
void combine_winograd(const Problem& p, const float* products, const float* C0,
                      float* D, cudaStream_t stream);
void scale_output(const Problem& p, const float* C0, float* D, cudaStream_t stream);
}  // namespace megafmm::detail

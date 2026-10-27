#include "internal.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <stdexcept>

namespace megafmm::detail {
namespace {
template<class T> __device__ T rounded(float x);
template<> __device__ __half rounded(float x) { return __float2half_rn(x); }
template<> __device__ __nv_bfloat16 rounded(float x) { return __float2bfloat16_rn(x); }

template<class T>
__device__ float load(const T* data, int64_t row, int64_t col,
                     int64_t rows, int64_t cols, int64_t ld, bool transpose) {
  if (row >= rows || col >= cols) return 0.0f;
  return float(data[transpose ? col * ld + row : row * ld + col]);
}

// One level of Winograd, padded logically to even M/N/K. Each warp computes
// the same 16x16 tile in all four output quadrants. Transform operands live
// only in warp-private shared memory; seven products stay in FP32 registers.
// No global scratch, atomics, cross-CTA dependencies, or cached input values.
template<class T>
__global__ void sw_pipeline(Problem p, const T* A, const T* B,
                             const float* C, float* D, int64_t tiles_n,
                             int64_t tile_count) {
  using namespace nvcuda;
  __shared__ __align__(32) T lhs[4][256];
  __shared__ __align__(32) T rhs[4][256];
  __shared__ __align__(32) float output[4][256];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t mh = (p.m + 1) / 2, nh = (p.n + 1) / 2, kh = (p.k + 1) / 2;
  for (int64_t tile = int64_t(blockIdx.x) * 4 + warp;
       tile < tile_count; tile += int64_t(gridDim.x) * 4) {
    const int64_t r0 = (tile / tiles_n) * 16, c0 = (tile % tiles_n) * 16;
    wmma::fragment<wmma::accumulator,16,16,16,float> products[7];
    for (int product = 0; product < 7; ++product) {
      wmma::fill_fragment(products[product], 0.0f);
      for (int64_t q = 0; q < kh; q += 16) {
        for (int i = lane; i < 256; i += 32) {
          int64_t ar = r0 + i / 16, ak = q + i % 16;
          int64_t bk = q + i / 16, bc = c0 + i % 16;
          // Mask the half-block boundary as well as the original matrix: a
          // padded position must never read the adjacent physical quadrant.
          float a[4]{}, b[4]{};
          if (ar < mh && ak < kh) {
            a[0]=load(A,ar,ak,p.m,p.k,p.lda,p.trans_a);
            a[1]=load(A,ar,ak+kh,p.m,p.k,p.lda,p.trans_a);
            a[2]=load(A,ar+mh,ak,p.m,p.k,p.lda,p.trans_a);
            a[3]=load(A,ar+mh,ak+kh,p.m,p.k,p.lda,p.trans_a);
          }
          if (bk < kh && bc < nh) {
            b[0]=load(B,bk,bc,p.k,p.n,p.ldb,p.trans_b);
            b[1]=load(B,bk,bc+nh,p.k,p.n,p.ldb,p.trans_b);
            b[2]=load(B,bk+kh,bc,p.k,p.n,p.ldb,p.trans_b);
            b[3]=load(B,bk+kh,bc+nh,p.k,p.n,p.ldb,p.trans_b);
          }
          float s1=a[2]+a[3], s2=s1-a[0], s3=a[0]-a[2], s4=a[1]-s2;
          float t1=b[1]-b[0], t2=b[3]-t1, t3=b[3]-b[1], t4=t2-b[2];
          float av = product==0 ? a[0] : product==1 ? a[1] : product==2 ? s4 :
                     product==3 ? a[3] : product==4 ? s1 : product==5 ? s2 : s3;
          float bv = product==0 ? b[0] : product==1 ? b[2] : product==2 ? b[3] :
                     product==3 ? t4 : product==4 ? t1 : product==5 ? t2 : t3;
          lhs[warp][i]=rounded<T>(av); rhs[warp][i]=rounded<T>(bv);
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag,lhs[warp],16);
        wmma::load_matrix_sync(b_frag,rhs[warp],16);
        wmma::mma_sync(products[product],a_frag,b_frag,products[product]);
        __syncwarp();
      }
    }
    for (int quadrant=0; quadrant<4; ++quadrant) {
      wmma::fragment<wmma::accumulator,16,16,16,float> result;
      for (int i=0; i<result.num_elements; ++i) {
        float u2=products[0].x[i]+products[5].x[i];
        float u3=u2+products[6].x[i], u4=u2+products[4].x[i];
        result.x[i]=quadrant==0 ? products[0].x[i]+products[1].x[i] :
                    quadrant==1 ? u4+products[2].x[i] :
                    quadrant==2 ? u3-products[3].x[i] : u3+products[4].x[i];
      }
      wmma::store_matrix_sync(output[warp],result,16,wmma::mem_row_major);
      __syncwarp();
      for (int i=lane; i<256; i+=32) {
        int64_t r=r0+i/16, c=c0+i%16;
        if (r>=mh || c>=nh) continue;
        r+=(quadrant/2)*mh; c+=(quadrant%2)*nh;
        if (r<p.m && c<p.n) {
          float value=p.alpha*output[warp][i];
          if (p.beta!=0.0f) value+=p.beta*C[r*p.ldc+c];
          D[r*p.ldd+c]=value;
        }
      }
      __syncwarp();
    }
  }
}
} // namespace

void run_sw_pipeline(const Problem& p,const void* A,const void* B,
                      const float* C,float* D,cudaStream_t stream) {
  if (p.dtype==DType::FP32) {
    run_sw_fp32(p,static_cast<const float*>(A),static_cast<const float*>(B),C,D,stream);
    return;
  }
  const int64_t rows=((p.m+1)/2+15)/16, cols=((p.n+1)/2+15)/16;
  const int blocks=int(std::min<int64_t>((rows*cols+3)/4,65535));
  if (p.dtype==DType::BF16)
    sw_pipeline<<<blocks,128,0,stream>>>(p,static_cast<const __nv_bfloat16*>(A),static_cast<const __nv_bfloat16*>(B),C,D,cols,rows*cols);
  else
    sw_pipeline<<<blocks,128,0,stream>>>(p,static_cast<const __half*>(A),static_cast<const __half*>(B),C,D,cols,rows*cols);
  auto status=cudaPeekAtLastError();
  if (status!=cudaSuccess) throw std::runtime_error(std::string("SW pipeline launch: ")+cudaGetErrorString(status));
}
} // namespace megafmm::detail

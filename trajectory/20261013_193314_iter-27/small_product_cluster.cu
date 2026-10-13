// Experimental product-parallel M1, variant 13: portable eight-CTA cluster.
// Seven 32-thread CTAs retain full P tiles; rank 7 only joins the two barriers.
// Rank 0 reads the seven final shared P fragments through DSM and applies the
// unchanged fixed FP32 U DAG. No global intermediate or caller workspace.
// Link with variant 9, which owns generic tail/stride/unaligned-input fallback.
#include "megafmm/gemm.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace megafmm {
// Variant 9 owns the generic tail/unaligned path and common problem contract.
// These forward declarations keep this scratch prototype independent of API edits.
bool small_product_can_implement(const Problem& p, int variant);
void prepare_small_product(const Problem& p, int variant);
void launch_small_product(const Problem& p, const void* A, const void* B,
                          const float* C0, float* D, cudaStream_t stream, int variant);

namespace {

using namespace cute;

template <class Element>
struct ClusterProductMmaTraits {
  static constexpr int BM = 16;
  static constexpr int BN = 16;
  static constexpr int BK = 16;
  static constexpr int Products = 7;
  static constexpr int Threads = 32;
  static constexpr int ClusterBlocks = 8;
  using Op = typename std::conditional<
      std::is_same<Element, cutlass::bfloat16_t>::value,
      SM80_16x8x16_F32BF16BF16F32_TN,
      SM80_16x8x16_F32F16F16F32_TN>::type;
  using Mma = decltype(make_tiled_mma(
      Op{}, Layout<Shape<_1,_1,_1>>{}, Tile<_16,_16,_16>{}));
  // Each warp has only its own A/B operand tile. The 256-element offsets
  // preserve the swizzle's 128-element period and 16-byte packed store alignment.
  using LayoutA = decltype(composition(Swizzle<1,3,3>{},
      Layout<Shape<_16,_16>, Stride<_16,_1>>{}));
  using LayoutB = decltype(composition(Swizzle<1,3,3>{},
      Layout<Shape<_16,_16>, Stride<_1,_16>>{}));
  static constexpr int ElementsA = cosize_v<LayoutA>;
  static constexpr int ElementsB = cosize_v<LayoutB>;
  static constexpr int WarpElements = ElementsA + ElementsB;
  static constexpr int SharedBytes =
      WarpElements * sizeof(Element) + BM * BN * sizeof(float);
  static_assert(ElementsA == BM * BK && ElementsB == BN * BK,
                "a product warp needs exactly one 1 KiB operand pair");
  static_assert(size(Mma{}) == 32, "one product must use one independent warp");
  static_assert(SharedBytes == 2 * 1024, "each CTA uses 1 KiB operands and 1 KiB P");
};

struct ClusterProductPackedEight {
  uint4 bits;
  CUTE_DEVICE uint16_t raw(int index) const {
    const uint32_t word = index < 2 ? bits.x : index < 4 ? bits.y
                                              : index < 6 ? bits.z : bits.w;
    return uint16_t(word >> (16 * (index & 1)));
  }
  CUTE_DEVICE void insert(int index, uint16_t value) {
    const uint32_t shifted = uint32_t(value) << (16 * (index & 1));
    if (index < 2) bits.x |= shifted;
    else if (index < 4) bits.y |= shifted;
    else if (index < 6) bits.z |= shifted;
    else bits.w |= shifted;
  }
};

template <class Element>
CUTE_DEVICE ClusterProductPackedEight cluster_product_read_eight(const Element* ptr) {
  // Host eligibility proves every vector has eight valid, aligned elements.
  return ClusterProductPackedEight{*reinterpret_cast<const uint4*>(ptr)};
}

template <class Element>
CUTE_DEVICE float cluster_product_unpack(const ClusterProductPackedEight& packed, int index) {
  return float(Element::bitcast(packed.raw(index)));
}

CUTE_DEVICE void cluster_product_store_eight(void* ptr, const ClusterProductPackedEight& packed) {
  const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
      :: "r"(address), "r"(packed.bits.x), "r"(packed.bits.y),
         "r"(packed.bits.z), "r"(packed.bits.w) : "memory");
}

// Quadrant indices are 0=11, 1=12, 2=21, 3=22. Under the host gate,
// half-M/N/K are divisible by 16, lda/ldb by 8, and A/B by 16 bytes.
// All lane vectors therefore fit entirely within one original quadrant.
template <int Quadrant, class Element>
CUTE_DEVICE ClusterProductPackedEight cluster_product_read_a_quadrant(
    const Problem& p, const Element* A, int64_t row_base, int64_t k_base, int lane) {
  const int64_t mh = p.m / 2;
  const int64_t kh = p.k / 2;
  const int64_t row = row_base + lane / 2 + (Quadrant / 2) * mh;
  const int64_t k = k_base + (lane & 1) * 8 + (Quadrant & 1) * kh;
  return cluster_product_read_eight(A + row * p.lda + k);
}

template <int Quadrant, class Element>
CUTE_DEVICE ClusterProductPackedEight cluster_product_read_b_quadrant(
    const Problem& p, const Element* B, int64_t col_base, int64_t k_base, int lane) {
  const int64_t nh = p.n / 2;
  const int64_t kh = p.k / 2;
  const int64_t k = k_base + lane / 2 + (Quadrant / 2) * kh;
  const int64_t col = col_base + (lane & 1) * 8 + (Quadrant & 1) * nh;
  return cluster_product_read_eight(B + k * p.ldb + col);
}

template <int Product, class Element>
CUTE_DEVICE ClusterProductPackedEight cluster_product_make_a(
    const Problem& p, const Element* A, int64_t row_base, int64_t k_base, int lane) {
  static_assert(Product >= 0 && Product < 7, "invalid Winograd product");
  if constexpr (Product == 0) {
    return cluster_product_read_a_quadrant<0>(p,A,row_base,k_base,lane); // A11
  } else if constexpr (Product == 1) {
    return cluster_product_read_a_quadrant<1>(p,A,row_base,k_base,lane); // A12
  } else if constexpr (Product == 3) {
    return cluster_product_read_a_quadrant<3>(p,A,row_base,k_base,lane); // A22
  } else {
    // Load only the dependencies of this product, and quantize only its final
    // operand. S2/S4 are formed from FP32 S1; they never read rounded S1/S2.
    ClusterProductPackedEight a11{make_uint4(0,0,0,0)}, a12{make_uint4(0,0,0,0)};
    ClusterProductPackedEight a21{make_uint4(0,0,0,0)}, a22{make_uint4(0,0,0,0)};
    if constexpr (Product == 2 || Product == 5 || Product == 6)
      a11 = cluster_product_read_a_quadrant<0>(p,A,row_base,k_base,lane);
    if constexpr (Product == 2)
      a12 = cluster_product_read_a_quadrant<1>(p,A,row_base,k_base,lane);
    a21 = cluster_product_read_a_quadrant<2>(p,A,row_base,k_base,lane);
    if constexpr (Product != 6)
      a22 = cluster_product_read_a_quadrant<3>(p,A,row_base,k_base,lane);
    ClusterProductPackedEight result{make_uint4(0,0,0,0)};
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
      float value;
      if constexpr (Product == 6) { // S3 = A11 - A21
        value = __fsub_rn(cluster_product_unpack<Element>(a11,i), cluster_product_unpack<Element>(a21,i));
      } else {
        const float s1 = __fadd_rn(cluster_product_unpack<Element>(a21,i), cluster_product_unpack<Element>(a22,i));
        if constexpr (Product == 4) { // S1
          value = s1;
        } else {
          const float s2 = __fsub_rn(s1, cluster_product_unpack<Element>(a11,i));
          if constexpr (Product == 5) value = s2; // S2
          else value = __fsub_rn(cluster_product_unpack<Element>(a12,i), s2); // S4
        }
      }
      result.insert(i, Element(value).raw());
    }
    return result;
  }
}

template <int Product, class Element>
CUTE_DEVICE ClusterProductPackedEight cluster_product_make_b(
    const Problem& p, const Element* B, int64_t col_base, int64_t k_base, int lane) {
  static_assert(Product >= 0 && Product < 7, "invalid Winograd product");
  if constexpr (Product == 0) {
    return cluster_product_read_b_quadrant<0>(p,B,col_base,k_base,lane); // B11
  } else if constexpr (Product == 1) {
    return cluster_product_read_b_quadrant<2>(p,B,col_base,k_base,lane); // B21
  } else if constexpr (Product == 2) {
    return cluster_product_read_b_quadrant<3>(p,B,col_base,k_base,lane); // B22
  } else {
    ClusterProductPackedEight b11{make_uint4(0,0,0,0)}, b12{make_uint4(0,0,0,0)};
    ClusterProductPackedEight b21{make_uint4(0,0,0,0)}, b22{make_uint4(0,0,0,0)};
    if constexpr (Product != 6)
      b11 = cluster_product_read_b_quadrant<0>(p,B,col_base,k_base,lane);
    b12 = cluster_product_read_b_quadrant<1>(p,B,col_base,k_base,lane);
    if constexpr (Product == 3)
      b21 = cluster_product_read_b_quadrant<2>(p,B,col_base,k_base,lane);
    if constexpr (Product != 4)
      b22 = cluster_product_read_b_quadrant<3>(p,B,col_base,k_base,lane);
    ClusterProductPackedEight result{make_uint4(0,0,0,0)};
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
      float value;
      if constexpr (Product == 6) { // T3 = B22 - B12
        value = __fsub_rn(cluster_product_unpack<Element>(b22,i), cluster_product_unpack<Element>(b12,i));
      } else {
        const float t1 = __fsub_rn(cluster_product_unpack<Element>(b12,i), cluster_product_unpack<Element>(b11,i));
        if constexpr (Product == 4) { // T1
          value = t1;
        } else {
          const float t2 = __fsub_rn(cluster_product_unpack<Element>(b22,i), t1);
          if constexpr (Product == 5) value = t2; // T2
          else value = __fsub_rn(t2, cluster_product_unpack<Element>(b21,i)); // T4
        }
      }
      result.insert(i, Element(value).raw());
    }
    return result;
  }
}

template <int Product, class Element, class SmemA, class SmemB, class Mma,
          class ThrMma, class CopyA, class CopyB, class Accumulator>
CUTE_DEVICE void cluster_product_compute(
    const Problem& p, const Element* A, const Element* B, int64_t row_base,
    int64_t col_base, int lane, SmemA sA, SmemB sB, Mma mma, ThrMma thr_mma,
    CopyA copy_a, CopyB copy_b, Accumulator& accumulator) {
  auto rA = thr_mma.partition_fragment_A(sA);
  auto rB = thr_mma.partition_fragment_B(sB);
  auto thr_copy_a = copy_a.get_thread_slice(lane);
  auto thr_copy_b = copy_b.get_thread_slice(lane);
  auto srcA = thr_copy_a.partition_S(sA);
  auto srcB = thr_copy_b.partition_S(sB);
  auto dstA = thr_copy_a.retile_D(rA);
  auto dstB = thr_copy_b.retile_D(rB);
  CUTE_NO_UNROLL
  for (int k_base = 0; k_base < int(p.k / 2); k_base += 16) {
    const auto a = cluster_product_make_a<Product>(p,A,row_base,k_base,lane);
    const auto b = cluster_product_make_b<Product>(p,B,col_base,k_base,lane);
    cluster_product_store_eight(&sA(lane/2,(lane&1)*8), a);
    cluster_product_store_eight(&sB((lane&1)*8,lane/2), b);
    // Only this warp reads or overwrites its 1 KiB operand pair. Synchronous
    // ldmatrix and register MMA use generic shared memory, with no async proxy.
    __syncwarp(0xffffffffu);
    copy(copy_a, srcA, dstA);
    copy(copy_b, srcB, dstB);
    gemm(mma, rA, rB, accumulator);
    __syncwarp(0xffffffffu);
  }
}

template <class Element>
__global__ __launch_bounds__(32)
void small_product_cluster_kernel(Problem p, const Element* __restrict__ A,
                                  const Element* __restrict__ B, const float* C0, float* D) {
  using T = ClusterProductMmaTraits<Element>;
  namespace cg = cooperative_groups;
  const cg::cluster_group cluster = cg::this_cluster();
  const int lane = threadIdx.x;
  const int product = int(cluster.block_rank());
  // clusterDim=(8,1,1): eight adjacent X blocks own the same output tile.
  const int64_t row_base = int64_t(blockIdx.y) * T::BM;
  const int64_t col_base = (int64_t(blockIdx.x) >> 3) * T::BN;
  __shared__ __align__(128) Element operands[T::WarpElements];
  __shared__ __align__(128) float local_products[T::BM * T::BN];
  if (product < T::Products) {
    typename T::Mma mma;
    auto thr_mma = mma.get_thread_slice(lane);
    auto coords = thr_mma.partition_C(make_identity_tensor(Shape<_16,_16>{}));
    auto sA = make_tensor(make_smem_ptr(operands), typename T::LayoutA{});
    auto sB = make_tensor(make_smem_ptr(operands + T::ElementsA), typename T::LayoutB{});
    auto accumulator = thr_mma.make_fragment_C(coords);
    clear(accumulator);
    static_assert(size(accumulator) == 8, "one product has eight FP32 values per lane");
    auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{}, mma);
    auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T,Element>{}, mma);
    // All 32 threads in a CTA select one product and run its complete K loop.
    switch (product) {
      case 0: cluster_product_compute<0>(p,A,B,row_base,col_base,lane,sA,sB,mma,thr_mma,copy_a,copy_b,accumulator); break;
      case 1: cluster_product_compute<1>(p,A,B,row_base,col_base,lane,sA,sB,mma,thr_mma,copy_a,copy_b,accumulator); break;
      case 2: cluster_product_compute<2>(p,A,B,row_base,col_base,lane,sA,sB,mma,thr_mma,copy_a,copy_b,accumulator); break;
      case 3: cluster_product_compute<3>(p,A,B,row_base,col_base,lane,sA,sB,mma,thr_mma,copy_a,copy_b,accumulator); break;
      case 4: cluster_product_compute<4>(p,A,B,row_base,col_base,lane,sA,sB,mma,thr_mma,copy_a,copy_b,accumulator); break;
      case 5: cluster_product_compute<5>(p,A,B,row_base,col_base,lane,sA,sB,mma,thr_mma,copy_a,copy_b,accumulator); break;
      case 6: cluster_product_compute<6>(p,A,B,row_base,col_base,lane,sA,sB,mma,thr_mma,copy_a,copy_b,accumulator); break;
    }
    CUTE_UNROLL
    for (int j = 0; j < size(accumulator); ++j)
      local_products[j * 32 + lane] = accumulator(j);
  }
  // All eight blocks, including dummy rank 7, participate. This publishes the
  // local P stores and ensures every target block exists before any DSM read.
  // CUDA 12.4 guide: #distributed-shared-memory, clusterHist_kernel protocol.
  cluster.sync();
  if (product == 0) {
    typename T::Mma mma;
    auto thr_mma = mma.get_thread_slice(lane);
    auto coords = thr_mma.partition_C(make_identity_tensor(Shape<_16,_16>{}));
    const float* p1_smem = cluster.map_shared_rank(local_products,0);
    const float* p2_smem = cluster.map_shared_rank(local_products,1);
    const float* p3_smem = cluster.map_shared_rank(local_products,2);
    const float* p4_smem = cluster.map_shared_rank(local_products,3);
    const float* p5_smem = cluster.map_shared_rank(local_products,4);
    const float* p6_smem = cluster.map_shared_rank(local_products,5);
    const float* p7_smem = cluster.map_shared_rank(local_products,6);
    // All CTAs use identical CuTe lane/value ordering. DSM slots are fragment
    // order j*32+lane, and rank 0 uses that same coords(j) for the four outputs.
    CUTE_NO_UNROLL
    for (int j = 0; j < 8; ++j) {
      const auto coord = coords(j);
      const int64_t row = row_base + get<0>(coord);
      const int64_t col = col_base + get<1>(coord);
      const int64_t mh = p.m / 2;
      const int64_t nh = p.n / 2;
      const int slot = j * 32 + lane;
      const float p1 = p1_smem[slot];
      const float p2 = p2_smem[slot];
      const float p3 = p3_smem[slot];
      const float p4 = p4_smem[slot];
      const float p5 = p5_smem[slot];
      const float p6 = p6_smem[slot];
      const float p7 = p7_smem[slot];
      const float u1 = __fadd_rn(p1,p2);
      const float u2 = __fadd_rn(p1,p6);
      const float u3 = __fadd_rn(u2,p7);
      const float u4 = __fadd_rn(u2,p5);
      const float u5 = __fadd_rn(u4,p3);
      const float u6 = __fsub_rn(u3,p4);
      const float u7 = __fadd_rn(u3,p5);
      float d11 = __fmul_rn(p.alpha,u1);
      float d12 = __fmul_rn(p.alpha,u5);
      float d21 = __fmul_rn(p.alpha,u6);
      float d22 = __fmul_rn(p.alpha,u7);
      if (p.beta != 0.0f) {
        // All four reads precede stores: exact C0==D with equal strides is safe.
        const float c11 = C0[row*p.ldc + col];
        const float c12 = C0[row*p.ldc + col + nh];
        const float c21 = C0[(row+mh)*p.ldc + col];
        const float c22 = C0[(row+mh)*p.ldc + col + nh];
        d11 = fmaf(p.beta,c11,d11);
        d12 = fmaf(p.beta,c12,d12);
        d21 = fmaf(p.beta,c21,d21);
        d22 = fmaf(p.beta,c22,d22);
      }
      D[row*p.ldd + col] = d11;
      D[row*p.ldd + col + nh] = d12;
      D[(row+mh)*p.ldd + col] = d21;
      D[(row+mh)*p.ldd + col + nh] = d22;
    }
  }
  // Every thread reaches this second barrier. Ranks 1..7 must stay resident
  // until rank 0 has finished all remote reads; none may exit prematurely.
  cluster.sync();
}

void configure_product_cluster(const Problem& p, cudaStream_t stream,
                               cudaLaunchAttribute& attribute, cudaLaunchConfig_t& config) {
  attribute = {};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = 8;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  config = {};
  config.gridDim = dim3{unsigned(8 * (p.n / 32)),unsigned(p.m / 32),1};
  config.blockDim = dim3{32,1,1};
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;
}

void check_product_cluster_status(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <class Element>
void prepare_small_product_cluster_typed(const Problem& p) {
  cudaLaunchAttribute attribute{};
  cudaLaunchConfig_t config{};
  configure_product_cluster(p,nullptr,attribute,config);
  int maximum_cluster_size = 0;
  check_product_cluster_status(cudaOccupancyMaxPotentialClusterSize(
      &maximum_cluster_size,small_product_cluster_kernel<Element>,&config),
      "query portable product cluster size");
  if (maximum_cluster_size < 8)
    throw std::invalid_argument("product cluster requires support for eight CTA clusters");
}

template <class Element>
void launch_small_product_cluster_typed(const Problem& p, const void* A, const void* B,
                                       const float* C0, float* D, cudaStream_t stream) {
  // Launch metadata is stack-owned and consumed by CUDA before this call returns.
  cudaLaunchAttribute attribute{};
  cudaLaunchConfig_t config{};
  configure_product_cluster(p,stream,attribute,config);
  check_product_cluster_status(cudaLaunchKernelEx(&config,small_product_cluster_kernel<Element>,
      p,static_cast<const Element*>(A),static_cast<const Element*>(B),C0,D),
      "launch portable product cluster");
}

bool cluster_product_problem_eligible(const Problem& p) {
  return (p.m % 32) == 0 && (p.n % 32) == 0 && (p.k % 32) == 0 &&
         (p.lda % 8) == 0 && (p.ldb % 8) == 0 &&
         (p.ldc % 8) == 0 && (p.ldd % 8) == 0;
}

bool cluster_product_addresses_eligible(const void* A, const void* B) {
  return (reinterpret_cast<uintptr_t>(A) & 15) == 0 &&
         (reinterpret_cast<uintptr_t>(B) & 15) == 0;
}

} // namespace

bool small_product_cluster_can_implement(const Problem& p, int variant) {
  // Includes the generic fallback; public Plan handles degenerate/overlap cases.
  return variant == 13 && small_product_can_implement(p,9);
}

void prepare_small_product_cluster(const Problem& p, int variant) {
  if (!small_product_cluster_can_implement(p,variant))
    throw std::invalid_argument("unsupported product cluster problem or variant");
  prepare_small_product(p,9);
  if (!cluster_product_problem_eligible(p)) return;
  // Capability/occupancy queries belong to prepare/create, never steady-state run.
  int device = 0;
  int cluster_launch = 0;
  check_product_cluster_status(cudaGetDevice(&device),"query product cluster device");
  check_product_cluster_status(cudaDeviceGetAttribute(
      &cluster_launch,cudaDevAttrClusterLaunch,device),"query product cluster support");
  if (!cluster_launch)
    throw std::invalid_argument("product cluster requires a Hopper cluster-capable device");
  if (p.dtype == DType::BF16) prepare_small_product_cluster_typed<cutlass::bfloat16_t>(p);
  else prepare_small_product_cluster_typed<cutlass::half_t>(p);
  // Standard portable size 8 needs no nonportable-size opt-in. Static storage
  // is only 2 KiB per CTA; metadata and fallback require no caller workspace.
}

void launch_small_product_cluster(const Problem& p, const void* A, const void* B,
                                  const float* C0, float* D, cudaStream_t stream,
                                  int variant) {
  if (!small_product_cluster_can_implement(p,variant))
    throw std::invalid_argument("unsupported product cluster problem or variant");
  if (!A || !B || !D || (p.beta != 0.0f && !C0))
    throw std::invalid_argument("product cluster requires nonnull input/output pointers");
  if ((reinterpret_cast<uintptr_t>(A) & 1) ||
      (reinterpret_cast<uintptr_t>(B) & 1) ||
      (reinterpret_cast<uintptr_t>(D) & 3) ||
      (p.beta != 0.0f && (reinterpret_cast<uintptr_t>(C0) & 3)))
    throw std::invalid_argument("product cluster pointers require element alignment");
  if (!cluster_product_problem_eligible(p) || !cluster_product_addresses_eligible(A,B)) {
    launch_small_product(p,A,B,C0,D,stream,9);
    return;
  }
  // No device query, allocation, synchronization or launch-error fallback here.
  // The prepared Plan is responsible for the current device's cluster support.
  if (p.dtype == DType::BF16)
    launch_small_product_cluster_typed<cutlass::bfloat16_t>(p,A,B,C0,D,stream);
  else launch_small_product_cluster_typed<cutlass::half_t>(p,A,B,C0,D,stream);
}

} // namespace megafmm

// Experimental formal policies28/29; original numerical gate remains independent.
// One-launch M1, dedicated producer WG, two final-operand banks, full seven P.
#include "megafmm/kernels.h"

#include <cute/tensor.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace megafmm {
namespace {
using namespace cute;

template <class Layout, int ContiguousMode>
// Keep CUDA12.4 constexpr work bounded; full geometry is independently checked
// in the frozen CPU prototype. This verifies four aligned edge vectors per slot.
constexpr bool pipeline_contiguous_eight_sentinels() {
  constexpr int rows = size<0>(Layout{});
  constexpr int cols = size<1>(Layout{});
  constexpr int slots = size<2>(Layout{});
  for (int slot = 0; slot < slots; ++slot) {
    for (int row_edge = 0; row_edge < 2; ++row_edge) {
      const int row = row_edge == 0 ? 0 : rows - (ContiguousMode == 0 ? 8 : 1);
      for (int col_edge = 0; col_edge < 2; ++col_edge) {
        const int col = col_edge == 0 ? 0 : cols - (ContiguousMode == 1 ? 8 : 1);
        const int base = Layout{}(make_coord(row,col,slot));
        if (base % 8 != 0) return false;
        for (int lane = 0; lane < 8; ++lane) {
          const int r = row + (ContiguousMode == 0 ? lane : 0);
          const int c = col + (ContiguousMode == 1 ? lane : 0);
          if (Layout{}(make_coord(r,c,slot)) != base + lane) return false;
        }
      }
    }
  }
  return true;
}

template <class Element, int Cols>
struct PipelineM1Traits {
  static_assert(Cols == 16 || Cols == 32,"only reviewed M1 tiles are supported");
  static constexpr int BM = 64;
  static constexpr int BN = Cols;
  static constexpr int BK = 32;
  static constexpr int Threads = 256;
  static constexpr int ProducerThreads = 128;
  static constexpr int ConsumerThreads = 128;
  static constexpr int Stages = 2;
  using Tile = Shape<Int<BM>,Int<BN>,Int<BK>>;
  // Identical atom/tiling construction to the existing M1. Each bank has its
  // own rank-3 tensor, retaining canonical GMMA descriptors for all seven P.
  using LayoutA = decltype(tile_to_shape(GMMA::Layout_K_SW32_Atom<Element>{},
      make_shape(Int<BM>{},Int<BK>{},Int<7>{})));
  using LayoutB = decltype(tile_to_shape(GMMA::Layout_MN_SW32_Atom<Element>{},
      make_shape(Int<BN>{},Int<BK>{},Int<7>{})));
  using Op = decltype(GMMA::ss_op_selector<Element,Element,float,Tile,
      GMMA::Major::K,GMMA::Major::MN>());
  using Mma = decltype(make_tiled_mma(Op{}));
  using Pipeline = cutlass::PipelineAsync<Stages>;
  static constexpr int ElementsA = cosize_v<LayoutA>;
  static constexpr int ElementsB = cosize_v<LayoutB>;
  static constexpr int BankElements = ElementsA + ElementsB;
  struct alignas(128) SharedStorage {
    Element operands[Stages * BankElements];
    typename Pipeline::SharedStorage barriers;
  };
  static constexpr size_t SharedBytes = sizeof(SharedStorage);
  static_assert(sizeof(Element) == 2,"vector helpers require 16-bit operands");
  static_assert(size(Mma{}) == ConsumerThreads,"one consumer WG owns the tile");
  static_assert(ElementsA == 7*BM*BK && ElementsB == 7*BN*BK,
      "compact product slots must have no padding or overlap");
  static_assert(BankElements * sizeof(Element) % 128 == 0,
      "each final bank and the barrier storage must retain 128B alignment");
  static_assert(pipeline_contiguous_eight_sentinels<LayoutA,1>(),
      "A edge vectors must be contiguous and 16B aligned in every product slot");
  static_assert(pipeline_contiguous_eight_sentinels<LayoutB,0>(),
      "B edge vectors must be contiguous and 16B aligned in every product slot");
};

struct PipelinePackedEight {
  uint4 bits;
  CUTE_DEVICE uint16_t raw(int lane) const {
    const uint32_t word = lane < 2 ? bits.x : lane < 4 ? bits.y
                                            : lane < 6 ? bits.z : bits.w;
    return uint16_t(word >> (16 * (lane & 1)));
  }
};

template <class Element>
CUTE_DEVICE PipelinePackedEight pipeline_read_eight(const Element* ptr) {
  PipelinePackedEight result;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(result.bits.x),"=r"(result.bits.y),
        "=r"(result.bits.z),"=r"(result.bits.w)
      : "l"(ptr) : "memory");
  return result;
}

template <class Element>
CUTE_DEVICE float pipeline_unpack(const PipelinePackedEight& packed,int lane) {
  return float(Element::bitcast(packed.raw(lane)));
}

CUTE_DEVICE void pipeline_store_eight(void* ptr,const PipelinePackedEight& v) {
  const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
      :: "r"(address),"r"(v.bits.x),"r"(v.bits.y),
         "r"(v.bits.z),"r"(v.bits.w) : "memory");
}

template <class Element>
CUTE_DEVICE uint32_t pipeline_round_pair(float even,float odd) {
  cutlass::Array<float,2> source;
  source[0] = even; source[1] = odd;
  using Converter = cutlass::NumericArrayConverter<
      Element,float,2,cutlass::FloatRoundStyle::round_to_nearest>;
  const auto rounded = Converter{}(source);
  static_assert(sizeof(rounded) == sizeof(uint32_t),"one pair is one packed word");
  // Same representation cast as the existing cooperative paired converter.
  return reinterpret_cast<const uint32_t&>(rounded);
}

CUTE_DEVICE void pipeline_put_word(PipelinePackedEight& out,int pair,uint32_t word) {
  if (pair == 0) out.bits.x = word;
  else if (pair == 1) out.bits.y = word;
  else if (pair == 2) out.bits.z = word;
  else out.bits.w = word;
}

// Each task owns eight K elements of one row. All256 tasks are assigned once
// to128 producer threads. Every original quadrant vector is read exactly once
// for this CTA/stage and every final seven-slot A element has one writer.
template <class Element,class SmemA>
CUTE_DEVICE void pipeline_fill_a(const Problem& p,const Element* A,
    int64_t row_base,int64_t k_base,int task,SmemA sA) {
  const int64_t mh = p.m/2, kh = p.k/2;
  const int row = task/4, k = (task&3)*8;
  const int64_t gr = row_base+row, gk = k_base+k;
  const auto a11 = pipeline_read_eight(A + gr*p.lda + gk);
  const auto a12 = pipeline_read_eight(A + gr*p.lda + gk + kh);
  const auto a21 = pipeline_read_eight(A + (gr+mh)*p.lda + gk);
  const auto a22 = pipeline_read_eight(A + (gr+mh)*p.lda + gk + kh);
  pipeline_store_eight(&sA(row,k,0),a11);
  pipeline_store_eight(&sA(row,k,1),a12);
  pipeline_store_eight(&sA(row,k,3),a22);
  PipelinePackedEight s1{make_uint4(0,0,0,0)},s2{make_uint4(0,0,0,0)};
  PipelinePackedEight s3{make_uint4(0,0,0,0)},s4{make_uint4(0,0,0,0)};
  CUTE_UNROLL
  for (int pair = 0; pair < 4; ++pair) {
    float f1[2],f2[2],f3[2],f4[2];
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int lane = 2*pair+j;
      const float x11 = pipeline_unpack<Element>(a11,lane);
      const float x12 = pipeline_unpack<Element>(a12,lane);
      const float x21 = pipeline_unpack<Element>(a21,lane);
      const float x22 = pipeline_unpack<Element>(a22,lane);
      f1[j] = __fadd_rn(x21,x22);
      f2[j] = __fsub_rn(f1[j],x11);
      f3[j] = __fsub_rn(x11,x21);
      f4[j] = __fsub_rn(x12,f2[j]);
    }
    pipeline_put_word(s1,pair,pipeline_round_pair<Element>(f1[0],f1[1]));
    pipeline_put_word(s2,pair,pipeline_round_pair<Element>(f2[0],f2[1]));
    pipeline_put_word(s3,pair,pipeline_round_pair<Element>(f3[0],f3[1]));
    pipeline_put_word(s4,pair,pipeline_round_pair<Element>(f4[0],f4[1]));
  }
  pipeline_store_eight(&sA(row,k,2),s4);
  pipeline_store_eight(&sA(row,k,4),s1);
  pipeline_store_eight(&sA(row,k,5),s2);
  pipeline_store_eight(&sA(row,k,6),s3);
}

// BN16:64 tasks; BN32:128 tasks. All use vector8 original reads/stores.
// A temporaries are dead before this helper starts; there is no raw shared bank.
template <class Element,int Cols,class SmemB>
CUTE_DEVICE void pipeline_fill_b(const Problem& p,const Element* B,
    int64_t col_base,int64_t k_base,int task,SmemB sB) {
  constexpr int VectorsPerRow = Cols/8;
  const int64_t nh = p.n/2, kh = p.k/2;
  const int k = task/VectorsPerRow, col = (task%VectorsPerRow)*8;
  const int64_t gk = k_base+k, gc = col_base+col;
  const auto b11 = pipeline_read_eight(B + gk*p.ldb + gc);
  const auto b12 = pipeline_read_eight(B + gk*p.ldb + gc + nh);
  const auto b21 = pipeline_read_eight(B + (gk+kh)*p.ldb + gc);
  const auto b22 = pipeline_read_eight(B + (gk+kh)*p.ldb + gc + nh);
  pipeline_store_eight(&sB(col,k,0),b11);
  pipeline_store_eight(&sB(col,k,1),b21);
  pipeline_store_eight(&sB(col,k,2),b22);
  PipelinePackedEight t1{make_uint4(0,0,0,0)},t2{make_uint4(0,0,0,0)};
  PipelinePackedEight t3{make_uint4(0,0,0,0)},t4{make_uint4(0,0,0,0)};
  CUTE_UNROLL
  for (int pair = 0; pair < 4; ++pair) {
    float f1[2],f2[2],f3[2],f4[2];
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int lane = 2*pair+j;
      const float x11 = pipeline_unpack<Element>(b11,lane);
      const float x12 = pipeline_unpack<Element>(b12,lane);
      const float x21 = pipeline_unpack<Element>(b21,lane);
      const float x22 = pipeline_unpack<Element>(b22,lane);
      f1[j] = __fsub_rn(x12,x11);
      f2[j] = __fsub_rn(x22,f1[j]);
      f3[j] = __fsub_rn(x22,x12);
      f4[j] = __fsub_rn(f2[j],x21);
    }
    pipeline_put_word(t1,pair,pipeline_round_pair<Element>(f1[0],f1[1]));
    pipeline_put_word(t2,pair,pipeline_round_pair<Element>(f2[0],f2[1]));
    pipeline_put_word(t3,pair,pipeline_round_pair<Element>(f3[0],f3[1]));
    pipeline_put_word(t4,pair,pipeline_round_pair<Element>(f4[0],f4[1]));
  }
  pipeline_store_eight(&sB(col,k,3),t4);
  pipeline_store_eight(&sB(col,k,4),t1);
  pipeline_store_eight(&sB(col,k,5),t2);
  pipeline_store_eight(&sB(col,k,6),t3);
}

template <class Element,int Cols>
__global__ __launch_bounds__(256,1)
void winograd_wgmma_pipeline(Problem p,const Element* __restrict__ A,
    const Element* __restrict__ B,const float* C0,float* D) {
  using T = PipelineM1Traits<Element,Cols>;
  using Pipeline = typename T::Pipeline;
  const int tid = int(threadIdx.x);
  const int64_t row_base = int64_t(blockIdx.x)*T::BM;
  const int64_t col_base = int64_t(blockIdx.y)*T::BN;
  extern __shared__ __align__(128) unsigned char raw_storage[];
  auto& storage = *reinterpret_cast<typename T::SharedStorage*>(raw_storage);
  typename Pipeline::Params params;
  params.role = tid < T::ProducerThreads ? Pipeline::ThreadCategory::Producer
                                        : Pipeline::ThreadCategory::Consumer;
  params.producer_arv_count = T::ProducerThreads;
  params.consumer_arv_count = T::ConsumerThreads;
  params.dst_blockid = 0; // Ordinary non-cluster CTA, destination is this block.
  Pipeline pipeline(storage.barriers,params);
  // Every one of the256 threads constructs the pipeline and reaches this only
  // CTA barrier. No role-local CTA barrier can deadlock the two role loops.
  __syncthreads();
  if (tid < T::ProducerThreads) {
    cutlass::arch::warpgroup_reg_dealloc<64>();
    auto state = cutlass::make_producer_start_state<Pipeline>();
    CUTE_NO_UNROLL
    for (int64_t k_base = 0; k_base < p.k/2; k_base += T::BK) {
      pipeline.producer_acquire(state);
      Element* bank = storage.operands + state.index()*T::BankElements;
      auto sA = make_tensor(make_smem_ptr(bank),typename T::LayoutA{});
      auto sB = make_tensor(make_smem_ptr(bank+T::ElementsA),typename T::LayoutB{});
      CUTE_UNROLL
      for (int task = tid; task < T::BM*T::BK/8; task += T::ProducerThreads)
        pipeline_fill_a(p,A,row_base,k_base,task,sA);
      CUTE_UNROLL
      for (int task = tid; task < T::BN*T::BK/8; task += T::ProducerThreads)
        pipeline_fill_b<Element,Cols>(p,B,col_base,k_base,task,sB);
      // Each writer fences its own generic shared stores into WGMMA's async
      // proxy before its full-barrier arrival. The128th arrival publishes all.
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      pipeline.producer_commit(state);
      ++state;
    }
    // Wait for final bank readers before the producer WG exits. This waits on
    // existing empty barriers and adds neither a CTA barrier nor a launch.
    pipeline.producer_tail(state);
  } else {
    // The producer releases registers collectively; this consumer WG requests
    // its own budget collectively before initializing any accumulator.
    // The compiled N16 role uses R0..R89; its reviewed96-register quota
    // avoids forcing the larger N32 reservation. Revalidate after compilation.
    if constexpr (Cols == 16) cutlass::arch::warpgroup_reg_alloc<96>();
    else cutlass::arch::warpgroup_reg_alloc<160>();
    typename T::Mma mma;
    auto thr_mma = mma.get_thread_slice(tid%T::ConsumerThreads);
    auto coords = thr_mma.partition_C(make_identity_tensor(
        make_shape(Int<T::BM>{},Int<T::BN>{})));
    auto p1 = thr_mma.make_fragment_C(coords);
    auto p2 = thr_mma.make_fragment_C(coords);
    auto p3 = thr_mma.make_fragment_C(coords);
    auto p4 = thr_mma.make_fragment_C(coords);
    auto p5 = thr_mma.make_fragment_C(coords);
    auto p6 = thr_mma.make_fragment_C(coords);
    auto p7 = thr_mma.make_fragment_C(coords);
    static_assert(size(p1) == T::BM*T::BN/T::ConsumerThreads,
        "each consumer has8/16 coordinates and a full P fragment per product");
    clear(p1); clear(p2); clear(p3); clear(p4);
    clear(p5); clear(p6); clear(p7);
    typename Pipeline::PipelineState state;
    CUTE_NO_UNROLL
    for (int64_t k_base = 0; k_base < p.k/2; k_base += T::BK) {
      pipeline.consumer_wait(state);
      Element* bank = storage.operands + state.index()*T::BankElements;
      auto sA = make_tensor(make_smem_ptr(bank),typename T::LayoutA{});
      auto sB = make_tensor(make_smem_ptr(bank+T::ElementsA),typename T::LayoutB{});
      auto tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA));
      auto tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB));
      warpgroup_fence_operand(p1); warpgroup_fence_operand(p2);
      warpgroup_fence_operand(p3); warpgroup_fence_operand(p4);
      warpgroup_fence_operand(p5); warpgroup_fence_operand(p6);
      warpgroup_fence_operand(p7);
      warpgroup_arrive();
      gemm(mma,tCrA(_,_,_,Int<0>{}),tCrB(_,_,_,Int<0>{}),p1);
      gemm(mma,tCrA(_,_,_,Int<1>{}),tCrB(_,_,_,Int<1>{}),p2);
      gemm(mma,tCrA(_,_,_,Int<2>{}),tCrB(_,_,_,Int<2>{}),p3);
      gemm(mma,tCrA(_,_,_,Int<3>{}),tCrB(_,_,_,Int<3>{}),p4);
      gemm(mma,tCrA(_,_,_,Int<4>{}),tCrB(_,_,_,Int<4>{}),p5);
      gemm(mma,tCrA(_,_,_,Int<5>{}),tCrB(_,_,_,Int<5>{}),p6);
      gemm(mma,tCrA(_,_,_,Int<6>{}),tCrB(_,_,_,Int<6>{}),p7);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(p1); warpgroup_fence_operand(p2);
      warpgroup_fence_operand(p3); warpgroup_fence_operand(p4);
      warpgroup_fence_operand(p5); warpgroup_fence_operand(p6);
      warpgroup_fence_operand(p7);
      // Every async reader is finished before any one of128 consumer releases.
      pipeline.consumer_release(state);
      ++state;
    }
    CUTE_UNROLL
    for (int i = 0; i < size(p1); ++i) {
      const auto coord = coords(i);
      const int64_t row = row_base+get<0>(coord),col = col_base+get<1>(coord);
      const int64_t mh = p.m/2,nh = p.n/2;
      const float u1 = __fadd_rn(p1(i),p2(i));
      const float u2 = __fadd_rn(p1(i),p6(i));
      const float u3 = __fadd_rn(u2,p7(i));
      const float u4 = __fadd_rn(u2,p5(i));
      const float u5 = __fadd_rn(u4,p3(i));
      const float u6 = __fsub_rn(u3,p4(i));
      const float u7 = __fadd_rn(u3,p5(i));
      float d11 = __fmul_rn(p.alpha,u1),d12 = __fmul_rn(p.alpha,u5);
      float d21 = __fmul_rn(p.alpha,u6),d22 = __fmul_rn(p.alpha,u7);
      if (p.beta != 0.0f) {
        // All four active C0 loads precede all D stores, including C0==D.
        const float c11 = C0[row*p.ldc+col];
        const float c12 = C0[row*p.ldc+col+nh];
        const float c21 = C0[(row+mh)*p.ldc+col];
        const float c22 = C0[(row+mh)*p.ldc+col+nh];
        d11 = __fmaf_rn(p.beta,c11,d11); d12 = __fmaf_rn(p.beta,c12,d12);
        d21 = __fmaf_rn(p.beta,c21,d21); d22 = __fmaf_rn(p.beta,c22,d22);
      }
      D[row*p.ldd+col] = d11; D[row*p.ldd+col+nh] = d12;
      D[(row+mh)*p.ldd+col] = d21; D[(row+mh)*p.ldd+col+nh] = d22;
    }
  }
}

template <class Element,int Cols>
void prepare_wgmma_pipeline_typed() {
  constexpr size_t smem = PipelineM1Traits<Element,Cols>::SharedBytes;
  static_assert(smem > 48*1024 && smem < 227*1024,"reviewed H100 shared budget");
  const auto status = cudaFuncSetAttribute(winograd_wgmma_pipeline<Element,Cols>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem));
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("WGMMA pipeline shared-memory opt-in: ")+
        cudaGetErrorString(status));
}

template <class Element,int Cols>
void launch_wgmma_pipeline_typed(const Problem& p,const void* A,const void* B,
    const float* C0,float* D,cudaStream_t stream) {
  using T = PipelineM1Traits<Element,Cols>;
  const dim3 grid(unsigned(p.m/(2*T::BM)),unsigned(p.n/(2*T::BN)));
  winograd_wgmma_pipeline<Element,Cols><<<grid,T::Threads,T::SharedBytes,stream>>>(
      p,static_cast<const Element*>(A),static_cast<const Element*>(B),C0,D);
}

template <int Cols>
bool wgmma_pipeline_native_binding(const Problem& p,const void* A,const void* B) {
  return p.m%128 == 0 && p.n%(2*Cols) == 0 && p.k%64 == 0 &&
      p.lda%8 == 0 && p.ldb%8 == 0 &&
      (reinterpret_cast<uintptr_t>(A)&15) == 0 &&
      (reinterpret_cast<uintptr_t>(B)&15) == 0;
}
} // namespace

// Internal configuration0/1: BN16/BN32, formal policies28/29.
bool wgmma_pipeline_can_implement(const Problem& p,int configuration) {
  if ((configuration != 0 && configuration != 1) || !megakernel_can_implement(p,0))
    return false;
  const int cols = configuration == 0 ? 16 : 32;
  return p.m%128 == 0 && p.n%(2*cols) == 0 && p.k%64 == 0 &&
      p.lda%8 == 0 && p.ldb%8 == 0;
}

bool wgmma_pipeline_full_tiles(const Problem& p,const void* A,const void* B,
    int configuration) {
  if (!wgmma_pipeline_can_implement(p,configuration) || !A || !B) return false;
  return configuration == 0 ? wgmma_pipeline_native_binding<16>(p,A,B)
                            : wgmma_pipeline_native_binding<32>(p,A,B);
}

void prepare_wgmma_pipeline(const Problem& p,int configuration) {
  if (!wgmma_pipeline_can_implement(p,configuration))
    throw std::invalid_argument("unsupported WGMMA pipeline problem/configuration");
  if (p.dtype == DType::BF16) {
    if (configuration == 0) prepare_wgmma_pipeline_typed<cutlass::bfloat16_t,16>();
    else prepare_wgmma_pipeline_typed<cutlass::bfloat16_t,32>();
  } else {
    if (configuration == 0) prepare_wgmma_pipeline_typed<cutlass::half_t,16>();
    else prepare_wgmma_pipeline_typed<cutlass::half_t,32>();
  }
}

void launch_wgmma_pipeline(const Problem& p,const void* A,const void* B,
    const float* C0,float* D,cudaStream_t stream,int configuration) {
  if (!wgmma_pipeline_can_implement(p,configuration))
    throw std::invalid_argument("unsupported WGMMA pipeline problem/configuration");
  if (!A || !B || !D || (p.beta != 0.0f && !C0))
    throw std::invalid_argument("WGMMA pipeline requires nonnull active pointers");
  // Current binding guards run on every call/capture. Graph replay inherits
  // fixed captured bindings and recomputes the fresh original inputs on device.
  // Generic span/overlap/alias checks and truthful fallback selection remain
  // the integrating Plan's contract. This helper launches only the native path.
  if (!wgmma_pipeline_full_tiles(p,A,B,configuration))
    throw std::invalid_argument("WGMMA pipeline requires native full tiles and AB16B");
  if (p.dtype == DType::BF16) {
    if (configuration == 0)
      launch_wgmma_pipeline_typed<cutlass::bfloat16_t,16>(p,A,B,C0,D,stream);
    else launch_wgmma_pipeline_typed<cutlass::bfloat16_t,32>(p,A,B,C0,D,stream);
  } else {
    if (configuration == 0)
      launch_wgmma_pipeline_typed<cutlass::half_t,16>(p,A,B,C0,D,stream);
    else launch_wgmma_pipeline_typed<cutlass::half_t,32>(p,A,B,C0,D,stream);
  }
}
} // namespace megafmm

// Scratch-only: not compiled, GPU-tested, timed, or numerically certified.
// One-launch M1 TMA raw aliases, dedicated producer WG, two final banks, full seven P.
#include "megafmm/kernels.h"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cuda.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace megafmm {
namespace {
using namespace cute;

template <class Layout, int ContiguousMode>
// Bound CUDA12.4 constexpr work: full geometry has an independent CPU model.
constexpr bool tma_alias_contiguous_eight_sentinels() {
  constexpr int rows = size<0>(Layout{});
  constexpr int cols = size<1>(Layout{});
  constexpr int slots = size<2>(Layout{});
  for (int slot = 0; slot < slots; ++slot) {
    for (int row_edge = 0; row_edge < 2; ++row_edge) {
      const int row = row_edge == 0 ? 0 : rows-(ContiguousMode == 0 ? 8 : 1);
      for (int col_edge = 0; col_edge < 2; ++col_edge) {
        const int col = col_edge == 0 ? 0 : cols-(ContiguousMode == 1 ? 8 : 1);
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

// Compare the pinned CuTe coordinates with TMA's dense row/column-major
// swizzle map. All pattern bases are512B aligned, so both use base offset0.
template <class Layout,int ContiguousMode,int SwizzleBits>
constexpr bool tma_alias_dense_box_sentinels_match() {
  constexpr int rows = size<0>(Layout{});
  constexpr int cols = size<1>(Layout{});
  for (int slot = 0; slot < 7; ++slot) {
    // Include atom/pattern transitions and tile ends without exhaustive CuTe
    // instantiation work. CPU checks independently enumerate every coordinate.
    for (int ri = 0; ri < 4; ++ri) {
      const int row = ri == 0 ? 0 : ri == 1 ? 7 : ri == 2 ? 8 : rows-1;
      for (int ci = 0; ci < 6; ++ci) {
        const int col = ci == 0 ? 0 : ci == 1 ? 7 : ci == 2 ? 8 :
            ci == 3 ? 15 : ci == 4 ? 16 : cols-1;
        const int plain = (ContiguousMode == 1 ? row*cols+col : col*rows+row)
            +slot*rows*cols;
        const int mask = (1<<SwizzleBits)-1;
        const int hardware = plain ^ (((plain>>6)&mask)<<3);
        if (Layout{}(make_coord(row,col,slot)) != hardware) return false;
      }
    }
  }
  return true;
}

template <class Element, int Cols>
struct TmaAliasM1Traits {
  static_assert(Cols == 16 || Cols == 32,"only reviewed M1 tiles are supported");
  static constexpr int BM = 64;
  static constexpr int BN = Cols;
  static constexpr int BK = 32;
  static constexpr int Threads = 256;
  static constexpr int ProducerThreads = 128;
  static constexpr int ConsumerThreads = 128;
  static constexpr int Stages = 2;
  using Tile = Shape<Int<BM>,Int<BN>,Int<BK>>;
  // Full 64B TMA inner boxes require SW64. BN16 B is a 32B inner box.
  // Separate rank-3 banks retain canonical GMMA descriptors for all seven P.
  using LayoutA = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},
      make_shape(Int<BM>{},Int<BK>{},Int<7>{})));
  using BAtom = std::conditional_t<BN == 16,
      GMMA::Layout_MN_SW32_Atom<Element>,GMMA::Layout_MN_SW64_Atom<Element>>;
  using LayoutB = decltype(tile_to_shape(BAtom{},
      make_shape(Int<BN>{},Int<BK>{},Int<7>{})));
  using Op = decltype(GMMA::ss_op_selector<Element,Element,float,Tile,
      GMMA::Major::K,GMMA::Major::MN>());
  using Mma = decltype(make_tiled_mma(Op{}));
  using Pipeline = cutlass::PipelineAsync<Stages>;
  static constexpr int ElementsA = cosize_v<LayoutA>;
  static constexpr int ElementsB = cosize_v<LayoutB>;
  static constexpr int BankElements = ElementsA + ElementsB;
  struct alignas(512) SharedStorage {
    Element operands[Stages * BankElements];
    typename Pipeline::SharedStorage barriers;
    alignas(8) uint64_t raw_done[Stages];
  };
  static constexpr size_t SharedBytes = sizeof(SharedStorage);
  static constexpr uint32_t RawBytes = 4*(BM*BK + BN*BK)*sizeof(Element);
  static_assert(BankElements*sizeof(Element)%512 == 0,
      "each bank must begin at the SW64 repeating-pattern boundary");
  static_assert(ElementsA*sizeof(Element)%512 == 0 &&
      BM*BK*sizeof(Element)%512 == 0 && BN*BK*sizeof(Element)%512 == 0,
      "A/B product slots must retain zero descriptor base offset");
  static_assert(sizeof(Element) == 2,"vector helpers require 16-bit operands");
  static_assert(size(Mma{}) == ConsumerThreads,"one consumer WG owns the tile");
  static_assert(ElementsA == 7*BM*BK && ElementsB == 7*BN*BK,
      "compact product slots must have no padding or overlap");
  static_assert(BankElements * sizeof(Element) % 128 == 0,
      "each final bank and the barrier storage must retain 128B alignment");
  static_assert(tma_alias_contiguous_eight_sentinels<LayoutA,1>(),
      "A edge vectors must be contiguous and 16B aligned in every product slot");
  static_assert(tma_alias_contiguous_eight_sentinels<LayoutB,0>(),
      "B edge vectors must be contiguous and 16B aligned in every product slot");
  static_assert(tma_alias_dense_box_sentinels_match<LayoutA,1,2>(),
      "A SW64 TMA box sentinels must match the GMMA seven-slot tensor");
  static_assert(tma_alias_dense_box_sentinels_match<LayoutB,0,BN == 16 ? 1 : 2>(),
      "B TMA swizzle sentinels must match the GMMA seven-slot tensor");
};

struct TmaAliasPackedEight {
  uint4 bits;
  CUTE_DEVICE uint16_t raw(int lane) const {
    const uint32_t word = lane < 2 ? bits.x : lane < 4 ? bits.y
                                            : lane < 6 ? bits.z : bits.w;
    return uint16_t(word >> (16 * (lane & 1)));
  }
};

template <class Element>
CUTE_DEVICE TmaAliasPackedEight tma_alias_read_eight(const Element* ptr) {
  TmaAliasPackedEight result;
  const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(result.bits.x),"=r"(result.bits.y),
        "=r"(result.bits.z),"=r"(result.bits.w)
      : "r"(address) : "memory");
  return result;
}

template <class Element>
CUTE_DEVICE float tma_alias_unpack(const TmaAliasPackedEight& packed,int lane) {
  return float(Element::bitcast(packed.raw(lane)));
}

CUTE_DEVICE void tma_alias_store_eight(void* ptr,const TmaAliasPackedEight& v) {
  const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
      :: "r"(address),"r"(v.bits.x),"r"(v.bits.y),
         "r"(v.bits.z),"r"(v.bits.w) : "memory");
}

template <class Element>
CUTE_DEVICE uint32_t tma_alias_round_pair(float even,float odd) {
  cutlass::Array<float,2> source;
  source[0] = even; source[1] = odd;
  using Converter = cutlass::NumericArrayConverter<
      Element,float,2,cutlass::FloatRoundStyle::round_to_nearest>;
  const auto rounded = Converter{}(source);
  static_assert(sizeof(rounded) == sizeof(uint32_t),"one pair is one packed word");
  // Same representation cast as the existing cooperative paired converter.
  return reinterpret_cast<const uint32_t&>(rounded);
}

CUTE_DEVICE void tma_alias_put_word(TmaAliasPackedEight& out,int pair,uint32_t word) {
  if (pair == 0) out.bits.x = word;
  else if (pair == 1) out.bits.y = word;
  else if (pair == 2) out.bits.z = word;
  else out.bits.w = word;
}

// Raw TMA slots: A11=0, A12=1, A22=3, A21 temporarily aliases S3=6.
// All four raw vectors are loaded before any transformed slot is overwritten.
// Only transformed slots2/4/5/6 receive generic stores; raw originals stay TMA bits.
template <class Element,class SmemA>
CUTE_DEVICE void tma_alias_fill_a(int task,SmemA sA) {
  const int row = task/4, k = (task&3)*8;
  const auto a11 = tma_alias_read_eight(&sA(row,k,0));
  const auto a12 = tma_alias_read_eight(&sA(row,k,1));
  const auto a21 = tma_alias_read_eight(&sA(row,k,6));
  const auto a22 = tma_alias_read_eight(&sA(row,k,3));
  TmaAliasPackedEight s1{make_uint4(0,0,0,0)},s2{make_uint4(0,0,0,0)};
  TmaAliasPackedEight s3{make_uint4(0,0,0,0)},s4{make_uint4(0,0,0,0)};
  CUTE_UNROLL
  for (int pair = 0; pair < 4; ++pair) {
    float f1[2],f2[2],f3[2],f4[2];
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int lane = 2*pair+j;
      const float x11 = tma_alias_unpack<Element>(a11,lane);
      const float x12 = tma_alias_unpack<Element>(a12,lane);
      const float x21 = tma_alias_unpack<Element>(a21,lane);
      const float x22 = tma_alias_unpack<Element>(a22,lane);
      f1[j] = __fadd_rn(x21,x22);
      f2[j] = __fsub_rn(f1[j],x11);
      f3[j] = __fsub_rn(x11,x21);
      f4[j] = __fsub_rn(x12,f2[j]);
    }
    tma_alias_put_word(s1,pair,tma_alias_round_pair<Element>(f1[0],f1[1]));
    tma_alias_put_word(s2,pair,tma_alias_round_pair<Element>(f2[0],f2[1]));
    tma_alias_put_word(s3,pair,tma_alias_round_pair<Element>(f3[0],f3[1]));
    tma_alias_put_word(s4,pair,tma_alias_round_pair<Element>(f4[0],f4[1]));
  }
  tma_alias_store_eight(&sA(row,k,2),s4);
  tma_alias_store_eight(&sA(row,k,4),s1);
  tma_alias_store_eight(&sA(row,k,5),s2);
  tma_alias_store_eight(&sA(row,k,6),s3);
}

// Raw TMA slots: B11=0, B21=1, B22=2, B12 temporarily aliases T1=4.
// All raw reads precede the T1 overwrite at the same uniquely owned coordinates.
template <class Element,int Cols,class SmemB>
CUTE_DEVICE void tma_alias_fill_b(int task,SmemB sB) {
  constexpr int VectorsPerRow = Cols/8;
  const int k = task/VectorsPerRow, col = (task%VectorsPerRow)*8;
  const auto b11 = tma_alias_read_eight(&sB(col,k,0));
  const auto b12 = tma_alias_read_eight(&sB(col,k,4));
  const auto b21 = tma_alias_read_eight(&sB(col,k,1));
  const auto b22 = tma_alias_read_eight(&sB(col,k,2));
  TmaAliasPackedEight t1{make_uint4(0,0,0,0)},t2{make_uint4(0,0,0,0)};
  TmaAliasPackedEight t3{make_uint4(0,0,0,0)},t4{make_uint4(0,0,0,0)};
  CUTE_UNROLL
  for (int pair = 0; pair < 4; ++pair) {
    float f1[2],f2[2],f3[2],f4[2];
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int lane = 2*pair+j;
      const float x11 = tma_alias_unpack<Element>(b11,lane);
      const float x12 = tma_alias_unpack<Element>(b12,lane);
      const float x21 = tma_alias_unpack<Element>(b21,lane);
      const float x22 = tma_alias_unpack<Element>(b22,lane);
      f1[j] = __fsub_rn(x12,x11);
      f2[j] = __fsub_rn(x22,f1[j]);
      f3[j] = __fsub_rn(x22,x12);
      f4[j] = __fsub_rn(f2[j],x21);
    }
    tma_alias_put_word(t1,pair,tma_alias_round_pair<Element>(f1[0],f1[1]));
    tma_alias_put_word(t2,pair,tma_alias_round_pair<Element>(f2[0],f2[1]));
    tma_alias_put_word(t3,pair,tma_alias_round_pair<Element>(f3[0],f3[1]));
    tma_alias_put_word(t4,pair,tma_alias_round_pair<Element>(f4[0],f4[1]));
  }
  tma_alias_store_eight(&sB(col,k,3),t4);
  tma_alias_store_eight(&sB(col,k,4),t1);
  tma_alias_store_eight(&sB(col,k,5),t2);
  tma_alias_store_eight(&sB(col,k,6),t3);
}

// Immutable descriptors are passed by value in kernel parameter space. Encoding
// uses current original bindings on each native ordinary invocation/capture.
struct alignas(64) TmaAliasDescriptors {
  CUtensorMap a;
  CUtensorMap b;
};
static_assert(sizeof(TmaAliasDescriptors) == 256 &&
    alignof(TmaAliasDescriptors) == 64 &&
    std::is_trivially_copyable<TmaAliasDescriptors>::value,
    "two 128B descriptors must remain aligned immutable kernel parameters");

template <class Element,int Cols>
__global__ __launch_bounds__(256,1)
void winograd_wgmma_tma_alias(Problem p,
    CUTE_GRID_CONSTANT const TmaAliasDescriptors maps,const float* C0,float* D) {
  using T = TmaAliasM1Traits<Element,Cols>;
  using Pipeline = typename T::Pipeline;
  const int tid = int(threadIdx.x);
  const int64_t row_base = int64_t(blockIdx.x)*T::BM;
  const int64_t col_base = int64_t(blockIdx.y)*T::BN;
  extern __shared__ __align__(512) unsigned char raw_storage[];
  auto& storage = *reinterpret_cast<typename T::SharedStorage*>(raw_storage);
  typename Pipeline::Params params;
  params.role = tid < T::ProducerThreads ? Pipeline::ThreadCategory::Producer
                                        : Pipeline::ThreadCategory::Consumer;
  params.producer_arv_count = T::ProducerThreads;
  params.consumer_arv_count = T::ConsumerThreads;
  params.dst_blockid = 0; // Ordinary non-cluster CTA, destination is this block.
  Pipeline pipeline(storage.barriers,params);
  if (tid == 0) {
    CUTE_UNROLL
    for (int bank = 0; bank < T::Stages; ++bank)
      cutlass::arch::ClusterTransactionBarrier::init(&storage.raw_done[bank],1);
    cutlass::arch::fence_barrier_init();
  }
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
      // Exactly eight tensor copies. The raw transaction barrier publishes
      // only raw slots; final-ready PipelineAsync publishes transformed slots.
      if (tid == 0) {
        auto* raw_done = &storage.raw_done[state.index()];
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            raw_done,T::RawBytes);
        constexpr uint64_t hint = uint64_t(TMA::CacheHintSm90::EVICT_NORMAL);
        const int32_t r = int32_t(row_base),c = int32_t(col_base);
        const int32_t k = int32_t(k_base),mh = int32_t(p.m/2);
        const int32_t nh = int32_t(p.n/2),kh = int32_t(p.k/2);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,0),k,r);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,1),k+kh,r);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,6),k,r+mh);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,3),k+kh,r+mh);
        SM90_TMA_LOAD_2D::copy(&maps.b,raw_done,hint,&sB(0,0,0),c,k);
        SM90_TMA_LOAD_2D::copy(&maps.b,raw_done,hint,&sB(0,0,4),c+nh,k);
        SM90_TMA_LOAD_2D::copy(&maps.b,raw_done,hint,&sB(0,0,1),c,k+kh);
        SM90_TMA_LOAD_2D::copy(&maps.b,raw_done,hint,&sB(0,0,2),c+nh,k+kh);
      }
      // Producer PipelineAsync starts phase1; raw barriers start phase0.
      // Both flip on bank reuse, so their wait phases remain complemented.
      cutlass::arch::ClusterTransactionBarrier::wait(
          &storage.raw_done[state.index()],state.phase()^1u);
      // Compiler ordering is separate from raw TMA's hardware completion.
      // The pinned wait uses volatile PTX without a compiler memory clobber.
      asm volatile("" ::: "memory");
      CUTE_UNROLL
      for (int task = tid; task < T::BM*T::BK/8; task += T::ProducerThreads)
        tma_alias_fill_a<Element>(task,sA);
      CUTE_UNROLL
      for (int task = tid; task < T::BN*T::BK/8; task += T::ProducerThreads)
        tma_alias_fill_b<Element,Cols>(task,sB);
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
    cutlass::arch::warpgroup_reg_alloc<Cols == 16 ? 96 : 160>();
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
void prepare_wgmma_tma_alias_typed() {
  constexpr size_t smem = TmaAliasM1Traits<Element,Cols>::SharedBytes;
  static_assert(smem > 48*1024 && smem < 227*1024,"reviewed H100 shared budget");
  const auto status = cudaFuncSetAttribute(winograd_wgmma_tma_alias<Element,Cols>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem));
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("WGMMA TMA alias shared-memory opt-in: ")+
        cudaGetErrorString(status));
}

// No pointer-bound descriptor cache: this host work stays in each timed run.
// Caller-native guards establish dimensions, strides and current AB16 before it.
template <int Cols>
TmaAliasDescriptors make_tma_alias_descriptors(const Problem& p,
    const void* A,const void* B) {
  TmaAliasDescriptors maps{};
  const CUtensorMapDataType type = p.dtype == DType::BF16
      ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  const cuuint64_t a_dims[2] = {cuuint64_t(p.k),cuuint64_t(p.m)};
  const cuuint64_t b_dims[2] = {cuuint64_t(p.n),cuuint64_t(p.k)};
  const cuuint64_t a_stride[1] = {cuuint64_t(p.lda)*2};
  const cuuint64_t b_stride[1] = {cuuint64_t(p.ldb)*2};
  const cuuint32_t a_box[2] = {32,64};
  const cuuint32_t b_box[2] = {Cols,32};
  const cuuint32_t step[2] = {1,1};
  const CUtensorMapSwizzle b_swizzle = Cols == 16
      ? CU_TENSOR_MAP_SWIZZLE_32B : CU_TENSOR_MAP_SWIZZLE_64B;
  const auto encode = [&](CUtensorMap* map,const void* ptr,
      const cuuint64_t* dims,const cuuint64_t* strides,
      const cuuint32_t* box,CUtensorMapSwizzle swizzle) {
    const auto result = cuTensorMapEncodeTiled(map,type,2,const_cast<void*>(ptr),
        dims,strides,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS)
      throw std::runtime_error("WGMMA TMA alias tensor-map encode error="+
          std::to_string(int(result)));
  };
  encode(&maps.a,A,a_dims,a_stride,a_box,CU_TENSOR_MAP_SWIZZLE_64B);
  encode(&maps.b,B,b_dims,b_stride,b_box,b_swizzle);
  return maps;
}

template <class Element,int Cols>
void launch_wgmma_tma_alias_typed(const Problem& p,const void* A,const void* B,
    const float* C0,float* D,cudaStream_t stream) {
  using T = TmaAliasM1Traits<Element,Cols>;
  const auto maps = make_tma_alias_descriptors<Cols>(p,A,B);
  const dim3 grid(unsigned(p.m/(2*T::BM)),unsigned(p.n/(2*T::BN)));
  winograd_wgmma_tma_alias<Element,Cols><<<grid,T::Threads,T::SharedBytes,stream>>>(
      p,maps,C0,D);
}

template <int Cols>
bool wgmma_tma_alias_native_binding(const Problem& p,const void* A,const void* B) {
  return p.m%128 == 0 && p.n%(2*Cols) == 0 && p.k%64 == 0 &&
      p.lda%8 == 0 && p.ldb%8 == 0 &&
      uint64_t(p.lda) < (uint64_t(1)<<39) &&
      uint64_t(p.ldb) < (uint64_t(1)<<39) &&
      (reinterpret_cast<uintptr_t>(A)&15) == 0 &&
      (reinterpret_cast<uintptr_t>(B)&15) == 0;
}
} // namespace

// Scratch configuration0/1: BN16/BN32, no formal public policy IDs assigned.
bool wgmma_tma_alias_can_implement(const Problem& p,int configuration) {
  if ((configuration != 0 && configuration != 1) || !megakernel_can_implement(p,0))
    return false;
  const int cols = configuration == 0 ? 16 : 32;
  return p.m%128 == 0 && p.n%(2*cols) == 0 && p.k%64 == 0 &&
      p.lda%8 == 0 && p.ldb%8 == 0 &&
      uint64_t(p.lda) < (uint64_t(1)<<39) &&
      uint64_t(p.ldb) < (uint64_t(1)<<39);
}

bool wgmma_tma_alias_full_tiles(const Problem& p,const void* A,const void* B,
    int configuration) {
  if (!wgmma_tma_alias_can_implement(p,configuration) || !A || !B) return false;
  return configuration == 0 ? wgmma_tma_alias_native_binding<16>(p,A,B)
                            : wgmma_tma_alias_native_binding<32>(p,A,B);
}

void prepare_wgmma_tma_alias(const Problem& p,int configuration) {
  if (!wgmma_tma_alias_can_implement(p,configuration))
    throw std::invalid_argument("unsupported WGMMA TMA alias problem/configuration");
  int device = 0,major = 0,minor = 0;
  auto check = [](cudaError_t status) {
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("WGMMA TMA alias preparation: ")+
          cudaGetErrorString(status));
  };
  check(cudaGetDevice(&device));
  check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));
  check(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));
  if (major != 9 || minor != 0)
    throw std::invalid_argument("WGMMA TMA alias requires the prepared SM90a device");
  if (p.dtype == DType::BF16) {
    if (configuration == 0) prepare_wgmma_tma_alias_typed<cutlass::bfloat16_t,16>();
    else prepare_wgmma_tma_alias_typed<cutlass::bfloat16_t,32>();
  } else {
    if (configuration == 0) prepare_wgmma_tma_alias_typed<cutlass::half_t,16>();
    else prepare_wgmma_tma_alias_typed<cutlass::half_t,32>();
  }
}

void launch_wgmma_tma_alias(const Problem& p,const void* A,const void* B,
    const float* C0,float* D,cudaStream_t stream,int configuration) {
  if (!wgmma_tma_alias_can_implement(p,configuration))
    throw std::invalid_argument("unsupported WGMMA TMA alias problem/configuration");
  if (!A || !B || !D || (p.beta != 0.0f && !C0))
    throw std::invalid_argument("WGMMA TMA alias requires nonnull active pointers");
  // Current binding guards run on every call/capture. Graph replay inherits
  // fixed captured bindings and recomputes the fresh original inputs on device.
  // Tensor maps are rebuilt for these actual pointers in the timed host call.
  // Generic span/overlap/alias checks and truthful fallback selection remain
  // the integrating Plan's contract. This helper launches only the native path.
  if (!wgmma_tma_alias_full_tiles(p,A,B,configuration))
    throw std::invalid_argument("WGMMA TMA alias requires native full tiles and AB16B");
  if (p.dtype == DType::BF16) {
    if (configuration == 0)
      launch_wgmma_tma_alias_typed<cutlass::bfloat16_t,16>(p,A,B,C0,D,stream);
    else launch_wgmma_tma_alias_typed<cutlass::bfloat16_t,32>(p,A,B,C0,D,stream);
  } else {
    if (configuration == 0)
      launch_wgmma_tma_alias_typed<cutlass::half_t,16>(p,A,B,C0,D,stream);
    else launch_wgmma_tma_alias_typed<cutlass::half_t,32>(p,A,B,C0,D,stream);
  }
}
} // namespace megafmm

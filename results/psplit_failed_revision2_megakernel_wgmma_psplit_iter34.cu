// Scratch-only P3+4 consumer split. No compile, GPU, performance or certification.
// One ordinary launch, zero global ST/P workspace, full seven P then fixed U.
// Preserves source30 vector producer and source-level per-P ascending K order.
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
constexpr bool psplit_contiguous_eight_sentinels() {
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
struct PSplitTraits {
  static_assert(Cols == 32,"this prototype uses BN32 only");
  static constexpr int BM = 64;
  static constexpr int BN = Cols;
  static constexpr int BK = 32;
  static constexpr int Threads = 384;
  static constexpr int ProducerThreads = 128;
  static constexpr int ConsumerThreads = 128; // each independent WG
  static constexpr int AllConsumerThreads = 256;
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
    // Float tail is an explicitly typed alternate member, activated by tail
    // stores only after both operand banks are uniformly retired. Barriers do
    // not overlap either member. Payload sizeof stays at the larger operands.
    union {
      Element operands[Stages * BankElements];
      float tail_products[7*BM*BN];
    };
    typename Pipeline::SharedStorage barriers;
  };
  static constexpr size_t SharedBytes = sizeof(SharedStorage);
  static_assert(sizeof(Element) == 2,"vector helpers require 16-bit operands");
  static_assert(size(Mma{}) == ConsumerThreads,"one WG owns all coordinates of each assigned P");
  static_assert(ElementsA == 7*BM*BK && ElementsB == 7*BN*BK,
      "compact product slots must have no padding or overlap");
  static_assert(BankElements * sizeof(Element) % 128 == 0,
      "each final bank and the barrier storage must retain 128B alignment");
  static_assert(psplit_contiguous_eight_sentinels<LayoutA,1>(),
      "A edge vectors must be contiguous and 16B aligned in every product slot");
  static_assert(psplit_contiguous_eight_sentinels<LayoutB,0>(),
      "B edge vectors must be contiguous and 16B aligned in every product slot");
};

struct PSplitPackedEight {
  uint4 bits;
  CUTE_DEVICE uint16_t raw(int lane) const {
    const uint32_t word = lane < 2 ? bits.x : lane < 4 ? bits.y
                                            : lane < 6 ? bits.z : bits.w;
    return uint16_t(word >> (16 * (lane & 1)));
  }
};

template <class Element>
CUTE_DEVICE PSplitPackedEight psplit_read_eight(const Element* ptr) {
  PSplitPackedEight result;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(result.bits.x),"=r"(result.bits.y),
        "=r"(result.bits.z),"=r"(result.bits.w)
      : "l"(ptr) : "memory");
  return result;
}

template <class Element>
CUTE_DEVICE float psplit_unpack(const PSplitPackedEight& packed,int lane) {
  return float(Element::bitcast(packed.raw(lane)));
}

CUTE_DEVICE void psplit_store_eight(void* ptr,const PSplitPackedEight& v) {
  const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
      :: "r"(address),"r"(v.bits.x),"r"(v.bits.y),
         "r"(v.bits.z),"r"(v.bits.w) : "memory");
}

template <class Element>
CUTE_DEVICE uint32_t psplit_round_pair(float even,float odd) {
  cutlass::Array<float,2> source;
  source[0] = even; source[1] = odd;
  using Converter = cutlass::NumericArrayConverter<
      Element,float,2,cutlass::FloatRoundStyle::round_to_nearest>;
  const auto rounded = Converter{}(source);
  static_assert(sizeof(rounded) == sizeof(uint32_t),"one pair is one packed word");
  // Same representation cast as the existing cooperative paired converter.
  return reinterpret_cast<const uint32_t&>(rounded);
}

CUTE_DEVICE void psplit_put_word(PSplitPackedEight& out,int pair,uint32_t word) {
  if (pair == 0) out.bits.x = word;
  else if (pair == 1) out.bits.y = word;
  else if (pair == 2) out.bits.z = word;
  else out.bits.w = word;
}

// Each task owns eight K elements of one row. All256 tasks are assigned once
// to128 producer threads. Every original quadrant vector is read exactly once
// for this CTA/stage and every final seven-slot A element has one writer.
template <class Element,class SmemA>
CUTE_DEVICE void psplit_fill_a(const Problem& p,const Element* A,
    int64_t row_base,int64_t k_base,int task,SmemA sA) {
  const int64_t mh = p.m/2, kh = p.k/2;
  const int row = task/4, k = (task&3)*8;
  const int64_t gr = row_base+row, gk = k_base+k;
  const auto a11 = psplit_read_eight(A + gr*p.lda + gk);
  const auto a12 = psplit_read_eight(A + gr*p.lda + gk + kh);
  const auto a21 = psplit_read_eight(A + (gr+mh)*p.lda + gk);
  const auto a22 = psplit_read_eight(A + (gr+mh)*p.lda + gk + kh);
  psplit_store_eight(&sA(row,k,0),a11);
  psplit_store_eight(&sA(row,k,1),a12);
  psplit_store_eight(&sA(row,k,3),a22);
  PSplitPackedEight s1{make_uint4(0,0,0,0)},s2{make_uint4(0,0,0,0)};
  PSplitPackedEight s3{make_uint4(0,0,0,0)},s4{make_uint4(0,0,0,0)};
  CUTE_UNROLL
  for (int pair = 0; pair < 4; ++pair) {
    float f1[2],f2[2],f3[2],f4[2];
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int lane = 2*pair+j;
      const float x11 = psplit_unpack<Element>(a11,lane);
      const float x12 = psplit_unpack<Element>(a12,lane);
      const float x21 = psplit_unpack<Element>(a21,lane);
      const float x22 = psplit_unpack<Element>(a22,lane);
      f1[j] = __fadd_rn(x21,x22);
      f2[j] = __fsub_rn(f1[j],x11);
      f3[j] = __fsub_rn(x11,x21);
      f4[j] = __fsub_rn(x12,f2[j]);
    }
    psplit_put_word(s1,pair,psplit_round_pair<Element>(f1[0],f1[1]));
    psplit_put_word(s2,pair,psplit_round_pair<Element>(f2[0],f2[1]));
    psplit_put_word(s3,pair,psplit_round_pair<Element>(f3[0],f3[1]));
    psplit_put_word(s4,pair,psplit_round_pair<Element>(f4[0],f4[1]));
  }
  psplit_store_eight(&sA(row,k,2),s4);
  psplit_store_eight(&sA(row,k,4),s1);
  psplit_store_eight(&sA(row,k,5),s2);
  psplit_store_eight(&sA(row,k,6),s3);
}

// BN16:64 tasks; BN32:128 tasks. All use vector8 original reads/stores.
// A temporaries are dead before this helper starts; there is no raw shared bank.
template <class Element,int Cols,class SmemB>
CUTE_DEVICE void psplit_fill_b(const Problem& p,const Element* B,
    int64_t col_base,int64_t k_base,int task,SmemB sB) {
  constexpr int VectorsPerRow = Cols/8;
  const int64_t nh = p.n/2, kh = p.k/2;
  const int k = task/VectorsPerRow, col = (task%VectorsPerRow)*8;
  const int64_t gk = k_base+k, gc = col_base+col;
  const auto b11 = psplit_read_eight(B + gk*p.ldb + gc);
  const auto b12 = psplit_read_eight(B + gk*p.ldb + gc + nh);
  const auto b21 = psplit_read_eight(B + (gk+kh)*p.ldb + gc);
  const auto b22 = psplit_read_eight(B + (gk+kh)*p.ldb + gc + nh);
  psplit_store_eight(&sB(col,k,0),b11);
  psplit_store_eight(&sB(col,k,1),b21);
  psplit_store_eight(&sB(col,k,2),b22);
  PSplitPackedEight t1{make_uint4(0,0,0,0)},t2{make_uint4(0,0,0,0)};
  PSplitPackedEight t3{make_uint4(0,0,0,0)},t4{make_uint4(0,0,0,0)};
  CUTE_UNROLL
  for (int pair = 0; pair < 4; ++pair) {
    float f1[2],f2[2],f3[2],f4[2];
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int lane = 2*pair+j;
      const float x11 = psplit_unpack<Element>(b11,lane);
      const float x12 = psplit_unpack<Element>(b12,lane);
      const float x21 = psplit_unpack<Element>(b21,lane);
      const float x22 = psplit_unpack<Element>(b22,lane);
      f1[j] = __fsub_rn(x12,x11);
      f2[j] = __fsub_rn(x22,f1[j]);
      f3[j] = __fsub_rn(x22,x12);
      f4[j] = __fsub_rn(f2[j],x21);
    }
    psplit_put_word(t1,pair,psplit_round_pair<Element>(f1[0],f1[1]));
    psplit_put_word(t2,pair,psplit_round_pair<Element>(f2[0],f2[1]));
    psplit_put_word(t3,pair,psplit_round_pair<Element>(f3[0],f3[1]));
    psplit_put_word(t4,pair,psplit_round_pair<Element>(f4[0],f4[1]));
  }
  psplit_store_eight(&sB(col,k,3),t4);
  psplit_store_eight(&sB(col,k,4),t1);
  psplit_store_eight(&sB(col,k,5),t2);
  psplit_store_eight(&sB(col,k,6),t3);
}


// Tail FP32 rendezvous aliases the complete drained TWO-bank operands payload.
// XOR permutation is a bijection within each row, does not add padding, and is
// separate from the WGMMA layouts used only before the first tail CTA barrier.
CUTE_DEVICE int psplit_tail_word(int product,int row,int col) {
  const int xor_mask = (row&1) | ((row&6)<<2);
  return product*2048+row*32+(col^xor_mask);
}

template <class Element,int Cols>
__global__ __launch_bounds__(384,1)
void winograd_wgmma_psplit(Problem p,const Element* __restrict__ A,
    const Element* __restrict__ B,const float* C0,float* D) {
  using T = PSplitTraits<Element,Cols>;
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
  params.consumer_arv_count = T::AllConsumerThreads;
  params.dst_blockid = 0;
  Pipeline pipeline(storage.barriers,params);
  typename T::Mma mma;
  auto thr_mma = mma.get_thread_slice(tid%T::ConsumerThreads);
  auto coords = thr_mma.partition_C(make_identity_tensor(make_shape(Int<T::BM>{},Int<T::BN>{})));
  // Do not construct an owning CuTe fragment before quota changes: its tuple
  // value-init may zero ArrayEngine despite an empty Tensor() ctor. decltype
  // obtains exactly the original fragment layout without constructing storage.
  // Plain float arrays WITHOUT initializer hold undefined values until the
  // owning consumer clears them after reg_alloc. Views initialize pointers only.
  // Actual compiler must prove scalarization and no local/spill/early P access.
  using QLayout = typename decltype(thr_mma.make_fragment_C(coords))::layout_type;
  float q0_data[cosize_v<QLayout>];
  float q1_data[cosize_v<QLayout>];
  float q2_data[cosize_v<QLayout>];
  float q3_data[cosize_v<QLayout>];
  auto q0 = make_tensor(make_rmem_ptr(q0_data),QLayout{});
  auto q1 = make_tensor(make_rmem_ptr(q1_data),QLayout{});
  auto q2 = make_tensor(make_rmem_ptr(q2_data),QLayout{});
  auto q3 = make_tensor(make_rmem_ptr(q3_data),QLayout{});
  static_assert(size(q0) == 16,"each assigned full P has 16 floats/thread");
  static_assert(7*T::BM*T::BN*sizeof(float) <= T::Stages*T::BankElements*sizeof(Element),
      "all seven P fit inside both drained operand banks without extra smem");
  // Initial PipelineAsync construction and full/empty init are collective384.
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
        psplit_fill_a(p,A,row_base,k_base,task,sA);
      CUTE_UNROLL
      for (int task = tid; task < T::BN*T::BK/8; task += T::ProducerThreads)
        psplit_fill_b<Element,Cols>(p,B,col_base,k_base,task,sB);
      // Each writer fences its own generic shared stores into WGMMA's async
      // proxy before its full-barrier arrival. The128th arrival publishes all.
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      pipeline.producer_commit(state);
      ++state;
    }
    // Wait for BOTH consumer WGs before any final-bank tail reuse. This waits on
    // existing empty barriers and adds neither a CTA barrier nor a launch.
    pipeline.producer_tail(state);
  } else {
    // Conservative source hypothesis, not actual compiler resource evidence.
    // Final pool need=128*(64+96+96)=32768; legal uniform initial quota88..96,
    // plus actual role-register bounds and no spills must be proved first.
    cutlass::arch::warpgroup_reg_alloc<96>();
    if (tid < 256) {
      clear(q0); clear(q1); clear(q2);
      typename Pipeline::PipelineState state;
      CUTE_NO_UNROLL
      for (int64_t k_base = 0; k_base < p.k/2; k_base += T::BK) {
        pipeline.consumer_wait(state);
        Element* bank = storage.operands+state.index()*T::BankElements;
        auto sA = make_tensor(make_smem_ptr(bank),typename T::LayoutA{});
        auto sB = make_tensor(make_smem_ptr(bank+T::ElementsA),typename T::LayoutB{});
        auto tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA));
        auto tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB));
        warpgroup_fence_operand(q0);
        warpgroup_fence_operand(q1);
        warpgroup_fence_operand(q2);
        warpgroup_arrive();
        gemm(mma,tCrA(_,_,_,Int<0>{}),tCrB(_,_,_,Int<0>{}),q0); // P1
        gemm(mma,tCrA(_,_,_,Int<1>{}),tCrB(_,_,_,Int<1>{}),q1); // P2
        gemm(mma,tCrA(_,_,_,Int<2>{}),tCrB(_,_,_,Int<2>{}),q2); // P3
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(q0);
        warpgroup_fence_operand(q1);
        warpgroup_fence_operand(q2);
        pipeline.consumer_release(state);
        ++state;
      }
    } else {
      clear(q0); clear(q1); clear(q2);
      clear(q3);
      typename Pipeline::PipelineState state;
      CUTE_NO_UNROLL
      for (int64_t k_base = 0; k_base < p.k/2; k_base += T::BK) {
        pipeline.consumer_wait(state);
        Element* bank = storage.operands+state.index()*T::BankElements;
        auto sA = make_tensor(make_smem_ptr(bank),typename T::LayoutA{});
        auto sB = make_tensor(make_smem_ptr(bank+T::ElementsA),typename T::LayoutB{});
        auto tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA));
        auto tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB));
        warpgroup_fence_operand(q0);
        warpgroup_fence_operand(q1);
        warpgroup_fence_operand(q2);
        warpgroup_fence_operand(q3);
        warpgroup_arrive();
        gemm(mma,tCrA(_,_,_,Int<3>{}),tCrB(_,_,_,Int<3>{}),q0); // P4
        gemm(mma,tCrA(_,_,_,Int<4>{}),tCrB(_,_,_,Int<4>{}),q1); // P5
        gemm(mma,tCrA(_,_,_,Int<5>{}),tCrB(_,_,_,Int<5>{}),q2); // P6
        gemm(mma,tCrA(_,_,_,Int<6>{}),tCrB(_,_,_,Int<6>{}),q3); // P7
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(q0);
        warpgroup_fence_operand(q1);
        warpgroup_fence_operand(q2);
        warpgroup_fence_operand(q3);
        pipeline.consumer_release(state);
        ++state;
      }
    }
  }
  // One lexical384-thread barrier waits for producer_tail and BOTH WG wait0/
  // releases, including both last banks. No early role return is permitted.
  __syncthreads();
  float* tail = storage.tail_products;
  if (tid >= 128) {
    const int first_product = tid < 256 ? 0 : 3;
    CUTE_UNROLL
    for (int i = 0; i < size(q0); ++i) {
      const auto c = coords(i);
      const int row = get<0>(c),col = get<1>(c);
      tail[psplit_tail_word(first_product+0,row,col)] = q0(i);
      tail[psplit_tail_word(first_product+1,row,col)] = q1(i);
      tail[psplit_tail_word(first_product+2,row,col)] = q2(i);
      if (tid >= 256) tail[psplit_tail_word(6,row,col)] = q3(i);
    }
  }
  // Complete all14336 FP32 P words before any generic U load across WGs.
  __syncthreads();
  for (int linear = tid; linear < T::BM*T::BN; linear += T::Threads) {
    const int local_row = linear/T::BN,local_col = linear%T::BN;
    const int64_t row = row_base+local_row,col = col_base+local_col;
    const int64_t mh = p.m/2,nh = p.n/2;
    // Each unique U owner loads all seven products once; no partial P sum,
    // atomics, changed contribution ordering, or output scratch is introduced.
    const float p1 = tail[psplit_tail_word(0,local_row,local_col)];
    const float p2 = tail[psplit_tail_word(1,local_row,local_col)];
    const float p3 = tail[psplit_tail_word(2,local_row,local_col)];
    const float p4 = tail[psplit_tail_word(3,local_row,local_col)];
    const float p5 = tail[psplit_tail_word(4,local_row,local_col)];
    const float p6 = tail[psplit_tail_word(5,local_row,local_col)];
    const float p7 = tail[psplit_tail_word(6,local_row,local_col)];
    const float u1 = __fadd_rn(p1,p2);
    const float u2 = __fadd_rn(p1,p6);
    const float u3 = __fadd_rn(u2,p7);
    const float u4 = __fadd_rn(u2,p5);
    const float u5 = __fadd_rn(u4,p3);
    const float u6 = __fsub_rn(u3,p4);
    const float u7 = __fadd_rn(u3,p5);
    float d11 = __fmul_rn(p.alpha,u1),d12 = __fmul_rn(p.alpha,u5);
    float d21 = __fmul_rn(p.alpha,u6),d22 = __fmul_rn(p.alpha,u7);
    if (p.beta != 0.0f) {
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

template <class Element>
void prepare_wgmma_psplit_typed() {
  constexpr size_t shared = PSplitTraits<Element,32>::SharedBytes;
  static_assert(shared > 48*1024 && shared < 227*1024,"H100 two-bank shared budget");
  const auto e = cudaFuncSetAttribute(winograd_wgmma_psplit<Element,32>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared));
  if (e != cudaSuccess) throw std::runtime_error(std::string("Psplit shared opt-in: ")+cudaGetErrorString(e));
}

template <class Element>
void launch_wgmma_psplit_typed(const Problem& p,const void* A,const void* B,
    const float* C0,float* D,cudaStream_t stream) {
  using T = PSplitTraits<Element,32>;
  const dim3 grid(unsigned(p.m/(2*T::BM)),unsigned(p.n/(2*T::BN)));
  winograd_wgmma_psplit<Element,32><<<grid,T::Threads,T::SharedBytes,stream>>>(
      p,static_cast<const Element*>(A),static_cast<const Element*>(B),C0,D);
}
} // namespace

bool wgmma_psplit_can_implement(const Problem& p,int configuration) {
  return configuration == 0 && megakernel_can_implement(p,0) &&
      p.m%128 == 0 && p.n%64 == 0 && p.k%64 == 0 && p.lda%8 == 0 && p.ldb%8 == 0;
}
bool wgmma_psplit_full_tiles(const Problem& p,const void* A,const void* B,int configuration) {
  return wgmma_psplit_can_implement(p,configuration) && A && B &&
      (reinterpret_cast<uintptr_t>(A)&15) == 0 && (reinterpret_cast<uintptr_t>(B)&15) == 0;
}
void prepare_wgmma_psplit(const Problem& p,int configuration) {
  if (!wgmma_psplit_can_implement(p,configuration))
    throw std::invalid_argument("unsupported Psplit metadata/configuration");
  if (p.dtype == DType::BF16) prepare_wgmma_psplit_typed<cutlass::bfloat16_t>();
  else prepare_wgmma_psplit_typed<cutlass::half_t>();
}
void launch_wgmma_psplit(const Problem& p,const void* A,const void* B,
    const float* C0,float* D,cudaStream_t stream,int configuration) {
  if (!wgmma_psplit_can_implement(p,configuration) || !wgmma_psplit_full_tiles(p,A,B,configuration))
    throw std::invalid_argument("Psplit requires native full tiles and current AB16B");
  if (!D || (p.beta != 0.0f && !C0)) throw std::invalid_argument("Psplit active output/source is null");
  // Native-only helper. Integrating Plan retains common current span/end,
  // overlap, scalar D/C0 alignment, exact C0==D, certification/range, degenerate,
  // invalid policy before empty/scale, truthful packed7/Dense fallback and Auto.
  // prepare is outside timing. Every run/capture reads current original AB;
  // replay fixed captured addresses still recomputes fresh ST on device.
  if (p.dtype == DType::BF16) launch_wgmma_psplit_typed<cutlass::bfloat16_t>(p,A,B,C0,D,stream);
  else launch_wgmma_psplit_typed<cutlass::half_t>(p,A,B,C0,D,stream);
}
} // namespace megafmm

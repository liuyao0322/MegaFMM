// Native51 BN32 / Native52 BN16 inplace SS TMA, BM64/BK32, two banks.
// Eight raw quadrant TMA boxes alias first four final planes. Each consumer pair
// loads all four words before overwriting; final RN ST cast once per bank stage.
// Both current A/B maps encoded per ordinary call; fresh four-instance CAN required.
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
// Keep CUDA12.4 constexpr work bounded; full geometry is independently checked
// in the frozen CPU prototype. This verifies four aligned edge vectors per slot.
constexpr bool psplit_tma_inplace_ss_contiguous_eight_sentinels() {
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
struct PSplitTmaInplaceSSTraits {
  static_assert(Cols == 16 || Cols == 32,"typed BN16/BN32 only");
  static_assert(std::is_same_v<Element,cutlass::half_t> ||
      std::is_same_v<Element,cutlass::bfloat16_t>,"FP16/BF16 operands only");
  static constexpr int BM = 64;
  static constexpr int BN = Cols;
  static constexpr int BK = 32;
  static constexpr int Threads = 384;
  static constexpr int ProducerThreads = 128;
  static constexpr int ConsumerThreads = 128; // each independent WG
  static constexpr int AllConsumerThreads = 256;
  static constexpr int Stages = 2;
  using Tile = Shape<Int<BM>,Int<BN>,Int<BK>>;
  // Shared layout uses SW64 A, SW64 BN32 B or SW32 BN16 B. Each bank has its
  // own rank-3 tensor, retaining canonical GMMA descriptors for all seven P.
  using LayoutA = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},
      make_shape(Int<BM>{},Int<BK>{},Int<7>{})));
  using LayoutB = std::conditional_t<Cols == 32,
      decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Element>{},make_shape(Int<BN>{},Int<BK>{},Int<7>{}))),
      decltype(tile_to_shape(GMMA::Layout_MN_SW32_Atom<Element>{},make_shape(Int<BN>{},Int<BK>{},Int<7>{})))>;
  using Op = decltype(GMMA::ss_op_selector<Element,Element,float,Tile,
      GMMA::Major::K,GMMA::Major::MN>());
  using Mma = decltype(make_tiled_mma(Op{}));
  using Pipeline = cutlass::PipelineAsync<Stages>;
  static constexpr int ElementsA = cosize_v<LayoutA>;
  static constexpr int ElementsB = cosize_v<LayoutB>;
  static constexpr int BankElements = ElementsA + ElementsB;
  struct alignas(1024) SharedStorage {
    // Float tail is an explicitly typed alternate member, activated by tail
    // stores only after both operand banks are uniformly retired. Barriers do
    // not overlap either member. Payload sizeof stays at the larger operands.
    union {
      Element operands[Stages * BankElements];
      float tail_products[7*BM*BN];
    };
    typename Pipeline::SharedStorage barriers;
    alignas(8) uint64_t raw_done[Stages];
  };
  static constexpr size_t SharedBytes = sizeof(SharedStorage);
  static constexpr uint32_t RawBytes = 4*(BM+BN)*BK*sizeof(Element);
  static_assert(RawBytes == (Cols==32 ? 24576 : 20480),"all eight raw boxes contribute TX bytes");
  static_assert(BM*BK*sizeof(Element)%512==0 && BN*BK*sizeof(Element)%512==0 &&
      ElementsA*sizeof(Element)%512==0 && BankElements*sizeof(Element)%512==0,
      "all absolute SW64/SW32 boxes have zero swizzle pattern offsets");
  static_assert(offsetof(SharedStorage,raw_done)>=offsetof(SharedStorage,barriers)+sizeof(typename Pipeline::SharedStorage),
      "raw barriers never overlap bank operands or pipeline barriers");
  static_assert(sizeof(Element) == 2,"vector helpers require 16-bit operands");
  static_assert(size(Mma{}) == ConsumerThreads,"one WG owns all coordinates of each assigned P");
  static_assert(ElementsA == 7*BM*BK && ElementsB == 7*BN*BK,
      "compact product slots must have no padding or overlap");
  static_assert(BankElements * sizeof(Element) % 128 == 0,
      "each final bank and the barrier storage must retain 128B alignment");
  static_assert(psplit_tma_inplace_ss_contiguous_eight_sentinels<LayoutA,1>(),
      "A edge vectors must be contiguous and 16B aligned in every product slot");
  static_assert(psplit_tma_inplace_ss_contiguous_eight_sentinels<LayoutB,0>(),
      "B edge vectors must be contiguous and 16B aligned in every product slot");
};

// Raw and final share the same flagged CuTe layout. Each pair task owns
// its coordinates in all seven planes. Four explicit loads are one asm statement;
// every mutating store is a later memory-clobbered statement.
CUTE_DEVICE uint4 psplit_tma_inplace_ss_load_four(const void* p0,const void* p1,
    const void* p2,const void* p3) {
  const uint32_t a0=uint32_t(__cvta_generic_to_shared(p0));
  const uint32_t a1=uint32_t(__cvta_generic_to_shared(p1));
  const uint32_t a2=uint32_t(__cvta_generic_to_shared(p2));
  const uint32_t a3=uint32_t(__cvta_generic_to_shared(p3));
  uint4 r;
  asm volatile("ld.shared.b32 %0,[%4]; ld.shared.b32 %1,[%5]; ld.shared.b32 %2,[%6]; ld.shared.b32 %3,[%7];"
      : "=r"(r.x),"=r"(r.y),"=r"(r.z),"=r"(r.w)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3) : "memory");
  return r;
}
CUTE_DEVICE void psplit_tma_inplace_ss_store_pair(void* p,uint32_t v) {
  const uint32_t a=uint32_t(__cvta_generic_to_shared(p));
  asm volatile("st.shared.b32 [%0],%1;" :: "r"(a),"r"(v) : "memory");
}
template<class Element>
CUTE_DEVICE float psplit_tma_inplace_ss_unpack_word(uint32_t x,int j) {
  return float(Element::bitcast(uint16_t(x>>(16*j))));
}
template<class Element>
CUTE_DEVICE uint32_t psplit_tma_inplace_ss_round_pair(float x,float y) {
  cutlass::Array<float,2> f; f[0]=x; f[1]=y;
  using C=cutlass::NumericArrayConverter<Element,float,2,cutlass::FloatRoundStyle::round_to_nearest>;
  const auto r=C{}(f);
  return reinterpret_cast<const uint32_t&>(r);
}
template<class Element,class SmemA>
CUTE_DEVICE void psplit_tma_inplace_ss_transform_a(int task,SmemA a) {
  const int row=task/16,k=(task%16)*2;
  const auto raw=psplit_tma_inplace_ss_load_four(&a(row,k,0),&a(row,k,1),&a(row,k,2),&a(row,k,3));
  float s1[2],s2[2],s3[2],s4[2];
  CUTE_UNROLL
  for(int j=0;j<2;++j) {
    const float a11=psplit_tma_inplace_ss_unpack_word<Element>(raw.x,j);
    const float a12=psplit_tma_inplace_ss_unpack_word<Element>(raw.y,j);
    const float a21=psplit_tma_inplace_ss_unpack_word<Element>(raw.z,j);
    const float a22=psplit_tma_inplace_ss_unpack_word<Element>(raw.w,j);
    s1[j]=__fadd_rn(a21,a22);
    s2[j]=__fsub_rn(s1[j],a11);
    s3[j]=__fsub_rn(a11,a21);
    s4[j]=__fsub_rn(a12,s2[j]);
  }
  // Raw A11/A12/A22 already reside in their unchanged final slots0/1/3.
  psplit_tma_inplace_ss_store_pair(&a(row,k,4),psplit_tma_inplace_ss_round_pair<Element>(s1[0],s1[1]));
  psplit_tma_inplace_ss_store_pair(&a(row,k,5),psplit_tma_inplace_ss_round_pair<Element>(s2[0],s2[1]));
  psplit_tma_inplace_ss_store_pair(&a(row,k,6),psplit_tma_inplace_ss_round_pair<Element>(s3[0],s3[1]));
  psplit_tma_inplace_ss_store_pair(&a(row,k,2),psplit_tma_inplace_ss_round_pair<Element>(s4[0],s4[1]));
}
template<class Element,int Cols,class SmemB>
CUTE_DEVICE void psplit_tma_inplace_ss_transform_b(int task,SmemB b) {
  const int k=task/(Cols/2),col=(task%(Cols/2))*2;
  const auto raw=psplit_tma_inplace_ss_load_four(&b(col,k,0),&b(col,k,1),&b(col,k,2),&b(col,k,3));
  // All raw loads precede the two bitcopy moves that overwrite raw B12/B21.
  psplit_tma_inplace_ss_store_pair(&b(col,k,1),raw.z);
  psplit_tma_inplace_ss_store_pair(&b(col,k,2),raw.w);
  float t1[2],t2[2],t3[2],t4[2];
  CUTE_UNROLL
  for(int j=0;j<2;++j) {
    const float b11=psplit_tma_inplace_ss_unpack_word<Element>(raw.x,j);
    const float b12=psplit_tma_inplace_ss_unpack_word<Element>(raw.y,j);
    const float b21=psplit_tma_inplace_ss_unpack_word<Element>(raw.z,j);
    const float b22=psplit_tma_inplace_ss_unpack_word<Element>(raw.w,j);
    t1[j]=__fsub_rn(b12,b11);
    t2[j]=__fsub_rn(b22,t1[j]);
    t3[j]=__fsub_rn(b22,b12);
    t4[j]=__fsub_rn(t2[j],b21);
  }
  psplit_tma_inplace_ss_store_pair(&b(col,k,4),psplit_tma_inplace_ss_round_pair<Element>(t1[0],t1[1]));
  psplit_tma_inplace_ss_store_pair(&b(col,k,5),psplit_tma_inplace_ss_round_pair<Element>(t2[0],t2[1]));
  psplit_tma_inplace_ss_store_pair(&b(col,k,6),psplit_tma_inplace_ss_round_pair<Element>(t3[0],t3[1]));
  psplit_tma_inplace_ss_store_pair(&b(col,k,3),psplit_tma_inplace_ss_round_pair<Element>(t4[0],t4[1]));
}
template<class Element,int Cols,class SmemA,class SmemB>
CUTE_DEVICE void psplit_tma_inplace_ss_transform_stage(SmemA a,SmemB b) {
  unsigned fresh_tid;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(fresh_tid) :: "memory");
  const int consumer=int(fresh_tid)-128;
  CUTE_UNROLL
  for(int task=consumer;task<64*32/2;task+=256)
    psplit_tma_inplace_ss_transform_a<Element>(task,a);
  CUTE_UNROLL
  for(int task=consumer;task<Cols*32/2;task+=256)
    psplit_tma_inplace_ss_transform_b<Element,Cols>(task,b);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  // All256 own writers must retire generic stores before either SS WG reads.
  asm volatile("barrier.sync 1,256;" ::: "memory");
}
struct alignas(64) TmaInplaceSSDescriptors { CUtensorMap a; CUtensorMap b; };

// Tail FP32 rendezvous aliases the complete drained TWO-bank operands payload.
// XOR permutation is a bijection within each row, does not add padding, and is
// separate from the WGMMA layouts used only before the first tail CTA barrier.
template <int Rows,int Cols>
CUTE_DEVICE int psplit_tma_inplace_ss_tail_word(int product,int row,int col) {
  static_assert(Cols >= 4 && (Cols&(Cols-1)) == 0,"power-of-two tail columns");
  const int xor_mask = ((row&2)>>1) | ((row&4)<<1);
  return product*Rows*Cols+row*Cols+(col^xor_mask);
}

template <class Element,int Cols>
__global__ __launch_bounds__(384,2)
void winograd_wgmma_psplit_tma_inplace_ss(Problem p,
    const __grid_constant__ TmaInplaceSSDescriptors maps,const float* C0,float* D) {
  using T = PSplitTmaInplaceSSTraits<Element,Cols>;
  using Pipeline = typename T::Pipeline;
  const int tid = int(threadIdx.x);
  const int64_t row_base = int64_t(blockIdx.x)*T::BM;
  const int64_t col_base = int64_t(blockIdx.y)*T::BN;
  extern __shared__ __align__(1024) unsigned char raw_storage[];
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
  if(tid==0) {
    CUTE_UNROLL
    for(int stage=0;stage<T::Stages;++stage) cutlass::arch::initialize_barrier(storage.raw_done[stage],1);
    cutlass::arch::fence_barrier_init();
  }
  // Initial PipelineAsync construction and full/empty init are collective384.
  __syncthreads();
  if (tid < T::ProducerThreads) {
    cutlass::arch::warpgroup_reg_dealloc<48>();
    auto state = cutlass::make_producer_start_state<Pipeline>();
    CUTE_NO_UNROLL
    for (int64_t k_base = 0; k_base < p.k/2; k_base += T::BK) {
      pipeline.producer_acquire(state);
      Element* bank = storage.operands + state.index()*T::BankElements;
      auto sA = make_tensor(make_smem_ptr(bank),typename T::LayoutA{});
      auto sB = make_tensor(make_smem_ptr(bank+T::ElementsA),typename T::LayoutB{});
      auto& raw_done=storage.raw_done[state.index()];
      if(cute::elect_one_sync() && tid<32) {
        cutlass::arch::set_barrier_transaction_bytes(raw_done,T::RawBytes);
        constexpr uint64_t hint=uint64_t(TMA::CacheHintSm90::EVICT_NORMAL);
        const int k=int(k_base),r=int(row_base),c=int(col_base);
        const int mh=int(p.m/2),nh=int(p.n/2),kh=int(p.k/2);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,0),k,r);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,1),k+kh,r);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,2),k,r+mh);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,3),k+kh,r+mh);
        SM90_TMA_LOAD_2D::copy(&maps.b,raw_done,hint,&sB(0,0,0),c,k);
        SM90_TMA_LOAD_2D::copy(&maps.b,raw_done,hint,&sB(0,0,1),c+nh,k);
        SM90_TMA_LOAD_2D::copy(&maps.b,raw_done,hint,&sB(0,0,2),c,k+kh);
        SM90_TMA_LOAD_2D::copy(&maps.b,raw_done,hint,&sB(0,0,3),c+nh,k+kh);
      }
      cutlass::arch::wait_barrier(raw_done,state.phase()^1);
      asm volatile("" ::: "memory");
      // Full publishes RAW operands. Consumerbar1 publishes final ST later.
      pipeline.producer_commit(state);
      ++state;
    }
    // Wait for BOTH consumer WGs before any final-bank tail reuse. This waits on
    // existing empty barriers and adds neither a CTA barrier nor a launch.
    pipeline.producer_tail(state);
    // Nonaligned CTA barrier permits distinct role-local call sites.
    asm volatile("barrier.sync 6,384;" ::: "memory");
  } else {
    // Do not construct an owning CuTe fragment before quota changes: its tuple
    // value-init may zero ArrayEngine despite an empty Tensor() ctor. decltype
    // obtains exactly the original fragment layout without constructing storage.
    // Plain float arrays WITHOUT initializer hold undefined values until the
    // owning consumer clears them after reg_alloc. Views initialize pointers only.
    // Actual compiler must prove scalarization and no local/spill/early P access.
    using QLayout = typename decltype(thr_mma.make_fragment_C(thr_mma.partition_C(
        make_identity_tensor(make_shape(Int<T::BM>{},Int<T::BN>{})))) )::layout_type;
    float q0_data[cosize_v<QLayout>];
    float q1_data[cosize_v<QLayout>];
    float q2_data[cosize_v<QLayout>];
    float q3_data[cosize_v<QLayout>];
    auto q0 = make_tensor(make_rmem_ptr(q0_data),QLayout{});
    auto q1 = make_tensor(make_rmem_ptr(q1_data),QLayout{});
    auto q2 = make_tensor(make_rmem_ptr(q2_data),QLayout{});
    auto q3 = make_tensor(make_rmem_ptr(q3_data),QLayout{});
    static_assert(size(q0) == T::BM*T::BN/T::ConsumerThreads,
        "each assigned full P has BM*BN/128 floats/thread (8 at BN16)");
    static_assert(7*T::BM*T::BN*sizeof(float) <= T::Stages*T::BankElements*sizeof(Element),
        "all seven P fit inside both drained operand banks without extra smem");
    // Source quotas are hypotheses until all four actual machine instances pass.
    constexpr int WG0Quota=Cols==32 ? 88 : 64;
    constexpr int WG1Quota=Cols==32 ? 104 : 72;
    if (tid < 256) {
      cutlass::arch::warpgroup_reg_alloc<WG0Quota>();
      clear(q0); clear(q1); clear(q2);
      typename Pipeline::PipelineState state;
      CUTE_NO_UNROLL
      for (int64_t k_base = 0; k_base < p.k/2; k_base += T::BK) {
        pipeline.consumer_wait(state);
        Element* bank = storage.operands+state.index()*T::BankElements;
        auto sA = make_tensor(make_smem_ptr(bank),typename T::LayoutA{});
        auto sB = make_tensor(make_smem_ptr(bank+T::ElementsA),typename T::LayoutB{});
        psplit_tma_inplace_ss_transform_stage<Element,Cols>(sA,sB);
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
      cutlass::arch::warpgroup_reg_alloc<WG1Quota>();
      clear(q0); clear(q1); clear(q2);
      clear(q3);
      typename Pipeline::PipelineState state;
      CUTE_NO_UNROLL
      for (int64_t k_base = 0; k_base < p.k/2; k_base += T::BK) {
        pipeline.consumer_wait(state);
        Element* bank = storage.operands+state.index()*T::BankElements;
        auto sA = make_tensor(make_smem_ptr(bank),typename T::LayoutA{});
        auto sB = make_tensor(make_smem_ptr(bank+T::ElementsA),typename T::LayoutB{});
        psplit_tma_inplace_ss_transform_stage<Element,Cols>(sA,sB);
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
    // All384 threads rendezvous after producer_tail and both consumer drains.
    asm volatile("barrier.sync 6,384;" ::: "memory");
    // Reconstruct fragment coordinates only after the complete operand drain.
    // A fresh volatile thread-id prevents carrying tail-only coordinate registers
    // through the asynchronous whole-K loops; address/layout math is unchanged.
    unsigned tail_tid;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tail_tid) :: "memory");
    auto tail_thr_mma = mma.get_thread_slice(int(tail_tid)%T::ConsumerThreads);
    auto coords = tail_thr_mma.partition_C(make_identity_tensor(
        make_shape(Int<T::BM>{},Int<T::BN>{})));
    float* tail = storage.tail_products;
    if (tid >= 128) {
      const int first_product = tid < 256 ? 0 : 3;
      CUTE_UNROLL
      for (int i = 0; i < size(q0); ++i) {
        const auto c = coords(i);
        const int row = get<0>(c),col = get<1>(c);
        tail[psplit_tma_inplace_ss_tail_word<T::BM,T::BN>(first_product+0,row,col)] = q0(i);
        tail[psplit_tma_inplace_ss_tail_word<T::BM,T::BN>(first_product+1,row,col)] = q1(i);
        tail[psplit_tma_inplace_ss_tail_word<T::BM,T::BN>(first_product+2,row,col)] = q2(i);
        if (tid >= 256) tail[psplit_tma_inplace_ss_tail_word<T::BM,T::BN>(6,row,col)] = q3(i);
      }
    }
  }
  // Complete all7*BM*BN FP32 P words before any generic U load across WGs.
  __syncthreads();
  float* tail = storage.tail_products;
  for (int linear = tid; linear < T::BM*T::BN; linear += T::Threads) {
    const int local_row = linear/T::BN,local_col = linear%T::BN;
    const int64_t row = row_base+local_row,col = col_base+local_col;
    const int64_t mh = p.m/2,nh = p.n/2;
    // Each unique U owner loads all seven products once; no partial P sum,
    // atomics, changed contribution ordering, or output scratch is introduced.
    const float p1 = tail[psplit_tma_inplace_ss_tail_word<T::BM,T::BN>(0,local_row,local_col)];
    const float p2 = tail[psplit_tma_inplace_ss_tail_word<T::BM,T::BN>(1,local_row,local_col)];
    const float p3 = tail[psplit_tma_inplace_ss_tail_word<T::BM,T::BN>(2,local_row,local_col)];
    const float p4 = tail[psplit_tma_inplace_ss_tail_word<T::BM,T::BN>(3,local_row,local_col)];
    const float p5 = tail[psplit_tma_inplace_ss_tail_word<T::BM,T::BN>(4,local_row,local_col)];
    const float p6 = tail[psplit_tma_inplace_ss_tail_word<T::BM,T::BN>(5,local_row,local_col)];
    const float p7 = tail[psplit_tma_inplace_ss_tail_word<T::BM,T::BN>(6,local_row,local_col)];
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

template<class Element,int Cols>
void prepare_wgmma_psplit_tma_inplace_ss_typed() {
  constexpr size_t bytes=PSplitTmaInplaceSSTraits<Element,Cols>::SharedBytes;
  static_assert(bytes>48*1024 && bytes<227*1024,"H100 two-bank shared budget");
  const auto e=cudaFuncSetAttribute(winograd_wgmma_psplit_tma_inplace_ss<Element,Cols>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes));
  if(e!=cudaSuccess) throw std::runtime_error(std::string("Psplit shared opt-in: ")+cudaGetErrorString(e));
}
template<int Cols>
TmaInplaceSSDescriptors make_tma_inplace_ss_descriptors(const Problem& p,const void* A,const void* B) {
  TmaInplaceSSDescriptors maps{};
  const auto type=p.dtype==DType::BF16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  const cuuint64_t adims[2]={cuuint64_t(p.k),cuuint64_t(p.m)},bdims[2]={cuuint64_t(p.n),cuuint64_t(p.k)};
  const cuuint64_t astride[1]={cuuint64_t(p.lda)*2},bstride[1]={cuuint64_t(p.ldb)*2};
  const cuuint32_t abox[2]={32,64},bbox[2]={Cols,32},step[2]={1,1};
  auto e=cuTensorMapEncodeTiled(&maps.a,type,2,const_cast<void*>(A),adims,astride,abox,step,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_64B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(e!=CUDA_SUCCESS) throw std::runtime_error("Inplace SS A tensor-map encode error="+std::to_string(int(e)));
  e=cuTensorMapEncodeTiled(&maps.b,type,2,const_cast<void*>(B),bdims,bstride,bbox,step,
      CU_TENSOR_MAP_INTERLEAVE_NONE,Cols==32 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_32B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(e!=CUDA_SUCCESS) throw std::runtime_error("Inplace SS B tensor-map encode error="+std::to_string(int(e)));
  return maps;
}
template<class Element,int Cols>
void launch_wgmma_psplit_tma_inplace_ss_typed(const Problem& p,const void* A,const void* B,
    const float* C0,float* D,cudaStream_t stream) {
  using T=PSplitTmaInplaceSSTraits<Element,Cols>;
  // Both current-input encodes are inside every ordinary call/capture.
  const auto maps=make_tma_inplace_ss_descriptors<Cols>(p,A,B);
  const dim3 grid(unsigned(p.m/(2*T::BM)),unsigned(p.n/(2*T::BN)));
  winograd_wgmma_psplit_tma_inplace_ss<Element,Cols><<<grid,T::Threads,T::SharedBytes,stream>>>(p,maps,C0,D);
}
} // namespace
bool wgmma_psplit_tma_inplace_ss_can_implement(const Problem& p,int columns,int configuration) {
  return configuration==0 && (columns==32 || columns==16) && megakernel_can_implement(p,0) &&
      p.m%128==0 && p.n%(2*columns)==0 && p.k%64==0 && p.lda%8==0 && p.ldb%8==0 &&
      uint64_t(p.lda)<(uint64_t(1)<<39) && uint64_t(p.ldb)<(uint64_t(1)<<39);
}
bool wgmma_psplit_tma_inplace_ss_full_tiles(const Problem& p,const void* A,const void* B,int columns,int configuration) {
  return wgmma_psplit_tma_inplace_ss_can_implement(p,columns,configuration) && A && B &&
      (reinterpret_cast<uintptr_t>(A)&15)==0 && (reinterpret_cast<uintptr_t>(B)&15)==0;
}
void prepare_wgmma_psplit_tma_inplace_ss(const Problem& p,int columns,int configuration) {
  if(!wgmma_psplit_tma_inplace_ss_can_implement(p,columns,configuration)) throw std::invalid_argument("unsupported inplace SS metadata/configuration");
  if(columns==32) {
    if(p.dtype==DType::BF16) prepare_wgmma_psplit_tma_inplace_ss_typed<cutlass::bfloat16_t,32>();
    else prepare_wgmma_psplit_tma_inplace_ss_typed<cutlass::half_t,32>();
  } else {
    if(p.dtype==DType::BF16) prepare_wgmma_psplit_tma_inplace_ss_typed<cutlass::bfloat16_t,16>();
    else prepare_wgmma_psplit_tma_inplace_ss_typed<cutlass::half_t,16>();
  }
}
void launch_wgmma_psplit_tma_inplace_ss(const Problem& p,const void* A,const void* B,const float* C0,float* D,
    cudaStream_t stream,int columns,int configuration) {
  if(!wgmma_psplit_tma_inplace_ss_full_tiles(p,A,B,columns,configuration)) throw std::invalid_argument("inplace SS requires native full tiles/currentAB16B");
  if(!D || (p.beta!=0.0f && !C0)) throw std::invalid_argument("inplace SS active output/source null");
  if(columns==32) {
    if(p.dtype==DType::BF16) launch_wgmma_psplit_tma_inplace_ss_typed<cutlass::bfloat16_t,32>(p,A,B,C0,D,stream);
    else launch_wgmma_psplit_tma_inplace_ss_typed<cutlass::half_t,32>(p,A,B,C0,D,stream);
  } else {
    if(p.dtype==DType::BF16) launch_wgmma_psplit_tma_inplace_ss_typed<cutlass::bfloat16_t,16>(p,A,B,C0,D,stream);
    else launch_wgmma_psplit_tma_inplace_ss_typed<cutlass::half_t,16>(p,A,B,C0,D,stream);
  }
}
} // namespace megafmm

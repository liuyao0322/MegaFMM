#include "megafmm/kernels.h"
#include <cute/tensor.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>
#include <type_traits>
#include <cstddef>
#include <iostream>
#include <set>
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
  using BAtom = std::conditional_t<Cols == 32,
      GMMA::Layout_MN_SW64_Atom<Element>,GMMA::Layout_MN_SW32_Atom<Element>>;
  using LayoutB = decltype(tile_to_shape(BAtom{},make_shape(Int<BN>{},Int<BK>{},Int<7>{})));
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

template<class Element,int Cols> void check(const char* dtype) {
 using T=PSplitTmaInplaceSSTraits<Element,Cols>;std::set<int> seenA,seenB;
 auto la=typename T::LayoutA{};auto lb=typename T::LayoutB{};
 for(int slot=0;slot<7;++slot) {
  for(int r=0;r<64;++r)for(int k=0;k<32;++k) {
   int a=la.layout_a()(2*la.layout_b()(make_coord(r,k,slot)))/2;int plain=slot*64*32+r*32+k;
   int bytes=2*plain;int hw=(bytes^(((bytes>>7)&3)<<4))/2;
   if(a!=hw || !seenA.insert(a).second)throw std::runtime_error("A TMA/GMMA byte map");
   if((k%2)==0 && la.layout_a()(2*la.layout_b()(make_coord(r,k+1,slot)))/2!=a+1)throw std::runtime_error("A packed pair");
  }
  for(int k=0;k<32;++k)for(int c=0;c<Cols;++c) {
   int b=lb.layout_a()(2*lb.layout_b()(make_coord(c,k,slot)))/2;int plain=slot*Cols*32+k*Cols+c;
   int bytes=2*plain;int hw=(bytes^(((bytes>>7)&(Cols==32?3:1))<<4))/2;
   if(b!=hw || !seenB.insert(b).second)throw std::runtime_error("B TMA/GMMA byte map");
   if((c%2)==0 && lb.layout_a()(2*lb.layout_b()(make_coord(c+1,k,slot)))/2!=b+1)throw std::runtime_error("B packed pair");
  }
 }
 for(int bank=0;bank<2;++bank) {
  int base=bank*T::BankElements*2;
  if(base%512 || (base+T::ElementsA*2)%512)throw std::runtime_error("absolute swizzle bases");
  std::set<int> pairA,pairB;
  for(int consumer=0;consumer<256;++consumer) {
   for(int task=consumer;task<64*32/2;task+=256) {
    int row=task/16,k=(task%16)*2;
    for(int slot=0;slot<7;++slot) {
     int word=la.layout_a()(base+la.layout_b()(make_coord(row,k,slot))*2)/4;
     if(!pairA.insert(word).second)throw std::runtime_error("A duplicate owner");
    }
   }
   for(int task=consumer;task<Cols*32/2;task+=256) {
    int k=task/(Cols/2),c=(task%(Cols/2))*2;
    for(int slot=0;slot<7;++slot) {
     int word=lb.layout_a()(base+T::ElementsA*2+lb.layout_b()(make_coord(c,k,slot))*2)/4;
     if(pairA.count(word) || !pairB.insert(word).second)throw std::runtime_error("B alias owner");
    }
   }
  }
  if(pairA.size()!=7*64*32/2 || pairB.size()!=7*Cols*32/2)throw std::runtime_error("pair coverage");
 }
 std::cout<<"LAYOUT "<<dtype<<" BN="<<Cols<<" BM="<<T::BM<<" BK="<<T::BK<<" ElementsA="<<T::ElementsA<<" ElementsB="<<T::ElementsB
 <<" BankBytes="<<T::BankElements*2<<" SharedBytes="<<T::SharedBytes<<" align="<<alignof(typename T::SharedStorage)
 <<" barrier_offset="<<offsetof(typename T::SharedStorage,barriers)<<" raw_done_offset="<<offsetof(typename T::SharedStorage,raw_done)
 <<" rawTX="<<T::RawBytes<<" Acoords="<<seenA.size()<<" Bcoords="<<seenB.size()<<" STATUS=PASS\n";
}
int main(){check<cutlass::half_t,32>("fp16");check<cutlass::bfloat16_t,32>("bf16");check<cutlass::half_t,16>("fp16");check<cutlass::bfloat16_t,16>("bf16");}

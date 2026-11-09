#pragma once
#include "internal.h"
#include <array>
#include <tuple>
#include <utility>

namespace megafmm::detail::sw_hopper {
// N is the A-row span; each consumer warp group computes 64 B columns.
// The same seven-product kernel handles both raw aligned and padded operands.
template<int N,int BK,int Stages,int Groups=1,int U3Components=0,
         bool Pending=false,int KTiles=0,int PackThreads=256,int PackCap=512,
         int Raster=0,bool Padded=false,bool SplitPack=!Padded,int PackVector=8>
struct KernelConfig {
  static_assert(N==8 || N==16 || N==32 || N==64 || N==96 || N==128 || N==160 || N==192,"unsupported WGMMA width");
  static_assert(BK==64 || BK==128 || BK==256,"unsupported K tile");
  static_assert(U3Components>=0 && U3Components<=N/2,"invalid shared U3 component count");
  static_assert(Stages>=2 && Stages<=8,"invalid pipeline depth");
  static_assert(Groups==1 || Groups==2,"one or two consumer warp groups");
  static_assert(PackThreads==128 || PackThreads==256 || PackThreads==512,"invalid pack block");
  static_assert(PackVector==2 || PackVector==8,"invalid pack vector width");
  static_assert(PackCap>=(SplitPack?2:1),"pack cap cannot cover the side grid");
  static_assert(!Pending || Stages>=3,"pending group needs at least three slots");
  static_assert(KTiles>=0 && KTiles<=4,"bounded K unrolling");
  static_assert(2*(64*Groups+N)*BK*Stages + 256 + Groups*U3Components*128*4<=232448,"shared memory exceeds Hopper opt-in budget");
  static constexpr int n=N,bk=BK,stages=Stages,groups=Groups,threads=128*(Groups+1);
  static constexpr bool guard_outputs=Padded || (N&(N-1))!=0;
  static constexpr int u3_components=U3Components;
  static constexpr bool shared_u3=U3Components>0,pending=Pending,padded=Padded;
  static constexpr bool split_pack=SplitPack;
  static constexpr int pack_vector=Padded?2:PackVector;
  static constexpr int k_tiles=KTiles,pack_threads=PackThreads,pack_cap=PackCap,raster=Raster;
  static constexpr int producer_regs=Groups==2?24:0;
  static constexpr int consumer_regs=Groups==2?(N<=64?144:240):0;
};
// A bounded registry: scheduling policies, never a matrix-size dispatch table.
using KernelConfigurations=std::tuple<
  KernelConfig<64,128,4,1,0,false,0,256,128,0,true>,
  KernelConfig<8,256,4,1,0,true,1,128,128>,
  KernelConfig<32,128,8,1,0,false,4,512,128,0,false,false,2>,
  KernelConfig<64,64,4,2,16,false,0,256,512,0,false,false>,
  KernelConfig<128,64,4,2,0,true,0,256,512,2>,
  KernelConfig<128,128,3,2,0,false,0,128,512,2>,
  KernelConfig<192,64,3,2,96,false,0,128,512,2>,
  KernelConfig<192,64,3,2,96,true,0,128,512,4>,
  KernelConfig<16,128,8,1,0,false,0,256,512,0,true>
>;
inline constexpr int configuration_count=std::tuple_size_v<KernelConfigurations>;
struct Configuration {
  int id,n,bk,stages,groups,threads,k_tiles,pack_threads,pack_cap,raster,u3_components,pack_vector;
  bool shared_u3,pending,padded,split_pack;
};
template<size_t I> constexpr Configuration configuration() {
  using C=std::tuple_element_t<I,KernelConfigurations>;
  return {int(I),C::n,C::bk,C::stages,C::groups,C::threads,C::k_tiles,
          C::pack_threads,C::pack_cap,C::raster,C::u3_components,C::pack_vector,C::shared_u3,C::pending,C::padded,C::split_pack};
}
template<size_t... I> constexpr auto registry(std::index_sequence<I...>) {
  return std::array<Configuration,sizeof...(I)>{configuration<I>()...};
}
inline constexpr auto configurations=registry(std::make_index_sequence<configuration_count>{});
bool eligible(const Problem&,int id);
size_t workspace_size(const Problem&,int id);
SWResources setup(int id);
void run(const Problem&,int id,const void*,const void*,float*,void*,cudaStream_t);
std::string name(int id);
}

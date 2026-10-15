// CPU-only CuTe address diagnostic. No device launch and no measured stalls.
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cstdio>
#include <set>
#include <map>
#include <cstdint>
#include <cassert>
using namespace cute;
template<class T,bool MN,int Rows> auto make_layout(){
 if constexpr(MN)return as_position_independent_swizzle_layout(tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<Int<Rows>,_128>{}));
 else return as_position_independent_swizzle_layout(tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<Int<Rows>,_128>{}));
}
template<class T,bool MN,int Rows,bool SideA,int TileN=0> void inspect(const char* label){
 auto layout=make_layout<T,MN,Rows>();
 int hist[33]={},peak=0;std::set<std::pair<int,int>> coordinates;
 for(int j=0;j<Rows;++j)for(int w=0;w<4;++w){
  std::set<int> words[32];
  for(int lane=0;lane<32;++lane){
   int tid=w*32+lane, r=SideA?j:tid%64, k=SideA?tid:2*j+tid/64;
   if constexpr(TileN>0){constexpr int TileK=128/TileN; r=tid/TileK+(j%(64/TileN))*TileN;k=tid%TileK+(j/(64/TileN))*TileK;}
   assert(r>=0&&r<Rows&&k>=0&&k<128);assert(coordinates.insert({r,k}).second);
   int byte=int(layout(r,k))*sizeof(T),word=byte/4;
   words[word%32].insert(word);
  }
  int maximum=0;for(auto const& bank:words)maximum=std::max(maximum,int(bank.size()));
  ++hist[maximum];peak=std::max(peak,maximum);
 }
 assert(coordinates.size()==Rows*128);
 printf("ACCESS label=%s maximum_distinct_words_per_bank=%d warp_instruction_hist=",label,peak);
 for(int j=1;j<=32;++j)if(hist[j])printf("%d:%d,",j,hist[j]);
 printf("\n");
}
int main(){
 inspect<uint16_t,false,32,true>("native_A");
 inspect<uint16_t,true,64,false>("native_B");
 inspect<int8_t,false,32,true>("int8_A");
 inspect<int8_t,false,64,false>("int8_B");
 inspect<uint16_t,true,64,false,16>("native_B_tileN16_K8");
 inspect<int8_t,false,64,false,16>("int8_B_tileN16_K8");
}

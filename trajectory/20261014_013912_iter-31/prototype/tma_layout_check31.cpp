#include <cute/tensor.hpp>
#include <cutlass/half.h>
#include <iostream>
#include <cstdint>
using namespace cute;
template<class Layout,int Major,int Bits> int check(const char* name) {
  int mismatch=0,old_mismatch=0,count=0,vectors=0;constexpr int rows=size<0>(Layout{}),cols=size<1>(Layout{}),slots=size<2>(Layout{});
  for(int s=0;s<slots;++s)for(int r=0;r<rows;++r)for(int c=0;c<cols;++c) {
    auto coord=make_coord(r,c,s);int raw=Layout{}.layout_b()(coord);int physical=Layout{}.layout_a()(raw*2)/2;
    int plain=(Major==1?r*cols+c:c*rows+r)+s*rows*cols;int expect=plain^(((plain>>6)&((1<<Bits)-1))<<3);
    if(physical!=expect)++mismatch;if(Layout{}(coord)!=expect)++old_mismatch;
    for(unsigned int base:{0u,512u,1024u,8192u}) if(unsigned(Layout{}.layout_a()(base+unsigned(raw*2)))!=base+unsigned(expect*2))++mismatch;
    if((Major==1?c:r)%8==0) { ++vectors;for(int lane=0;lane<8;++lane) {auto next=make_coord(r+(Major==0?lane:0),c+(Major==1?lane:0),s);int nr=Layout{}.layout_b()(next);if(int(Layout{}.layout_a()(nr*2)/2)!=physical+lane)++mismatch;}if(physical%8)++mismatch;}
    ++count;
  }
  std::cout<<name<<" elements="<<count<<" vectors="<<vectors<<" physical_mismatches="<<mismatch<<" old_operator_mismatches="<<old_mismatch<<"\n";return mismatch!=0;
}
int main(){using E=cutlass::half_t;using A=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<E>{},make_shape(Int<64>{},Int<32>{},Int<7>{})));using B16=decltype(tile_to_shape(GMMA::Layout_MN_SW32_Atom<E>{},make_shape(Int<16>{},Int<32>{},Int<7>{})));using B32=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<E>{},make_shape(Int<32>{},Int<32>{},Int<7>{})));return check<A,1,2>("A")|check<B16,0,1>("B16")|check<B32,0,2>("B32");}

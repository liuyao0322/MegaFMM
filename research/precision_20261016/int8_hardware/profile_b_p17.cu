// One frozen P17 BF16 B pack for profiling, not a correctness/timing acceptance.
#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#include "p17/pack.cuh"
int main(){try{
 Stream stream;Buffer a(Count*2),b(Count*2),ad(Count*8),bd(Count*8),workspace(int8_research::WorkBytes);
 fill<<<512,256,0,stream.s>>>(a.as<__nv_bfloat16>(),b.as<__nv_bfloat16>(),ad.as<double>(),bd.as<double>(),0,1901901);
 auto w=int8_research::Work(workspace.p);
 int8_research::inside_partial<<<dim3(8,32),256,0,stream.s>>>(a.as<cute::bfloat16_t>(),b.as<cute::bfloat16_t>(),w.partial);
 int8_research::inside_finalize<<<128,256,0,stream.s>>>(w.partial,w.inside);
 int8_research::pack<cute::bfloat16_t,false,true><<<7*64,256,0,stream.s>>>(b.as<cute::bfloat16_t>(),static_cast<cute::bfloat16_t*>(w.hb),w.qb,w.eb,w.inside);
 check(cudaStreamSynchronize(stream.s));puts("PROFILE_B_P17 done=1 diagnostic_only=1");return 0;
}catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

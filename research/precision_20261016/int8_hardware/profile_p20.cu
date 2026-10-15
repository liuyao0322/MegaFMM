// Three frozen P20 kernels on one fresh BF16 input; diagnostic only.
#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#include "p20/corrected.cuh"
int main(){try{
 Stream stream;Buffer a(Count*2),b(Count*2),ad(Count*8),bd(Count*8),out(Count*4),workspace(int8_research::WorkBytes);
 fill<<<512,256,0,stream.s>>>(a.as<__nv_bfloat16>(),b.as<__nv_bfloat16>(),ad.as<double>(),bd.as<double>(),0,1901901);
 int8_research::init();
 int8_research::run(megafmm::DType::BF16,a.p,b.p,out.as<float>(),workspace.p,stream.s,true);
 check(cudaStreamSynchronize(stream.s));puts("PROFILE_P20 done=1 diagnostic_only=1");return 0;
}catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

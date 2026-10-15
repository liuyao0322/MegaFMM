#pragma once
namespace native_fusion {
template<class T> __device__ __forceinline__ void float_mma(uint64_t a,uint64_t b,float* d,cute::GMMA::ScaleOut scale){
 using namespace cute;
 if constexpr(std::is_same_v<T,cute::half_t>)
  SM90_64x32x16_F32F16F16_SS<GMMA::Major::MN,GMMA::Major::K>::fma(a,b,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],scale);
 else
  SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::MN,GMMA::Major::K>::fma(a,b,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],scale);
}
}

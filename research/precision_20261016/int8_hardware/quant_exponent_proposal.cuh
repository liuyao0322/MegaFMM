#pragma once
// Input is a maximum of rounded native FP16/BF16 magnitudes, initialized at +0.
// Very tiny values and exceptional bit patterns retain the original expression.
__device__ __forceinline__ int native_high_exponent(float peak){
 unsigned u=__float_as_uint(peak);
 if(u==0)return 0;
 if(u>=0x04800000u&&u<=0x7f7fffffu)
  return int(u>>23)-133+int((u&0x7fffffu)>0x7e0000u);
 return peak>0?int(ceilf(log2f(peak/127.f))):0;
}

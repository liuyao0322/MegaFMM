#pragma once
#include <cuda_runtime.h>
namespace int8_research {
// The scale itself is normal in the common branch; multiply preserves RN
// overflow/underflow semantics. Other exponents retain the general operation.
__device__ __forceinline__ float scale_pow2(float x,int e){
 if(e>=-126 && e<=127)return __fmul_rn(x,__int_as_float((e+127)<<23));
 return ldexpf(x,e);
}
}

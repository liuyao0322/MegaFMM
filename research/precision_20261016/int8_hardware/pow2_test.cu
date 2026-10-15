#include "pow2.cuh"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return x;}
__global__ void verify(unsigned* errors){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<(1<<20);i+=gridDim.x*blockDim.x){
  uint32_t bits=mix(i);int e=int(mix(i+91273)%601)-300;
  // Explicitly include zero, smallest/largest subnormal, normal extrema.
  if(i<601*8){constexpr uint32_t b[8]={0U,0x80000000U,1U,0x007fffffU,0x00800000U,0x7f7fffffU,0x3f800000U,0xbf800000U};bits=b[i/601];e=i%601-300;}
  float x=__uint_as_float(bits),a=int8_research::scale_pow2(x,e),b=ldexpf(x,e);
  if(__float_as_uint(a)!=__float_as_uint(b) && !(isnan(a)&&isnan(b)))atomicAdd(errors,1);
 }
}
int main(){try{unsigned* d;ck(cudaMalloc(&d,4));ck(cudaMemset(d,0,4));verify<<<128,256>>>(d);ck(cudaGetLastError());unsigned errors;
 ck(cudaMemcpy(&errors,d,4,cudaMemcpyDeviceToHost));ck(cudaFree(d));printf("POW2 cases=1048576 exponents=-300..300 errors=%u\n",errors);return errors?2:0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

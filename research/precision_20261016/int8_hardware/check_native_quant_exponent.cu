#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "quant_exponent_proposal.cuh"
struct Result{unsigned cases[2],fast[2],fallback[2],zeros[2],errors[2],first[2];};
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);}}
__global__ void check_exponents(Result* out){
 unsigned code=blockIdx.x*blockDim.x+threadIdx.x;
 for(int dt=0;dt<2;++dt){
  if(code>=(dt==0?0x7c00u:0x7f80u))continue;
  float peak;
  if(dt==1)peak=__uint_as_float(code<<16);
  else{
   unsigned exponent=code>>10,mantissa=code&1023u;
   peak=exponent?__uint_as_float(((exponent+112)<<23)|(mantissa<<13)):ldexpf(float(mantissa),-24);
  }
  int old=peak>0?int(ceilf(log2f(peak/127.f))):0;
  int proposed=native_high_exponent(peak);
  unsigned u=__float_as_uint(peak);
  atomicAdd(&out->cases[dt],1);
  if(u==0)atomicAdd(&out->zeros[dt],1);
  else if(u>=0x04800000u&&u<=0x7f7fffffu)atomicAdd(&out->fast[dt],1);
  else atomicAdd(&out->fallback[dt],1);
  if(old!=proposed){atomicAdd(&out->errors[dt],1);atomicMin(&out->first[dt],code);}
 }
}
int main(){
 Result initial{};initial.first[0]=initial.first[1]=65536;Result* device=nullptr;
 ck(cudaMalloc(&device,sizeof(Result)));ck(cudaMemcpy(device,&initial,sizeof(Result),cudaMemcpyHostToDevice));
 check_exponents<<<128,256>>>(device);ck(cudaGetLastError());
 Result result;ck(cudaMemcpy(&result,device,sizeof(Result),cudaMemcpyDeviceToHost));ck(cudaFree(device));
 for(int dt=0;dt<2;++dt)printf("QUANT_EXP dtype=%s cases=%u fast=%u fallback=%u zeros=%u errors=%u first_error_code=%u\n",
  dt==0?"fp16":"bf16",result.cases[dt],result.fast[dt],result.fallback[dt],result.zeros[dt],result.errors[dt],result.first[dt]);
 return (result.errors[0]||result.errors[1])?2:0;
}

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
int main(){void* p=nullptr; auto e=cudaMalloc(&p,8388608);if(e!=cudaSuccess)return 2;
 for(int t=0;t<2;++t){CUtensorMap map{};uint64_t d[4]={64,t?512ULL:1024ULL,t?8ULL:16ULL,8};uint64_t s[3]={t?1024ULL:2048ULL,128,524288};uint32_t b[4]={64,64,4,1},es[4]={1,1,1,1};auto r=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,t?4:3,p,d,s,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);const char* name=nullptr;cuGetErrorName(r,&name);std::printf("transformed=%d status=%d name=%s\n",t,int(r),name?name:"");if(r!=CUDA_SUCCESS)return 3;}
 cudaFree(p);}

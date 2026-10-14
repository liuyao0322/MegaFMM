// Isolated internal tile diagnostic; not a matrix-size benchmark.
#include "../solution/sw_pipeline.cu"
#include <cstdio>
#include <vector>
using namespace cute;
__global__ void diagnostic(float* out) {
  __shared__ __align__(128) half_t a[2048],b[1024];
  constexpr auto la=tile_to_shape(GMMA::Layout_K_INTER_Atom<half_t>{},Shape<_64,_32>{});
  constexpr auto lb=tile_to_shape(GMMA::Layout_K_INTER_Atom<half_t>{},Shape<_32,_32>{});
  if(threadIdx.x==0)printf("addr a=%u b=%u\n",cast_smem_ptr_to_uint(a),cast_smem_ptr_to_uint(b));
  for(int i=threadIdx.x;i<2048;i+=256)a[la(i/32,i%32)]=half_t(float((i/32+3*(i%32))%17-8)/8);
  for(int i=threadIdx.x;i<1024;i+=256)b[lb(i/32,i%32)]=half_t(float((2*(i/32)+i%32)%13-6)/8);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if(threadIdx.x>=128) {
    auto ta=make_tensor(make_smem_ptr(a),la);auto tb=make_tensor(make_smem_ptr(b),lb);
    float d[16]={};warpgroup_arrive();
    megafmm::detail::mma_term<half_t>(ta,tb,d);
    warpgroup_commit_batch();warpgroup_wait<0>();
    for(int j=0;j<16;++j){int mn=GMMA::CLayout_64x32{}(threadIdx.x-128,j);out[(mn%64)*32+mn/64]=d[j];}
  }
}
int main(){float* d;cudaMalloc(&d,8192);diagnostic<<<1,256>>>(d);auto e=cudaDeviceSynchronize();
  printf("CUDA %s\n",cudaGetErrorString(e));std::vector<float> v(2048);cudaMemcpy(v.data(),d,8192,cudaMemcpyDeviceToHost);
  int errors=0;for(int m=0;m<64;++m)for(int n=0;n<32;++n){float ref=0;for(int k=0;k<32;++k)ref+=float((m+3*k)%17-8)*float((2*n+k)%13-6)/64;
  if(v[m*32+n]!=ref){if(errors<12)printf("m%d n%d got%f ref%f\n",m,n,v[m*32+n],ref);++errors;}}
  for(int n=0;n<32;++n){float ref=0;for(int k=0;k<32;++k)ref+=float((3*k)%17-8)*float((2*n+k)%13-6)/64;printf("n%d %.6f ref %.6f\n",n,v[n],ref);}
  printf("errors=%d\n",errors);cudaFree(d);return errors || e!=cudaSuccess;}

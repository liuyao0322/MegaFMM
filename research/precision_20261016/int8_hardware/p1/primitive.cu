// Isolate signed INT8 Hopper layout, exact K128 partial dot, and rescaling.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class Int8> __global__ void int8_dot(const Int8* A,const Int8* BT,int32_t* D,float* F,int exponent){
 using namespace cute;
 __shared__ __align__(128) Int8 smem[64*128+32*128];
 constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<Int8>{},Shape<_64,_128>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_K_SW128_Atom<Int8>{},Shape<_32,_128>{});
 auto a=make_tensor(make_smem_ptr(smem),la);
 auto b=make_tensor(make_smem_ptr(smem+64*128),lb);
 for(int i=threadIdx.x;i<64*128;i+=128)a(i/128,i%128)=A[i];
 for(int i=threadIdx.x;i<32*128;i+=128)b(i/128,i%128)=BT[i];
 __syncthreads();
 asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
 auto da=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(a,Shape<_64,_32>{}));
 auto db=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(b,Shape<_32,_32>{}));
 uint32_t d[16]={};
 warpgroup_arrive();
 CUTE_UNROLL
 for(int k=0;k<4;++k){
  SM90_64x32x32_S32S8S8_SS_TN::fma(uint64_t(da(0,0,k)),uint64_t(db(0,0,k)),
   d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],
   k?GMMA::ScaleOut::One:GMMA::ScaleOut::Zero);
 }
 warpgroup_commit_batch();warpgroup_wait<0>();
 CUTE_UNROLL
 for(int j=0;j<16;++j){
  int mn=GMMA::CLayout_64x32{}(threadIdx.x,j),m=mn%64,n=mn/64;
  int32_t value=static_cast<int32_t>(d[j]);D[m*32+n]=value;
  F[m*32+n]=ldexpf(float(value),exponent);
 }
}

int main(){try{
 int8_t *a,*b;int32_t* d;float* f;
 ck(cudaMalloc(&a,64*128));ck(cudaMalloc(&b,32*128));ck(cudaMalloc(&d,64*32*4));ck(cudaMalloc(&f,64*32*4));
 std::mt19937 gen(1901);std::uniform_int_distribution<int> dist(-127,127);
 for(int pattern=0;pattern<6;++pattern)for(int trial=0;trial<3;++trial){
  std::vector<int8_t> ha(64*128),hb(32*128);
  for(int i=0;i<int(ha.size());++i)ha[i]=pattern==0?dist(gen):pattern==1?127:pattern==2?-127:pattern==3?0:pattern==4?(i%2?127:-127):(i%255)-127;
  for(int i=0;i<int(hb.size());++i)hb[i]=pattern==0?dist(gen):pattern==1?127:pattern==2?127:pattern==3?0:pattern==4?(i%2?-127:127):((i*11)%255)-127;
  ck(cudaMemcpy(a,ha.data(),ha.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(b,hb.data(),hb.size(),cudaMemcpyHostToDevice));
  int exponent=(trial-1)*20;
  int8_dot<int8_t><<<1,128>>>(a,b,d,f,exponent);ck(cudaGetLastError());
  std::vector<int32_t> hd(64*32);std::vector<float> hf(64*32);
  ck(cudaMemcpy(hd.data(),d,hd.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hf.data(),f,hf.size()*4,cudaMemcpyDeviceToHost));
  int errors=0;int64_t max_abs=0;
  for(int m=0;m<64;++m)for(int n=0;n<32;++n){
   int64_t ref=0;for(int k=0;k<128;++k)ref+=int64_t(ha[m*128+k])*hb[n*128+k];
   max_abs=std::max(max_abs,std::abs(ref));
   errors+=hd[m*32+n]!=ref || double(hf[m*32+n])!=std::ldexp(double(ref),exponent);
  }
  printf("PRIMITIVE pattern=%d trial=%d exponent=%d max_abs=%lld errors=%d\n",pattern,trial,exponent,(long long)max_abs,errors);
  if(errors)return 2;
 }
 ck(cudaFree(a));ck(cudaFree(b));ck(cudaFree(d));ck(cudaFree(f));
 return 0;
}catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

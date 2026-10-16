#include "internal.h"
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <vector>
#include <cstdio>
namespace megafmm::detail {
#include "small_debug.cuh"
}
int main(){using T=cute::bfloat16_t; int M=256,N=128,K=256;
std::vector<T>a(M*K),b(K*N);for(int i=0;i<M*K;++i)a[i]=T(float((i*13)%7-3));for(int i=0;i<K*N;++i)b[i]=T(float((i*17)%7-3));
T *A,*B;float*D;cudaMalloc(&A,a.size()*2);cudaMalloc(&B,b.size()*2);cudaMalloc(&D,M*N*4);cudaMemcpy(A,a.data(),a.size()*2,cudaMemcpyHostToDevice);cudaMemcpy(B,b.data(),b.size()*2,cudaMemcpyHostToDevice);
cudaFuncSetAttribute(megafmm::detail::small_sw<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,7*80*64*2);
megafmm::detail::small_sw<T><<<1,128,7*80*64*2>>>(A,B,D,M,N,K);printf("sync %s\n",cudaGetErrorString(cudaDeviceSynchronize()));
float out[7];cudaMemcpy(out,D,28,cudaMemcpyDeviceToHost);for(int p=0;p<7;++p)printf("p%d=%.9g\n",p,out[p]);
double ref[7]={};for(int k=0;k<K/2;++k){float a0=float(a[k]),a1=float(a[k+K/2]),a2=float(a[(M/2)*K+k]),a3=float(a[(M/2)*K+k+K/2]);float b0=float(b[k*N]),b1=float(b[k*N+N/2]),b2=float(b[(k+K/2)*N]),b3=float(b[(k+K/2)*N+N/2]);float s1=a2+a3,s2=s1-a0,t1=b1-b0,t2=b3-t1;float av[7]={a0,a1,a1-s2,a3,s1,s2,a0-a2},bv[7]={b0,b2,b3,t2-b2,t1,t2,b3-b1};for(int p=0;p<7;++p)ref[p]+=double(av[p])*bv[p];}for(int p=0;p<7;++p)printf("ref%d=%.9g\n",p,ref[p]);}

#include "sw_seven.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <cstdio>
namespace megafmm::detail {
namespace {
constexpr size_t LeafBudget=size_t(64)<<20;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> __device__ float value(T x);
template<> __device__ float value(__nv_bfloat16 x){return __bfloat162float(x);}
template<> __device__ float value(__half x){return __half2float(x);}
template<class T> __device__ uint32_t pair(float a,float b){uint32_t x;if constexpr(std::is_same_v<T,__nv_bfloat16>)asm("cvt.rn.bf16x2.f32 %0,%2,%1;":"=r"(x):"f"(a),"f"(b));else asm("cvt.rn.f16x2.f32 %0,%2,%1;":"=r"(x):"f"(a),"f"(b));return x;}
template<class T> __global__ void transform_seven(const T* A,const T* B,T* out,int mh,int nh,int kh){
 int64_t sa=int64_t(mh)*kh,sb=int64_t(kh)*nh;
 for(int side=0;side<2;++side){int cols=side?nh:kh,halfrows=side?kh:mh;int64_t plane=side?sb:sa;const T* input=side?B:A;T* dst=out+(side?4*sa:0);
  for(int64_t i=2*(int64_t(blockIdx.x)*256+threadIdx.x);i<plane;i+=2*256*gridDim.x){int r=i/cols,c=i%cols;int64_t at=int64_t(r)*2*cols+c;float v[4][2];
   #pragma unroll
   for(int l=0;l<2;++l){float x0=value(input[at+l]),x1=value(input[at+cols+l]),x2=value(input[at+int64_t(halfrows)*2*cols+l]),x3=value(input[at+int64_t(halfrows)*2*cols+cols+l]);
    if(side==0){v[0][l]=__fadd_rn(x2,x3);v[1][l]=__fsub_rn(v[0][l],x0);v[2][l]=__fsub_rn(x0,x2);v[3][l]=__fsub_rn(x1,v[1][l]);}
    else{v[0][l]=__fsub_rn(x1,x0);v[1][l]=__fsub_rn(x3,v[0][l]);v[2][l]=__fsub_rn(x3,x1);v[3][l]=__fsub_rn(v[1][l],x2);}}
   #pragma unroll
   for(int j=0;j<4;++j)reinterpret_cast<uint32_t*>(dst)[(j*plane+i)/2]=pair<T>(v[j][0],v[j][1]);
  }
 }
}
__global__ void combine_five(float* D,const float* P,int mh,int nh){
 for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<int64_t(mh)*nh;i+=int64_t(gridDim.x)*256){int r=i/nh,c=i%nh;int64_t at=int64_t(r)*2*nh+c;float p=P[i];D[at+int64_t(mh)*2*nh+nh]=__fadd_rn(D[at+int64_t(mh)*2*nh],p);D[at+nh]=__fadd_rn(D[at+nh],p);}
}
}
struct SWSevenState {
 Problem p;std::array<std::unique_ptr<Plan>,7> leaf;
 size_t sa,sb,st_bytes,p_bytes,leaf_offset,total;
};
bool seven_eligible(const Problem& p,size_t limit){
 if(p.m<4096 || p.n<4096 || p.k<4096 || p.m%256 || p.n%256 || p.k%256 || p.m>32768 || p.n>32768 || p.k>32768 || p.trans_a || p.trans_b || p.lda!=p.k || p.ldb!=p.n || p.ldd!=p.n || p.alpha!=1 || p.beta!=0)return false;
 return size_t(2)*p.k*(p.m+p.n)+size_t(p.m)*p.n+LeafBudget<=limit;
}
std::shared_ptr<SWSevenState> make_seven(const Problem& p){
 auto s=std::make_shared<SWSevenState>();s->p=p;s->sa=size_t(p.m/2)*(p.k/2);s->sb=size_t(p.k/2)*(p.n/2);s->st_bytes=8*(s->sa+s->sb);s->p_bytes=size_t(p.m)*p.n;s->leaf_offset=s->st_bytes+s->p_bytes;size_t scratch=0;
 for(int j=0;j<7;++j){Problem q;q.m=p.m/2;q.n=p.n/2;q.k=p.k/2;q.dtype=p.dtype;q.lda=(j==2 || j>=4)?q.k:p.k;q.ldb=(j==3 || j>=4)?q.n:p.n;q.ldc=q.ldd=j==4?q.n:p.n;q.alpha=j==3?-1:1;q.beta=(j==0 || j==4)?0:1;
 Policy policy;policy.algorithm=Algorithm::Dense;policy.workspace_limit=LeafBudget;
 s->leaf[j]=create_plan(q,policy);scratch=std::max(scratch,s->leaf[j]->workspace_bytes());}
 s->total=s->leaf_offset+scratch;return s;
}
size_t seven_workspace(const SWSevenState& s){return s.total;}
void execute_seven(const SWSevenState& s,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream,bool tuning){
 int mh=s.p.m/2,nh=s.p.n/2,kh=s.p.k/2;
 const char* a=static_cast<const char*>(A);const char* b=static_cast<const char*>(B);char* w=static_cast<char*>(workspace);
 float* temp=reinterpret_cast<float*>(w+s.st_bytes);void* scratch=w+s.leaf_offset;
 const void* av[7]={a,a+kh*2,w+6*s.sa,a+(size_t(mh)*s.p.k+kh)*2,w,w+2*s.sa,w+4*s.sa};

 const char* t=w+8*s.sa;
 const void* bv[7]={b,b+size_t(kh)*s.p.n*2,b+(size_t(kh)*s.p.n+nh)*2,t+6*s.sb,t,t+2*s.sb,t+4*s.sb};
 float* dv[7]={D,D,D+nh,D+size_t(mh)*s.p.n,temp,D+nh,D+size_t(mh)*s.p.n};
 const float* cv[7]={nullptr,D,D+nh,D+size_t(mh)*s.p.n,nullptr,D,D+nh};
 auto leaf=[&](int j){execute_seven_leaf(*s.leaf[j],av[j],bv[j],cv[j],dv[j],scratch,stream,tuning);};
 leaf(0);
 if(s.p.dtype==DType::BF16)transform_seven<<<128,256,0,stream>>>(static_cast<const __nv_bfloat16*>(A),static_cast<const __nv_bfloat16*>(B),reinterpret_cast<__nv_bfloat16*>(workspace),mh,nh,kh);
 else transform_seven<<<128,256,0,stream>>>(static_cast<const __half*>(A),static_cast<const __half*>(B),reinterpret_cast<__half*>(workspace),mh,nh,kh);
 check(cudaPeekAtLastError());leaf(5);leaf(6);leaf(4);
 combine_five<<<256,256,0,stream>>>(D,temp,mh,nh);check(cudaPeekAtLastError());
 leaf(2);leaf(3);leaf(1);
}
void run_seven(const SWSevenState& s,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream){execute_seven(s,A,B,D,workspace,stream,false);}
void tune_seven(SWSevenState& s,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream){execute_seven(s,A,B,D,workspace,stream,true);}
}

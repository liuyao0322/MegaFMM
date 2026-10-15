#include "internal.h"
#include <algorithm>
#include <stdexcept>
namespace megafmm::detail {
namespace {
size_t align256(size_t bytes) {return (bytes+255)&~size_t(255);}
struct Layout {
  int64_t k,n;size_t b,d,bytes;
  explicit Layout(const Problem& p):k((p.k+7)&~int64_t(7)),n((p.n+7)&~int64_t(7)),
      b(align256(size_t(p.m)*k*2)),d(b+align256(size_t(k)*n*2)),
      bytes(d+align256(size_t(p.m)*n*4)) {}
};
__global__ void pack(Problem p,const uint16_t* A,const uint16_t* B,const float* C,
                     uint16_t* pa,uint16_t* pb,float* pc,int64_t kp,int64_t np) {
  const int64_t ac=p.m*kp,bc=kp*np,cc=p.beta!=0 ? p.m*np:0;
  for(int64_t at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;at<ac+bc+cc;at+=int64_t(gridDim.x)*blockDim.x) {
    if(at<ac) {
      int64_t r=at/kp,q=at%kp;
      pa[at]=q<p.k ? A[p.trans_a ? q*p.lda+r:r*p.lda+q]:0;
    } else if(at<ac+bc) {
      int64_t i=at-ac,q=i/np,c=i%np;
      pb[i]=q<p.k && c<p.n ? B[p.trans_b ? c*p.ldb+q:q*p.ldb+c]:0;
    } else {
      int64_t i=at-ac-bc,r=i/np,c=i%np;
      pc[i]=c<p.n ? C[r*p.ldc+c]:0.0f;
    }
  }
}
__global__ void unpack(Problem p,const float* packed,float* D,int64_t np) {
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<p.m*p.n;i+=int64_t(gridDim.x)*blockDim.x)
    D[(i/p.n)*p.ldd+i%p.n]=packed[(i/p.n)*np+i%p.n];
}
}
size_t hopper_workspace(const Problem& p,int id) {
  if(id<5 || id>=HopperCandidateCount || p.m<=0 || p.n<=0 || p.k<=0 ||
      std::max({p.m,p.n,p.k})>32768) return 0;
  if(!p.trans_a && !p.trans_b && p.k%8==0 && p.n%8==0 && p.lda%8==0 &&
      p.ldb%8==0 && p.ldd%4==0 && (p.beta==0 || p.ldc%4==0)) return 0;
  return Layout(p).bytes;
}
bool padded_hopper(int base,const Problem& p,const void* A,const void* B,const float* C,float* D,
                    void* workspace,int sm_count,cudaStream_t stream,int operation) {
  if(operation==2) return true; // Inner kernels are prepared by Plan creation.
  if(!hopper_workspace(p,base+5) || !workspace) return false;
  Layout l(p);Problem aligned=p;
  aligned.k=l.k;aligned.n=l.n;aligned.lda=l.k;aligned.ldb=l.n;
  aligned.ldc=aligned.ldd=l.n;aligned.trans_a=aligned.trans_b=false;
  char* bytes=static_cast<char*>(workspace);
  auto pa=reinterpret_cast<uint16_t*>(bytes),pb=reinterpret_cast<uint16_t*>(bytes+l.b);
  auto pd=reinterpret_cast<float*>(bytes+l.d);
  if(!hopper_dense(base,aligned,pa,pb,pd,pd,nullptr,sm_count,stream,0)) return false;
  if(operation==0) return true;
  pack<<<std::min<int64_t>((p.m*l.k+l.k*l.n+(p.beta!=0 ? p.m*l.n:0)+255)/256,4096),256,0,stream>>>
      (p,static_cast<const uint16_t*>(A),static_cast<const uint16_t*>(B),C,pa,pb,pd,l.k,l.n);
  if(!hopper_dense(base,aligned,pa,pb,pd,pd,nullptr,sm_count,stream,1))
    throw std::runtime_error("padded Hopper binding changed");
  unpack<<<std::min<int64_t>((p.m*p.n+255)/256,4096),256,0,stream>>>(p,pd,D,l.n);
  auto e=cudaPeekAtLastError();if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
  return true;
}
}

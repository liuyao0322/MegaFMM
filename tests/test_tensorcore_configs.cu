#include "sw_hopper.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace megafmm;
namespace tc=megafmm::detail::sw_hopper;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {
  void* p=nullptr;
  explicit Buffer(size_t n){check(cudaMalloc(&p,n));}
  ~Buffer(){cudaFree(p);}
};
template<class T>T cast(float x);
template<>__half cast(float x){return __float2half(x);}
template<>__nv_bfloat16 cast(float x){return __float2bfloat16(x);}
template<class T>void exercise(int id,int m,int n,int k,DType dtype) {
  Problem p;p.m=m;p.n=n;p.k=k;p.lda=k;p.ldb=n;p.ldd=n;p.dtype=dtype;
  if(!tc::eligible(p,id))throw std::runtime_error("test requested ineligible config");
  const size_t count=size_t(m)*n,ws=tc::workspace_size(p,id);
  Buffer a(size_t(m)*k*sizeof(T)),b(size_t(k)*n*sizeof(T)),d((count+64)*4),w(ws+512);
  auto output=static_cast<float*>(d.p)+32;auto scratch=static_cast<char*>(w.p)+256;
  std::vector<T> ha(size_t(m)*k),hb(size_t(k)*n);std::vector<float> hd(count+64);
  cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  for(int seed=1;seed<=3;++seed) {
    std::fill(ha.begin(),ha.end(),cast<T>(0));
    for(int r=0;r<m;++r)ha[size_t(r)*k+(r*13+seed*7)%k]=cast<T>((r+seed)%2?1:-1);
    for(int q=0;q<k;++q)for(int c=0;c<n;++c)hb[size_t(q)*n+c]=cast<T>(int((q*17+c+seed)%7)-3);
    check(cudaMemcpyAsync(a.p,ha.data(),ha.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
    check(cudaMemcpyAsync(b.p,hb.data(),hb.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
    check(cudaMemsetAsync(d.p,0x5a,(count+64)*4,stream));check(cudaMemsetAsync(w.p,0xa5,ws+512,stream));
    tc::run(p,id,a.p,b.p,output,scratch,stream);
    check(cudaMemcpyAsync(hd.data(),d.p,(count+64)*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
    auto bits=reinterpret_cast<const unsigned*>(hd.data());
    for(int i=0;i<32;++i)if(bits[i]!=0x5a5a5a5a || bits[count+32+i]!=0x5a5a5a5a)throw std::runtime_error("output guard");
    for(int r=0;r<m;++r)for(int c=0;c<n;++c) {
      int q=(r*13+seed*7)%k;float ref=((r+seed)%2?1:-1)*(int((q*17+c+seed)%7)-3);
      if(hd[32+size_t(r)*n+c]!=ref) {
        std::fprintf(stderr,"id=%d %dx%dx%d %s seed=%d at %d,%d got=%g ref=%g\n",id,m,n,k,dtype_name(dtype),seed,r,c,hd[32+size_t(r)*n+c],ref);
        throw std::runtime_error("original AB mismatch");
      }
    }
    unsigned guards[128];check(cudaMemcpy(guards,w.p,256,cudaMemcpyDeviceToHost));check(cudaMemcpy(guards+64,scratch+ws,256,cudaMemcpyDeviceToHost));
    for(auto v:guards)if(v!=0xa5a5a5a5)throw std::runtime_error("workspace guard");
  }
  check(cudaStreamDestroy(stream));
  std::printf("PASS id=%d %dx%dx%d %s fresh=3\n",id,m,n,k,dtype_name(dtype));
}
int main(int argc,char** argv) {try {
  std::setvbuf(stdout,nullptr,_IOLBF,0);
  int only=argc>=2?std::stoi(argv[1]):-1;
  bool short_run=argc>=3 && std::string(argv[2])=="--short";
  for(const auto& c:tc::configurations) {
    if(only>=0 && c.id!=only)continue;
    auto resources=tc::setup(c.id);
    if(resources.resident<1)throw std::runtime_error("unlaunchable config");
    std::printf("RESOURCE id=%d registers=%d shared=%zu resident=%d\n",c.id,resources.registers,resources.shared,resources.resident);
    int k=c.k_tiles?2*c.k_tiles*c.bk:6*c.bk;
    for(auto dims:(short_run?std::vector<std::pair<int,int>>{{256,256}}:std::vector<std::pair<int,int>>{{256,256},{400,512}})) {
      int m=(!c.padded && !(c.n&(c.n-1)))?((dims.first+2*c.n-1)/(2*c.n)*2*c.n):dims.first;
      exercise<__nv_bfloat16>(c.id,m,dims.second,k,DType::BF16);
      exercise<__half>(c.id,m,dims.second,k,DType::FP16);
    }
    if(c.padded && !short_run)for(auto dims:std::vector<std::pair<int,int>>{{1,1},{129,257},{514,770}}) {
      exercise<__nv_bfloat16>(c.id,dims.first,dims.second,259,DType::BF16);
      exercise<__half>(c.id,dims.first,dims.second,259,DType::FP16);
    }
    if(!short_run && !c.k_tiles && !c.padded) {
      exercise<__nv_bfloat16>(c.id,256,256,8192,DType::BF16);
      exercise<__half>(c.id,256,256,8192,DType::FP16);
    }
  }
  return 0;
} catch(const std::exception& e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

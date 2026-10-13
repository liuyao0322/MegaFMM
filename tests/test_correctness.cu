#include "megafmm/gemm.h"
#include "megafmm/kernels.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t status) { if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
void require(bool condition,const std::string& message) { if (!condition) throw std::runtime_error(message); }
struct Buffer {
  void* data=nullptr;
  explicit Buffer(size_t bytes) { if (bytes) check(cudaMalloc(&data,bytes)); }
  ~Buffer() { if(data) cudaFree(data); }
  Buffer(const Buffer&)=delete;
  Buffer& operator=(const Buffer&)=delete;
  template<class T> T* as() const { return static_cast<T*>(data); }
};
struct Stream {
  cudaStream_t value=nullptr;
  Stream() { check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking)); }
  ~Stream() { if(value)cudaStreamDestroy(value); }
};
struct Graph {
  cudaGraph_t value=nullptr;cudaGraphExec_t executable=nullptr;
  ~Graph() { if(executable)cudaGraphExecDestroy(executable);if(value)cudaGraphDestroy(value); }
};
template<class T> T round_value(float v);
template<> __half round_value(float v) { return __float2half_rn(v); }
template<> __nv_bfloat16 round_value(float v) { return __float2bfloat16_rn(v); }
template<class T> double double_value(T v) { return double(float(v)); }
template<> double double_value(__nv_bfloat16 v) { return double(__bfloat162float(v)); }
struct Metrics { double maxabs=0, ef=0, scaled=0, normalized=0, nearzero=0; size_t nonfinite=0; };
Metrics metrics(const std::vector<float>& actual,const std::vector<double>& reference,
                int m,int n,int ldd,double floor,double rtol,double normab) {
  Metrics out; double err2=0,ref2=0;
  for(int i=0;i<m;++i) for(int j=0;j<n;++j) {
    double a=actual[size_t(i)*ldd+j],r=reference[size_t(i)*n+j];
    if(!std::isfinite(a) || !std::isfinite(r)) { ++out.nonfinite; continue; }
    double e=std::abs(a-r); err2+=e*e; ref2+=r*r;
    out.maxabs=std::max(out.maxabs,e);
    out.scaled=std::max(out.scaled,e/(floor+rtol*std::abs(r)));
    if(std::abs(r)<=floor) out.nearzero=std::max(out.nearzero,e);
  }
  out.ef=std::sqrt(err2/std::max(ref2,std::numeric_limits<double>::min()));
  out.normalized=std::sqrt(err2)/std::max(normab,std::numeric_limits<double>::min());
  return out;
}
enum class Pattern { Integers, Zero, Identity, Cancellation, Uniform, Positive, Range, HighHalf, M2BlockBasis, OwnerRows };
const char* pattern_name(Pattern p) {
  switch(p) {
    case Pattern::Integers:return "integers";case Pattern::Zero:return "zero";
    case Pattern::Identity:return "identity";case Pattern::Cancellation:return "cancellation";
    case Pattern::Uniform:return "uniform";case Pattern::Positive:return "positive";
    case Pattern::Range:return "range";case Pattern::HighHalf:return "fp16_upper_range";
    case Pattern::M2BlockBasis:return "m2_noncommuting_blocks";
    case Pattern::OwnerRows:return "distinct_eight_row_owners";
  } return "unknown";
}
std::string algorithm_name(megafmm::Algorithm algorithm) {
  switch(algorithm) {
    case megafmm::Algorithm::Auto:return "auto";case megafmm::Algorithm::Dense:return "dense";
    case megafmm::Algorithm::WinogradLt:return "lt";case megafmm::Algorithm::WinogradHopper:return "hopper";
    case megafmm::Algorithm::Megakernel:return "mega";
  } return "unknown";
}
const char* batched_name(megafmm::Algorithm algorithm,int variant) {
  if(algorithm==megafmm::Algorithm::WinogradLt)return "winograd_lt_batched";
  return variant>=7 ? "winograd_sm80_batched_k64":"winograd_sm80_batched";
}
const char* warp_mma_name(int variant,const megafmm::Problem& p) {
  if(variant==26 || variant==27)
    return p.m%64==0 && p.n%(variant==26 ? 32:64)==0 && p.k%32==0 &&
           p.lda%8==0 && p.ldb%8==0 ?
           (variant==26 ? "megakernel_cooperative_32x16":"megakernel_cooperative_32x32"):
           "megakernel_warp_mma_packed";
  if(variant==24 || variant==25) {
    if(p.k==128 && p.m%32==0 && p.n%32==0 && p.lda%8==0 && p.ldb%8==0 &&
       p.ldd%2==0 && p.alpha==1.0f && p.beta==0.0f)
      return variant==24 ? "megakernel_warp_mma_8x8_k128_no_unroll":"megakernel_warp_mma_8x8_k128_unroll";
    return warp_mma_name(19,p);
  }
  if(variant==20 || variant==21 || variant==22 || variant==23) {
    if(p.m%32==0 && p.n%32==0 && p.k%32==0 && p.lda%8==0 && p.ldb%8==0 &&
       p.ldd%2==0 && p.alpha==1.0f && p.beta==0.0f) {
      if(variant>=22)
        return variant==22 ? "megakernel_warp_mma_coalesced_16x8":"megakernel_warp_mma_coalesced_8x8";
      return variant==20 ? "megakernel_warp_mma_direct_16x8":"megakernel_warp_mma_direct_8x8";
    }
    return warp_mma_name(16,p);
  }
  if(variant==19) {
    if(p.m%32==0 && p.n%32==0 && p.k%32==0 && p.lda%8==0 && p.ldb%8==0 &&
       p.ldd%2==0 && p.alpha==1.0f && p.beta==0.0f)
      return "megakernel_warp_mma_8x8_prefetch";
    return warp_mma_name(16,p);
  }
  if(variant==17 || variant==18)return p.m%32==0 && p.n%32==0 && p.k%32==0 &&
                         p.lda%8==0 && p.ldb%8==0 ?
                         (variant==18 ? "megakernel_stage_m2":"megakernel_warp_mma_cpasync"):
                         "megakernel_warp_mma_packed";
  if(variant==14 || variant==15 || variant==16) {
    bool full=p.m%32==0 && p.n%16==0 && p.k%32==0 && p.lda%8==0 && p.ldb%8==0;
    if(!full)return "megakernel_warp_mma_packed";
    if(p.ldd%2!=0 || (p.beta!=0.0f && p.ldc%2!=0))return "megakernel_warp_mma_16x8_vec4";
    if(variant>=15 && p.alpha==1.0f && p.beta==0.0f)
      return variant==16 ? "megakernel_warp_mma_16x8_prefetch":"megakernel_warp_mma_16x8_identity";
    return "megakernel_warp_mma_16x8_epilogue";
  }
  if(variant==13)return p.m%32==0 && p.n%32==0 && p.k%32==0 &&
                         p.lda%8==0 && p.ldb%8==0 && p.ldc%8==0 && p.ldd%8==0 ?
                         "megakernel_product_cluster":"megakernel_product_warps";
  if(variant==11 || variant==12)return p.m%32==0 && p.n%16==0 && p.k%32==0 && p.lda%8==0 && p.ldb%8==0 ?
                         (variant==11 ? "megakernel_warp_mma_16x8":"megakernel_warp_mma_16x8_vec4"):
                         "megakernel_warp_mma_packed";
  if(variant==10)return p.m%32==0 && p.n%32==0 && p.k%32==0 &&
                         p.lda%8==0 && p.ldb%8==0 && p.ldc%8==0 && p.ldd%8==0 ?
                         "megakernel_product_warps_fast":"megakernel_product_warps";
  if(variant==9)return "megakernel_product_warps";
  if(variant<=6)return "megakernel_warp_mma";
  if(variant==8 && p.m%32==0 && p.n%32==0 && p.k%32==0)return "megakernel_warp_mma_fast";
  return "megakernel_warp_mma_packed";
}
const char* lt_config_kind(const megafmm::Plan& plan) {
  const auto name=plan.algorithm_name();
  if(name=="dense_cublaslt")return "dense";
  if(name=="winograd_lt_v1")return "leaf";
  if(name=="winograd_lt_batched")return "batch7";
  return "none";
}
std::string check_lt_config(const megafmm::Plan& plan,const void* A,const void* B,
                            const float* C0,float* D,void* workspace,
                            const std::string& kind,bool tuned) {
  const auto config=plan.lt_algorithm_config(A,B,C0,D,workspace);
  if(kind=="none") {require(config=="none","custom/degenerate Lt metadata not none");return config;}
  size_t begin=0;int records=0;
  do {
    size_t end=config.find('|',begin);
    auto record=config.substr(begin,end==std::string::npos ? end:end-begin);
    auto prefix=kind=="leaf" ? "leaf"+std::to_string(records):kind;
    require(record.rfind(prefix+":",0)==0,"Lt metadata operator identity incorrect");
    record=record.substr(prefix.size()+1);
    auto value=[&](const std::string& key) {
      auto at=record.find(key+"=");
      require(at!=std::string::npos && (at==0 || record[at-1]==';'),"Lt metadata field missing: "+key);
      at+=key.size()+1;return record.substr(at,record.find(';',at)-at);
    };
    for(const char* attr:{"id","tile","split_k","reduction","swizzle","stages","custom","inner","cluster"})
      require(!value(attr).empty(),"empty Lt algorithm attribute");
    auto bound=value("bound_candidate");auto slash=bound.find('/');
    require(slash!=std::string::npos,"Lt metadata candidate pool missing");
    auto index=std::stoull(bound.substr(0,slash)),pool=std::stoull(bound.substr(slash+1));
    auto selected=std::stoull(value("tuned_candidate"));
    require(index<pool && selected<pool,"Lt metadata candidate index out of range");
    require(std::stoull(value("bound_workspace"))<=plan.workspace_bytes(),"Lt bound workspace exceeds allocation");
    require(value("tuned")== (tuned ? "1":"0"),"Lt metadata tuned state incorrect");
    const auto& p=plan.problem();auto descriptor_max=std::max({p.m,p.n,p.k});
    if(kind!="dense")descriptor_max/=2;
    require(value("tune")== (descriptor_max<=4096 ? "4w/3x10median":"1w/1x3"),"Lt actual descriptor tuning profile incorrect");
    auto score=value("tuning_ms");
    if(!tuned)require(score=="NA(untuned)","untuned Lt metadata reports a score");
    else if(index!=selected)require(score=="NA(rebound)","rebound Lt metadata reports another candidate's score");
    else {auto ms=std::stod(score);require(std::isfinite(ms) && ms>=0,"invalid Lt tuning score");}
    for(const char* alignment:{"align0","align1","align2","align3"})
      require(std::stoull(value(alignment))>0,"invalid Lt alignment requirement");
    ++records;if(end==std::string::npos)break;begin=end+1;
  } while(true);
  require(records==(kind=="leaf" ? 7:1),"Lt metadata operator count incorrect");
  return config;
}
template<class T> std::vector<double> m2_integer_reference(const megafmm::Problem& p,
                                                         const std::vector<T>& a,const std::vector<T>& b) {
  // Independent exact arithmetic oracle for small integer operands: form each
  // stage's seven bilinear products, apply the expanded 14 coefficients, then
  // compare against ordinary AB. This does not model Tensor Core dot rounding.
  const int coefficients[4][7]={{1,1,0,0,0,0,0},{1,0,1,0,1,1,0},
                                {1,0,0,-1,0,1,1},{1,0,0,0,1,1,1}};
  const int mh=int(p.m/2),nh=int(p.n/2),kh=int(p.k/2);
  std::vector<double> result(size_t(p.m)*p.n);
  for(int i=0;i<mh;++i) for(int j=0;j<nh;++j) {
    double c[4]={};
    for(int stage=0;stage<kh;stage+=16) {
      double products[7]={};
      for(int l=stage;l<std::min(stage+16,kh);++l) {
        const double a11=double_value(a[size_t(i)*p.lda+l]),a12=double_value(a[size_t(i)*p.lda+l+kh]);
        const double a21=double_value(a[size_t(i+mh)*p.lda+l]),a22=double_value(a[size_t(i+mh)*p.lda+l+kh]);
        const double b11=double_value(b[size_t(l)*p.ldb+j]),b12=double_value(b[size_t(l)*p.ldb+j+nh]);
        const double b21=double_value(b[size_t(l+kh)*p.ldb+j]),b22=double_value(b[size_t(l+kh)*p.ldb+j+nh]);
        const double s1=a21+a22,s2=s1-a11,s3=a11-a21,s4=a12-s2;
        const double t1=b12-b11,t2=b22-t1,t3=b22-b12,t4=t2-b21;
        products[0]+=a11*b11;products[1]+=a12*b21;products[2]+=s4*b22;
        products[3]+=a22*t4;products[4]+=s1*t1;products[5]+=s2*t2;products[6]+=s3*t3;
      }
      for(int product=0;product<7;++product) for(int quadrant=0;quadrant<4;++quadrant)
        if(coefficients[quadrant][product])c[quadrant]+=coefficients[quadrant][product]*products[product];
    }
    result[size_t(i)*p.n+j]=c[0];result[size_t(i)*p.n+j+nh]=c[1];
    result[size_t(i+mh)*p.n+j]=c[2];result[size_t(i+mh)*p.n+j+nh]=c[3];
  }
  return result;
}
template<class T> Metrics run_case(megafmm::Problem p,megafmm::Algorithm algorithm,
                                  Pattern pattern,bool alias,bool certified,uint32_t seed=1234567,int variant=0,
                                  bool do_tune=true,size_t workspace_limit=size_t(8)<<30,
                                  const char* expected_route=nullptr) {
  int ar=p.trans_a ? p.k:p.m, ac=p.trans_a ? p.m:p.k;
  int br=p.trans_b ? p.n:p.k, bc=p.trans_b ? p.k:p.n;
  std::vector<T> a(size_t(ar)*p.lda),b(size_t(br)*p.ldb);
  std::vector<float> c(size_t(p.m)*p.ldc,0),result(size_t(p.m)*p.ldd,-777.0f);
  std::mt19937 rng(seed);
  auto value=[&](int i,int j,bool left) {
    if(pattern==Pattern::OwnerRows) {
      const int rows=left ? int(p.m/2):int(p.k/2),cols=left ? int(p.k/2):int(p.n/2);
      const int block=(i/rows)*2+j/cols,row=i%rows,col=j%cols;
      const int inner=left ? col:row;
      if(inner%16>=2)return 0.0f;
      const int r=1+row%8,g=1+row/8,v=1+col%2,h=1+col/8,s=1+inner/16;
      const int a[4][2]={{r,g},{g,-r},{-r,g+1},{g+1,r}};
      const int b[4][2]={{s*v,h},{s*(v+1),-h},{-s*v,h+1},{s*(v+2),-(h+1)}};
      return float(left ? a[block][inner%16]:b[block][inner%16]);
    }
    if(pattern==Pattern::M2BlockBasis) {
      // Nonsymmetric 2x2 blocks do not commute. Embed their contracted axes
      // into each K16 stage, with different exact integer stage weights.
      const int left_blocks[4][4]={{1,2,0,1},{0,1,3,0},{2,0,1,-1},{1,-1,2,1}};
      const int right_blocks[4][4]={{0,2,1,0},{1,0,-2,1},{2,1,0,-1},{-1,1,2,0}};
      const int rows=left ? int(p.m/2):int(p.k/2),cols=left ? int(p.k/2):int(p.n/2);
      const int block=(i/rows)*2+j/cols,row=i%rows,col=j%cols;
      const int contracted=left ? col:row;
      if(contracted%16>=2)return 0.0f;
      const int at=(left ? row%2:contracted%16)*2+(left ? contracted%16:col%2);
      return float((left ? left_blocks[block][at]:right_blocks[block][at])*(1+contracted/16));
    }
    if(pattern==Pattern::Zero) return 0.0f;
    if(pattern==Pattern::Identity && left) return i==j ? 1.0f:0.0f;
    if(pattern==Pattern::Integers || pattern==Pattern::Identity) return float(int(rng()%5)-2);
    if(pattern==Pattern::Cancellation) {
      // Exact dyadic perturbations and opposite neighboring terms exercise near-zero outputs.
      return (j%2 ? -1.0f:1.0f)*(1.0f+float((i+j)%3)/32.0f);
    }
    float v=float(rng()>>8)/16777216.0f;
    if(pattern==Pattern::Positive) return v;
    if(pattern==Pattern::Range) return (2*v-1)*16;
    if(pattern==Pattern::HighHalf) return (left ? 65504.0f:0.00006103515625f)*(j%2 ? -1.0f:1.0f);
    return 2*v-1;
  };
  for(int i=0;i<p.m;++i) for(int j=0;j<p.k;++j) {
    size_t at=p.trans_a ? size_t(j)*p.lda+i:size_t(i)*p.lda+j;
    a[at]=round_value<T>(value(i,j,true));
  }
  for(int i=0;i<p.k;++i) for(int j=0;j<p.n;++j) {
    size_t at=p.trans_b ? size_t(j)*p.ldb+i:size_t(i)*p.ldb+j;
    b[at]=round_value<T>(value(i,j,false));
  }
  for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j) c[size_t(i)*p.ldc+j]=float(int(rng()%9)-4)/4;
  double anorm=0,bnorm=0;
  for(int i=0;i<p.m;++i) for(int l=0;l<p.k;++l) {
    double v=double_value(a[p.trans_a ? size_t(l)*p.lda+i:size_t(i)*p.lda+l]);anorm+=v*v;
  }
  for(int l=0;l<p.k;++l) for(int j=0;j<p.n;++j) {
    double v=double_value(b[p.trans_b ? size_t(j)*p.ldb+l:size_t(l)*p.ldb+j]);bnorm+=v*v;
  }
  std::vector<double> reference(size_t(p.m)*p.n);
  // FP64 plain AB from the original rounded inputs, with no transformed operands.
  for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j) {
    double sum=0;
    for(int l=0;l<p.k;++l)
      sum+=double_value(a[p.trans_a ? size_t(l)*p.lda+i:size_t(i)*p.lda+l]) *
           double_value(b[p.trans_b ? size_t(j)*p.ldb+l:size_t(l)*p.ldb+j]);
    reference[size_t(i)*p.n+j]=double(p.alpha)*sum+double(p.beta)*c[size_t(i)*p.ldc+j];
  }
  std::vector<double> stage_reference;
  if(algorithm==megafmm::Algorithm::Megakernel && variant==18 && expected_route &&
     std::string(expected_route)=="megakernel_stage_m2" &&
     (pattern==Pattern::Integers || pattern==Pattern::M2BlockBasis)) {
    stage_reference=m2_integer_reference(p,a,b);
    for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j) {
      auto& stage=stage_reference[size_t(i)*p.n+j];
      stage=double(p.alpha)*stage+double(p.beta)*c[size_t(i)*p.ldc+j];
      require(stage==reference[size_t(i)*p.n+j],"M2 expanded stage reference disagrees with independent FP64 AB");
    }
  }
  Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dc(c.size()*sizeof(float)),dd(result.size()*sizeof(float));
  check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
  check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
  check(cudaMemcpy(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
  check(cudaMemcpy(dd.data,result.data(),result.size()*sizeof(float),cudaMemcpyHostToDevice));
  megafmm::Policy policy;policy.algorithm=algorithm;policy.certified_range=certified;
  if(algorithm==megafmm::Algorithm::WinogradHopper)policy.hopper_variant=variant;
  if(algorithm==megafmm::Algorithm::Megakernel)policy.mega_variant=variant;
  if(algorithm==megafmm::Algorithm::WinogradLt)policy.lt_variant=variant;
  policy.workspace_limit=workspace_limit;
  auto plan=megafmm::create_plan(p,policy);
  if(expected_route)require(plan->algorithm_name()==expected_route,
                           std::string("unexpected route: expected ")+expected_route+", got "+plan->algorithm_name());
  if(expected_route && (std::string(expected_route).find("megakernel_warp_mma")==0 ||
                        std::string(expected_route).find("megakernel_cooperative")==0 ||
                        std::string(expected_route).find("megakernel_product_warps")==0 ||
                        std::string(expected_route)=="megakernel_product_cluster" ||
                        std::string(expected_route)=="megakernel_stage_m2"))
    require(plan->workspace_bytes()==0,"warp MMA unexpectedly requires workspace");
  Buffer workspace(plan->workspace_bytes());
  float* output=alias ? dc.as<float>():dd.as<float>();
  require(!alias || p.ldc==p.ldd,"test alias needs equal C/D strides");
  if(pattern==Pattern::Integers) {
    check_lt_config(*plan,da.data,db.data,p.beta==0 ? nullptr:dc.as<float>(),output,workspace.data,
                    lt_config_kind(*plan),false);
    // A representative alias binding must remain byte-identical after reading metadata.
    if(alias && algorithm==megafmm::Algorithm::Dense && p.m==128 && p.n==128 && p.k==128) {
      check(cudaMemcpy(result.data(),output,result.size()*sizeof(float),cudaMemcpyDeviceToHost));
      require(std::memcmp(result.data(),c.data(),result.size()*sizeof(float))==0,"Lt metadata modified C0/D");
    }
  }
  if(do_tune) {
    megafmm::tune(*plan,da.data,db.data,p.beta==0 ? nullptr:dc.as<float>(),output,workspace.data);
    if(pattern==Pattern::Integers)
      check_lt_config(*plan,da.data,db.data,p.beta==0 ? nullptr:dc.as<float>(),output,workspace.data,
                      lt_config_kind(*plan),true);
    if(alias && pattern==Pattern::Integers) {
      check(cudaMemcpy(result.data(),output,result.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
        require(double(result[size_t(i)*p.ldd+j])==reference[size_t(i)*p.n+j],
                "tuned in-place alpha/beta output incorrect");
    }
  }
  // Restore C0 after tuning when C0 and D share storage.
  if(alias) check(cudaMemcpy(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
  for(int repeat=0;repeat<2;++repeat) {
    if(alias) check(cudaMemcpy(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
    megafmm::run(*plan,da.data,db.data,p.beta==0 ? nullptr:dc.as<float>(),output,workspace.data);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(result.data(),output,result.size()*sizeof(float),cudaMemcpyDeviceToHost));
  }
  double floor=p.dtype==megafmm::DType::BF16 ? 1e-3:1e-4,rtol=p.dtype==megafmm::DType::BF16 ? 1e-2:1e-3;
  Metrics e=metrics(result,reference,p.m,p.n,p.ldd,floor,rtol,std::sqrt(anorm*bnorm));
  if(!stage_reference.empty())
    for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
      require(double(result[size_t(i)*p.ldd+j])==stage_reference[size_t(i)*p.n+j],
              "M2 GPU integer output disagrees with expanded per-stage contribution reference");
  std::cout << "CASE " << algorithm_name(algorithm) << " variant=" << variant
            << " actual=" << plan->algorithm_name() << ' '
            << megafmm::dtype_name(p.dtype) << ' ' << pattern_name(pattern) << ' '
            << p.m << ',' << p.n << ',' << p.k << " maxabs=" << e.maxabs << " eF=" << e.ef
            << " scaled=" << e.scaled << " normalized=" << e.normalized << " nonfinite=" << e.nonfinite << '\n';
  for(int i=0;i<p.m;++i) for(int j=p.n;j<p.ldd;++j)
    require(result[size_t(i)*p.ldd+j]==(alias ? c[size_t(i)*p.ldc+j]:-777.0f),"output padding overwritten");
  return e;
}
megafmm::Problem problem(int m,int n,int k,megafmm::DType dtype) {
  megafmm::Problem p;p.m=m;p.n=n;p.k=k;p.lda=k;p.ldb=n;p.ldc=n;p.ldd=n;p.dtype=dtype;return p;
}
template<class T> void structural_suite(megafmm::DType dtype) {
  const std::vector<megafmm::Algorithm> algorithms={megafmm::Algorithm::Dense,megafmm::Algorithm::WinogradLt,
    megafmm::Algorithm::WinogradHopper,megafmm::Algorithm::Megakernel,megafmm::Algorithm::Auto};
  for(auto algo:algorithms) {
    for(auto pat:{Pattern::Integers,Pattern::Zero,Pattern::Identity,Pattern::Cancellation}) {
      auto e=run_case<T>(problem(128,128,128,dtype),algo,pat,false,true);
      require(e.nonfinite==0 && e.maxabs==0,"exact structural DAG test failed");
    }
    for(auto dims:{std::vector<int>{65,97,33},std::vector<int>{1,1,1},std::vector<int>{128,64,256},std::vector<int>{130,66,258}}) {
      auto e=run_case<T>(problem(dims[0],dims[1],dims[2],dtype),algo,Pattern::Integers,false,true);
      require(e.nonfinite==0 && e.maxabs==0,"rectangular/boundary structural test failed");
    }
    auto p=problem(31,47,29,dtype);p.lda+=3;p.ldb+=5;p.ldc+=7;p.ldd+=9;p.alpha=.5f;p.beta=-.25f;
    auto e=run_case<T>(p,algo,Pattern::Integers,false,true);
    require(e.nonfinite==0 && e.maxabs==0,"stride alpha/beta test failed");
    p.ldc=p.ldd;
    e=run_case<T>(p,algo,Pattern::Integers,true,true);
    require(e.nonfinite==0 && e.maxabs==0,"C0/D alias test failed");
    // These aligned shapes reach forced SW/Hopper/mega rather than relying on boundary fallback.
    p=problem(128,128,128,dtype);p.alpha=.5f;p.beta=-.25f;
    e=run_case<T>(p,algo,Pattern::Integers,false,true);
    require(e.nonfinite==0 && e.maxabs==0,"fast-path alpha/beta test failed");
    e=run_case<T>(p,algo,Pattern::Integers,true,true);
    require(e.nonfinite==0 && e.maxabs==0,"fast-path C0/D alias test failed");
    p.lda+=8;p.ldb+=8;p.ldc+=8;p.ldd+=8;
    e=run_case<T>(p,algo,Pattern::Integers,false,true);
    require(e.nonfinite==0 && e.maxabs==0,"aligned padded-stride alpha/beta test failed");
    e=run_case<T>(p,algo,Pattern::Integers,true,true);
    require(e.nonfinite==0 && e.maxabs==0,"aligned padded-stride C0/D alias test failed");
    for(bool ta:{false,true}) for(bool tb:{false,true}) {
      p=problem(17,23,19,dtype);p.trans_a=ta;p.trans_b=tb;
      p.lda=(ta ? p.m:p.k)+5;p.ldb=(tb ? p.k:p.n)+7;p.alpha=-.5f;p.beta=.25f;
      e=run_case<T>(p,algo,Pattern::Integers,false,true);
      require(e.nonfinite==0 && e.maxabs==0,"transpose fallback test failed");
    }
    // Ordinary random and larger magnitudes exercise dense fallback under an unknown range declaration.
    for(auto pat:{Pattern::Uniform,Pattern::Positive,Pattern::Range,Pattern::HighHalf}) {
      p=problem(37,43,31,dtype);
      auto reference_dense=run_case<T>(p,megafmm::Algorithm::Dense,pat,false,false,9876543);
      e=run_case<T>(p,algo,pat,false,false,9876543);
      double floor=dtype==megafmm::DType::BF16 ? 1e-3:1e-4;
      require(e.nonfinite==0 && e.ef<=3*reference_dense.ef+floor,"unknown-range dense fallback numerical gate failed");
    }
  }
  for(auto candidate:{std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,1},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,2},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,3},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,4},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,5},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,6},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,7},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,8},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,9},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,10},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,11},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,12},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,13},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,14},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,15},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,16},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,17},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,18},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,19},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,20},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,21},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,22},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,23},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,24},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,25},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,26},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,27},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,1},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,2}}) {
    auto e=run_case<T>(problem(256,256,256,dtype),candidate.first,Pattern::Integers,false,true,1234567,candidate.second);
    require(e.nonfinite==0 && e.maxabs==0,"candidate variant structural test failed");
  }
  for(int variant:{2,3}) {
    auto p=problem(256,256,256,dtype);
    p.alpha=.5f;p.beta=-.25f;
    auto e=run_case<T>(p,megafmm::Algorithm::Megakernel,Pattern::Integers,true,true,1234567,variant);
    require(e.nonfinite==0 && e.maxabs==0,"streaming megakernel alpha/beta alias test failed");
  }
}
template<class T> void batched_suite(megafmm::DType dtype,bool lt_only=false) {
  for(auto candidate:{std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradLt,1},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,5},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,6},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,7},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,8}}) {
  const auto algorithm=candidate.first;const int variant=candidate.second;
  if(lt_only && algorithm!=megafmm::Algorithm::WinogradLt)continue;
  auto exact=[dtype,algorithm,variant](megafmm::Problem p,Pattern pattern,bool alias=false,bool certified=true,
                    bool expect_batched=true,size_t workspace_limit=size_t(8)<<30,bool do_tune=false) {
    const char* route=!expect_batched ? "dense_cublaslt":batched_name(algorithm,variant);
    std::cout << "BEGIN_BATCHED dtype=" << megafmm::dtype_name(dtype)
              << " algorithm=" << algorithm_name(algorithm) << " variant=" << variant
              << " pattern=" << pattern_name(pattern) << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
              << " certified_range=" << certified << " workspace_limit=" << workspace_limit
              << " expected=" << route << std::endl;
    auto e=run_case<T>(p,algorithm,pattern,alias,certified,1234567,variant,
                       do_tune,workspace_limit,route);
    require(e.nonfinite==0 && e.maxabs==0,"batched structural/alias/fallback test failed");
  };
  // Small exact operands exercise all seven slot mappings and the fixed U DAG.
  // Tune one case; the remaining scenarios also cover the untuned route.
  for(auto pattern:{Pattern::Integers,Pattern::Zero,Pattern::Identity,Pattern::Cancellation})
    exact(problem(128,128,128,dtype),pattern,false,true,true,size_t(8)<<30,pattern==Pattern::Integers);
  exact(problem(128,64,256,dtype),Pattern::Integers);
  exact(problem(256,256,256,dtype),Pattern::Integers);
  auto p=problem(128,128,128,dtype);p.alpha=.5f;p.beta=-.25f;
  exact(p,Pattern::Integers);
  // Verify tune's output before restoring the original C0 for repeated run calls.
  exact(p,Pattern::Integers,true,true,true,size_t(8)<<30,true);
  p.lda+=8;p.ldb+=8;p.ldc+=8;p.ldd+=8;
  exact(p,Pattern::Integers);
  exact(p,Pattern::Integers,true);
  // These cases must select Dense even though a batched route was requested.
  exact(problem(128,128,128,dtype),Pattern::Integers,false,false,false);
  exact(problem(65,97,33,dtype),Pattern::Integers,false,true,false);
  for(auto dims:{std::vector<int>{130,128,128},std::vector<int>{128,130,128},std::vector<int>{128,128,130}}) {
    p=problem(dims[0],dims[1],dims[2],dtype);p.lda=(p.lda+7)/8*8;p.ldb=(p.ldb+7)/8*8;
    exact(p,Pattern::Integers,false,true,false);
  }
  exact(problem(128,128,128,dtype),Pattern::Integers,false,true,false,0);
  p=problem(128,128,128,dtype);p.lda+=1;
  exact(p,Pattern::Integers,false,true,false);
  p=problem(128,128,128,dtype);p.ldb+=1;
  exact(p,Pattern::Integers,false,true,false);
  p=problem(128,128,128,dtype);p.trans_a=true;
  exact(p,Pattern::Integers,false,true,false);
  p=problem(128,128,128,dtype);p.trans_b=true;
  exact(p,Pattern::Integers,false,true,false);
  }
}
template<class T> void k64_batched_tail_suite(megafmm::DType dtype,int variant) {
  require(variant==7 || variant==8,"K64 tail suite needs Hopper7/8");
  for(int scenario=0;scenario<4;++scenario) {
    const int parent_k=scenario==0 ? 16:scenario==1 ? 48:144;
    auto p=problem(80,112,parent_k,dtype);
    const bool alias=scenario==3;
    if(alias) {
      p.lda+=8;p.ldb+=8;p.ldc+=8;p.ldd+=8;
      p.alpha=.5f;p.beta=-.25f;
    }
    auto leaf=problem(p.m/2,p.n/2,p.k/2,dtype);
    require(megafmm::small_batched_k64_can_implement(leaf,variant),"K64 leaf tail rejected");
    require(megafmm::small_batched_k64_workspace_bytes(leaf,variant)==0,
            "K64 leaf unexpectedly requires workspace");
    // Parent uniform A/B/P slots still require caller-owned workspace.
    std::cout << "BEGIN_K64_BATCHED_TAIL dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " leaf_k=" << leaf.k << " alias=" << alias
              << " alpha=" << p.alpha << " beta=" << p.beta << std::endl;
    auto e=run_case<T>(p,megafmm::Algorithm::WinogradHopper,Pattern::Integers,alias,true,
                       1234567,variant,alias,size_t(8)<<30,batched_name(megafmm::Algorithm::WinogradHopper,variant));
    require(e.nonfinite==0 && e.maxabs==0,"K64 batched partial K/M/N or alias/tune output incorrect");
  }
}
template<class T> void warp_mma_suite(megafmm::DType dtype) {
  for(int variant=4;variant<=27;++variant) {
    auto exact=[dtype,variant](megafmm::Problem p,Pattern pattern,bool alias=false,bool certified=true,
                              bool expect_warp=true,bool do_tune=false) {
      const char* route=expect_warp ? warp_mma_name(variant,p):"dense_cublaslt";
      std::cout << "BEGIN_WARP_MMA dtype=" << megafmm::dtype_name(dtype) << " variant=" << variant
                << " pattern=" << pattern_name(pattern) << " shape=" << p.m << ',' << p.n << ',' << p.k
                << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
                << " certified_range=" << certified << " expected=" << route << std::endl;
      auto e=run_case<T>(p,megafmm::Algorithm::Megakernel,pattern,alias,certified,1234567,variant,
                         do_tune,0,route);
      require(e.nonfinite==0 && e.maxabs==0,"warp MMA structural/alias/fallback test failed");
    };
    for(auto pattern:{Pattern::Integers,Pattern::Zero,Pattern::Identity,Pattern::Cancellation})
      exact(problem(128,128,128,dtype),pattern);
    if(variant>=7) {
      exact(problem(256,256,256,dtype),Pattern::Integers);
      auto full=problem(128,128,128,dtype);full.alpha=.5f;full.beta=-.25f;
      full.lda+=8;full.ldb+=8;full.ldc+=8;full.ldd+=8;
      exact(full,Pattern::Integers,true,true,true,true);
      if(variant==10 || variant==13 || variant==14 || variant==15 || variant==16 || variant==17 || variant==18 || variant==19 || variant==20 || variant==21 || variant==22 || variant==23 || variant==24 || variant==25 || variant==26 || variant==27) {
        auto output_padded=problem(128,128,128,dtype);output_padded.alpha=.5f;output_padded.beta=-.25f;
        output_padded.ldc+=1;output_padded.ldd+=1;
        exact(output_padded,Pattern::Integers,true,true,true,true);
      }
      if(variant==14 || variant==15 || variant==16 || variant==17 || variant==18 || variant==19 || variant==20 || variant==21 || variant==22 || variant==23 || variant==24 || variant==25 || variant==26 || variant==27) {
        auto output_padded=problem(128,128,128,dtype);output_padded.alpha=.5f;output_padded.beta=-.25f;
        output_padded.ldd+=1;
        exact(output_padded,Pattern::Integers);
        output_padded=problem(128,128,128,dtype);output_padded.alpha=.5f;output_padded.beta=-.25f;
        output_padded.ldc+=1;
        exact(output_padded,Pattern::Integers);
        output_padded=problem(128,128,128,dtype);output_padded.ldc+=1;
        exact(output_padded,Pattern::Integers);
      }
      if(variant==15 || variant==16 || variant==19 || variant==20 || variant==21 || variant==22 || variant==23 || variant==24 || variant==25 || variant==26 || variant==27) {
        auto identity=problem(128,128,128,dtype);
        identity.lda+=8;identity.ldb+=8;identity.ldc+=8;identity.ldd+=8;
        exact(identity,Pattern::Integers,false,true,true,true);
        auto alpha_only=problem(128,128,128,dtype);alpha_only.alpha=.5f;
        exact(alpha_only,Pattern::Integers);
        auto negative_zero=problem(128,128,128,dtype);negative_zero.beta=-0.0f;negative_zero.ldc+=1;
        exact(negative_zero,Pattern::Integers);
      }
    }
    // Partial output/K tiles, including a final CTA with inactive owner warps.
    exact(problem(80,80,48,dtype),Pattern::Integers);
    auto p=problem(80,112,48,dtype);
    exact(p,Pattern::Integers);
    p.alpha=.5f;p.beta=-.25f;
    exact(p,Pattern::Integers);
    exact(p,Pattern::Integers,true,true,true,true);
    p.lda+=8;p.ldb+=8;p.ldc+=8;p.ldd+=8;
    exact(p,Pattern::Integers);
    exact(p,Pattern::Integers,true);
    exact(problem(128,128,128,dtype),Pattern::Integers,false,false,false);
    exact(problem(65,97,33,dtype),Pattern::Integers,false,true,false);
    for(auto dims:{std::vector<int>{130,128,128},std::vector<int>{128,130,128},std::vector<int>{128,128,130}}) {
      p=problem(dims[0],dims[1],dims[2],dtype);p.lda=(p.lda+7)/8*8;p.ldb=(p.ldb+7)/8*8;
      exact(p,Pattern::Integers,false,true,false);
    }
    p=problem(128,128,128,dtype);p.lda+=1;
    exact(p,Pattern::Integers,false,true,false);
    p=problem(128,128,128,dtype);p.ldb+=1;
    exact(p,Pattern::Integers,false,true,false);
    p=problem(128,128,128,dtype);p.trans_a=true;
    exact(p,Pattern::Integers,false,true,false);
    p=problem(128,128,128,dtype);p.trans_b=true;
    exact(p,Pattern::Integers,false,true,false);
  }
}
template<class T> void prefetch_short_k_suite(megafmm::DType dtype) {
  // Half-K holds one, two, or three MMA tiles: the next-bank prefetch executes
  // zero, one, or two times, including the final iteration that must not load.
  for(auto mn:{std::pair<int,int>{128,128},std::pair<int,int>{32,16},std::pair<int,int>{32,48}})
  for(int k:{32,64,96}) {
    auto p=problem(mn.first,mn.second,k,dtype);
    std::cout << "BEGIN_PREFETCH_SHORT_K dtype=" << megafmm::dtype_name(dtype)
              << " variant=16 shape=" << p.m << ',' << p.n << ',' << p.k << std::endl;
    auto e=run_case<T>(p,megafmm::Algorithm::Megakernel,Pattern::Integers,false,true,
                       1234567,16,false,0,"megakernel_warp_mma_16x8_prefetch");
    require(e.nonfinite==0 && e.maxabs==0,"prefetch short-K final-bank test failed");
  }
}
template<class T> void cpasync_short_k_suite(megafmm::DType dtype) {
  // One, two, and three K16 stages exercise first publish, bank reuse, and
  // the final iteration without a speculative copy beyond the original K.
  for(auto mn:{std::pair<int,int>{128,128},std::pair<int,int>{32,32}})
  for(int k:{32,64,96}) {
    auto p=problem(mn.first,mn.second,k,dtype);
    std::cout << "BEGIN_CPASYNC_SHORT_K dtype=" << megafmm::dtype_name(dtype)
              << " variant=17 shape=" << p.m << ',' << p.n << ',' << p.k << std::endl;
    auto e=run_case<T>(p,megafmm::Algorithm::Megakernel,Pattern::Integers,false,true,
                       1234567,17,false,0,"megakernel_warp_mma_cpasync");
    require(e.nonfinite==0 && e.maxabs==0,"cp.async first/last-bank short-K output incorrect");
  }
}
template<class T> void prefetch_8x8_suite(megafmm::DType dtype) {
  // Distinct logical rows 0..7 and 8..15 expose skipped/repeated owners while
  // physical MMA rows 8..15 are zero padding. Distinct columns expose both
  // stored slots. Stage weights exercise zero/one/two next-bank prefetches.
  for(auto mn:{std::pair<int,int>{32,32},std::pair<int,int>{128,128}})
  for(int k:{32,64,96}) {
    auto p=problem(mn.first,mn.second,k,dtype);
    if(k==64) {p.beta=-0.0f;p.ldc+=1;}
    if(k==96) {p.lda+=8;p.ldb+=8;p.ldd+=8;}
    std::cout << "BEGIN_8X8_PREFETCH dtype=" << megafmm::dtype_name(dtype)
              << " variant=19 shape=" << p.m << ',' << p.n << ',' << p.k
              << " beta=" << p.beta << " null_c0=1" << std::endl;
    auto e=run_case<T>(p,megafmm::Algorithm::Megakernel,Pattern::OwnerRows,false,true,
                       1234567,19,false,0,"megakernel_warp_mma_8x8_prefetch");
    require(e.nonfinite==0 && e.maxabs==0,"8x8 logical owner row/column or next-bank output incorrect");
  }
  for(int n:{16,48}) {
    auto p=problem(32,n,64,dtype);p.beta=-0.0f;p.ldc+=1;
    auto e=run_case<T>(p,megafmm::Algorithm::Megakernel,Pattern::OwnerRows,false,true,
                       1234567,19,false,0,"megakernel_warp_mma_16x8_prefetch");
    require(e.nonfinite==0 && e.maxabs==0,"8x8 narrow-N identity fallback output incorrect");
  }
}
template<class T> void stage_m2_suite(megafmm::DType dtype) {
  // One/two/three independent K16 products must each be reset before their
  // fourteen contributions update the long C state. Nonsymmetric blocks and
  // unequal stage weights expose swapped products, signs and repeated stages.
  for(auto mn:{std::pair<int,int>{128,128},std::pair<int,int>{32,32}})
  for(int k:{32,64,96}) {
    auto p=problem(mn.first,mn.second,k,dtype);
    const auto pattern=mn.first==32 ? Pattern::M2BlockBasis:Pattern::Integers;
    std::cout << "BEGIN_STAGE_M2 dtype=" << megafmm::dtype_name(dtype)
              << " variant=18 shape=" << p.m << ',' << p.n << ',' << p.k
              << " pattern=" << pattern_name(pattern) << std::endl;
    auto e=run_case<T>(p,megafmm::Algorithm::Megakernel,pattern,false,true,
                       1234567,18,false,0,"megakernel_stage_m2");
    require(e.nonfinite==0 && e.maxabs==0,"M2 stage-product reset/contribution output incorrect");
  }
}
template<class T> void direct_register_suite(megafmm::DType dtype,int variant) {
  require(variant>=20 && variant<=25,"register suite needs Mega20..25");
  auto exact=[dtype,variant](megafmm::Problem p,bool alias=false) {
    std::cout << (variant>=24 ? "BEGIN_K128_REGISTER dtype=":variant>=22 ? "BEGIN_COALESCED_REGISTER dtype=":"BEGIN_DIRECT_REGISTER dtype=") << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
              << " expected=" << warp_mma_name(variant,p) << std::endl;
    auto e=run_case<T>(p,megafmm::Algorithm::Megakernel,Pattern::OwnerRows,alias,true,
                       1234567,variant,false,0,warp_mma_name(variant,p));
    require(e.nonfinite==0 && e.maxabs==0,"direct-register owner/stage/identity fallback output incorrect");
  };
  // Different logical eight-row owners, columns, and K16 stage weights test
  // both halves of M16 and the zero-padded physical lower rows of M8. K128
  // uses four distinct stage weights; K96/K160 retain the original19 route.
  for(auto mn:{std::pair<int,int>{32,32},std::pair<int,int>{128,128}})
  for(int k:(variant>=24 ? std::vector<int>{96,128,160}:std::vector<int>{32,64,96})) {
    auto p=problem(mn.first,mn.second,k,dtype);
    if(k==64 || (variant>=24 && k==128)) {p.beta=-0.0f;p.ldc+=1;}
    if(k==96 || (variant>=24 && k==128)) {p.lda+=8;p.ldb+=8;p.ldd+=8;}
    exact(p);
  }
  auto odd_d=problem(128,128,variant>=24 ? 128:64,dtype);odd_d.ldd+=1;
  exact(odd_d);
  auto active_beta=problem(128,128,variant>=24 ? 128:64,dtype);active_beta.beta=-.25f;
  exact(active_beta,true);
}
template<class T> void direct_native_bitwise_suite(megafmm::DType dtype,int variant) {
  require(variant>=20 && variant<=25,"native bitwise suite needs Mega20..25");
  const int reference_variant=(variant==20 || variant==22) ? 15:19;
  const std::vector<std::pair<int,int>> shapes=variant>=24 ?
      std::vector<std::pair<int,int>>{{32,32},{128,128},{128,256}}:
      std::vector<std::pair<int,int>>{{128,128},{256,256},{128,256}};
  for(auto mn:shapes)
  for(int k:(variant>=24 ? std::vector<int>{128}:std::vector<int>{32,64,96}))
  for(int layout=0;layout<4;++layout) {
    // Cross both address phases with packed/padded strides. Both signed zeros
    // are used at each shape/K, always with a null C0 and identity alpha.
    const int pad=layout>=2 ? 8:0,shift=(layout&1) ? 8:0;
    auto p=problem(mn.first,mn.second,k,dtype);
    p.lda+=pad;p.ldb+=pad;p.ldd+=pad;
    p.beta=(layout==1 || layout==2) ? -0.0f:0.0f;
    uint32_t seed=775533u+uint32_t(p.m*31+p.n*17+p.k*7+layout*131+variant*137)+
                  (dtype==megafmm::DType::BF16 ? 0u:918273u);
    std::mt19937 rng(seed);
    std::vector<T> a(size_t(p.m)*p.lda,round_value<T>(3.25f)),
                   b(size_t(p.k)*p.ldb,round_value<T>(-2.75f));
    auto random_input=[&]() {return round_value<T>(2.0f*(float(rng()>>8)/16777216.0f)-1.0f);};
    for(int i=0;i<p.m;++i) for(int l=0;l<p.k;++l)a[size_t(i)*p.lda+l]=random_input();
    for(int l=0;l<p.k;++l) for(int j=0;j<p.n;++j)b[size_t(l)*p.ldb+j]=random_input();
    // Original rounded inputs are copied unchanged to both implementations.
    // Extra eight elements allow a genuinely 16B-compatible rebound address.
    std::vector<T> a_storage(a.size()+8,round_value<T>(5.5f)),
                   b_storage(b.size()+8,round_value<T>(-6.5f));
    std::copy(a.begin(),a.end(),a_storage.begin()+shift);
    std::copy(b.begin(),b.end(),b_storage.begin()+shift);
    const size_t output_count=size_t(p.m)*p.ldd;
    std::vector<float> initial(output_count+2),actual(initial.size()),baseline(initial.size());
    for(size_t at=0;at<initial.size();++at)
      initial[at]=float(int((at+seed)%1024)-512)/8.0f;
    for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
      initial[size_t(i)*p.ldd+j]=std::numeric_limits<float>::quiet_NaN();
    Buffer da(a_storage.size()*sizeof(T)),db(b_storage.size()*sizeof(T)),
           dd(initial.size()*sizeof(float));
    const T* left=da.as<T>()+shift,*right=db.as<T>()+shift;
    std::cout << (variant>=24 ? "BEGIN_K128_NATIVE_BITWISE dtype=":variant>=22 ? "BEGIN_COALESCED_NATIVE_BITWISE dtype=":"BEGIN_DIRECT_NATIVE_BITWISE dtype=") << megafmm::dtype_name(dtype)
              << " variant=" << variant << " reference_variant=" << reference_variant
              << " shape=" << p.m << ',' << p.n << ',' << p.k << " seed=" << seed
              << " pad=" << pad << " input_shift_bytes=" << shift*sizeof(T)
              << " beta_signbit=" << std::signbit(p.beta) << " null_c0=1" << std::endl;
    for(int current:{reference_variant,variant}) {
      megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
      policy.certified_range=true;policy.mega_variant=current;policy.workspace_limit=0;
      auto plan=megafmm::create_plan(p,policy);
      const char* expected=current==variant ?
          warp_mma_name(variant,p):
          (reference_variant==15 ? "megakernel_warp_mma_16x8_identity":"megakernel_warp_mma_8x8_prefetch");
      require(plan->algorithm_name()==expected,"native bitwise fixture selected a fallback route");
      require(plan->workspace_bytes()==0,"native bitwise route unexpectedly requires workspace");
      check(cudaMemcpy(da.data,a_storage.data(),a_storage.size()*sizeof(T),cudaMemcpyHostToDevice));
      check(cudaMemcpy(db.data,b_storage.data(),b_storage.size()*sizeof(T),cudaMemcpyHostToDevice));
      check(cudaMemcpy(dd.data,initial.data(),initial.size()*sizeof(float),cudaMemcpyHostToDevice));
      require(plan->lt_algorithm_config(left,right,nullptr,dd.as<float>(),nullptr)=="none",
              "native bitwise metadata not none");
      megafmm::run(*plan,left,right,nullptr,dd.as<float>(),nullptr);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),dd.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i) {
        for(int j=0;j<p.n;++j)
          require(std::isfinite(actual[size_t(i)*p.ldd+j]),"native bitwise returned a nonfinite/unwritten output");
        for(int j=int(p.n);j<p.ldd;++j)
          require(actual[size_t(i)*p.ldd+j]==initial[size_t(i)*p.ldd+j],"native bitwise overwrote D padding");
      }
      for(size_t at=output_count;at<actual.size();++at)
        require(actual[at]==initial[at],"native bitwise overwrote the trailing D guard");
      if(current==reference_variant)baseline=actual;
      else for(int i=0;i<p.m;++i)
        require(std::memcmp(baseline.data()+size_t(i)*p.ldd,actual.data()+size_t(i)*p.ldd,
                            size_t(p.n)*sizeof(float))==0,
                "register native output bits differ from the matching M16/M8 shared-operand implementation");
    }
    std::cout << (variant>=24 ? "PASS_K128_NATIVE_BITWISE dtype=":variant>=22 ? "PASS_COALESCED_NATIVE_BITWISE dtype=":"PASS_DIRECT_NATIVE_BITWISE dtype=") << megafmm::dtype_name(dtype)
              << " variant=" << variant << " reference_variant=" << reference_variant
              << " shape=" << p.m << ',' << p.n << ',' << p.k << " pad=" << pad
              << " input_shift_bytes=" << shift*sizeof(T) << " beta_signbit=" << std::signbit(p.beta)
              << " null_c0=1 padding_guard=1\n";
  }
}
template<class T> void cooperative_suite(megafmm::DType dtype,int variant) {
  require(variant==26 || variant==27,"cooperative suite needs Mega26/27");
  auto exact=[dtype,variant](megafmm::Problem p,Pattern pattern,bool alias=false) {
    std::cout << "BEGIN_COOPERATIVE dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
              << " null_c0=" << (p.beta==0.0f) << " expected=" << warp_mma_name(variant,p) << std::endl;
    auto e=run_case<T>(p,megafmm::Algorithm::Megakernel,pattern,alias,true,
                       1234567,variant,false,0,warp_mma_name(variant,p));
    require(e.nonfinite==0 && e.maxabs==0,"cooperative native/packed fallback output incorrect");
  };
  for(int k:{32,64,96})
    exact(problem(64,variant==26 ? 32:64,k,dtype),Pattern::OwnerRows);
  // The same M64/N32 geometry is native26 and packed27; N64 is native both.
  exact(problem(64,variant==26 ? 64:32,64,dtype),Pattern::Integers);
  exact(problem(512,512,32,dtype),Pattern::Integers);
  auto padded=problem(128,256,64,dtype);padded.alpha=.5f;padded.beta=-.25f;
  padded.lda+=8;padded.ldb+=8;padded.ldc+=1;padded.ldd+=1;
  exact(padded,Pattern::Integers,true);
  auto null_c0=problem(128,128,96,dtype);null_c0.beta=-0.0f;
  null_c0.ldc+=1;null_c0.ldd+=1;
  exact(null_c0,Pattern::OwnerRows);
  auto tail=problem(80,112,48,dtype);tail.alpha=.5f;tail.beta=-.25f;
  exact(tail,Pattern::Integers,true);
}
template<class T> void cooperative_native_bitwise_suite(megafmm::DType dtype,int variant) {
  require(variant==26 || variant==27,"cooperative native bits needs Mega26/27");
  for(auto mn:{std::pair<int,int>{64,variant==26 ? 32:64},std::pair<int,int>{128,256},
               std::pair<int,int>{512,512}})
  for(int k:{32,64,96}) {
    auto p=problem(mn.first,mn.second,k,dtype);
    const bool alias=k==64;
    const int input_shift=k==96 ? 8:0,output_shift=k==96 ? 1:0;
    if(k!=32) {p.lda+=8;p.ldb+=8;p.ldc+=1;p.ldd+=1;p.alpha=.375f;p.beta=-.625f;}
    if(k==32 && mn.first==64)p.beta=-0.0f;
    uint32_t seed=663311u+uint32_t(p.m*31+p.n*17+p.k*7+variant*137)+
                  (dtype==megafmm::DType::BF16 ? 0u:918273u);
    std::mt19937 rng(seed);
    std::vector<T> a(size_t(p.m)*p.lda,round_value<T>(3.25f)),
                   b(size_t(p.k)*p.ldb,round_value<T>(-2.75f));
    auto random_input=[&]() {return round_value<T>(2.0f*(float(rng()>>8)/16777216.0f)-1.0f);};
    for(int i=0;i<p.m;++i) for(int l=0;l<p.k;++l)a[size_t(i)*p.lda+l]=random_input();
    for(int l=0;l<p.k;++l) for(int j=0;j<p.n;++j)b[size_t(l)*p.ldb+j]=random_input();
    std::vector<T> a_storage(a.size()+8,round_value<T>(5.5f)),
                   b_storage(b.size()+8,round_value<T>(-6.5f));
    std::copy(a.begin(),a.end(),a_storage.begin()+input_shift);
    std::copy(b.begin(),b.end(),b_storage.begin()+input_shift);
    const size_t output_count=size_t(p.m)*p.ldd;
    std::vector<float> initial(output_count+2),actual(initial.size()),baseline(initial.size());
    for(size_t at=0;at<initial.size();++at)initial[at]=float(int((at+seed)%1024)-512)/8.0f;
    auto c=initial;
    for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j) {
      initial[size_t(i)*p.ldd+j]=std::numeric_limits<float>::quiet_NaN();
      c[size_t(i)*p.ldc+j]=float(int(rng()%257)-128)/64.0f;
    }
    Buffer da(a_storage.size()*sizeof(T)),db(b_storage.size()*sizeof(T)),
           dc((c.size()+1)*sizeof(float)),dd((initial.size()+1)*sizeof(float));
    const T* left=da.as<T>()+input_shift,*right=db.as<T>()+input_shift;
    float* source=dc.as<float>()+output_shift;
    float* output=alias ? source:dd.as<float>()+output_shift;
    std::cout << "BEGIN_COOPERATIVE_NATIVE_BITWISE dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " reference_variant=7 shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
              << " null_c0=" << (p.beta==0.0f) << " input_shift_bytes=" << input_shift*sizeof(T)
              << " output_shift_bytes=" << output_shift*sizeof(float) << " seed=" << seed << std::endl;
    for(int current:{7,variant}) {
      megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
      policy.certified_range=true;policy.mega_variant=current;policy.workspace_limit=0;
      auto plan=megafmm::create_plan(p,policy);
      require(plan->algorithm_name()==warp_mma_name(current,p),"cooperative bits selected another route");
      require(plan->workspace_bytes()==0,"cooperative bits unexpectedly requires workspace");
      check(cudaMemcpy(da.data,a_storage.data(),a_storage.size()*sizeof(T),cudaMemcpyHostToDevice));
      check(cudaMemcpy(db.data,b_storage.data(),b_storage.size()*sizeof(T),cudaMemcpyHostToDevice));
      check(cudaMemcpy(source,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
      if(!alias)check(cudaMemcpy(output,initial.data(),initial.size()*sizeof(float),cudaMemcpyHostToDevice));
      megafmm::run(*plan,left,right,p.beta==0.0f ? nullptr:source,output,nullptr);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i) {
        for(int j=0;j<p.n;++j)
          require(std::isfinite(actual[size_t(i)*p.ldd+j]),"cooperative bits returned a nonfinite/unwritten output");
        for(int j=int(p.n);j<p.ldd;++j)
          require(actual[size_t(i)*p.ldd+j]==initial[size_t(i)*p.ldd+j],"cooperative bits overwrote D padding");
      }
      for(size_t at=output_count;at<actual.size();++at)
        require(actual[at]==initial[at],"cooperative bits overwrote the trailing D guard");
      if(current==7)baseline=actual;
      else for(int i=0;i<p.m;++i)
        require(std::memcmp(baseline.data()+size_t(i)*p.ldd,actual.data()+size_t(i)*p.ldd,
                            size_t(p.n)*sizeof(float))==0,"cooperative native bits differ from Mega7");
    }
    std::cout << "PASS_COOPERATIVE_NATIVE_BITWISE dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " reference_variant=7 shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
              << " null_c0=" << (p.beta==0.0f) << " input_shift_bytes=" << input_shift*sizeof(T)
              << " output_shift_bytes=" << output_shift*sizeof(float) << " padding_guard=1\n";
  }
}
template<class T> void warp_mma_bitwise_suite(megafmm::DType dtype) {
  // M1 requires complete-P accumulation followed by the fixed U DAG. Mega18
  // uses separately rounded stage contributions and is intentionally excluded.
  for(auto dims:{std::vector<int>{128,128,128},std::vector<int>{80,112,48},
                std::vector<int>{32,16,32},std::vector<int>{32,48,32}})
  for(int scenario=0;scenario<3;++scenario) {
    auto p=problem(dims[0],dims[1],dims[2],dtype);
    bool alias=scenario==2;
    if(scenario!=0) {p.alpha=.375f;p.beta=-.625f;}
    uint32_t seed=424242u+uint32_t(p.m*31+p.n*17+p.k*7+scenario*131)+
                  (dtype==megafmm::DType::BF16 ? 0u:918273u);
    std::mt19937 rng(seed);
    std::vector<T> a(size_t(p.m)*p.lda),b(size_t(p.k)*p.ldb);
    std::vector<float> c(size_t(p.m)*p.ldc),actual(size_t(p.m)*p.ldd),baseline(actual.size());
    // Fresh exact dyadic inputs retain nontrivial rounding in transformed S/T.
    for(auto& value:a)value=round_value<T>(float(int(rng()%2049)-1024)/1024.0f);
    for(auto& value:b)value=round_value<T>(float(int(rng()%2049)-1024)/1024.0f);
    for(auto& value:c)value=float(int(rng()%257)-128)/64.0f;
    Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dc(c.size()*sizeof(float)),
           dd(actual.size()*sizeof(float));
    check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
    std::cout << "BEGIN_MMA_BITWISE dtype=" << megafmm::dtype_name(dtype)
              << " shape=" << p.m << ',' << p.n << ',' << p.k << " alpha=" << p.alpha
              << " beta=" << p.beta << " alias=" << alias << " seed=" << seed << std::endl;
    for(int variant:{4,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27}) {
      megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
      policy.certified_range=true;policy.mega_variant=variant;
      auto plan=megafmm::create_plan(p,policy);
      require(plan->algorithm_name()==warp_mma_name(variant,p),"MMA bitwise test selected another route");
      require(plan->workspace_bytes()==0,"MMA bitwise test unexpectedly requires workspace");
      check(cudaMemcpy(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
      if(!alias)check(cudaMemset(dd.data,0xff,actual.size()*sizeof(float)));
      float* output=alias ? dc.as<float>():dd.as<float>();
      megafmm::run(*plan,da.data,db.data,p.beta==0 ? nullptr:dc.as<float>(),output,nullptr);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(float value:actual)require(std::isfinite(value),"MMA bitwise test returned nonfinite output");
      if(variant==4)baseline=actual;
      else require(std::memcmp(baseline.data(),actual.data(),actual.size()*sizeof(float))==0,
                   "Mega4 vs Mega"+std::to_string(variant)+" output bits differ on fresh dyadic inputs");
    }
    std::cout << "PASS_MMA_BITWISE dtype=" << megafmm::dtype_name(dtype)
              << " shape=" << p.m << ',' << p.n << ',' << p.k << " alpha=" << p.alpha
              << " beta=" << p.beta << " alias=" << alias << " variants=4,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27\n";
  }
}
template<class T> void cluster_graph_suite(megafmm::DType dtype,int variant=13) {
  require(variant==13 || variant==17 || variant==18 || variant==19 || variant==20 || variant==21 || variant==22 || variant==23 || variant==24 || variant==25 || variant==26 || variant==27,
          "graph suite needs cluster13, cpasync17, stage18, 8x8prefetch19, register20..25 or cooperative26/27");
  for(auto dims:{std::vector<int>{32,32,32},std::vector<int>{96,96,96},std::vector<int>{80,112,48}}) {
    if(variant==26 || variant==27) {
      if(dims[0]==32)dims={64,variant==26 ? 32:64,32};
      else if(dims[0]==96)dims={128,128,96};
    }
    if((variant==24 || variant==25) && dims[0]%32==0 && dims[1]%32==0)dims[2]=128;
    auto p=problem(dims[0],dims[1],dims[2],dtype);p.alpha=.5f;p.beta=-.25f;
    if(variant>=19 && variant<=25) {p.alpha=1.0f;p.beta=-0.0f;}
    p.lda+=8;p.ldb+=8;p.ldc+=8;p.ldd+=8;
    if(variant==26 || variant==27) {p.ldc+=1;p.ldd+=1;}
    std::mt19937 rng(998877u+uint32_t(p.m*31+p.n*17+p.k*7));
    std::vector<T> a(size_t(p.m)*p.lda),b(size_t(p.k)*p.ldb);
    std::vector<float> c(size_t(p.m)*p.ldc,-777.0f),actual(c.size());
    for(int i=0;i<p.m;++i) for(int l=0;l<p.k;++l)
      a[size_t(i)*p.lda+l]=round_value<T>(float(int(rng()%5)-2));
    for(int l=0;l<p.k;++l) for(int j=0;j<p.n;++j)
      b[size_t(l)*p.ldb+j]=round_value<T>(float(int(rng()%5)-2));
    std::vector<double> reference_ab(size_t(p.m)*p.n),stage_ab;
    Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dc(c.size()*sizeof(float));
    check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
    check(cudaMemcpy(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaDeviceSynchronize());
    megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
    policy.certified_range=true;policy.mega_variant=variant;
    auto plan=megafmm::create_plan(p,policy);
    require(plan->algorithm_name()==warp_mma_name(variant,p),"megakernel graph selected another route");
    require(plan->workspace_bytes()==0,"megakernel graph unexpectedly requires workspace");
    std::cout << (variant==13 ? "BEGIN_CLUSTER_GRAPH dtype=":variant==17 ? "BEGIN_CPASYNC_GRAPH dtype=":variant==18 ? "BEGIN_STAGE_M2_GRAPH dtype=":variant==19 ? "BEGIN_8X8_PREFETCH_GRAPH dtype=":variant>=26 ? "BEGIN_COOPERATIVE_GRAPH dtype=":variant>=24 ? "BEGIN_K128_REGISTER_GRAPH dtype=":variant>=22 ? "BEGIN_COALESCED_REGISTER_GRAPH dtype=":"BEGIN_DIRECT_REGISTER_GRAPH dtype=") << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k << " actual=" << plan->algorithm_name()
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << (p.beta!=0.0f)
              << " null_c0=" << (p.beta==0.0f) << " refresh_ab=1" << std::endl;
    Stream stream;Graph graph;
    check(cudaStreamSynchronize(stream.value));
    check(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));
    megafmm::run(*plan,da.data,db.data,p.beta==0 ? nullptr:dc.as<float>(),dc.as<float>(),nullptr,stream.value);
    check(cudaStreamEndCapture(stream.value,&graph.value));
    check(cudaGraphInstantiate(&graph.executable,graph.value,0));
    for(int replay=0;replay<3;++replay) {
      for(int i=0;i<p.m;++i) for(int l=0;l<p.k;++l)
        a[size_t(i)*p.lda+l]=round_value<T>(float(int(rng()%5)-2));
      for(int l=0;l<p.k;++l) for(int j=0;j<p.n;++j)
        b[size_t(l)*p.ldb+j]=round_value<T>(float(int(rng()%5)-2));
      // Recompute FP64 plain AB from this replay's original rounded inputs.
      std::fill(reference_ab.begin(),reference_ab.end(),0.0);
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
        for(int l=0;l<p.k;++l)
          reference_ab[size_t(i)*p.n+j]+=double_value(a[size_t(i)*p.lda+l])*double_value(b[size_t(l)*p.ldb+j]);
      if(variant==18) {
        stage_ab=m2_integer_reference(p,a,b);
        require(stage_ab==reference_ab,"M2 graph stage-expanded reference disagrees with fresh FP64 AB");
      }
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
        c[size_t(i)*p.ldc+j]=float((i+j+3*replay)%9-4)/4.0f;
      // Update the same captured A/B addresses and restore the immutable host
      // C0 snapshot on the launch stream. Readback goes into actual, never c.
      check(cudaMemcpyAsync(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice,stream.value));
      check(cudaMemcpyAsync(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice,stream.value));
      check(cudaMemcpyAsync(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice,stream.value));
      check(cudaGraphLaunch(graph.executable,stream.value));
      check(cudaStreamSynchronize(stream.value));
      check(cudaMemcpy(actual.data(),dc.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j) {
        double reference=double(p.alpha)*reference_ab[size_t(i)*p.n+j]+double(p.beta)*c[size_t(i)*p.ldc+j];
        require(double(actual[size_t(i)*p.ldd+j])==reference,"megakernel graph replay alpha/beta output incorrect");
        if(variant==18)
          require(double(actual[size_t(i)*p.ldd+j])==double(p.alpha)*stage_ab[size_t(i)*p.n+j]+double(p.beta)*c[size_t(i)*p.ldc+j],
                  "M2 graph replay disagrees with independent stage-expanded reference");
      }
      for(int i=0;i<p.m;++i) for(int j=p.n;j<p.ldd;++j)
        require(actual[size_t(i)*p.ldd+j]==-777.0f,"megakernel graph replay overwrote output padding");
    }
    std::cout << (variant==13 ? "PASS_CLUSTER_GRAPH dtype=":variant==17 ? "PASS_CPASYNC_GRAPH dtype=":variant==18 ? "PASS_STAGE_M2_GRAPH dtype=":variant==19 ? "PASS_8X8_PREFETCH_GRAPH dtype=":variant>=26 ? "PASS_COOPERATIVE_GRAPH dtype=":variant>=24 ? "PASS_K128_REGISTER_GRAPH dtype=":variant>=22 ? "PASS_COALESCED_REGISTER_GRAPH dtype=":"PASS_DIRECT_REGISTER_GRAPH dtype=") << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k << " actual=" << plan->algorithm_name()
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << (p.beta!=0.0f)
              << " null_c0=" << (p.beta==0.0f) << " refresh_ab=1 replays=3\n";
  }
}
template<class T> void epilogue_output_rebind_suite(megafmm::DType dtype,int variant=14) {
  for(bool identity:{false,true}) {
  if(identity && variant!=15 && variant!=16 && variant!=19 && variant!=20 && variant!=21 && variant!=22 && variant!=23 && variant!=24 && variant!=25 && variant!=26 && variant!=27)continue;
  auto p=problem(128,128,128,dtype);
  if(!identity) {p.alpha=.5f;p.beta=-.25f;}
  std::mt19937 rng(776655u);
  std::vector<T> a(size_t(p.m)*p.lda),b(size_t(p.k)*p.ldb);
  std::vector<float> c(size_t(p.m)*p.ldc),actual(c.size());
  for(auto& value:a)value=round_value<T>(float(int(rng()%5)-2));
  for(auto& value:b)value=round_value<T>(float(int(rng()%5)-2));
  for(auto& value:c)value=float(int(rng()%9)-4)/4.0f;
  std::vector<double> reference(size_t(p.m)*p.n);
  for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j) {
    double sum=0;
    for(int l=0;l<p.k;++l)
      sum+=double_value(a[size_t(i)*p.lda+l])*double_value(b[size_t(l)*p.ldb+j]);
    reference[size_t(i)*p.n+j]=double(p.alpha)*sum+double(p.beta)*c[size_t(i)*p.ldc+j];
  }
  Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dc((c.size()+1)*sizeof(float)),
         dd((actual.size()+1)*sizeof(float));
  check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
  check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
  policy.certified_range=true;policy.mega_variant=variant;
  auto plan=megafmm::create_plan(p,policy);
  require(plan->algorithm_name()==warp_mma_name(variant,p),"output rebind selected another route");
  require(plan->workspace_bytes()==0,"output rebind unexpectedly requires workspace");
  for(int mode=0;mode<3;++mode) {
    if(identity && mode==1)continue;
    std::cout << "BEGIN_EPILOGUE_OUTPUT_REBIND dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " alpha=" << p.alpha << " beta=" << p.beta
              << " mode=" << mode << " alias=" << (mode==2) << std::endl;
    for(int shift:{0,1,0}) {
      // D only, active C0 only, and exact C0=D each lose 8-byte alignment.
      float* source=dc.as<float>()+(mode==0 ? 0:shift);
      float* output=mode==2 ? source:dd.as<float>()+(mode==0 ? shift:0);
      check(cudaMemset(dc.data,0xff,(c.size()+1)*sizeof(float)));
      check(cudaMemset(dd.data,0xff,(actual.size()+1)*sizeof(float)));
      check(cudaMemcpy(source,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
      megafmm::run(*plan,da.data,db.data,p.beta==0.0f ? nullptr:source,output,nullptr);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(size_t i=0;i<actual.size();++i)
        require(double(actual[i])==reference[i],"same-plan 4-byte output/C0 rebind returned incorrect alpha/beta output");
    }
    std::cout << "PASS_EPILOGUE_OUTPUT_REBIND dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " alpha=" << p.alpha << " beta=" << p.beta
              << " mode=" << mode << " alias=" << (mode==2)
              << " aligned_then_float_aligned_then_aligned=1\n";
  }
  }
}
template<class T> void rebind_suite(megafmm::DType dtype,bool batched_only=false,int hopper_filter=-1,int mega_filter=-1) {
  // Reuse a tuned plan with changed addresses: an aligned preferred SW route must
  // safely choose Dense when the new input views have only element alignment.
  auto p=problem(256,256,(mega_filter==24 || mega_filter==25) ? 128:256,dtype);
  std::vector<T> a(size_t(p.m)*p.k),b(size_t(p.k)*p.n);
  for(int i=0;i<p.m;++i) for(int l=0;l<p.k;++l)
    a[size_t(i)*p.k+l]=round_value<T>(float(i%5-2));
  for(int l=0;l<p.k;++l) for(int j=0;j<p.n;++j)
    b[size_t(l)*p.n+j]=round_value<T>(float(j%7-3));
  Buffer da((a.size()+8)*sizeof(T)),db((b.size()+8)*sizeof(T)),dd(size_t(p.m)*p.n*sizeof(float));
  std::vector<float> actual(size_t(p.m)*p.n);
  for(auto candidate:{std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Dense,0},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradLt,0},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradLt,1},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,0},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,5},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,6},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,7},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,8},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,3},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,4},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,5},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,6},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,7},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,8},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,9},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,10},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,11},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,12},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,13},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,14},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,15},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,16},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,17},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,18},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,19},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,20},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,21},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,22},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,23},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,24},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,25},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,26},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,27}}) {
    auto algo=candidate.first;int variant=candidate.second;
    bool is_batched=(algo==megafmm::Algorithm::WinogradLt && variant==1) ||
                    (algo==megafmm::Algorithm::WinogradHopper && variant>=5);
    if(batched_only && !(algo==megafmm::Algorithm::WinogradLt && variant==1))continue;
    if(hopper_filter>=0 && (algo!=megafmm::Algorithm::WinogradHopper || variant!=hopper_filter))continue;
    if(mega_filter>=0 && (algo!=megafmm::Algorithm::Megakernel || variant!=mega_filter))continue;
    megafmm::Policy policy;policy.algorithm=algo;policy.certified_range=true;
    if(algo==megafmm::Algorithm::WinogradLt)policy.lt_variant=variant;
    if(algo==megafmm::Algorithm::WinogradHopper)policy.hopper_variant=variant;
    if(algo==megafmm::Algorithm::Megakernel)policy.mega_variant=variant;
    auto plan=megafmm::create_plan(p,policy);Buffer workspace(plan->workspace_bytes());
    if(is_batched)require(plan->algorithm_name()==batched_name(algo,variant),"batched rebind test selected Dense");
    if(algo==megafmm::Algorithm::Megakernel && variant>=4) {
      require(plan->algorithm_name()==warp_mma_name(variant,p),"warp MMA rebind test selected another route");
      require(plan->workspace_bytes()==0,"warp MMA rebind unexpectedly requires workspace");
    }
    check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
    megafmm::tune(*plan,da.data,db.data,nullptr,dd.as<float>(),workspace.data);
    const auto original_config=check_lt_config(*plan,da.data,db.data,nullptr,dd.as<float>(),workspace.data,
                                               lt_config_kind(*plan),true);
    for(int shift:{0,1,8,0}) {
      T* left=da.as<T>()+shift,*right=db.as<T>()+shift;
      check(cudaMemcpy(left,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
      check(cudaMemcpy(right,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
      const bool dense_fallback=shift==1 && algo!=megafmm::Algorithm::Dense;
      auto config=check_lt_config(*plan,left,right,nullptr,dd.as<float>(),workspace.data,
                                  dense_fallback ? "dense":lt_config_kind(*plan),!dense_fallback);
      if(shift==0)require(config==original_config,"readonly metadata changed after pointer rebind");
      megafmm::run(*plan,left,right,nullptr,dd.as<float>(),workspace.data);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),dd.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
        require(actual[size_t(i)*p.n+j]==float(p.k*(i%5-2)*(j%7-3)),
                "same-plan pointer alignment rebind returned an incorrect product");
    }
    std::cout << "PASS_POINTER_REBIND dtype=" << megafmm::dtype_name(dtype)
              << " algorithm=" << algorithm_name(algo) << " variant=" << variant
              << " aligned_then_element_aligned_then_aligned=1\n";
  }
}
void batched_policy_rejection_suite(bool lt_only=false) {
  for(auto algorithm:{megafmm::Algorithm::Auto,megafmm::Algorithm::Dense,megafmm::Algorithm::WinogradLt,
                      megafmm::Algorithm::WinogradHopper,megafmm::Algorithm::Megakernel})
  for(int scenario=0;scenario<4;++scenario) for(int family=0;family<3;++family)
  for(int variant:{-1,family==0 ? 2:family==1 ? 9:28}) {
    if(lt_only && family!=0)continue;
    megafmm::Policy policy;policy.algorithm=algorithm;policy.certified_range=true;
    if(family==0)policy.lt_variant=variant;
    else if(family==1)policy.hopper_variant=variant;
    else policy.mega_variant=variant;
    auto p=problem(128,128,128,megafmm::DType::BF16);
    if(scenario==1)p.m=0;
    if(scenario==2)p.alpha=0;
    if(scenario==3)p.k=0;
    bool rejected=false;
    try { auto plan=megafmm::create_plan(p,policy); }
    catch(const std::exception&) { rejected=true; }
    require(rejected,family==0 ? "invalid Lt variant accepted":
                     family==1 ? "invalid Hopper variant accepted":"invalid Mega variant accepted");
  }
}
void degenerate_suite() {
  for(auto algo:{megafmm::Algorithm::Dense,megafmm::Algorithm::WinogradLt,megafmm::Algorithm::WinogradHopper,
                 megafmm::Algorithm::Megakernel,megafmm::Algorithm::Auto}) {
    megafmm::Policy policy;policy.algorithm=algo;policy.certified_range=true;
    for(int zero_dim:{0,1}) {
      auto p=problem(zero_dim==0 ? 0:9,zero_dim==1 ? 0:11,17,megafmm::DType::BF16);
      auto plan=megafmm::create_plan(p,policy);
      require(plan->lt_algorithm_config(nullptr,nullptr,nullptr,nullptr,nullptr)=="none","empty metadata not none");
      megafmm::run(*plan,nullptr,nullptr,nullptr,nullptr,nullptr);
      check(cudaDeviceSynchronize());
    }
    for(bool alpha_zero:{false,true}) for(float beta:{0.0f,.5f,-1.0f}) {
      auto p=problem(7,11,alpha_zero ? 13:0,megafmm::DType::FP16);p.alpha=alpha_zero ? 0:1;p.beta=beta;
      std::vector<float> c(size_t(p.m)*p.n),out(c.size());
      for(size_t i=0;i<c.size();++i)c[i]=float(int(i%9)-4)/4;
      Buffer dc(c.size()*sizeof(float)),dd(c.size()*sizeof(float));
      check(cudaMemcpy(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
      auto plan=megafmm::create_plan(p,policy);Buffer workspace(plan->workspace_bytes());
      require(plan->lt_algorithm_config(nullptr,nullptr,beta==0 ? nullptr:dc.as<float>(),dd.as<float>(),workspace.data)=="none",
              "degenerate metadata not none");
      if(beta!=0) {
        bool invalid=false;
        try {plan->lt_algorithm_config(nullptr,nullptr,nullptr,dd.as<float>(),workspace.data);}
        catch(const std::exception&) {invalid=true;}
        require(invalid,"metadata accepted missing active C0");
      }
      megafmm::run(*plan,nullptr,nullptr,beta==0 ? nullptr:dc.as<float>(),dd.as<float>(),workspace.data);
      check(cudaDeviceSynchronize());check(cudaMemcpy(out.data(),dd.data,out.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(size_t i=0;i<c.size();++i)require(out[i]==beta*c[i],"degenerate beta C0 failed");
    }
  }
}
void rejection_suite() {
  auto rejected=[](megafmm::Problem p) {
    try { auto plan=megafmm::create_plan(p); } catch(const std::exception&) { return true; }
    return false;
  };
  auto p=problem(16,16,16,megafmm::DType::BF16);p.m=-1;require(rejected(p),"negative dimension accepted");
  p=problem(16,16,16,megafmm::DType::BF16);p.lda=15;require(rejected(p),"short lda accepted");
  p=problem(16,16,16,megafmm::DType::BF16);p.ldb=15;require(rejected(p),"short ldb accepted");
  p=problem(16,16,16,megafmm::DType::BF16);p.ldd=15;require(rejected(p),"short ldd accepted");
  p=problem(16,16,16,megafmm::DType::BF16);p.beta=1;p.ldc=15;require(rejected(p),"short ldc accepted");
  p=problem(16,16,16,megafmm::DType::BF16);p.lda=std::numeric_limits<int64_t>::max();require(rejected(p),"address overflow accepted");
  p=problem(16,16,16,megafmm::DType::BF16);
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::WinogradLt;policy.certified_range=true;
  auto plan=megafmm::create_plan(p,policy);
  Buffer memory(16384),workspace(plan->workspace_bytes());
  bool overlap_rejected=false;
  try { megafmm::run(*plan,memory.data,memory.as<char>()+4096,nullptr,memory.as<float>(),workspace.data); }
  catch(const std::exception&) { overlap_rejected=true; }
  require(overlap_rejected,"A/D overlap accepted");
  overlap_rejected=false;
  try {plan->lt_algorithm_config(memory.data,memory.as<char>()+4096,nullptr,memory.as<float>(),workspace.data);}
  catch(const std::exception&) {overlap_rejected=true;}
  require(overlap_rejected,"metadata accepted A/D overlap");
  if(plan->workspace_bytes()) {
    overlap_rejected=false;
    try { megafmm::run(*plan,workspace.data,memory.data,nullptr,memory.as<float>()+2048,workspace.data); }
    catch(const std::exception&) { overlap_rejected=true; }
    require(overlap_rejected,"A/workspace overlap accepted");
  }
  p.beta=1;plan=megafmm::create_plan(p,policy);
  Buffer beta_workspace(plan->workspace_bytes());
  overlap_rejected=false;
  try { megafmm::run(*plan,memory.data,memory.as<char>()+4096,memory.as<float>()+2048,
                     memory.as<float>()+2049,beta_workspace.data); }
  catch(const std::exception&) { overlap_rejected=true; }
  require(overlap_rejected,"partial C0/D overlap accepted");
  p.ldc=17;plan=megafmm::create_plan(p,policy);
  Buffer unequal_stride_workspace(plan->workspace_bytes());
  overlap_rejected=false;
  try { megafmm::run(*plan,memory.data,memory.as<char>()+4096,memory.as<float>()+2048,
                     memory.as<float>()+2048,unequal_stride_workspace.data); }
  catch(const std::exception&) { overlap_rejected=true; }
  require(overlap_rejected,"exact C0/D alias with unequal strides accepted");
  p.ldc=p.ldd;plan=megafmm::create_plan(p,policy);
  Buffer c_overlap_workspace(plan->workspace_bytes());
  if(plan->workspace_bytes()) {
    overlap_rejected=false;
    try { megafmm::run(*plan,memory.data,memory.as<char>()+4096,c_overlap_workspace.as<float>(),
                       memory.as<float>()+2048,c_overlap_workspace.data); }
    catch(const std::exception&) { overlap_rejected=true; }
    require(overlap_rejected,"C0/workspace overlap accepted");
  }
  overlap_rejected=false;
  try {
    const void* overflowing_a=reinterpret_cast<const void*>(std::numeric_limits<uintptr_t>::max()-1);
    megafmm::run(*plan,overflowing_a,memory.as<char>()+4096,memory.as<float>()+1024,
                 memory.as<float>()+2048,c_overlap_workspace.data);
  } catch(const std::exception&) { overlap_rejected=true; }
  require(overlap_rejected,"runtime pointer range overflow accepted");
}
template<class T> bool strict_numerics(megafmm::DType dtype,bool k128_only=false) {
  bool pass=true;double floor=dtype==megafmm::DType::BF16 ? 1e-3:1e-4;
  struct Candidate { megafmm::Algorithm algorithm;int variant; };
  std::vector<Candidate> candidates={{megafmm::Algorithm::WinogradLt,0},{megafmm::Algorithm::WinogradLt,1}};
  for(int variant=0;variant<=8;++variant)candidates.push_back({megafmm::Algorithm::WinogradHopper,variant});
  for(int variant=0;variant<=23;++variant)candidates.push_back({megafmm::Algorithm::Megakernel,variant});
  candidates.push_back({megafmm::Algorithm::Megakernel,26});
  candidates.push_back({megafmm::Algorithm::Megakernel,27});
  if(k128_only)candidates={{megafmm::Algorithm::Megakernel,24},{megafmm::Algorithm::Megakernel,25}};
  for(auto pat:{Pattern::Uniform,Pattern::Positive,Pattern::Range,Pattern::Cancellation,Pattern::HighHalf}) {
    std::vector<std::vector<double>> ef(candidates.size());
    std::vector<double> dense_ef;
    std::vector<size_t> failures(candidates.size(),0),nonfinite(candidates.size(),0);
    // Near the FP16 limit is outside the certified fast-path range and must use fallback.
    bool certified=pat!=Pattern::HighHalf;
    for(int trial=0;trial<3;++trial) {
      // 256 cubed gives variant 3 complete output tiles and therefore exercises
      // its new beta=0 fast kernel rather than only its general fallback.
      auto p=problem(256,256,k128_only ? 128:256,dtype);uint32_t seed=1234567+trial*918273;
      auto dense=run_case<T>(p,megafmm::Algorithm::Dense,pat,false,certified,seed,0,!k128_only);
      dense_ef.push_back(dense.ef);
      for(size_t candidate=0;candidate<candidates.size();++candidate) {
        const auto algo=candidates[candidate].algorithm;
        const int variant=candidates[candidate].variant;
        // Kernel numerical coverage does not require retuning seven leaves for
        // every input pattern. The original group retains tuned Dense; the K128
        // group shares an untuned Dense reference and verifies its actual route.
        const char* expected_route=(k128_only || (algo==megafmm::Algorithm::Megakernel && (variant==26 || variant==27))) ?
            (certified ? warp_mma_name(variant,p):"dense_cublaslt"):nullptr;
        auto sw=run_case<T>(p,algo,pat,false,certified,seed,variant,false,size_t(8)<<30,expected_route);
        bool finite=dense.nonfinite==0 && sw.nonfinite==0;
        bool gate=finite && sw.ef<=3*dense.ef+floor;
        pass=gate && pass;failures[candidate]+=!gate;nonfinite[candidate]+=dense.nonfinite+sw.nonfinite;
        ef[candidate].push_back(sw.ef);
        std::cout << "NUMERICAL_GATE dtype=" << megafmm::dtype_name(dtype)
                  << " algorithm=" << algorithm_name(algo) << " variant=" << variant
                  << " shape=" << p.m << ',' << p.n << ',' << p.k << " k128_only=" << k128_only
                  << " pattern=" << pattern_name(pat)
                  << " trial=" << trial << " certified_range=" << certified << " pass=" << gate
                  << " finite=" << finite << " nonfinite=" << dense.nonfinite+sw.nonfinite
                  << " dense_ef=" << dense.ef << " solution_ef=" << sw.ef
                  << " maxabs=" << sw.maxabs << " scaled=" << sw.scaled
                  << " normalized=" << sw.normalized << " nearzero_abs=" << sw.nearzero
                  << " floor=" << floor << " limit=" << 3*dense.ef+floor << '\n';
      }
    }
    std::sort(dense_ef.begin(),dense_ef.end());
    for(size_t candidate=0;candidate<candidates.size();++candidate) {
      auto& errors=ef[candidate];std::sort(errors.begin(),errors.end());
      std::cout << "NUMERICAL_DISTRIBUTION dtype=" << megafmm::dtype_name(dtype)
                << " algorithm=" << algorithm_name(candidates[candidate].algorithm)
                << " variant=" << candidates[candidate].variant << " pattern=" << pattern_name(pat)
                << " shape=256,256," << (k128_only ? 128:256) << " k128_only=" << k128_only
                << " trials=" << errors.size() << " failures=" << failures[candidate]
                << " nonfinite=" << nonfinite[candidate]
                << " ef_min=" << errors.front() << " ef_median=" << errors[errors.size()/2]
                << " ef_max=" << errors.back() << " dense_ef_median=" << dense_ef[dense_ef.size()/2]
                << " gate_floor=" << floor << '\n';
    }
  }
  return pass;
}
} // namespace
int main(int argc,char** argv) {
  try {
    std::cout << std::unitbuf;std::cerr << std::unitbuf;
    const std::string usage="usage: test_correctness [--strict-numerics] [--smoke-batched | --smoke-candidates [--mega-only] [--mega-variant 0..27 | --hopper-variant 0..8]]";
    bool numeric=false, candidates_only=false,mega_only=false,batched_only=false;
    int hopper_filter=-1,mega_filter=-1;
    for(int i=1;i<argc;++i) {
      if(std::string(argv[i])=="--strict-numerics")numeric=true;
      else if(std::string(argv[i])=="--smoke-batched")batched_only=true;
      else if(std::string(argv[i])=="--smoke-candidates")candidates_only=true;
      else if(std::string(argv[i])=="--mega-only")mega_only=true;
      else if(std::string(argv[i])=="--mega-variant") {
        require(i+1<argc,usage);
        std::string value=argv[++i];size_t consumed=0;
        mega_filter=std::stoi(value,&consumed);
        require(consumed==value.size() && mega_filter>=0 && mega_filter<=27,usage);
      }
      else if(std::string(argv[i])=="--hopper-variant") {
        require(i+1<argc,usage);
        std::string value=argv[++i];size_t consumed=0;
        hopper_filter=std::stoi(value,&consumed);
        require(consumed==value.size() && hopper_filter>=0 && hopper_filter<=8,usage);
      } else throw std::runtime_error(usage);
    }
    require((!mega_only && hopper_filter<0 && mega_filter<0) || candidates_only,
            "candidate filters require --smoke-candidates; "+usage);
    require(!mega_only || hopper_filter<0,"--mega-only and --hopper-variant cannot be combined; "+usage);
    require(mega_filter<0 || hopper_filter<0,"--mega-variant and --hopper-variant cannot be combined; "+usage);
    require(!batched_only || !candidates_only,"--smoke-batched and --smoke-candidates cannot be combined; "+usage);
    if(batched_only) {
      batched_suite<__nv_bfloat16>(megafmm::DType::BF16,true);
      batched_suite<__half>(megafmm::DType::FP16,true);
      rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,true);
      rebind_suite<__half>(megafmm::DType::FP16,true);
      batched_policy_rejection_suite(true);
      std::cout << "PASS batched Lt slot mappings, rectangles, strides, alpha/beta alias, Dense fallback, rebind and policy rejection\n";
      return 0;
    }
    if(candidates_only) {
      auto check_candidates=[mega_only,hopper_filter,mega_filter](auto input_tag,megafmm::DType dtype) {
        using T=decltype(input_tag);
        for(auto candidate:{std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,0},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,1},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,2},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,3},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,4},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,5},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,6},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,7},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,8},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,9},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,10},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,11},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,12},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,13},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,14},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,15},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,16},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,17},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,18},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,19},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,20},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,21},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,22},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,23},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,24},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,25},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,26},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::Megakernel,27},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,0},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,1},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,2},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,3},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,4},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,5},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,6},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,7},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,8}}) {
          if(mega_only && candidate.first!=megafmm::Algorithm::Megakernel)continue;
          if(hopper_filter>=0 && (candidate.first!=megafmm::Algorithm::WinogradHopper ||
                                  candidate.second!=hopper_filter))continue;
          if(mega_filter>=0 && (candidate.first!=megafmm::Algorithm::Megakernel ||
                                candidate.second!=mega_filter))continue;
          bool warp_mma=candidate.first==megafmm::Algorithm::Megakernel && candidate.second>=4;
          for(int scenario=0;scenario<15;++scenario) {
            if(scenario==2 && candidate.first!=megafmm::Algorithm::Megakernel && candidate.second<5)continue;
            if(scenario>=3 && !warp_mma)continue;
            if(scenario==6 && candidate.second<7)continue;
            if(scenario==7 && candidate.second!=10 && candidate.second!=13 && candidate.second!=14 && candidate.second!=15 && candidate.second!=16 && candidate.second!=17 && candidate.second!=18 && candidate.second!=19 && candidate.second!=20 && candidate.second!=21 && candidate.second!=22 && candidate.second!=23 && candidate.second!=24 && candidate.second!=25 && candidate.second!=26 && candidate.second!=27)continue;
            if((scenario==8 || scenario==9) && candidate.second!=11 && candidate.second!=12 && candidate.second!=14 && candidate.second!=15 && candidate.second!=16 && candidate.second!=17 && candidate.second!=18 && candidate.second!=19 && candidate.second!=20 && candidate.second!=21 && candidate.second!=22 && candidate.second!=23 && candidate.second!=24 && candidate.second!=25 && candidate.second!=26 && candidate.second!=27)continue;
            if((scenario==10 || scenario==11) && candidate.second!=13)continue;
            if(scenario>=12 && candidate.second!=14 && candidate.second!=15 && candidate.second!=16 && candidate.second!=17 && candidate.second!=18 && candidate.second!=19 && candidate.second!=20 && candidate.second!=21 && candidate.second!=22 && candidate.second!=23 && candidate.second!=24 && candidate.second!=25 && candidate.second!=26 && candidate.second!=27)continue;
            auto p=scenario==2 ? problem(128,64,256,dtype):problem(256,256,256,dtype);
            if(scenario==3 || scenario==4)p=problem(80,112,48,dtype);
            if(scenario==5)p=problem(80,80,48,dtype);
            if(scenario==6 || scenario==7)p=problem(128,128,128,dtype);
            if(scenario==8)p=problem(32,16,32,dtype);
            if(scenario==9)p=problem(32,48,32,dtype);
            if(scenario==10)p=problem(32,32,32,dtype);
            if(scenario==11)p=problem(96,96,96,dtype);
            if(scenario>=12)p=problem(128,128,128,dtype);
            if(scenario==4 || scenario==6) {p.lda+=8;p.ldb+=8;p.ldc+=8;p.ldd+=8;}
            if(scenario==7) {p.ldc+=1;p.ldd+=1;}
            if(scenario==12 || scenario==14)p.ldc+=1;
            if(scenario==13)p.ldd+=1;
            bool alias=scenario==1 || scenario==4 || scenario==6 || scenario==7 || scenario==9;
            if(alias) {p.alpha=.5f;p.beta=-.25f;}
            if(scenario==12 || scenario==13) {p.alpha=.5f;p.beta=-.25f;}
            std::cout << "BEGIN_CANDIDATE dtype=" << megafmm::dtype_name(dtype)
                      << " algorithm=" << algorithm_name(candidate.first) << " variant=" << candidate.second
                      << " shape=" << p.m << ',' << p.n << ',' << p.k
                      << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias << std::endl;
            megafmm::Policy policy;
            policy.algorithm=candidate.first;policy.certified_range=true;
            if(candidate.first==megafmm::Algorithm::Megakernel)policy.mega_variant=candidate.second;
            if(candidate.first==megafmm::Algorithm::WinogradHopper)policy.hopper_variant=candidate.second;
            auto planned=megafmm::create_plan(p,policy);
            require(planned->algorithm_name().find("dense")==std::string::npos,
                    "supported candidate unexpectedly fell back to Dense");
            const char* route=warp_mma ? warp_mma_name(candidate.second,p):
                              candidate.first==megafmm::Algorithm::WinogradHopper && candidate.second>=5 ?
                                batched_name(candidate.first,candidate.second):nullptr;
            auto e=run_case<T>(p,candidate.first,Pattern::Integers,alias,true,1234567,candidate.second,
                               true,size_t(8)<<30,route);
            require(e.nonfinite==0 && e.maxabs==0,"candidate beta0/alias/rectangular/partial CTA smoke failed");
          }
        }
      };
      check_candidates(__nv_bfloat16{},megafmm::DType::BF16);
      check_candidates(__half{},megafmm::DType::FP16);
      if(mega_filter==13) {
        cluster_graph_suite<__nv_bfloat16>(megafmm::DType::BF16);
        cluster_graph_suite<__half>(megafmm::DType::FP16);
      }
      if(mega_filter==14 || mega_filter==15 || mega_filter==16 || mega_filter==17 || mega_filter==18 || mega_filter==19 || mega_filter==20 || mega_filter==21 || mega_filter==22 || mega_filter==23 || mega_filter==24 || mega_filter==25 || mega_filter==26 || mega_filter==27) {
        epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
      }
      if(mega_filter==16) {
        prefetch_short_k_suite<__nv_bfloat16>(megafmm::DType::BF16);
        prefetch_short_k_suite<__half>(megafmm::DType::FP16);
      }
      if(mega_filter==17) {
        cpasync_short_k_suite<__nv_bfloat16>(megafmm::DType::BF16);
        cpasync_short_k_suite<__half>(megafmm::DType::FP16);
        cluster_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,17);
        cluster_graph_suite<__half>(megafmm::DType::FP16,17);
        rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,false,-1,17);
        rebind_suite<__half>(megafmm::DType::FP16,false,-1,17);
      }
      if(mega_filter==18) {
        stage_m2_suite<__nv_bfloat16>(megafmm::DType::BF16);
        stage_m2_suite<__half>(megafmm::DType::FP16);
        cluster_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,18);
        cluster_graph_suite<__half>(megafmm::DType::FP16,18);
        rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,false,-1,18);
        rebind_suite<__half>(megafmm::DType::FP16,false,-1,18);
      }
      if(mega_filter==19) {
        prefetch_8x8_suite<__nv_bfloat16>(megafmm::DType::BF16);
        prefetch_8x8_suite<__half>(megafmm::DType::FP16);
        cluster_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,19);
        cluster_graph_suite<__half>(megafmm::DType::FP16,19);
        rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,false,-1,19);
        rebind_suite<__half>(megafmm::DType::FP16,false,-1,19);
      }
      if(mega_filter==20 || mega_filter==21 || mega_filter==22 || mega_filter==23 || mega_filter==24 || mega_filter==25) {
        direct_register_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        direct_register_suite<__half>(megafmm::DType::FP16,mega_filter);
        direct_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        direct_native_bitwise_suite<__half>(megafmm::DType::FP16,mega_filter);
        cluster_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        cluster_graph_suite<__half>(megafmm::DType::FP16,mega_filter);
        rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,false,-1,mega_filter);
        rebind_suite<__half>(megafmm::DType::FP16,false,-1,mega_filter);
      }
      if(mega_filter==26 || mega_filter==27) {
        cooperative_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        cooperative_suite<__half>(megafmm::DType::FP16,mega_filter);
        cooperative_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        cooperative_native_bitwise_suite<__half>(megafmm::DType::FP16,mega_filter);
        cluster_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        cluster_graph_suite<__half>(megafmm::DType::FP16,mega_filter);
        rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,false,-1,mega_filter);
        rebind_suite<__half>(megafmm::DType::FP16,false,-1,mega_filter);
      }
      if(hopper_filter==7 || hopper_filter==8) {
        k64_batched_tail_suite<__nv_bfloat16>(megafmm::DType::BF16,hopper_filter);
        k64_batched_tail_suite<__half>(megafmm::DType::FP16,hopper_filter);
        rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,false,hopper_filter);
        rebind_suite<__half>(megafmm::DType::FP16,false,hopper_filter);
      }
      std::cout << "PASS selected candidate kernel paths, beta0, alpha/beta alias and rectangular fallback\n";
      return 0;
    }
    structural_suite<__nv_bfloat16>(megafmm::DType::BF16);
    structural_suite<__half>(megafmm::DType::FP16);
    batched_suite<__nv_bfloat16>(megafmm::DType::BF16);
    batched_suite<__half>(megafmm::DType::FP16);
    for(int variant:{7,8}) {
      k64_batched_tail_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      k64_batched_tail_suite<__half>(megafmm::DType::FP16,variant);
    }
    warp_mma_suite<__nv_bfloat16>(megafmm::DType::BF16);
    warp_mma_suite<__half>(megafmm::DType::FP16);
    prefetch_short_k_suite<__nv_bfloat16>(megafmm::DType::BF16);
    prefetch_short_k_suite<__half>(megafmm::DType::FP16);
    cpasync_short_k_suite<__nv_bfloat16>(megafmm::DType::BF16);
    cpasync_short_k_suite<__half>(megafmm::DType::FP16);
    prefetch_8x8_suite<__nv_bfloat16>(megafmm::DType::BF16);
    prefetch_8x8_suite<__half>(megafmm::DType::FP16);
    for(int variant:{20,21,22,23,24,25}) {
      direct_register_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      direct_register_suite<__half>(megafmm::DType::FP16,variant);
      direct_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      direct_native_bitwise_suite<__half>(megafmm::DType::FP16,variant);
      cluster_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      cluster_graph_suite<__half>(megafmm::DType::FP16,variant);
      epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
      if(variant>=24) {
        rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,false,-1,variant);
        rebind_suite<__half>(megafmm::DType::FP16,false,-1,variant);
      }
    }
    for(int variant:{26,27}) {
      cooperative_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      cooperative_suite<__half>(megafmm::DType::FP16,variant);
      cooperative_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      cooperative_native_bitwise_suite<__half>(megafmm::DType::FP16,variant);
      cluster_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      cluster_graph_suite<__half>(megafmm::DType::FP16,variant);
      epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
    }
    stage_m2_suite<__nv_bfloat16>(megafmm::DType::BF16);
    stage_m2_suite<__half>(megafmm::DType::FP16);
    warp_mma_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16);
    warp_mma_bitwise_suite<__half>(megafmm::DType::FP16);
    cluster_graph_suite<__nv_bfloat16>(megafmm::DType::BF16);
    cluster_graph_suite<__half>(megafmm::DType::FP16);
    cluster_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,17);
    cluster_graph_suite<__half>(megafmm::DType::FP16,17);
    cluster_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,18);
    cluster_graph_suite<__half>(megafmm::DType::FP16,18);
    cluster_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,19);
    cluster_graph_suite<__half>(megafmm::DType::FP16,19);
    rebind_suite<__nv_bfloat16>(megafmm::DType::BF16);
    rebind_suite<__half>(megafmm::DType::FP16);
    epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16);
    epilogue_output_rebind_suite<__half>(megafmm::DType::FP16);
    epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,15);
    epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,15);
    epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,16);
    epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,16);
    epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,17);
    epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,17);
    epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,18);
    epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,18);
    epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,19);
    epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,19);
    degenerate_suite();rejection_suite();batched_policy_rejection_suite();
    std::cout << "PASS structural DAG, fallback numerics, strides/transposes, alias, degenerate and rejection tests\n";
    bool pass=true;
    if(numeric) {
      bool bf16_pass=strict_numerics<__nv_bfloat16>(megafmm::DType::BF16);
      bool fp16_pass=strict_numerics<__half>(megafmm::DType::FP16);
      bool k128_bf16_pass=strict_numerics<__nv_bfloat16>(megafmm::DType::BF16,true);
      bool k128_fp16_pass=strict_numerics<__half>(megafmm::DType::FP16,true);
      pass=bf16_pass && fp16_pass && k128_bf16_pass && k128_fp16_pass;
    }
    else std::cout << "Forced SW mixed-precision certification is a separate gate: use --strict-numerics or megafmm_bench\n";
    return pass ? 0:2;
  } catch(const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n';return 1; }
}

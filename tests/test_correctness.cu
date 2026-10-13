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
  if(variant==9)return "winograd_fused_input_32x32";
  if(variant==10)return "winograd_fused_input_32x64";
  if(variant==11)return "winograd_packed_input_32x32";
  if(variant==12)return "winograd_packed_input_32x64";
  return variant>=7 ? "winograd_sm80_batched_k64":"winograd_sm80_batched";
}
bool fused_input_supported(const megafmm::Problem& p,int variant) {
  return (variant==9 || variant==10) && !p.trans_a && !p.trans_b &&
         p.m>0 && p.n>0 && p.k>0 && p.m<=32768 && p.n<=32768 && p.k<=32768 &&
         p.m%64==0 && p.n%(variant==9 ? 64:128)==0 && p.k%64==0 &&
         p.lda%8==0 && p.ldb%8==0;
}
size_t fused_product_bytes(const megafmm::Problem& p) {
  return size_t(7)*size_t(p.m/2)*size_t(p.n/2)*sizeof(float);
}
size_t fused_reserved_bytes(const megafmm::Problem& p) {
  return (fused_product_bytes(p)+255)/256*256;
}
bool packed_input_supported(const megafmm::Problem& p,int variant) {
  return (variant==11 || variant==12) && !p.trans_a && !p.trans_b &&
         p.m>0 && p.n>0 && p.k>0 && p.m<=32768 && p.n<=32768 && p.k<=32768 &&
         p.m%64==0 && p.n%(variant==11 ? 64:128)==0 && p.k%64==0 &&
         p.lda%8==0 && p.ldb%8==0;
}
size_t packed_align256(size_t bytes) {return (bytes+255)/256*256;}
struct PackedTestLayout {size_t a_bytes,b_bytes,p_bytes,b_offset,p_offset,payload;};
PackedTestLayout packed_test_layout(const megafmm::Problem& p) {
  const size_t mh=size_t(p.m/2),nh=size_t(p.n/2),kh=size_t(p.k/2);
  const size_t a=7*mh*kh*2,b=7*kh*nh*2,c=7*mh*nh*sizeof(float);
  const size_t bo=packed_align256(a),po=packed_align256(bo+b);
  return {a,b,c,bo,po,packed_align256(po+c)};
}
void check_packed_workspace(const megafmm::Plan& plan,const megafmm::Problem& p,
                            size_t limit=size_t(8)<<30) {
  const auto layout=packed_test_layout(p);
  require(plan.workspace_bytes()>=layout.payload && plan.workspace_bytes()<=limit,
          "packed input reserved workspace must cover aligned A7+B7+P and obey cap");
}
void check_packed_metadata(const megafmm::Problem& p,int variant) {
  require(megafmm::packed_leaf_can_implement(p,variant-11)==packed_input_supported(p,variant),
          "packed input metadata guard disagrees with independent shape/stride reference");
  if(!packed_input_supported(p,variant))return;
  const auto meta=megafmm::packed_leaf_metadata(p,variant-11);
  const auto layout=packed_test_layout(p);
  const int64_t mh=p.m/2,nh=p.n/2,kh=p.k/2;
  require(meta.mh==mh && meta.nh==nh && meta.kh==kh &&
          meta.a_row_stride==kh && meta.b_row_stride==nh && meta.p_row_stride==nh &&
          meta.a_product_stride==mh*kh && meta.b_product_stride==kh*nh &&
          meta.p_product_stride==mh*nh && meta.a_bytes==layout.a_bytes &&
          meta.b_bytes==layout.b_bytes && meta.p_bytes==layout.p_bytes &&
          meta.compact_workspace_bytes==layout.payload &&
          megafmm::packed_leaf_workspace_bytes(p,variant-11)==layout.payload,
          "packed input metadata is not seven compact row-major A/B/P slots");
}
const char* warp_mma_name(int variant,const megafmm::Problem& p) {
  if(variant==44) {
    if(p.lda%8!=0 || p.ldb%8!=0)return "dense_cublaslt";
    // A single reusable stageA retains RS-A39's N32/BK64 parent geometry.
    return p.m%128==0 && p.n%64==0 && p.k%128==0 ?
           "megakernel_wgmma_psplit_rsa_microbatch_64x32":
           "megakernel_warp_mma_packed";
  }
  if(variant==42 || variant==43) {
    if(p.lda%8!=0 || p.ldb%8!=0)return "dense_cublaslt";
    // The N16 atom launches two column CTAs at minimum parent N64. N32 falls back.
    return p.m%128==0 && p.n%64==0 && p.k%64==0 ?
           (variant==42 ? "megakernel_wgmma_psplit_rsa_bn16_64x16":
                          "megakernel_wgmma_psplit_ss_bn16_64x16"):
           "megakernel_warp_mma_packed";
  }
  if(variant==41) {
    if(p.lda%8!=0 || p.ldb%8!=0)return "dense_cublaslt";
    // Hybrid S explicitly retains N32 and one native stage per parent K64.
    return p.m%128==0 && p.n%64==0 && p.k%64==0 ?
           "megakernel_wgmma_psplit_hybrid_s_64x32":
           "megakernel_warp_mma_packed";
  }
  if(variant==40) {
    if(p.lda%8!=0 || p.ldb%8!=0)return "dense_cublaslt";
    // BK32 RS-A explicitly retains N32 and one native stage per parent K64.
    return p.m%128==0 && p.n%64==0 && p.k%64==0 ?
           "megakernel_wgmma_psplit_rsa_bk32_64x32":
           "megakernel_warp_mma_packed";
  }
  if(variant==39) {
    if(p.lda%8!=0 || p.ldb%8!=0)return "dense_cublaslt";
    // RS-A keeps N32 and the BK64 SS parent-K domain, independent of parity.
    return p.m%128==0 && p.n%64==0 && p.k%128==0 ?
           "megakernel_wgmma_psplit_rsa_64x32":
           "megakernel_warp_mma_packed";
  }
  if(variant==38) {
    if(p.lda%8!=0 || p.ldb%8!=0)return "dense_cublaslt";
    // Parent K128 supplies one complete BK64 stage in each SW leaf.
    return p.m%128==0 && p.n%64==0 && p.k%128==0 ?
           "megakernel_wgmma_psplit_bk64_64x32":
           "megakernel_warp_mma_packed";
  }
  if(variant==36) {
    if(p.lda%8!=0 || p.ldb%8!=0)return "dense_cublaslt";
    // Both product consumers retain the complete M64N32K32 atom.
    return p.m%128==0 && p.n%64==0 && p.k%64==0 ?
           "megakernel_wgmma_psplit_64x32":
           "megakernel_warp_mma_packed";
  }
  if(variant==34 || variant==35) {
    if(p.lda%8!=0 || p.ldb%8!=0)return "dense_cublaslt";
    return p.m%128==0 && p.n%(variant==34 ? 32:64)==0 && p.k%64==0 ?
           (variant==34 ? "megakernel_wgmma_wait1_three_stage_64x16":"megakernel_wgmma_wait1_three_stage_64x32"):
           "megakernel_warp_mma_packed";
  }
  if(variant==32 || variant==33) {
    if(p.lda%8!=0 || p.ldb%8!=0)return "dense_cublaslt";
    return p.m%128==0 && p.n%(variant==32 ? 32:64)==0 && p.k%64==0 ?
           (variant==32 ? "megakernel_wgmma_wait1_64x16":"megakernel_wgmma_wait1_64x32"):
           "megakernel_warp_mma_packed";
  }
  if(variant==30 || variant==31) {
    if(p.lda%8!=0 || p.ldb%8!=0)return "dense_cublaslt";
    return p.m%128==0 && p.n%(variant==30 ? 32:64)==0 && p.k%64==0 &&
           uint64_t(p.lda)<(uint64_t(1)<<39) && uint64_t(p.ldb)<(uint64_t(1)<<39) ?
           (variant==30 ? "megakernel_wgmma_tma_alias_64x16":"megakernel_wgmma_tma_alias_64x32"):
           "megakernel_warp_mma_packed";
  }
  if(variant==28 || variant==29) {
    if(p.lda%8!=0 || p.ldb%8!=0)return "dense_cublaslt";
    return p.m%128==0 && p.n%(variant==28 ? 32:64)==0 && p.k%64==0 &&
           p.lda%8==0 && p.ldb%8==0 ?
           (variant==28 ? "megakernel_wgmma_pipeline_64x16":"megakernel_wgmma_pipeline_64x32"):
           "megakernel_warp_mma_packed";
  }
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
template<class T> std::vector<double> m1_integer_reference(const megafmm::Problem& p,
                                                         const std::vector<T>& a,const std::vector<T>& b) {
  // Independent exact-integer oracle: accumulate all seven complete-K dots,
  // then apply the fixed U dependencies. This is not a Tensor Core bit oracle.
  const int mh=int(p.m/2),nh=int(p.n/2),kh=int(p.k/2);
  std::vector<double> result(size_t(p.m)*p.n);
  for(int i=0;i<mh;++i)for(int j=0;j<nh;++j) {
    double products[7]={};
    for(int l=0;l<kh;++l) {
      const double a11=double_value(a[size_t(i)*p.lda+l]),a12=double_value(a[size_t(i)*p.lda+l+kh]);
      const double a21=double_value(a[size_t(i+mh)*p.lda+l]),a22=double_value(a[size_t(i+mh)*p.lda+l+kh]);
      const double b11=double_value(b[size_t(l)*p.ldb+j]),b12=double_value(b[size_t(l)*p.ldb+j+nh]);
      const double b21=double_value(b[size_t(l+kh)*p.ldb+j]),b22=double_value(b[size_t(l+kh)*p.ldb+j+nh]);
      const double s1=a21+a22,s2=s1-a11,s3=a11-a21,s4=a12-s2;
      const double t1=b12-b11,t2=b22-t1,t3=b22-b12,t4=t2-b21;
      const double left[7]={a11,a12,s4,a22,s1,s2,s3};
      const double right[7]={b11,b21,b22,t4,t1,t2,t3};
      for(int product=0;product<7;++product)products[product]+=left[product]*right[product];
    }
    const double u1=products[0]+products[1],u2=products[0]+products[5];
    const double u3=u2+products[6],u4=u2+products[4];
    const double u5=u4+products[2],u6=u3-products[3],u7=u3+products[4];
    result[size_t(i)*p.n+j]=u1;result[size_t(i)*p.n+j+nh]=u5;
    result[size_t(i+mh)*p.n+j]=u6;result[size_t(i+mh)*p.n+j+nh]=u7;
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
  if(algorithm==megafmm::Algorithm::Megakernel &&
     (variant==28 || variant==29 || variant==30 || variant==31 || variant==32 || variant==33 || variant==34 || variant==35 || variant==36 || variant==38 || variant==39 || variant==40 || variant==41 || variant==42 || variant==43 || variant==44) && expected_route &&
     (std::string(expected_route).find("megakernel_wgmma_pipeline")==0 ||
      std::string(expected_route).find("megakernel_wgmma_tma_alias")==0 ||
      std::string(expected_route).find("megakernel_wgmma_wait1")==0 ||
      std::string(expected_route).find("megakernel_wgmma_psplit")==0) &&
     (pattern==Pattern::Integers || pattern==Pattern::OwnerRows)) {
    stage_reference=m1_integer_reference(p,a,b);
    for(int i=0;i<p.m;++i)for(int j=0;j<p.n;++j) {
      auto& fixed=stage_reference[size_t(i)*p.n+j];
      fixed=double(p.alpha)*fixed+double(p.beta)*c[size_t(i)*p.ldc+j];
      require(fixed==reference[size_t(i)*p.n+j],"M1 complete-P fixed DAG disagrees with independent FP64 AB");
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
  if(expected_route && std::string(expected_route).find("winograd_fused_input")==0)
    require(plan->workspace_bytes()==fused_reserved_bytes(p),"fused input plan must reserve only seven P slots");
  if(expected_route && std::string(expected_route).find("winograd_packed_input")==0)
    check_packed_workspace(*plan,p,workspace_limit);
  if(expected_route && (std::string(expected_route).find("megakernel_warp_mma")==0 ||
                        std::string(expected_route).find("megakernel_cooperative")==0 ||
                        std::string(expected_route).find("megakernel_wgmma_pipeline")==0 ||
                        std::string(expected_route).find("megakernel_wgmma_tma_alias")==0 ||
                        std::string(expected_route).find("megakernel_wgmma_wait1")==0 ||
                        std::string(expected_route).find("megakernel_wgmma_psplit")==0 ||
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
              "GPU integer output disagrees with independent fixed DAG reference");
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
template<class T> void wgmma_pipeline_suite(megafmm::DType dtype,int variant) {
  require(variant==28 || variant==29,"pipeline suite needs Mega28/29");
  auto exact=[dtype,variant](megafmm::Problem p,Pattern pattern,bool alias=false,bool certified=true) {
    const char* route=certified ? warp_mma_name(variant,p):"dense_cublaslt";
    std::cout << "BEGIN_WGMMA_PIPELINE dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
              << " null_c0=" << (p.beta==0.0f) << " expected=" << route << std::endl;
    auto e=run_case<T>(p,megafmm::Algorithm::Megakernel,pattern,alias,certified,
                       1234567,variant,false,0,route);
    require(e.nonfinite==0 && e.maxabs==0,"pipeline native/packed/Dense integer output incorrect");
  };
  // Parent K64/128/192/256 gives 1/2/3/4 stages: prime, drain, bank reuse and
  // phase wrap are all observable with distinct K16 owner contributions.
  for(int k:{64,128,192,256})
    exact(problem(128,variant==28 ? 32:64,k,dtype),Pattern::OwnerRows);
  exact(problem(256,128,64,dtype),Pattern::Integers);
  auto padded=problem(128,256,128,dtype);padded.alpha=.5f;padded.beta=-.25f;
  padded.lda+=8;padded.ldb+=8;padded.ldc+=1;padded.ldd+=1;
  exact(padded,Pattern::Integers,true);
  auto null_c0=problem(128,128,192,dtype);null_c0.beta=-0.0f;
  null_c0.ldc+=1;null_c0.ldd+=1;exact(null_c0,Pattern::OwnerRows);
  auto tail=problem(80,112,48,dtype);tail.alpha=.5f;tail.beta=-.25f;
  exact(tail,Pattern::Integers,true);
  exact(problem(128,64,32,dtype),Pattern::Integers);
  if(variant==29)exact(problem(128,32,64,dtype),Pattern::Integers);
  auto odd_stride=problem(128,128,128,dtype);odd_stride.lda+=1;
  exact(odd_stride,Pattern::Integers);
  exact(problem(128,128,128,dtype),Pattern::Integers,false,false);
}
template<class T> void wgmma_pipeline_native_bitwise_suite(megafmm::DType dtype) {
  // Only Mega29 and Mega0 select the same M64N32K32 WGMMA atom and full-P
  // chronology. Mega28's N16 atom is intentionally excluded from this claim.
  for(auto mn:{std::pair<int,int>{128,64},std::pair<int,int>{128,256},std::pair<int,int>{256,256}})
  for(int k:{64,128,192}) {
    auto p=problem(mn.first,mn.second,k,dtype);
    const bool alias=k==128;
    const int input_shift=k==192 ? 8:0,output_shift=k==192 ? 1:0;
    if(k!=64) {p.lda+=8;p.ldb+=8;p.ldc+=1;p.ldd+=1;p.alpha=.375f;p.beta=-.625f;}
    if(k==64 && mn.second==64)p.beta=-0.0f;
    uint32_t seed=331166u+uint32_t(p.m*31+p.n*17+p.k*7)+
                  (dtype==megafmm::DType::BF16 ? 0u:918273u);
    std::mt19937 rng(seed);
    std::vector<T> a(size_t(p.m)*p.lda,round_value<T>(3.25f)),b(size_t(p.k)*p.ldb,round_value<T>(-2.75f));
    auto random_input=[&]() {return round_value<T>(2.0f*(float(rng()>>8)/16777216.0f)-1.0f);};
    for(int i=0;i<p.m;++i)for(int l=0;l<p.k;++l)a[size_t(i)*p.lda+l]=random_input();
    for(int l=0;l<p.k;++l)for(int j=0;j<p.n;++j)b[size_t(l)*p.ldb+j]=random_input();
    std::vector<T> a_storage(a.size()+8,round_value<T>(5.5f)),b_storage(b.size()+8,round_value<T>(-6.5f));
    std::copy(a.begin(),a.end(),a_storage.begin()+input_shift);
    std::copy(b.begin(),b.end(),b_storage.begin()+input_shift);
    const size_t output_count=size_t(p.m)*p.ldd;
    std::vector<float> initial(output_count+2),actual(initial.size()),baseline(initial.size());
    for(size_t at=0;at<initial.size();++at)initial[at]=float(int((at+seed)%1024)-512)/8.0f;
    auto c=initial;
    for(int i=0;i<p.m;++i)for(int j=0;j<p.n;++j) {
      initial[size_t(i)*p.ldd+j]=std::numeric_limits<float>::quiet_NaN();
      c[size_t(i)*p.ldc+j]=float(int(rng()%257)-128)/64.0f;
    }
    Buffer da(a_storage.size()*sizeof(T)),db(b_storage.size()*sizeof(T)),
           dc((c.size()+1)*sizeof(float)),dd((initial.size()+1)*sizeof(float));
    const T* left=da.as<T>()+input_shift,*right=db.as<T>()+input_shift;
    float* source=dc.as<float>()+output_shift;
    float* output=alias ? source:dd.as<float>()+output_shift;
    std::cout << "BEGIN_WGMMA_PIPELINE_NATIVE_BITWISE dtype=" << megafmm::dtype_name(dtype)
              << " variant=29 reference_variant=0 shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
              << " null_c0=" << (p.beta==0.0f) << " input_shift_bytes=" << input_shift*sizeof(T)
              << " output_shift_bytes=" << output_shift*sizeof(float) << " seed=" << seed << std::endl;
    const bool identity_epilogue=p.alpha==1.0f && p.beta==0.0f;
    // First compare raw U with identical WGMMA atoms. Then test the specified
    // RN alpha multiply followed by active-beta FMA, independently of the old
    // Mega0 source's implicit epilogue contraction. Active epilogues need a
    // third run; identity cases already exercise both claims in the second.
    for(int stage=0;stage<(identity_epilogue ? 2:3);++stage) {
      const int current=stage==0 ? 0:29;
      auto current_problem=p;
      if(stage<2) {current_problem.alpha=1.0f;current_problem.beta=0.0f;}
      float* current_output=stage==2 ? output:dd.as<float>()+output_shift;
      megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
      policy.certified_range=true;policy.mega_variant=current;policy.workspace_limit=0;
      auto plan=megafmm::create_plan(current_problem,policy);
      require(plan->algorithm_name()==(current==0 ? "megakernel_m1":warp_mma_name(29,p)),
              "pipeline bits selected another route");
      require(plan->workspace_bytes()==0,"pipeline bits unexpectedly requires workspace");
      check(cudaMemcpy(da.data,a_storage.data(),a_storage.size()*sizeof(T),cudaMemcpyHostToDevice));
      check(cudaMemcpy(db.data,b_storage.data(),b_storage.size()*sizeof(T),cudaMemcpyHostToDevice));
      check(cudaMemcpy(source,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
      if(stage!=2 || !alias)check(cudaMemcpy(current_output,initial.data(),initial.size()*sizeof(float),cudaMemcpyHostToDevice));
      require(plan->lt_algorithm_config(left,right,current_problem.beta==0.0f ? nullptr:source,current_output,nullptr)=="none",
              "pipeline bits metadata unexpectedly uses Dense");
      megafmm::run(*plan,left,right,current_problem.beta==0.0f ? nullptr:source,current_output,nullptr);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),current_output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i) {
        for(int j=0;j<p.n;++j)require(std::isfinite(actual[size_t(i)*p.ldd+j]),"pipeline bits returned nonfinite/unwritten output");
        for(int j=int(p.n);j<p.ldd;++j)require(actual[size_t(i)*p.ldd+j]==initial[size_t(i)*p.ldd+j],"pipeline bits overwrote D padding");
      }
      for(size_t at=output_count;at<actual.size();++at)require(actual[at]==initial[at],"pipeline bits overwrote trailing D guard");
      if(stage==0)baseline=actual;
      else for(int i=0;i<p.m;++i)for(int j=0;j<p.n;++j) {
        const size_t at=size_t(i)*p.ldd+j;
        float expected=baseline[at];
        if(stage==2 || identity_epilogue) {
          // The volatile float store forces a separate FP32 RN multiply;
          // std::fma(float,float,float) supplies the single-rounding beta FMA.
          const volatile float scaled=p.alpha*baseline[at];
          expected=p.beta==0.0f ? float(scaled):std::fma(p.beta,c[size_t(i)*p.ldc+j],float(scaled));
        }
        uint32_t expected_bits=0,actual_bits=0;
        std::memcpy(&expected_bits,&expected,sizeof(expected_bits));
        std::memcpy(&actual_bits,&actual[at],sizeof(actual_bits));
        if(expected_bits!=actual_bits) {
          std::cerr << "WGMMA_PIPELINE_BIT_MISMATCH dtype=" << megafmm::dtype_name(dtype)
                    << " variant=29 phase=" << (stage==1 ? "raw_u":"contract_epilogue")
                    << " shape=" << p.m << ',' << p.n << ',' << p.k
                    << " row=" << i << " col=" << j << " byte_index=" << at*sizeof(float)
                    << " expected_bits=0x" << std::hex << expected_bits << " actual_bits=0x" << actual_bits << std::dec
                    << " raw_u=" << baseline[at] << " alpha=" << p.alpha << " beta=" << p.beta
                    << " c0=" << c[size_t(i)*p.ldc+j] << '\n';
          throw std::runtime_error(stage==1 ? "pipeline raw U bits differ from matching Mega0 WGMMA atom":
                                   "pipeline output bits differ from specified host RN-multiply/FMA oracle");
        }
      }
    }
    std::cout << "PASS_WGMMA_PIPELINE_NATIVE_BITWISE dtype=" << megafmm::dtype_name(dtype)
              << " variant=29 reference_variant=0 shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
              << " null_c0=" << (p.beta==0.0f)
              << " reference=mega0_sameatom_raw_u+specified_host_epilogue"
              << " raw_u_bits=1 explicit_host_epilogue=1 padding_guard=1\n";
  }
}
template<class T> void wgmma_pipeline_input_rebind_suite(megafmm::DType dtype,int variant) {
  auto p=problem(128,128,128,dtype);
  std::vector<T> a(size_t(p.m)*p.lda),b(size_t(p.k)*p.ldb);
  for(int i=0;i<p.m;++i)for(int l=0;l<p.k;++l)a[size_t(i)*p.lda+l]=round_value<T>(float(i%5-2));
  for(int l=0;l<p.k;++l)for(int j=0;j<p.n;++j)b[size_t(l)*p.ldb+j]=round_value<T>(float(j%7-3));
  std::vector<float> initial(size_t(p.m)*p.ldd+2,-777.0f),actual(initial.size());
  Buffer da((a.size()+8)*sizeof(T)),db((b.size()+8)*sizeof(T)),dd(initial.size()*sizeof(float));
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
  policy.certified_range=true;policy.mega_variant=variant;policy.workspace_limit=0;
  auto plan=megafmm::create_plan(p,policy);
  require(plan->algorithm_name()==warp_mma_name(variant,p) && plan->workspace_bytes()==0,
          "pipeline rebind plan is not a native zero-workspace route");
  for(auto shifts:std::vector<std::pair<int,int>>{{0,0},{1,0},{0,1},{1,1},{8,8},{0,0}}) {
    T* left=da.as<T>()+shifts.first,*right=db.as<T>()+shifts.second;
    check(cudaMemcpy(left,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
    check(cudaMemcpy(right,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
    check(cudaMemcpy(dd.data,initial.data(),initial.size()*sizeof(float),cudaMemcpyHostToDevice));
    const bool dense=shifts.first==1 || shifts.second==1;
    check_lt_config(*plan,left,right,nullptr,dd.as<float>(),nullptr,dense ? "dense":"none",false);
    megafmm::run(*plan,left,right,nullptr,dd.as<float>(),nullptr);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(actual.data(),dd.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(int i=0;i<p.m;++i)for(int j=0;j<p.n;++j)
      require(actual[size_t(i)*p.ldd+j]==float(p.k*(i%5-2)*(j%7-3)),"pipeline current AB alignment fallback output incorrect");
    for(size_t at=size_t(p.m)*p.ldd;at<actual.size();++at)
      require(actual[at]==initial[at],"pipeline rebind overwrote trailing D sentinel");
  }
  std::cout << "PASS_WGMMA_PIPELINE_INPUT_REBIND dtype=" << megafmm::dtype_name(dtype)
            << " variant=" << variant << " bindings=6 element_only_A_B_both_dense=1 aligned_recovery=1 trailing_guard=1\n";
}
template<class T> void wgmma_pipeline_graph_suite(megafmm::DType dtype,int variant) {
  for(int scenario=0;scenario<3;++scenario) {
    auto p=scenario==0 ? problem(128,variant==28 ? 32:64,64,dtype):
           scenario==1 ? problem(128,256,192,dtype):problem(80,112,48,dtype);
    p.lda+=8;p.ldb+=8;p.ldc+=1;p.ldd+=1;
    const bool active_beta=scenario!=0;
    if(active_beta) {p.alpha=.5f;p.beta=-.25f;} else p.beta=-0.0f;
    std::mt19937 rng(117799u+uint32_t(p.m*31+p.n*17+p.k*7));
    std::vector<T> a(size_t(p.m)*p.lda),b(size_t(p.k)*p.ldb);
    std::vector<float> c(size_t(p.m)*p.ldc+2,-777.0f),actual(c.size());
    std::vector<double> reference_ab(size_t(p.m)*p.n);
    Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dc(c.size()*sizeof(float));
    megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
    policy.certified_range=true;policy.mega_variant=variant;policy.workspace_limit=0;
    auto plan=megafmm::create_plan(p,policy);
    require(plan->algorithm_name()==warp_mma_name(variant,p) && plan->workspace_bytes()==0,
            "pipeline graph route/workspace incorrect");
    std::cout << "BEGIN_WGMMA_PIPELINE_GRAPH dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " actual=" << plan->algorithm_name() << " alias=" << active_beta
              << " null_c0=" << !active_beta << " refresh_ab=1" << std::endl;
    Stream stream;Graph graph;
    check(cudaStreamSynchronize(stream.value));
    check(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));
    megafmm::run(*plan,da.data,db.data,active_beta ? dc.as<float>():nullptr,dc.as<float>(),nullptr,stream.value);
    check(cudaStreamEndCapture(stream.value,&graph.value));
    size_t nodes=0;check(cudaGraphGetNodes(graph.value,nullptr,&nodes));
    require(nodes==1,"pipeline full invocation did not capture exactly one kernel node");
    cudaGraphNode_t node=nullptr;size_t one=1;check(cudaGraphGetNodes(graph.value,&node,&one));
    cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
    require(type==cudaGraphNodeTypeKernel,"pipeline captured node is not a kernel");
    check(cudaGraphInstantiate(&graph.executable,graph.value,0));
    for(int replay=0;replay<3;++replay) {
      for(int i=0;i<p.m;++i)for(int l=0;l<p.k;++l)a[size_t(i)*p.lda+l]=round_value<T>(float(int(rng()%5)-2));
      for(int l=0;l<p.k;++l)for(int j=0;j<p.n;++j)b[size_t(l)*p.ldb+j]=round_value<T>(float(int(rng()%5)-2));
      std::fill(reference_ab.begin(),reference_ab.end(),0.0);
      for(int i=0;i<p.m;++i)for(int j=0;j<p.n;++j)for(int l=0;l<p.k;++l)
        reference_ab[size_t(i)*p.n+j]+=double_value(a[size_t(i)*p.lda+l])*double_value(b[size_t(l)*p.ldb+j]);
      if(scenario<2)require(m1_integer_reference(p,a,b)==reference_ab,"pipeline graph independent complete-P DAG differs from fresh AB");
      for(int i=0;i<p.m;++i)for(int j=0;j<p.n;++j)c[size_t(i)*p.ldc+j]=float((i+j+3*replay)%9-4)/4.0f;
      // Refresh original rounded inputs at captured addresses, and restore the
      // immutable C0 snapshot before replay. Capture never precomputes S/T.
      check(cudaMemcpyAsync(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice,stream.value));
      check(cudaMemcpyAsync(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice,stream.value));
      check(cudaMemcpyAsync(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice,stream.value));
      check(cudaGraphLaunch(graph.executable,stream.value));check(cudaStreamSynchronize(stream.value));
      check(cudaMemcpy(actual.data(),dc.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i)for(int j=0;j<p.n;++j)
        require(double(actual[size_t(i)*p.ldd+j])==double(p.alpha)*reference_ab[size_t(i)*p.n+j]+double(p.beta)*c[size_t(i)*p.ldc+j],
                "pipeline graph output differs from fresh original-input FP64 AB");
      for(int i=0;i<p.m;++i)for(int j=int(p.n);j<p.ldd;++j)
        require(actual[size_t(i)*p.ldd+j]==-777.0f,"pipeline graph overwrote output padding");
      for(size_t at=size_t(p.m)*p.ldd;at<actual.size();++at)
        require(actual[at]==-777.0f,"pipeline graph overwrote trailing D sentinel");
    }
    std::cout << "PASS_WGMMA_PIPELINE_GRAPH dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " actual=" << plan->algorithm_name() << " alias=" << active_beta
              << " null_c0=" << !active_beta << " refresh_ab=1 replays=3 kernel_nodes=1\n";
  }
}
void wgmma_pipeline_policy_suite(int variant) {
  for(auto algorithm:{megafmm::Algorithm::Auto,megafmm::Algorithm::Dense,megafmm::Algorithm::WinogradLt,
                      megafmm::Algorithm::WinogradHopper,megafmm::Algorithm::Megakernel})
  for(int scenario=0;scenario<4;++scenario) {
    auto p=problem(128,128,128,megafmm::DType::BF16);
    if(scenario==1)p.m=0;if(scenario==2)p.alpha=0;if(scenario==3)p.k=0;
    megafmm::Policy invalid;invalid.algorithm=algorithm;invalid.mega_variant=45;invalid.certified_range=true;
    bool rejected=false;try {auto plan=megafmm::create_plan(p,invalid);}catch(const std::exception&) {rejected=true;}
    require(rejected,"invalid Mega45 accepted before normal/empty/alpha0/K0 dispatch");
  }
  megafmm::Policy automatic;automatic.algorithm=megafmm::Algorithm::Auto;
  automatic.mega_variant=variant;automatic.certified_range=true;
  auto p=problem(128,128,128,megafmm::DType::BF16);
  require(megafmm::create_plan(p,automatic)->algorithm_name()=="dense_cublaslt","pipeline candidate enabled uncertified Auto SW");
  std::cout << "PASS_WGMMA_PIPELINE_POLICY variant=" << variant
            << " invalid45_cases=20 validate_before_empty_scale=1 auto_dense=1\n";
}
const char* wgmma_scheduled_family(int variant) {
  return variant==44 ? "PSPLIT_MICROBATCH":variant==42 ? "PSPLIT_RSA_BN16":variant==43 ? "PSPLIT_SS_BN16":variant==41 ? "PSPLIT_HYBRID_S":variant==40 ? "PSPLIT_RSA_BK32":variant==39 ? "PSPLIT_RSA":variant==38 ? "PSPLIT_BK64":variant>=36 ? "PSPLIT":variant>=34 ? "WAIT1_THREE_STAGE":variant>=32 ? "WAIT1":"TMA_ALIAS";
}
bool wgmma_scheduled_extended_bk32(int variant) {
  return variant==40 || variant==41 || variant==42 || variant==43;
}
bool wgmma_scheduled_bk64(int variant) {
  return variant==38 || variant==39 || variant==44;
}
bool wgmma_scheduled_n16(int variant) {
  // This controls parent-N32 fixtures of older N16 routes. New42/43 N16 atoms
  // require parent N64 for two column CTAs and deliberately do not use parity.
  return variant==30 || variant==32 || variant==34;
}
bool psplit_bn16_route_enabled(int variant) {
  require(variant==42 || variant==43,"BN16 enable probe requires variant42/43");
  static int cached[2]={-1,-1};
  if(cached[variant-42]>=0)return cached[variant-42]!=0;
  int enabled_dtypes=0,disabled_dtypes=0;
  for(auto dtype:{megafmm::DType::BF16,megafmm::DType::FP16}) {
    auto p=problem(128,64,64,dtype);
    megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
    policy.mega_variant=variant;policy.certified_range=true;policy.workspace_limit=0;
    try {
      auto plan=megafmm::create_plan(p,policy);
      require(plan->algorithm_name()==warp_mma_name(variant,p) && plan->workspace_bytes()==0,
              "BN16 registered route silently substituted another algorithm");
      ++enabled_dtypes;
    } catch(const std::invalid_argument& e) {
      const std::string message=e.what();
      if(message.find("disabled")==std::string::npos ||
         message.find(std::to_string(variant))==std::string::npos)throw;
      ++disabled_dtypes;
    }
  }
  require(enabled_dtypes==2 || disabled_dtypes==2,"BN16 enable state differs between dtypes");
  cached[variant-42]=enabled_dtypes==2;
  std::cout << "BN16_ROUTE_SCOPE variant=" << variant << " enabled=" << cached[variant-42]
            << " dtypes=2 metadata_only=1 native_runs=0\n";
  return cached[variant-42]!=0;
}
void psplit_bn16_blocked_policy_suite(int variant) {
  require(!psplit_bn16_route_enabled(variant),"BN16 blocked policy requires an explicitly disabled route");
  size_t rejected_cases=0,auto_dense_cases=0;
  for(auto dtype:{megafmm::DType::BF16,megafmm::DType::FP16}) {
    megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
    policy.mega_variant=variant;policy.certified_range=true;policy.workspace_limit=0;
    for(int scenario=0;scenario<4;++scenario) {
      auto p=problem(128,64,64,dtype);
      if(scenario==1)p.m=0;if(scenario==2)p.alpha=0;if(scenario==3)p.k=0;
      bool rejected=false;
      try {auto plan=megafmm::create_plan(p,policy);}
      catch(const std::invalid_argument& e) {
        const std::string message=e.what();
        rejected=message.find("disabled")!=std::string::npos &&
                 message.find(std::to_string(variant))!=std::string::npos;
      }
      require(rejected,"disabled BN16 route accepted before normal/empty/alpha0/K0 dispatch");
      ++rejected_cases;
    }
    policy.algorithm=megafmm::Algorithm::Auto;
    auto p=problem(128,64,64,dtype);
    require(megafmm::create_plan(p,policy)->algorithm_name()=="dense_cublaslt",
            "Auto BN16 metadata did not retain Dense policy");
    ++auto_dense_cases;
  }
  std::cout << "PASS_WGMMA_PSPLIT_BN16_BLOCKED_POLICY variant=" << variant
            << " forced_rejections=" << rejected_cases << " validate_before_empty_scale=1"
            << " auto_dense_dtypes=" << auto_dense_cases << " native_runs=0\n";
}
bool psplit_microbatch_route_enabled() {
  static int cached=-1;
  if(cached>=0)return cached!=0;
  int enabled_dtypes=0,disabled_dtypes=0;
  for(auto dtype:{megafmm::DType::BF16,megafmm::DType::FP16}) {
    auto p=problem(128,64,128,dtype);
    megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
    policy.mega_variant=44;policy.certified_range=true;policy.workspace_limit=0;
    try {
      auto plan=megafmm::create_plan(p,policy);
      require(plan->algorithm_name()==warp_mma_name(44,p) && plan->workspace_bytes()==0,
              "microbatch44 registered route silently substituted another algorithm");
      ++enabled_dtypes;
    } catch(const std::invalid_argument& e) {
      const std::string message=e.what();
      if(message.find("disabled")==std::string::npos || message.find("44")==std::string::npos)throw;
      ++disabled_dtypes;
    }
  }
  require(enabled_dtypes==2 || disabled_dtypes==2,"microbatch44 enable state differs between dtypes");
  cached=enabled_dtypes==2;
  std::cout << "MICROBATCH_ROUTE_SCOPE variant=44 enabled=" << cached
            << " dtypes=2 metadata_only=1 native_runs=0\n";
  return cached!=0;
}
void psplit_microbatch_blocked_policy_suite() {
  require(!psplit_microbatch_route_enabled(),"microbatch blocked policy requires an explicitly disabled44 route");
  size_t rejected_cases=0,auto_dense_cases=0;
  for(auto dtype:{megafmm::DType::BF16,megafmm::DType::FP16}) {
    megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
    policy.mega_variant=44;policy.certified_range=true;policy.workspace_limit=0;
    for(int scenario=0;scenario<4;++scenario) {
      auto p=problem(128,64,128,dtype);
      if(scenario==1)p.m=0;if(scenario==2)p.alpha=0;if(scenario==3)p.k=0;
      bool rejected=false;
      try {auto plan=megafmm::create_plan(p,policy);}
      catch(const std::invalid_argument& e) {
        const std::string message=e.what();
        rejected=message.find("disabled")!=std::string::npos && message.find("44")!=std::string::npos;
      }
      require(rejected,"disabled microbatch44 accepted before normal/empty/alpha0/K0 dispatch");
      ++rejected_cases;
    }
    policy.algorithm=megafmm::Algorithm::Auto;
    auto p=problem(128,64,128,dtype);
    require(megafmm::create_plan(p,policy)->algorithm_name()=="dense_cublaslt",
            "Auto microbatch metadata did not retain Dense policy");
    ++auto_dense_cases;
  }
  std::cout << "PASS_WGMMA_PSPLIT_MICROBATCH_BLOCKED_POLICY variant=44 forced_rejections=" << rejected_cases
            << " validate_before_empty_scale=1 auto_dense_dtypes=" << auto_dense_cases << " native_runs=0\n";
}
std::vector<int> wgmma_scheduled_parent_ks(int variant,bool bitwise=false) {
  if(wgmma_scheduled_extended_bk32(variant))return {64,128,192,256,384,448,512,576,640,1024,1088,1152,2112,2176};
  if(wgmma_scheduled_bk64(variant))return {128,256,384,512,768,896,1024,1152,1280,2048,2176,2304,4224,4352};
  if(variant>=36)return {64,128,192,256,1024,1088,1152,2048,2112,2176};
  if(variant>=34)return {64,128,192,256,384,448,1024,1088,1152,2112,2176};
  if(variant>=32)return {64,128,192,256,1024,1088,2048,2112};
  return bitwise ? std::vector<int>{64,128,192}:std::vector<int>{64,128,192,256};
}
template<class T> void wgmma_scheduled_suite(megafmm::DType dtype,int variant) {
  require(variant==30 || variant==31 || variant==32 || variant==33 || variant==34 || variant==35 || variant==36 || variant==38 || variant==39 || variant==40 || variant==41 || variant==42 || variant==43 || variant==44,"scheduled WGMMA suite needs enabled Mega30..36 or Mega38..44");
  auto exact=[dtype,variant](megafmm::Problem p,Pattern pattern,bool alias=false,bool certified=true) {
    const char* route=certified ? warp_mma_name(variant,p):"dense_cublaslt";
    std::cout << "BEGIN_WGMMA_" << wgmma_scheduled_family(variant) << " dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
              << " null_c0=" << (p.beta==0.0f) << " expected=" << route << std::endl;
    auto e=run_case<T>(p,megafmm::Algorithm::Megakernel,pattern,alias,certified,
                       1234567,variant,false,0,route);
    require(e.nonfinite==0 && e.maxabs==0,"scheduled WGMMA native/packed/Dense integer output incorrect");
  };
  // Parent K64/128/192/256 gives 1/2/3/4 stages. WAIT1 also exercises
  // K1024/1088/2048/2112 (16/17/32/33 stages), including long prime/drain
  // and repeated bank phase wraps. Long cases use bounded random integers:
  // OwnerRows grows its K16 coefficients past the BF16 exact-integer range.
  // Three-stage candidates add stage counts6/7/18/34 around bank phase wraps.
  // BK64 candidates38/39 keep N32 and double parent K per stage: stage8 is K1024.
  // Extended BK32 routes retain these parent stages; BN16_42/43 require parentN64, stage16 K1024.
  for(int k:wgmma_scheduled_parent_ks(variant))
    exact(problem(128,wgmma_scheduled_n16(variant) ? 32:64,k,dtype),
          k<=256 ? Pattern::OwnerRows:Pattern::Integers);
  exact(problem(256,128,wgmma_scheduled_bk64(variant) ? 128:64,dtype),Pattern::Integers);
  auto padded=problem(128,256,128,dtype);padded.alpha=.5f;padded.beta=-.25f;
  padded.lda+=8;padded.ldb+=8;padded.ldc+=1;padded.ldd+=1;
  exact(padded,Pattern::Integers,true);
  auto null_c0=problem(128,128,wgmma_scheduled_bk64(variant) ? 384:192,dtype);null_c0.beta=-0.0f;
  null_c0.ldc+=1;null_c0.ldd+=1;exact(null_c0,wgmma_scheduled_bk64(variant) ? Pattern::Integers:Pattern::OwnerRows);
  auto tail=problem(80,112,48,dtype);tail.alpha=.5f;tail.beta=-.25f;
  exact(tail,Pattern::Integers,true);
  exact(problem(128,64,32,dtype),Pattern::Integers);
  if(!wgmma_scheduled_n16(variant))exact(problem(128,32,64,dtype),Pattern::Integers);
  if(wgmma_scheduled_bk64(variant) || wgmma_scheduled_extended_bk32(variant))
  for(int k:(wgmma_scheduled_extended_bk32(variant) ? std::vector<int>{32,96,160}:std::vector<int>{64,192,320})) {
    // Partial BK32/BK64 leaves use actual P7, or Dense when input ld is odd.
    auto partial=problem(128,64,k,dtype);partial.alpha=.5f;partial.beta=-.25f;
    partial.lda+=8;partial.ldb+=8;partial.ldc+=1;partial.ldd+=1;
    exact(partial,Pattern::Integers,true);
    partial.lda+=1;exact(partial,Pattern::Integers,true);
  }
  auto odd_stride=problem(128,128,128,dtype);odd_stride.lda+=1;
  exact(odd_stride,Pattern::Integers);
  exact(problem(128,128,128,dtype),Pattern::Integers,false,false);
}
template<class T> void wgmma_scheduled_native_bitwise_suite(megafmm::DType dtype,int variant) {
  // Each candidate is compared only with the corresponding same-atom vector
  // pipeline: TMA30/WAIT1_32/three-stage34 N16 vs Mega28, and the corresponding
  // N32 candidates31/33/35 and enabled product-split36 vs Mega29; BK64_38
  // retains the same N32 atom and is compared with BK32 product-split36.
  // Microbatch44 retains N32/BK64 and compares raw U with RS-A39.
  // RS-A39 compares with BK64 SS38; BK32 RS-A40 and Hybrid S41 use SS36, explicitly N32. These
  // compare raw U and final D; no separately observable seven-P slots are claimed.
  require(variant==30 || variant==31 || variant==32 || variant==33 || variant==34 || variant==35 || variant==36 || variant==38 || variant==39 || variant==40 || variant==41 || variant==42 || variant==43 || variant==44,"scheduled WGMMA bits needs enabled Mega30..36 or Mega38..44");
  const bool bn16_candidate=variant==42 || variant==43;
  const bool sameatom_bn16=variant==42 && psplit_bn16_route_enabled(43);
  const bool crossatom_bn16=bn16_candidate && !sameatom_bn16;
  const char* bitwise_suffix=crossatom_bn16 ? "_CROSSATOM_BITWISE":"_NATIVE_BITWISE";
  const int reference_variant=sameatom_bn16 ? 43:bn16_candidate ? 36:wgmma_scheduled_extended_bk32(variant) ? 36:variant==44 ? 39:variant==39 ? 38:variant==38 ? 36:variant>=36 ? 29:28+variant%2;
  const int identity_k=wgmma_scheduled_bk64(variant) ? 128:64;
  const int alias_k=wgmma_scheduled_bk64(variant) ? 256:128;
  const int shift_k=wgmma_scheduled_bk64(variant) ? 384:192;
  for(auto mn:{std::pair<int,int>{128,wgmma_scheduled_n16(variant) ? 32:64},std::pair<int,int>{128,256},std::pair<int,int>{256,256}})
  for(int k:wgmma_scheduled_parent_ks(variant,true)) {
    auto p=problem(mn.first,mn.second,k,dtype);
    const bool alias=k==alias_k;
    const int input_shift=k==shift_k ? 8:0,output_shift=k==shift_k ? 1:0;
    if(k!=identity_k) {p.lda+=8;p.ldb+=8;p.ldc+=1;p.ldd+=1;p.alpha=.375f;p.beta=-.625f;}
    if(k==identity_k && mn.second==64)p.beta=-0.0f;
    uint32_t seed=331166u+uint32_t(p.m*31+p.n*17+p.k*7)+
                  (dtype==megafmm::DType::BF16 ? 0u:918273u);
    std::mt19937 rng(seed);
    std::vector<T> a(size_t(p.m)*p.lda,round_value<T>(3.25f)),b(size_t(p.k)*p.ldb,round_value<T>(-2.75f));
    auto random_input=[&]() {return round_value<T>(2.0f*(float(rng()>>8)/16777216.0f)-1.0f);};
    for(int i=0;i<p.m;++i)for(int l=0;l<p.k;++l)a[size_t(i)*p.lda+l]=random_input();
    for(int l=0;l<p.k;++l)for(int j=0;j<p.n;++j)b[size_t(l)*p.ldb+j]=random_input();
    std::vector<T> a_storage(a.size()+8,round_value<T>(5.5f)),b_storage(b.size()+8,round_value<T>(-6.5f));
    std::copy(a.begin(),a.end(),a_storage.begin()+input_shift);
    std::copy(b.begin(),b.end(),b_storage.begin()+input_shift);
    const size_t output_count=size_t(p.m)*p.ldd;
    std::vector<float> initial(output_count+2),actual(initial.size()),baseline(initial.size());
    for(size_t at=0;at<initial.size();++at)initial[at]=float(int((at+seed)%1024)-512)/8.0f;
    auto c=initial;
    for(int i=0;i<p.m;++i)for(int j=0;j<p.n;++j) {
      initial[size_t(i)*p.ldd+j]=std::numeric_limits<float>::quiet_NaN();
      c[size_t(i)*p.ldc+j]=float(int(rng()%257)-128)/64.0f;
    }
    Buffer da(a_storage.size()*sizeof(T)),db(b_storage.size()*sizeof(T)),
           dc((c.size()+1)*sizeof(float)),dd((initial.size()+1)*sizeof(float));
    const T* left=da.as<T>()+input_shift,*right=db.as<T>()+input_shift;
    float* source=dc.as<float>()+output_shift;
    float* output=alias ? source:dd.as<float>()+output_shift;
    std::cout << "BEGIN_WGMMA_" << wgmma_scheduled_family(variant) << bitwise_suffix << " dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " reference_variant=" << reference_variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
              << " null_c0=" << (p.beta==0.0f) << " input_shift_bytes=" << input_shift*sizeof(T)
              << " output_shift_bytes=" << output_shift*sizeof(float) << " seed=" << seed << std::endl;
    const bool identity_epilogue=p.alpha==1.0f && p.beta==0.0f;
    // New42 uses enabled SS43 as an N16 same-atom reference. SS43, or RS42
    // without enabled SS43, uses SS36 in a explicitly labeled N16/N32 cross-
    // atom bit check. No tolerance is relaxed if a cross-atom comparison fails.
    // The final output also uses the separate host RN-multiply/FMA oracle.
    // Active epilogues need three runs; identity cases need two.
    for(int stage=0;stage<(identity_epilogue ? 2:3);++stage) {
      const int current=stage==0 ? reference_variant:variant;
      auto current_problem=p;
      if(stage<2) {current_problem.alpha=1.0f;current_problem.beta=0.0f;}
      float* current_output=stage==2 ? output:dd.as<float>()+output_shift;
      megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
      policy.certified_range=true;policy.mega_variant=current;policy.workspace_limit=0;
      auto plan=megafmm::create_plan(current_problem,policy);
      require(plan->algorithm_name()==warp_mma_name(current,current_problem),
              "scheduled WGMMA bits selected another route");
      require(plan->workspace_bytes()==0,"scheduled WGMMA bits unexpectedly requires workspace");
      check(cudaMemcpy(da.data,a_storage.data(),a_storage.size()*sizeof(T),cudaMemcpyHostToDevice));
      check(cudaMemcpy(db.data,b_storage.data(),b_storage.size()*sizeof(T),cudaMemcpyHostToDevice));
      check(cudaMemcpy(source,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
      if(stage!=2 || !alias)check(cudaMemcpy(current_output,initial.data(),initial.size()*sizeof(float),cudaMemcpyHostToDevice));
      require(plan->lt_algorithm_config(left,right,current_problem.beta==0.0f ? nullptr:source,current_output,nullptr)=="none",
              "scheduled WGMMA bits metadata unexpectedly uses Dense");
      megafmm::run(*plan,left,right,current_problem.beta==0.0f ? nullptr:source,current_output,nullptr);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),current_output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i) {
        for(int j=0;j<p.n;++j)require(std::isfinite(actual[size_t(i)*p.ldd+j]),"scheduled WGMMA bits returned nonfinite/unwritten output");
        for(int j=int(p.n);j<p.ldd;++j)require(actual[size_t(i)*p.ldd+j]==initial[size_t(i)*p.ldd+j],"scheduled WGMMA bits overwrote D padding");
      }
      for(size_t at=output_count;at<actual.size();++at)require(actual[at]==initial[at],"scheduled WGMMA bits overwrote trailing D guard");
      if(stage==0)baseline=actual;
      else for(int i=0;i<p.m;++i)for(int j=0;j<p.n;++j) {
        const size_t at=size_t(i)*p.ldd+j;
        float expected=baseline[at];
        if(stage==2 || identity_epilogue) {
          // The volatile float store forces a separate FP32 RN multiply;
          // std::fma(float,float,float) supplies the single-rounding beta FMA.
          const volatile float scaled=p.alpha*baseline[at];
          expected=p.beta==0.0f ? float(scaled):std::fma(p.beta,c[size_t(i)*p.ldc+j],float(scaled));
        }
        uint32_t expected_bits=0,actual_bits=0;
        std::memcpy(&expected_bits,&expected,sizeof(expected_bits));
        std::memcpy(&actual_bits,&actual[at],sizeof(actual_bits));
        if(expected_bits!=actual_bits) {
          std::cerr << "WGMMA_" << wgmma_scheduled_family(variant) << "_BIT_MISMATCH dtype=" << megafmm::dtype_name(dtype)
                    << " variant=" << variant << " reference_variant=" << reference_variant << " phase=" << (stage==1 ? "raw_u":"contract_epilogue")
                    << " shape=" << p.m << ',' << p.n << ',' << p.k
                    << " row=" << i << " col=" << j << " byte_index=" << at*sizeof(float)
                    << " expected_bits=0x" << std::hex << expected_bits << " actual_bits=0x" << actual_bits << std::dec
                    << " raw_u=" << baseline[at] << " alpha=" << p.alpha << " beta=" << p.beta
                    << " c0=" << c[size_t(i)*p.ldc+j] << '\n';
          throw std::runtime_error(stage==1 ? (crossatom_bn16 ? "BN16 raw U bits differ from cross-atom N32 SS reference":"scheduled WGMMA raw U bits differ from matching WGMMA atom"):
                                   "scheduled WGMMA output bits differ from specified host RN-multiply/FMA oracle");
        }
      }
    }
    std::cout << "PASS_WGMMA_" << wgmma_scheduled_family(variant) << bitwise_suffix << " dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " reference_variant=" << reference_variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
              << " null_c0=" << (p.beta==0.0f)
              << (sameatom_bn16 ? " reference=sameatom_N16_rsa_vs_ss43_raw_u+specified_host_epilogue":
                  crossatom_bn16 ? " reference=crossatom_N16_vs_N32_ss36_raw_u+specified_host_epilogue":
                  wgmma_scheduled_extended_bk32(variant) ? " reference=sameatom_product_split_bk32_ss_raw_u+specified_host_epilogue":
                  variant==44 ? " reference=sameatom_N32_bk64_rsa39_raw_u+specified_host_epilogue":
                  variant==39 ? " reference=sameatom_product_split_bk64_ss_raw_u+specified_host_epilogue":
                  variant==38 ? " reference=sameatom_product_split_bk32_raw_u+specified_host_epilogue":
                                 " reference=sameatom_vector_pipeline_raw_u+specified_host_epilogue")
              << " raw_u_bits=1 explicit_host_epilogue=1 padding_guard=1\n";
  }
}
template<class T> void wgmma_scheduled_input_rebind_suite(megafmm::DType dtype,int variant) {
  auto p=problem(128,128,128,dtype);
  std::vector<T> a(size_t(p.m)*p.lda),b(size_t(p.k)*p.ldb);
  std::vector<float> initial(size_t(p.m)*p.ldd+2,-777.0f),actual(initial.size());
  Buffer da0((a.size()+8)*sizeof(T)),da1((a.size()+8)*sizeof(T)),
         db0((b.size()+8)*sizeof(T)),db1((b.size()+8)*sizeof(T)),dd(initial.size()*sizeof(float));
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
  policy.certified_range=true;policy.mega_variant=variant;policy.workspace_limit=0;
  auto plan=megafmm::create_plan(p,policy);
  require(plan->algorithm_name()==warp_mma_name(variant,p) && plan->workspace_bytes()==0,
          "scheduled WGMMA rebind plan is not a native zero-workspace route");
  struct Binding {int a_buffer,b_buffer,a_shift,b_shift;};
  const std::vector<Binding> bindings={{0,0,0,0},{1,0,0,0},{0,1,0,0},{1,1,0,0},
                                      {0,0,8,8},{1,1,8,8},{0,0,1,0},{0,1,0,1},
                                      {1,1,1,1},{0,0,0,0}};
  for(size_t phase=0;phase<bindings.size();++phase) {
    const auto binding=bindings[phase];
    T* left=(binding.a_buffer ? da1:da0).template as<T>()+binding.a_shift;
    T* right=(binding.b_buffer ? db1:db0).template as<T>()+binding.b_shift;
    // Fresh data at independent A/B allocations and 16B address phases
    // independently exercise both current input pointers and aligned recovery.
    for(int i=0;i<p.m;++i)for(int l=0;l<p.k;++l)
      a[size_t(i)*p.lda+l]=round_value<T>(float((i+3*l+int(phase))%5-2));
    for(int l=0;l<p.k;++l)for(int j=0;j<p.n;++j)
      b[size_t(l)*p.ldb+j]=round_value<T>(float((2*l+j+int(phase))%7-3));
    check(cudaMemcpy(left,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
    check(cudaMemcpy(right,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
    check(cudaMemcpy(dd.data,initial.data(),initial.size()*sizeof(float),cudaMemcpyHostToDevice));
    const bool dense=binding.a_shift==1 || binding.b_shift==1;
    check_lt_config(*plan,left,right,nullptr,dd.as<float>(),nullptr,dense ? "dense":"none",false);
    megafmm::run(*plan,left,right,nullptr,dd.as<float>(),nullptr);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(actual.data(),dd.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(int i=0;i<p.m;++i)for(int j=0;j<p.n;++j) {
      double expected=0;
      for(int l=0;l<p.k;++l)expected+=double_value(a[size_t(i)*p.lda+l])*double_value(b[size_t(l)*p.ldb+j]);
      require(double(actual[size_t(i)*p.ldd+j])==expected,"scheduled WGMMA current-pointer map rebind output incorrect");
    }
    for(size_t at=size_t(p.m)*p.ldd;at<actual.size();++at)
      require(actual[at]==initial[at],"scheduled WGMMA rebind overwrote trailing D sentinel");
  }
  std::cout << "PASS_WGMMA_" << wgmma_scheduled_family(variant) << "_INPUT_REBIND dtype=" << megafmm::dtype_name(dtype)
            << " variant=" << variant << " bindings=10 native_bindings=7 dense_bindings=3"
            << " independent_A_B_allocations=1 aligned16_phase=1 fresh_AB_each_binding=1"
            << " element_only_A_B_both_dense=1 aligned_recovery=1 trailing_guard=1\n";
}
template<class T> void wgmma_scheduled_graph_suite(megafmm::DType dtype,int variant) {
  const std::vector<int> tail_ks=wgmma_scheduled_extended_bk32(variant) ?
                               std::vector<int>{128,256,384,448,512,576,640,1024,1088,1152,2112,2176,32,96,160}:
                               wgmma_scheduled_bk64(variant) ?
                               std::vector<int>{256,512,768,896,1024,1152,1280,2048,2176,2304,4224,4352,64,192,320}:
                               variant>=36 ? std::vector<int>{1024,1088,1152,2048,2112,2176}:
                               variant>=34 ? std::vector<int>{384,448,1088,1152,2112,2176}:
                                                            std::vector<int>{1088,2112};
  for(int scenario=0;scenario<((wgmma_scheduled_bk64(variant) || wgmma_scheduled_extended_bk32(variant)) ? 18:variant>=34 ? 9:variant>=32 ? 5:3);++scenario) {
    auto p=scenario==0 ? problem(128,wgmma_scheduled_n16(variant) ? 32:64,wgmma_scheduled_bk64(variant) ? 128:64,dtype):
           scenario==1 ? problem(128,256,wgmma_scheduled_bk64(variant) ? 384:192,dtype):
           scenario==2 ? problem(80,112,48,dtype):
           problem(128,wgmma_scheduled_n16(variant) ? 32:64,tail_ks[scenario-3],dtype);
    p.lda+=8;p.ldb+=8;p.ldc+=1;p.ldd+=1;
    const bool active_beta=scenario!=0;
    if(active_beta) {p.alpha=.5f;p.beta=-.25f;} else p.beta=-0.0f;
    std::mt19937 rng(117799u+uint32_t(p.m*31+p.n*17+p.k*7));
    std::vector<T> a(size_t(p.m)*p.lda),b(size_t(p.k)*p.ldb);
    std::vector<float> c(size_t(p.m)*p.ldc+2,-777.0f),actual(c.size());
    std::vector<double> reference_ab(size_t(p.m)*p.n);
    const int input_shift=scenario==1 ? 8:0,output_shift=scenario==1 ? 1:0;
    Buffer da((a.size()+8)*sizeof(T)),db((b.size()+8)*sizeof(T)),dc((c.size()+1)*sizeof(float));
    T* left=da.as<T>()+input_shift,*right=db.as<T>()+input_shift;
    float* output=dc.as<float>()+output_shift;
    megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
    policy.certified_range=true;policy.mega_variant=variant;policy.workspace_limit=0;
    auto plan=megafmm::create_plan(p,policy);
    require(plan->algorithm_name()==warp_mma_name(variant,p) && plan->workspace_bytes()==0,
            "scheduled WGMMA graph route/workspace incorrect");
    std::cout << "BEGIN_WGMMA_" << wgmma_scheduled_family(variant) << "_GRAPH dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " actual=" << plan->algorithm_name() << " alias=" << active_beta
              << " null_c0=" << !active_beta << " input_shift_bytes=" << input_shift*sizeof(T)
              << " output_shift_bytes=" << output_shift*sizeof(float) << " refresh_ab=1" << std::endl;
    Stream stream;Graph graph;
    check(cudaStreamSynchronize(stream.value));
    check(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));
    megafmm::run(*plan,left,right,active_beta ? output:nullptr,output,nullptr,stream.value);
    check(cudaStreamEndCapture(stream.value,&graph.value));
    size_t nodes=0;check(cudaGraphGetNodes(graph.value,nullptr,&nodes));
    require(nodes==1,"scheduled WGMMA full invocation did not capture exactly one kernel node");
    cudaGraphNode_t node=nullptr;size_t one=1;check(cudaGraphGetNodes(graph.value,&node,&one));
    cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
    require(type==cudaGraphNodeTypeKernel,"scheduled WGMMA captured node is not a kernel");
    check(cudaGraphInstantiate(&graph.executable,graph.value,0));
    for(int replay=0;replay<3;++replay) {
      for(int i=0;i<p.m;++i)for(int l=0;l<p.k;++l)a[size_t(i)*p.lda+l]=round_value<T>(float(int(rng()%5)-2));
      for(int l=0;l<p.k;++l)for(int j=0;j<p.n;++j)b[size_t(l)*p.ldb+j]=round_value<T>(float(int(rng()%5)-2));
      std::fill(reference_ab.begin(),reference_ab.end(),0.0);
      for(int i=0;i<p.m;++i)for(int j=0;j<p.n;++j)for(int l=0;l<p.k;++l)
        reference_ab[size_t(i)*p.n+j]+=double_value(a[size_t(i)*p.lda+l])*double_value(b[size_t(l)*p.ldb+j]);
      if(scenario!=2)require(m1_integer_reference(p,a,b)==reference_ab,"scheduled WGMMA graph independent complete-P DAG differs from fresh AB");
      for(int i=0;i<p.m;++i)for(int j=0;j<p.n;++j)c[size_t(i)*p.ldc+j]=float((i+j+3*replay)%9-4)/4.0f;
      // Refresh original rounded inputs at captured addresses, and restore the
      // immutable C0 snapshot before replay. Capture never precomputes S/T.
      check(cudaMemcpyAsync(left,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice,stream.value));
      check(cudaMemcpyAsync(right,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice,stream.value));
      check(cudaMemcpyAsync(output,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice,stream.value));
      check(cudaGraphLaunch(graph.executable,stream.value));check(cudaStreamSynchronize(stream.value));
      check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i)for(int j=0;j<p.n;++j)
        require(double(actual[size_t(i)*p.ldd+j])==double(p.alpha)*reference_ab[size_t(i)*p.n+j]+double(p.beta)*c[size_t(i)*p.ldc+j],
                "scheduled WGMMA graph output differs from fresh original-input FP64 AB");
      for(int i=0;i<p.m;++i)for(int j=int(p.n);j<p.ldd;++j)
        require(actual[size_t(i)*p.ldd+j]==-777.0f,"scheduled WGMMA graph overwrote output padding");
      for(size_t at=size_t(p.m)*p.ldd;at<actual.size();++at)
        require(actual[at]==-777.0f,"scheduled WGMMA graph overwrote trailing D sentinel");
    }
    std::cout << "PASS_WGMMA_" << wgmma_scheduled_family(variant) << "_GRAPH dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " actual=" << plan->algorithm_name() << " alias=" << active_beta
              << " null_c0=" << !active_beta << " refresh_ab=1 fixed_capture_AB_addresses=1 replays=3 kernel_nodes=1\n";
  }
}
void wgmma_scheduled_policy_suite(int variant) {
  for(auto algorithm:{megafmm::Algorithm::Auto,megafmm::Algorithm::Dense,megafmm::Algorithm::WinogradLt,
                      megafmm::Algorithm::WinogradHopper,megafmm::Algorithm::Megakernel})
  for(int scenario=0;scenario<4;++scenario) {
    auto p=problem(128,128,128,megafmm::DType::BF16);
    if(scenario==1)p.m=0;if(scenario==2)p.alpha=0;if(scenario==3)p.k=0;
    megafmm::Policy invalid;invalid.algorithm=algorithm;invalid.mega_variant=45;invalid.certified_range=true;
    bool rejected=false;try {auto plan=megafmm::create_plan(p,invalid);}catch(const std::exception&) {rejected=true;}
    require(rejected,"invalid Mega45 accepted before normal/empty/alpha0/K0 dispatch");
  }
  for(auto dtype:{megafmm::DType::BF16,megafmm::DType::FP16}) {
    megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
    policy.mega_variant=variant;policy.certified_range=true;policy.workspace_limit=0;
    for(int zero_dim:{0,1}) {
      auto p=problem(zero_dim==0 ? 0:9,zero_dim==1 ? 0:11,17,dtype);
      auto plan=megafmm::create_plan(p,policy);
      require(plan->workspace_bytes()==0 && plan->lt_algorithm_config(nullptr,nullptr,nullptr,nullptr,nullptr)=="none",
              "scheduled WGMMA empty plan unexpectedly requires inputs/workspace");
      megafmm::run(*plan,nullptr,nullptr,nullptr,nullptr,nullptr);
    }
    for(bool alpha_zero:{false,true})for(float beta:{0.0f,.5f}) {
      auto p=problem(7,11,alpha_zero ? 13:0,dtype);p.alpha=alpha_zero ? 0:1;p.beta=beta;
      std::vector<float> c(size_t(p.m)*p.n),actual(c.size());
      for(size_t at=0;at<c.size();++at)c[at]=float(int(at%9)-4)/4.0f;
      Buffer dc(c.size()*sizeof(float)),dd(c.size()*sizeof(float));
      check(cudaMemcpy(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
      check(cudaMemset(dd.data,0xff,c.size()*sizeof(float)));
      auto plan=megafmm::create_plan(p,policy);
      require(plan->workspace_bytes()==0 &&
              plan->lt_algorithm_config(nullptr,nullptr,beta==0 ? nullptr:dc.as<float>(),dd.as<float>(),nullptr)=="none",
              "scheduled WGMMA scale plan unexpectedly requires maps/workspace");
      megafmm::run(*plan,nullptr,nullptr,beta==0 ? nullptr:dc.as<float>(),dd.as<float>(),nullptr);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),dd.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(size_t at=0;at<c.size();++at)require(actual[at]==beta*c[at],"scheduled WGMMA degenerate beta*C0 output incorrect");
    }
    auto p=problem(128,128,128,dtype);policy.algorithm=megafmm::Algorithm::Auto;
    require(megafmm::create_plan(p,policy)->algorithm_name()=="dense_cublaslt","scheduled WGMMA candidate enabled uncertified Auto SW");
  }
  std::cout << "PASS_WGMMA_" << wgmma_scheduled_family(variant) << "_POLICY variant=" << variant
            << " invalid45_cases=20 validate_before_empty_scale=1 empty_cases=4 scale_cases=8 auto_dense_dtypes=2\n";
}
template<class T> void wgmma_tma_alias_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_suite<T>(dtype,variant);
}
template<class T> void wgmma_tma_alias_native_bitwise_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_native_bitwise_suite<T>(dtype,variant);
}
template<class T> void wgmma_tma_alias_input_rebind_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_input_rebind_suite<T>(dtype,variant);
}
template<class T> void wgmma_tma_alias_graph_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_graph_suite<T>(dtype,variant);
}
void wgmma_tma_alias_policy_suite(int variant) {wgmma_scheduled_policy_suite(variant);}
template<class T> void wgmma_wait1_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_suite<T>(dtype,variant);
}
template<class T> void wgmma_wait1_native_bitwise_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_native_bitwise_suite<T>(dtype,variant);
}
template<class T> void wgmma_wait1_input_rebind_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_input_rebind_suite<T>(dtype,variant);
}
template<class T> void wgmma_wait1_graph_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_graph_suite<T>(dtype,variant);
}
void wgmma_wait1_policy_suite(int variant) {wgmma_scheduled_policy_suite(variant);}
template<class T> void wgmma_three_stage_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_suite<T>(dtype,variant);
}
template<class T> void wgmma_three_stage_native_bitwise_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_native_bitwise_suite<T>(dtype,variant);
}
template<class T> void wgmma_three_stage_input_rebind_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_input_rebind_suite<T>(dtype,variant);
}
template<class T> void wgmma_three_stage_graph_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_graph_suite<T>(dtype,variant);
}
void wgmma_three_stage_policy_suite(int variant) {wgmma_scheduled_policy_suite(variant);}
template<class T> void wgmma_psplit_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_native_bitwise_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_native_bitwise_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_input_rebind_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_input_rebind_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_graph_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_graph_suite<T>(dtype,variant);
}
void wgmma_psplit_policy_suite(int variant) {wgmma_scheduled_policy_suite(variant);}
template<class T> void wgmma_psplit_bk64_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_bk64_native_bitwise_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_native_bitwise_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_bk64_input_rebind_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_input_rebind_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_bk64_graph_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_graph_suite<T>(dtype,variant);
}
void wgmma_psplit_bk64_policy_suite(int variant) {wgmma_scheduled_policy_suite(variant);}
template<class T> void wgmma_psplit_rsa_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_rsa_native_bitwise_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_native_bitwise_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_rsa_input_rebind_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_input_rebind_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_rsa_graph_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_graph_suite<T>(dtype,variant);
}
void wgmma_psplit_rsa_policy_suite(int variant) {wgmma_scheduled_policy_suite(variant);}
template<class T> void wgmma_psplit_rsa_bk32_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_rsa_bk32_native_bitwise_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_native_bitwise_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_rsa_bk32_input_rebind_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_input_rebind_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_rsa_bk32_graph_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_graph_suite<T>(dtype,variant);
}
void wgmma_psplit_rsa_bk32_policy_suite(int variant) {wgmma_scheduled_policy_suite(variant);}
template<class T> void wgmma_psplit_hybrid_s_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_hybrid_s_native_bitwise_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_native_bitwise_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_hybrid_s_input_rebind_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_input_rebind_suite<T>(dtype,variant);
}
template<class T> void wgmma_psplit_hybrid_s_graph_suite(megafmm::DType dtype,int variant) {
  wgmma_scheduled_graph_suite<T>(dtype,variant);
}
void wgmma_psplit_hybrid_s_policy_suite(int variant) {wgmma_scheduled_policy_suite(variant);}
void psplit_disabled37_policy_suite() {
  size_t rejected_cases=0,auto_dense_cases=0;
  for(auto dtype:{megafmm::DType::BF16,megafmm::DType::FP16}) {
    megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Megakernel;
    policy.mega_variant=37;policy.certified_range=true;policy.workspace_limit=0;
    for(int scenario=0;scenario<4;++scenario) {
      auto p=problem(128,128,128,dtype);
      if(scenario==1)p.m=0;if(scenario==2)p.alpha=0;if(scenario==3)p.k=0;
      bool rejected=false;
      try {auto plan=megafmm::create_plan(p,policy);}
      catch(const std::invalid_argument& e) {
        // A specific disabled37 error is required before shape or degenerate
        // dispatch; no native or packed route may silently replace this request.
        rejected=std::string(e.what()).find("37")!=std::string::npos;
      }
      require(rejected,"disabled Mega37 accepted before normal/empty/alpha0/K0 dispatch");
      ++rejected_cases;
    }
    policy.algorithm=megafmm::Algorithm::Auto;
    auto p=problem(128,128,128,dtype);
    require(megafmm::create_plan(p,policy)->algorithm_name()=="dense_cublaslt",
            "Auto Mega37 metadata did not retain Dense policy");
    ++auto_dense_cases;
  }
  std::cout << "PASS_WGMMA_PSPLIT_DISABLED_POLICY variant=37 forced_rejections=" << rejected_cases
            << " validate_before_empty_scale=1 auto_dense_dtypes=" << auto_dense_cases
            << " native_runs=0 register_spill_gate_blocked=1\n";
}
template<class T> void fused_input_product_slots(megafmm::DType dtype,int variant) {
  auto p=problem(64,variant==9 ? 64:128,64,dtype);
  p.lda+=8;p.ldb+=8;
  std::vector<T> a(size_t(p.m)*p.lda,round_value<T>(0)),
                 b(size_t(p.k)*p.ldb,round_value<T>(0));
  std::mt19937 rng(928374u+unsigned(variant));
  for(int i=0;i<p.m;++i)for(int k=0;k<p.k;++k)
    a[size_t(i)*p.lda+k]=round_value<T>(float(int(rng()%7)-3));
  for(int k=0;k<p.k;++k)for(int j=0;j<p.n;++j)
    b[size_t(k)*p.ldb+j]=round_value<T>(float(int(rng()%7)-3));
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::WinogradHopper;
  policy.certified_range=true;policy.hopper_variant=variant;
  auto plan=megafmm::create_plan(p,policy);
  require(plan->algorithm_name()==batched_name(policy.algorithm,variant),"P-slot fixture is not native");
  require(plan->workspace_bytes()==fused_reserved_bytes(p),"P-slot fixture is not P-only");
  Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dd(size_t(p.m)*p.n*sizeof(float)),
         workspace(plan->workspace_bytes());
  check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
  check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
  megafmm::run(*plan,da.data,db.data,nullptr,dd.as<float>(),workspace.data);
  check(cudaDeviceSynchronize());
  const int mh=int(p.m/2),nh=int(p.n/2),kh=int(p.k/2);
  std::vector<float> actual(size_t(7)*mh*nh);
  check(cudaMemcpy(actual.data(),workspace.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
  // These integer transforms are exactly representable in either input dtype.
  // Independent original-quadrant FP64 dots check every compact P slot directly.
  for(int row=0;row<mh;++row)for(int col=0;col<nh;++col) {
    double reference[7]={};
    for(int k=0;k<kh;++k) {
      const double a11=double_value(a[size_t(row)*p.lda+k]);
      const double a12=double_value(a[size_t(row)*p.lda+k+kh]);
      const double a21=double_value(a[size_t(row+mh)*p.lda+k]);
      const double a22=double_value(a[size_t(row+mh)*p.lda+k+kh]);
      const double b11=double_value(b[size_t(k)*p.ldb+col]);
      const double b12=double_value(b[size_t(k)*p.ldb+col+nh]);
      const double b21=double_value(b[size_t(k+kh)*p.ldb+col]);
      const double b22=double_value(b[size_t(k+kh)*p.ldb+col+nh]);
      const double s1=a21+a22,s2=s1-a11,s3=a11-a21,s4=a12-s2;
      const double t1=b12-b11,t2=b22-t1,t3=b22-b12,t4=t2-b21;
      const double left[7]={a11,a12,s4,a22,s1,s2,s3};
      const double right[7]={b11,b21,b22,t4,t1,t2,t3};
      for(int product=0;product<7;++product)reference[product]+=left[product]*right[product];
    }
    for(int product=0;product<7;++product)
      require(double(actual[size_t(product)*mh*nh+size_t(row)*nh+col])==reference[product],
              "fused input compact product slot differs from independent FP64 integer dot");
  }
  std::cout << "PASS_FUSED_INPUT_PRODUCTS dtype=" << megafmm::dtype_name(dtype)
            << " variant=" << variant << " slots=7 elements=" << actual.size()
            << " native=1 reference=original_quadrant_fp64_integer_dot\n";
}
template<class T> void fused_input_suite(megafmm::DType dtype,int variant) {
  require(variant==9 || variant==10,"fused input suite needs Hopper9/10");
  fused_input_product_slots<T>(dtype,variant);
  auto exact=[dtype,variant](megafmm::Problem p,bool alias=false,bool certified=true,
                            size_t limit=size_t(8)<<30,bool tune=false) {
    const bool native=certified && fused_input_supported(p,variant) && fused_reserved_bytes(p)<=limit;
    const char* route=native ? batched_name(megafmm::Algorithm::WinogradHopper,variant):"dense_cublaslt";
    std::cout << "BEGIN_FUSED_INPUT dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
              << " null_c0=" << (p.beta==0.0f) << " expected=" << route << std::endl;
    require(megafmm::fused_leaf_can_implement(p,variant-9)==fused_input_supported(p,variant),
            "fused input metadata guard disagrees with independent shape/stride reference");
    if(fused_input_supported(p,variant))
      require(megafmm::fused_leaf_product_workspace_bytes(p,variant-9)==fused_product_bytes(p),
              "fused input helper product payload is not seven compact FP32 slots");
    auto e=run_case<T>(p,megafmm::Algorithm::WinogradHopper,Pattern::Integers,alias,certified,
                       1234567,variant,tune,limit,route);
    require(e.nonfinite==0 && e.maxabs==0,"fused input native/Dense fallback integer output incorrect");
  };
  const int n=variant==9 ? 64:128;
  // Leaf K32/64/96 corresponds to parent K64/128/192. Parent K32/96 is unsupported.
  for(int k:{64,128,192})exact(problem(64,n,k,dtype));
  exact(problem(512,1024,64,dtype));
  auto padded=problem(128,256,128,dtype);padded.lda+=8;padded.ldb+=8;
  padded.ldc+=1;padded.ldd+=1;padded.alpha=.5f;padded.beta=-.25f;
  exact(padded,true,true,size_t(8)<<30,true);
  auto null_c0=problem(128,128,64,dtype);null_c0.beta=-0.0f;null_c0.ldc+=1;null_c0.ldd+=1;
  exact(null_c0);
  for(int k:{32,96})exact(problem(64,n,k,dtype));
  exact(problem(80,112,48,dtype));
  if(variant==10)exact(problem(64,64,64,dtype));
  auto p=problem(128,128,128,dtype);
  exact(p,false,false);
  exact(p,false,true,0);
  exact(p,false,true,fused_reserved_bytes(p)-1);
  p.lda+=1;exact(p);
}
template<class T> void fused_input_native_bitwise_suite(megafmm::DType dtype,int variant) {
  require(variant==9 || variant==10,"fused native bits needs Hopper9/10");
  for(auto mn:{std::pair<int,int>{64,variant==9 ? 64:128},std::pair<int,int>{128,256},
               std::pair<int,int>{512,1024}})
  for(int k:{64,128,192}) {
    auto p=problem(mn.first,mn.second,k,dtype);
    const int input_shift=k==192 ? 8:0,output_shift=k==192 ? 1:0;
    if(k!=64) {p.lda+=8;p.ldb+=8;p.ldc+=1;p.ldd+=1;}
    if(k==128)p.beta=-0.0f;
    require(fused_input_supported(p,variant),"fused random fixture is not native");
    uint32_t seed=553311u+uint32_t(p.m*31+p.n*17+p.k*7+variant*137)+
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
    for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
      initial[size_t(i)*p.ldd+j]=std::numeric_limits<float>::quiet_NaN();
    Buffer da(a_storage.size()*sizeof(T)),db(b_storage.size()*sizeof(T)),
           dd((initial.size()+1)*sizeof(float));
    const T* left=da.as<T>()+input_shift,*right=db.as<T>()+input_shift;
    float* output=dd.as<float>()+output_shift;
    std::cout << "BEGIN_FUSED_INPUT_NATIVE_BITWISE dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " reference_variant=7 shape=" << p.m << ',' << p.n << ',' << p.k
              << " input_shift_bytes=" << input_shift*sizeof(T) << " output_shift_bytes=" << output_shift*sizeof(float)
              << " beta_signbit=" << std::signbit(p.beta) << " null_c0=1 seed=" << seed << std::endl;
    for(bool fused:{false,true}) {
      megafmm::Policy policy;policy.certified_range=true;
      policy.algorithm=fused ? megafmm::Algorithm::WinogradHopper:megafmm::Algorithm::Megakernel;
      policy.hopper_variant=variant;policy.mega_variant=7;
      auto plan=megafmm::create_plan(p,policy);Buffer workspace(plan->workspace_bytes());
      require(plan->algorithm_name()==(fused ? batched_name(policy.algorithm,variant):"megakernel_warp_mma_packed"),
              "fused random bitwise fixture selected another route");
      require(plan->workspace_bytes()==(fused ? fused_reserved_bytes(p):0),"fused random bitwise workspace mismatch");
      check(cudaMemcpy(da.data,a_storage.data(),a_storage.size()*sizeof(T),cudaMemcpyHostToDevice));
      check(cudaMemcpy(db.data,b_storage.data(),b_storage.size()*sizeof(T),cudaMemcpyHostToDevice));
      check(cudaMemcpy(output,initial.data(),initial.size()*sizeof(float),cudaMemcpyHostToDevice));
      require(plan->lt_algorithm_config(left,right,nullptr,output,workspace.data)=="none",
              "fused random bitwise metadata not none");
      megafmm::run(*plan,left,right,nullptr,output,workspace.data);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i) {
        for(int j=0;j<p.n;++j)
          require(std::isfinite(actual[size_t(i)*p.ldd+j]),"fused random bits returned nonfinite/unwritten output");
        for(int j=int(p.n);j<p.ldd;++j)
          require(actual[size_t(i)*p.ldd+j]==initial[size_t(i)*p.ldd+j],"fused random bits overwrote D padding");
      }
      for(size_t at=output_count;at<actual.size();++at)
        require(actual[at]==initial[at],"fused random bits overwrote trailing D guard");
      if(!fused)baseline=actual;
      else for(int i=0;i<p.m;++i)
        require(std::memcmp(baseline.data()+size_t(i)*p.ldd,actual.data()+size_t(i)*p.ldd,
                            size_t(p.n)*sizeof(float))==0,"fused-input final M1 output bits differ from Mega7");
    }
    std::cout << "PASS_FUSED_INPUT_NATIVE_BITWISE dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " reference_variant=7 shape=" << p.m << ',' << p.n << ',' << p.k
              << " input_shift_bytes=" << input_shift*sizeof(T) << " output_shift_bytes=" << output_shift*sizeof(float)
              << " beta_signbit=" << std::signbit(p.beta) << " null_c0=1 padding_guard=1\n";
  }
}
template<class T> void fused_input_output_rebind_suite(megafmm::DType dtype,int variant) {
  auto p=problem(128,128,128,dtype);p.alpha=.5f;p.beta=-.25f;
  std::mt19937 rng(776655u);
  std::vector<T> a(size_t(p.m)*p.lda),b(size_t(p.k)*p.ldb);
  std::vector<float> c(size_t(p.m)*p.ldc),actual(c.size());
  for(auto& value:a)value=round_value<T>(float(int(rng()%5)-2));
  for(auto& value:b)value=round_value<T>(float(int(rng()%5)-2));
  for(auto& value:c)value=float(int(rng()%9)-4)/4.0f;
  std::vector<double> reference(size_t(p.m)*p.n);
  for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j) {
    double sum=0;
    for(int l=0;l<p.k;++l)sum+=double_value(a[size_t(i)*p.lda+l])*double_value(b[size_t(l)*p.ldb+j]);
    reference[size_t(i)*p.n+j]=double(p.alpha)*sum+double(p.beta)*c[size_t(i)*p.ldc+j];
  }
  Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dc((c.size()+1)*sizeof(float)),
         dd((actual.size()+1)*sizeof(float));
  check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
  check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::WinogradHopper;
  policy.certified_range=true;policy.hopper_variant=variant;
  auto plan=megafmm::create_plan(p,policy);Buffer workspace(plan->workspace_bytes()+256);
  require(plan->algorithm_name()==batched_name(policy.algorithm,variant),"fused output rebind selected another route");
  require(plan->workspace_bytes()==fused_reserved_bytes(p),"fused output rebind is not P-only");
  require(megafmm::fused_leaf_full_tiles(p,da.data,db.data,workspace.as<float>(),variant-9),
          "fused low-level aligned binding rejected");
  require(!megafmm::fused_leaf_full_tiles(p,da.data,db.data,workspace.as<float>()+1,variant-9),
          "fused low-level P+4 binding accepted");
  bool rejected=false;
  try {megafmm::run(*plan,da.data,db.data,dc.as<float>(),dd.as<float>(),workspace.as<float>()+1);}
  catch(const std::exception&) {rejected=true;}
  require(rejected,"fused Plan accepted workspace without required 256-byte alignment");
  for(int mode=0;mode<3;++mode) {
    for(int shift:{0,1,0}) {
      float* source=dc.as<float>()+(mode==0 ? 0:shift);
      float* output=mode==2 ? source:dd.as<float>()+(mode==0 ? shift:0);
      check(cudaMemset(dc.data,0xff,(c.size()+1)*sizeof(float)));
      check(cudaMemset(dd.data,0xff,(actual.size()+1)*sizeof(float)));
      check(cudaMemcpy(source,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
      require(plan->lt_algorithm_config(da.data,db.data,source,output,workspace.data)=="none",
              "fused scalar output rebind unexpectedly chose Dense");
      megafmm::run(*plan,da.data,db.data,source,output,workspace.data);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(size_t i=0;i<actual.size();++i)
        require(double(actual[i])==reference[i],"fused same-plan D/C0+4 rebind returned incorrect output");
    }
    std::cout << "PASS_FUSED_INPUT_OUTPUT_REBIND dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " mode=" << mode << " alias=" << (mode==2)
              << " aligned_then_float_aligned_then_aligned=1 workspace_alignment_rejection=1\n";
  }
}
template<class T> void fused_input_graph_suite(megafmm::DType dtype,int variant) {
  for(bool active_beta:{false,true}) {
    auto p=active_beta ? problem(128,256,128,dtype):problem(64,variant==9 ? 64:128,64,dtype);
    p.lda+=8;p.ldb+=8;p.ldc+=1;p.ldd+=1;
    if(active_beta) {p.alpha=.5f;p.beta=-.25f;} else p.beta=-0.0f;
    std::mt19937 rng(998877u+uint32_t(p.m*31+p.n*17+p.k*7));
    std::vector<T> a(size_t(p.m)*p.lda),b(size_t(p.k)*p.ldb);
    std::vector<float> c(size_t(p.m)*p.ldc,-777.0f),actual(c.size());
    std::vector<double> reference_ab(size_t(p.m)*p.n);
    Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dc(c.size()*sizeof(float));
    megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::WinogradHopper;
    policy.certified_range=true;policy.hopper_variant=variant;
    auto plan=megafmm::create_plan(p,policy);Buffer workspace(plan->workspace_bytes());
    require(plan->algorithm_name()==batched_name(policy.algorithm,variant),"fused graph selected another route");
    require(plan->workspace_bytes()==fused_reserved_bytes(p),"fused graph is not P-only");
    std::cout << "BEGIN_FUSED_INPUT_GRAPH dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << active_beta
              << " null_c0=" << !active_beta << " refresh_ab=1" << std::endl;
    Stream stream;Graph graph;
    check(cudaStreamSynchronize(stream.value));
    check(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));
    megafmm::run(*plan,da.data,db.data,active_beta ? dc.as<float>():nullptr,dc.as<float>(),workspace.data,stream.value);
    check(cudaStreamEndCapture(stream.value,&graph.value));
    check(cudaGraphInstantiate(&graph.executable,graph.value,0));
    for(int replay=0;replay<3;++replay) {
      for(int i=0;i<p.m;++i) for(int l=0;l<p.k;++l)
        a[size_t(i)*p.lda+l]=round_value<T>(float(int(rng()%5)-2));
      for(int l=0;l<p.k;++l) for(int j=0;j<p.n;++j)
        b[size_t(l)*p.ldb+j]=round_value<T>(float(int(rng()%5)-2));
      std::fill(reference_ab.begin(),reference_ab.end(),0.0);
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
        for(int l=0;l<p.k;++l)
          reference_ab[size_t(i)*p.n+j]+=double_value(a[size_t(i)*p.lda+l])*double_value(b[size_t(l)*p.ldb+j]);
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
        c[size_t(i)*p.ldc+j]=float((i+j+3*replay)%9-4)/4.0f;
      // Captured pointers stay fixed. Refresh original rounded A/B and restore
      // the immutable host C0 snapshot on the replay stream before each launch.
      check(cudaMemcpyAsync(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice,stream.value));
      check(cudaMemcpyAsync(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice,stream.value));
      check(cudaMemcpyAsync(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice,stream.value));
      check(cudaGraphLaunch(graph.executable,stream.value));
      check(cudaStreamSynchronize(stream.value));
      check(cudaMemcpy(actual.data(),dc.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
        require(double(actual[size_t(i)*p.ldd+j])==double(p.alpha)*reference_ab[size_t(i)*p.n+j]+double(p.beta)*c[size_t(i)*p.ldc+j],
                "fused graph replay disagrees with fresh original-input FP64 AB");
      for(int i=0;i<p.m;++i) for(int j=p.n;j<p.ldd;++j)
        require(actual[size_t(i)*p.ldd+j]==-777.0f,"fused graph replay overwrote D padding");
    }
    std::cout << "PASS_FUSED_INPUT_GRAPH dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " alias=" << active_beta << " null_c0=" << !active_beta << " refresh_ab=1 replays=3\n";
  }
}
template<class T> void packed_input_product_slots(megafmm::DType dtype,int variant) {
  auto p=problem(64,variant==11 ? 64:128,64,dtype);
  p.lda+=8;p.ldb+=8;
  std::vector<T> a(size_t(p.m)*p.lda,round_value<T>(0)),
                 b(size_t(p.k)*p.ldb,round_value<T>(0));
  std::mt19937 rng(928374u+unsigned(variant));
  for(int i=0;i<p.m;++i)for(int k=0;k<p.k;++k)
    a[size_t(i)*p.lda+k]=round_value<T>(float(int(rng()%7)-3));
  for(int k=0;k<p.k;++k)for(int j=0;j<p.n;++j)
    b[size_t(k)*p.ldb+j]=round_value<T>(float(int(rng()%7)-3));
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::WinogradHopper;
  policy.certified_range=true;policy.hopper_variant=variant;
  auto plan=megafmm::create_plan(p,policy);
  require(plan->algorithm_name()==batched_name(policy.algorithm,variant),"P-slot fixture is not native");
  check_packed_workspace(*plan,p);
  Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dd(size_t(p.m)*p.n*sizeof(float)),
         workspace(plan->workspace_bytes());
  check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
  check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
  megafmm::run(*plan,da.data,db.data,nullptr,dd.as<float>(),workspace.data);
  check(cudaDeviceSynchronize());
  const int mh=int(p.m/2),nh=int(p.n/2),kh=int(p.k/2);
  std::vector<float> actual(size_t(7)*mh*nh);
  const auto layout=packed_test_layout(p);
  check(cudaMemcpy(actual.data(),workspace.as<char>()+layout.p_offset,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
  std::vector<T> actual_a(layout.a_bytes/sizeof(T)),actual_b(layout.b_bytes/sizeof(T));
  check(cudaMemcpy(actual_a.data(),workspace.data,layout.a_bytes,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(actual_b.data(),workspace.as<char>()+layout.b_offset,layout.b_bytes,cudaMemcpyDeviceToHost));
  for(int row=0;row<mh;++row)for(int k=0;k<kh;++k) {
    const double a11=double_value(a[size_t(row)*p.lda+k]);
    const double a12=double_value(a[size_t(row)*p.lda+k+kh]);
    const double a21=double_value(a[size_t(row+mh)*p.lda+k]);
    const double a22=double_value(a[size_t(row+mh)*p.lda+k+kh]);
    const double s1=a21+a22,s2=s1-a11,s3=a11-a21,s4=a12-s2;
    const double expected[7]={a11,a12,s4,a22,s1,s2,s3};
    for(int product=0;product<7;++product)
      require(double_value(actual_a[size_t(product)*mh*kh+size_t(row)*kh+k])==expected[product],
              "packed A7 slot differs from independent original-quadrant integer transform");
  }
  for(int k=0;k<kh;++k)for(int col=0;col<nh;++col) {
    const double b11=double_value(b[size_t(k)*p.ldb+col]);
    const double b12=double_value(b[size_t(k)*p.ldb+col+nh]);
    const double b21=double_value(b[size_t(k+kh)*p.ldb+col]);
    const double b22=double_value(b[size_t(k+kh)*p.ldb+col+nh]);
    const double t1=b12-b11,t2=b22-t1,t3=b22-b12,t4=t2-b21;
    const double expected[7]={b11,b21,b22,t4,t1,t2,t3};
    for(int product=0;product<7;++product)
      require(double_value(actual_b[size_t(product)*kh*nh+size_t(k)*nh+col])==expected[product],
              "packed B7 slot differs from independent original-quadrant integer transform");
  }
  // These integer transforms are exactly representable in either input dtype.
  // Independent original-quadrant FP64 dots check every compact P slot directly.
  for(int row=0;row<mh;++row)for(int col=0;col<nh;++col) {
    double reference[7]={};
    for(int k=0;k<kh;++k) {
      const double a11=double_value(a[size_t(row)*p.lda+k]);
      const double a12=double_value(a[size_t(row)*p.lda+k+kh]);
      const double a21=double_value(a[size_t(row+mh)*p.lda+k]);
      const double a22=double_value(a[size_t(row+mh)*p.lda+k+kh]);
      const double b11=double_value(b[size_t(k)*p.ldb+col]);
      const double b12=double_value(b[size_t(k)*p.ldb+col+nh]);
      const double b21=double_value(b[size_t(k+kh)*p.ldb+col]);
      const double b22=double_value(b[size_t(k+kh)*p.ldb+col+nh]);
      const double s1=a21+a22,s2=s1-a11,s3=a11-a21,s4=a12-s2;
      const double t1=b12-b11,t2=b22-t1,t3=b22-b12,t4=t2-b21;
      const double left[7]={a11,a12,s4,a22,s1,s2,s3};
      const double right[7]={b11,b21,b22,t4,t1,t2,t3};
      for(int product=0;product<7;++product)reference[product]+=left[product]*right[product];
    }
    for(int product=0;product<7;++product)
      require(double(actual[size_t(product)*mh*nh+size_t(row)*nh+col])==reference[product],
              "packed input compact product slot differs from independent FP64 integer dot");
  }
  std::cout << "PASS_PACKED_INPUT_PRODUCTS dtype=" << megafmm::dtype_name(dtype)
            << " variant=" << variant << " slots=7 elements=" << actual.size()
            << " native=1 input_slots_verified=14 reference=original_quadrant_fp64_integer_dot\n";
}
template<class T> void packed_input_suite(megafmm::DType dtype,int variant) {
  require(variant==11 || variant==12,"packed input suite needs Hopper11/12");
  packed_input_product_slots<T>(dtype,variant);
  auto exact=[dtype,variant](megafmm::Problem p,bool alias=false,bool certified=true,
                            size_t limit=size_t(8)<<30,bool tune=false) {
    const bool native=certified && packed_input_supported(p,variant) && packed_test_layout(p).payload<=limit;
    const char* route=native ? batched_name(megafmm::Algorithm::WinogradHopper,variant):"dense_cublaslt";
    std::cout << "BEGIN_PACKED_INPUT dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << alias
              << " null_c0=" << (p.beta==0.0f) << " expected=" << route << std::endl;
    check_packed_metadata(p,variant);
    auto e=run_case<T>(p,megafmm::Algorithm::WinogradHopper,Pattern::Integers,alias,certified,
                       1234567,variant,tune,limit,route);
    require(e.nonfinite==0 && e.maxabs==0,"packed input native/Dense fallback integer output incorrect");
  };
  const int n=variant==11 ? 64:128;
  // Leaf K32/64/96 corresponds to parent K64/128/192. Parent K32/96 is unsupported.
  for(int k:{64,128,192})exact(problem(64,n,k,dtype));
  exact(problem(512,1024,64,dtype));
  auto padded=problem(128,256,128,dtype);padded.lda+=8;padded.ldb+=8;
  padded.ldc+=1;padded.ldd+=1;padded.alpha=.5f;padded.beta=-.25f;
  exact(padded,true,true,size_t(8)<<30,true);
  auto null_c0=problem(128,128,64,dtype);null_c0.beta=-0.0f;null_c0.ldc+=1;null_c0.ldd+=1;
  exact(null_c0);
  for(int k:{32,96})exact(problem(64,n,k,dtype));
  exact(problem(80,112,48,dtype));
  if(variant==12)exact(problem(64,64,64,dtype));
  auto p=problem(128,128,128,dtype);
  exact(p,false,false);
  exact(p,false,true,0);
  exact(p,false,true,packed_test_layout(p).payload-1);
  exact(p,false,true,packed_test_layout(p).payload,true);
  auto odd_b=p;odd_b.ldb+=1;exact(odd_b);
  for(bool ta:{false,true})for(bool tb:{false,true}) {
    if(!ta && !tb)continue;
    auto transposed=p;transposed.trans_a=ta;transposed.trans_b=tb;exact(transposed);
  }
  p.lda+=1;exact(p);
  const auto large=problem(1024,1024,1024,dtype);
  check_packed_metadata(large,variant);
  require(packed_test_layout(large).payload==size_t(14)*1024*1024,
          "1024 packed payload must be exactly 14 MiB before Dense scratch");
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::WinogradHopper;
  policy.certified_range=true;policy.hopper_variant=variant;
  auto plan=megafmm::create_plan(large,policy);
  require(plan->algorithm_name()==batched_name(policy.algorithm,variant),"1024 packed metadata selected Dense");
  check_packed_workspace(*plan,large,policy.workspace_limit);
  std::cout << "PASS_PACKED_INPUT_WORKSPACE dtype=" << megafmm::dtype_name(dtype)
            << " variant=" << variant << " shape=1024,1024,1024 payload_bytes="
            << packed_test_layout(large).payload << " reserved_bytes=" << plan->workspace_bytes()
            << " cap_bytes=" << policy.workspace_limit << " payload_alignment=256\n";
}
template<class T> void packed_input_native_bitwise_suite(megafmm::DType dtype,int variant) {
  require(variant==11 || variant==12,"packed native bits needs Hopper11/12");
  for(auto mn:{std::pair<int,int>{64,variant==11 ? 64:128},std::pair<int,int>{128,256},
               std::pair<int,int>{512,1024}})
  for(int k:{64,128,192}) {
    auto p=problem(mn.first,mn.second,k,dtype);
    const int input_shift=k==192 ? 8:0,output_shift=k==192 ? 1:0;
    if(k!=64) {p.lda+=8;p.ldb+=8;p.ldc+=1;p.ldd+=1;}
    if(k==128)p.beta=-0.0f;
    require(packed_input_supported(p,variant),"packed random fixture is not native");
    uint32_t seed=553311u+uint32_t(p.m*31+p.n*17+p.k*7+variant*137)+
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
    std::vector<float> initial(output_count+2),actual(initial.size()),baseline(initial.size()),fused_output(initial.size());
    std::vector<float> fused_products(size_t(7)*(p.m/2)*(p.n/2)),packed_products(fused_products.size());
    for(size_t at=0;at<initial.size();++at)initial[at]=float(int((at+seed)%1024)-512)/8.0f;
    for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
      initial[size_t(i)*p.ldd+j]=std::numeric_limits<float>::quiet_NaN();
    Buffer da(a_storage.size()*sizeof(T)),db(b_storage.size()*sizeof(T)),
           dd((initial.size()+1)*sizeof(float));
    const T* left=da.as<T>()+input_shift,*right=db.as<T>()+input_shift;
    float* output=dd.as<float>()+output_shift;
    std::cout << "BEGIN_PACKED_INPUT_NATIVE_BITWISE dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " reference_variant=7 paired_fused_variant=" << variant-2 << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " input_shift_bytes=" << input_shift*sizeof(T) << " output_shift_bytes=" << output_shift*sizeof(float)
              << " beta_signbit=" << std::signbit(p.beta) << " null_c0=1 seed=" << seed << std::endl;
    for(int route:{0,1,2}) {
      const bool native_leaf=route!=0,paired_fused=route==1,packed=route==2;
      megafmm::Policy policy;policy.certified_range=true;
      policy.algorithm=native_leaf ? megafmm::Algorithm::WinogradHopper:megafmm::Algorithm::Megakernel;
      policy.hopper_variant=paired_fused ? variant-2:variant;policy.mega_variant=7;
      auto plan=megafmm::create_plan(p,policy);Buffer workspace(plan->workspace_bytes());
      require(plan->algorithm_name()==(native_leaf ? batched_name(policy.algorithm,policy.hopper_variant):"megakernel_warp_mma_packed"),
              "packed random bitwise fixture selected another route");
      if(packed)check_packed_workspace(*plan,p);
      else if(paired_fused)require(plan->workspace_bytes()==fused_reserved_bytes(p),"paired fused fixture is not P-only");
      else require(plan->workspace_bytes()==0,"packed random Mega7 reference unexpectedly requires workspace");
      check(cudaMemcpy(da.data,a_storage.data(),a_storage.size()*sizeof(T),cudaMemcpyHostToDevice));
      check(cudaMemcpy(db.data,b_storage.data(),b_storage.size()*sizeof(T),cudaMemcpyHostToDevice));
      check(cudaMemcpy(output,initial.data(),initial.size()*sizeof(float),cudaMemcpyHostToDevice));
      require(plan->lt_algorithm_config(left,right,nullptr,output,workspace.data)=="none",
              "packed random bitwise metadata not none");
      megafmm::run(*plan,left,right,nullptr,output,workspace.data);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i) {
        for(int j=0;j<p.n;++j)
          require(std::isfinite(actual[size_t(i)*p.ldd+j]),"packed random bits returned nonfinite/unwritten output");
        for(int j=int(p.n);j<p.ldd;++j)
          require(actual[size_t(i)*p.ldd+j]==initial[size_t(i)*p.ldd+j],"packed random bits overwrote D padding");
      }
      for(size_t at=output_count;at<actual.size();++at)
        require(actual[at]==initial[at],"packed random bits overwrote trailing D guard");
      if(route==0)baseline=actual;
      else for(int i=0;i<p.m;++i)
        require(std::memcmp(baseline.data()+size_t(i)*p.ldd,actual.data()+size_t(i)*p.ldd,
                            size_t(p.n)*sizeof(float))==0,"matched-control final M1 output bits differ from Mega7");
      if(paired_fused) {
        fused_output=actual;
        check(cudaMemcpy(fused_products.data(),workspace.data,fused_products.size()*sizeof(float),cudaMemcpyDeviceToHost));
      } else if(packed) {
        const auto layout=packed_test_layout(p);
        check(cudaMemcpy(packed_products.data(),workspace.as<char>()+layout.p_offset,
                         packed_products.size()*sizeof(float),cudaMemcpyDeviceToHost));
        require(std::memcmp(fused_products.data(),packed_products.data(),
                            packed_products.size()*sizeof(float))==0,
                "matched packed/fused complete seven P slots differ on the same rounded random inputs");
        for(int i=0;i<p.m;++i)
          require(std::memcmp(fused_output.data()+size_t(i)*p.ldd,actual.data()+size_t(i)*p.ldd,
                              size_t(p.n)*sizeof(float))==0,
                  "matched packed/fused final D bits differ on the same rounded random inputs");
      }
    }
    std::cout << "PASS_PACKED_INPUT_NATIVE_BITWISE dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " reference_variant=7 paired_fused_variant=" << variant-2 << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " input_shift_bytes=" << input_shift*sizeof(T) << " output_shift_bytes=" << output_shift*sizeof(float)
              << " beta_signbit=" << std::signbit(p.beta) << " null_c0=1 padding_guard=1 paired_p_slots=7 paired_final_d=1\n";
  }
}
template<class T> void packed_input_output_rebind_suite(megafmm::DType dtype,int variant) {
  auto p=problem(128,128,128,dtype);p.alpha=.5f;p.beta=-.25f;
  std::mt19937 rng(776655u);
  std::vector<T> a(size_t(p.m)*p.lda),b(size_t(p.k)*p.ldb);
  std::vector<float> c(size_t(p.m)*p.ldc),actual(c.size());
  for(auto& value:a)value=round_value<T>(float(int(rng()%5)-2));
  for(auto& value:b)value=round_value<T>(float(int(rng()%5)-2));
  for(auto& value:c)value=float(int(rng()%9)-4)/4.0f;
  std::vector<double> reference(size_t(p.m)*p.n);
  for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j) {
    double sum=0;
    for(int l=0;l<p.k;++l)sum+=double_value(a[size_t(i)*p.lda+l])*double_value(b[size_t(l)*p.ldb+j]);
    reference[size_t(i)*p.n+j]=double(p.alpha)*sum+double(p.beta)*c[size_t(i)*p.ldc+j];
  }
  Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dc((c.size()+1)*sizeof(float)),
         dd((actual.size()+1)*sizeof(float));
  check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
  check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::WinogradHopper;
  policy.certified_range=true;policy.hopper_variant=variant;
  auto plan=megafmm::create_plan(p,policy);Buffer workspace(plan->workspace_bytes()+256);
  require(plan->algorithm_name()==batched_name(policy.algorithm,variant),"packed output rebind selected another route");
  check_packed_workspace(*plan,p);
  const auto layout=packed_test_layout(p);
  const void* a7=workspace.data;
  const void* b7=workspace.as<char>()+layout.b_offset;
  const float* products=reinterpret_cast<float*>(workspace.as<char>()+layout.p_offset);
  require(megafmm::packed_leaf_original_input_guard(p,da.data,db.data,variant-11),
          "packed original aligned A/B rejected");
  require(!megafmm::packed_leaf_original_input_guard(p,da.as<T>()+1,db.data,variant-11) &&
          !megafmm::packed_leaf_original_input_guard(p,da.data,db.as<T>()+1,variant-11),
          "packed original A+2/B+2 accepted");
  require(megafmm::packed_leaf_full_tiles(p,a7,b7,products,variant-11),
          "packed low-level aligned binding rejected");
  require(!megafmm::packed_leaf_full_tiles(p,workspace.as<char>()+2,b7,products,variant-11) &&
          !megafmm::packed_leaf_full_tiles(p,a7,static_cast<const char*>(b7)+2,products,variant-11) &&
          !megafmm::packed_leaf_full_tiles(p,a7,b7,products+1,variant-11),
          "packed low-level A7+2/B7+2/P+4 binding accepted");
  require(!megafmm::packed_leaf_full_tiles(p,nullptr,b7,products,variant-11) &&
          !megafmm::packed_leaf_full_tiles(p,a7,nullptr,products,variant-11) &&
          !megafmm::packed_leaf_full_tiles(p,a7,b7,nullptr,variant-11),
          "packed low-level null compact binding accepted");
  const void* overflowing=reinterpret_cast<const void*>(std::numeric_limits<uintptr_t>::max()-15);
  require(!megafmm::packed_leaf_full_tiles(p,overflowing,b7,products,variant-11) &&
          !megafmm::packed_leaf_full_tiles(p,a7,overflowing,products,variant-11) &&
          !megafmm::packed_leaf_full_tiles(p,a7,b7,static_cast<const float*>(overflowing),variant-11),
          "packed low-level compact pointer range overflow accepted");
  bool rejected=false;
  try {megafmm::run(*plan,da.data,db.data,dc.as<float>(),dd.as<float>(),workspace.as<float>()+1);}
  catch(const std::exception&) {rejected=true;}
  require(rejected,"packed Plan accepted workspace without required 256-byte alignment");
  for(int mode=0;mode<3;++mode) {
    for(int shift:{0,1,0}) {
      float* source=dc.as<float>()+(mode==0 ? 0:shift);
      float* output=mode==2 ? source:dd.as<float>()+(mode==0 ? shift:0);
      check(cudaMemset(dc.data,0xff,(c.size()+1)*sizeof(float)));
      check(cudaMemset(dd.data,0xff,(actual.size()+1)*sizeof(float)));
      check(cudaMemcpy(source,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
      require(plan->lt_algorithm_config(da.data,db.data,source,output,workspace.data)=="none",
              "packed scalar output rebind unexpectedly chose Dense");
      megafmm::run(*plan,da.data,db.data,source,output,workspace.data);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(size_t i=0;i<actual.size();++i)
        require(double(actual[i])==reference[i],"packed same-plan D/C0+4 rebind returned incorrect output");
    }
    std::cout << "PASS_PACKED_INPUT_OUTPUT_REBIND dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " mode=" << mode << " alias=" << (mode==2)
              << " aligned_then_float_aligned_then_aligned=1 workspace_alignment_rejection=1\n";
  }
}
template<class T> void packed_input_graph_suite(megafmm::DType dtype,int variant) {
  for(bool active_beta:{false,true}) {
    auto p=active_beta ? problem(128,256,128,dtype):problem(64,variant==11 ? 64:128,64,dtype);
    p.lda+=8;p.ldb+=8;p.ldc+=1;p.ldd+=1;
    if(active_beta) {p.alpha=.5f;p.beta=-.25f;} else p.beta=-0.0f;
    std::mt19937 rng(998877u+uint32_t(p.m*31+p.n*17+p.k*7));
    std::vector<T> a(size_t(p.m)*p.lda),b(size_t(p.k)*p.ldb);
    std::vector<float> c(size_t(p.m)*p.ldc,-777.0f),actual(c.size());
    std::vector<double> reference_ab(size_t(p.m)*p.n);
    Buffer da(a.size()*sizeof(T)),db(b.size()*sizeof(T)),dc(c.size()*sizeof(float));
    megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::WinogradHopper;
    policy.certified_range=true;policy.hopper_variant=variant;
    auto plan=megafmm::create_plan(p,policy);Buffer workspace(plan->workspace_bytes());
    require(plan->algorithm_name()==batched_name(policy.algorithm,variant),"packed graph selected another route");
    check_packed_workspace(*plan,p);
    std::cout << "BEGIN_PACKED_INPUT_GRAPH dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta << " alias=" << active_beta
              << " null_c0=" << !active_beta << " refresh_ab=1" << std::endl;
    Stream stream;Graph graph;
    check(cudaStreamSynchronize(stream.value));
    check(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));
    megafmm::run(*plan,da.data,db.data,active_beta ? dc.as<float>():nullptr,dc.as<float>(),workspace.data,stream.value);
    check(cudaStreamEndCapture(stream.value,&graph.value));
    check(cudaGraphInstantiate(&graph.executable,graph.value,0));
    for(int replay=0;replay<3;++replay) {
      for(int i=0;i<p.m;++i) for(int l=0;l<p.k;++l)
        a[size_t(i)*p.lda+l]=round_value<T>(float(int(rng()%5)-2));
      for(int l=0;l<p.k;++l) for(int j=0;j<p.n;++j)
        b[size_t(l)*p.ldb+j]=round_value<T>(float(int(rng()%5)-2));
      std::fill(reference_ab.begin(),reference_ab.end(),0.0);
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
        for(int l=0;l<p.k;++l)
          reference_ab[size_t(i)*p.n+j]+=double_value(a[size_t(i)*p.lda+l])*double_value(b[size_t(l)*p.ldb+j]);
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
        c[size_t(i)*p.ldc+j]=float((i+j+3*replay)%9-4)/4.0f;
      // Captured pointers stay fixed. Refresh original rounded A/B and restore
      // the immutable host C0 snapshot on the replay stream before each launch.
      check(cudaMemcpyAsync(da.data,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice,stream.value));
      check(cudaMemcpyAsync(db.data,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice,stream.value));
      check(cudaMemcpyAsync(dc.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice,stream.value));
      check(cudaGraphLaunch(graph.executable,stream.value));
      check(cudaStreamSynchronize(stream.value));
      check(cudaMemcpy(actual.data(),dc.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.m;++i) for(int j=0;j<p.n;++j)
        require(double(actual[size_t(i)*p.ldd+j])==double(p.alpha)*reference_ab[size_t(i)*p.n+j]+double(p.beta)*c[size_t(i)*p.ldc+j],
                "packed graph replay disagrees with fresh original-input FP64 AB");
      for(int i=0;i<p.m;++i) for(int j=p.n;j<p.ldd;++j)
        require(actual[size_t(i)*p.ldd+j]==-777.0f,"packed graph replay overwrote D padding");
    }
    std::cout << "PASS_PACKED_INPUT_GRAPH dtype=" << megafmm::dtype_name(dtype)
              << " variant=" << variant << " shape=" << p.m << ',' << p.n << ',' << p.k
              << " alias=" << active_beta << " null_c0=" << !active_beta << " refresh_ab=1 replays=3\n";
  }
}
template<class T> void packed_input_policy_suite(megafmm::DType dtype,int variant) {
  megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::WinogradHopper;
  policy.certified_range=true;policy.hopper_variant=variant;
  auto p=problem(128,128,128,dtype);p.beta=.5f;
  auto plan=megafmm::create_plan(p,policy);
  check_packed_workspace(*plan,p,policy.workspace_limit);
  const size_t ab=size_t(p.m)*p.lda*sizeof(T),bb=size_t(p.k)*p.ldb*sizeof(T),
               cb=size_t(p.m)*p.ldc*sizeof(float);
  Buffer da(ab),db(bb),dc(cb+sizeof(float)),dd(cb+sizeof(float)),workspace(plan->workspace_bytes()+256);
  auto reject=[&](const megafmm::Plan& current,const void* a,const void* b,const float* c,
                  float* d,void* work,const char* reason) {
    bool runtime=false,metadata=false;
    try {megafmm::run(current,a,b,c,d,work);}catch(const std::exception&) {runtime=true;}
    try {(void)current.lt_algorithm_config(a,b,c,d,work);}catch(const std::exception&) {metadata=true;}
    require(runtime && metadata,std::string("packed run/metadata accepted ")+reason);
  };
  reject(*plan,nullptr,db.data,dc.as<float>(),dd.as<float>(),workspace.data,"null A");
  reject(*plan,da.data,nullptr,dc.as<float>(),dd.as<float>(),workspace.data,"null B");
  reject(*plan,da.data,db.data,nullptr,dd.as<float>(),workspace.data,"missing active C0");
  reject(*plan,da.data,db.data,dc.as<float>(),dd.as<float>(),nullptr,"null workspace");
  reject(*plan,da.data,db.data,dc.as<float>(),dd.as<float>(),workspace.as<char>()+4,"workspace+4");
  reject(*plan,da.data,db.data,dc.as<float>(),da.as<float>(),workspace.data,"A/D overlap");
  reject(*plan,da.data,db.data,dc.as<float>(),db.as<float>(),workspace.data,"B/D overlap");
  reject(*plan,workspace.data,db.data,dc.as<float>(),dd.as<float>(),workspace.data,"A/workspace overlap");
  reject(*plan,da.data,workspace.data,dc.as<float>(),dd.as<float>(),workspace.data,"B/workspace overlap");
  reject(*plan,da.data,db.data,workspace.as<float>(),dd.as<float>(),workspace.data,"C0/workspace overlap");
  reject(*plan,da.data,db.data,dc.as<float>(),workspace.as<float>(),workspace.data,"D/workspace overlap");
  reject(*plan,da.data,db.data,dc.as<float>(),dc.as<float>()+1,workspace.data,"partial C0/D overlap");
  const void* overflowing=reinterpret_cast<const void*>(std::numeric_limits<uintptr_t>::max()-15);
  reject(*plan,overflowing,db.data,dc.as<float>(),dd.as<float>(),workspace.data,"A pointer range overflow");
  reject(*plan,da.data,overflowing,dc.as<float>(),dd.as<float>(),workspace.data,"B pointer range overflow");
  auto unequal=p;unequal.ldc+=1;
  auto unequal_plan=megafmm::create_plan(unequal,policy);Buffer unequal_workspace(unequal_plan->workspace_bytes());
  reject(*unequal_plan,da.data,db.data,dc.as<float>(),dc.as<float>(),unequal_workspace.data,
         "equal C0/D pointers with unequal strides");
  for(int zero_dim:{0,1}) {
    auto empty=problem(zero_dim==0 ? 0:9,zero_dim==1 ? 0:11,17,dtype);
    auto current=megafmm::create_plan(empty,policy);
    require(current->workspace_bytes()==0 && current->lt_algorithm_config(nullptr,nullptr,nullptr,nullptr,nullptr)=="none",
            "packed empty plan requires data/workspace");
    megafmm::run(*current,nullptr,nullptr,nullptr,nullptr,nullptr);
  }
  for(bool alpha_zero:{false,true})for(float beta:{0.0f,-0.0f,.5f,-1.0f}) {
    auto scale=problem(7,11,alpha_zero ? 13:0,dtype);scale.alpha=alpha_zero ? 0:1;scale.beta=beta;
    std::vector<float> c(size_t(scale.m)*scale.n),actual(c.size());
    for(size_t i=0;i<c.size();++i)c[i]=float(int(i%9)-4)/4;
    Buffer source(c.size()*sizeof(float)),output(c.size()*sizeof(float));
    check(cudaMemcpy(source.data,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaMemset(output.data,0xff,actual.size()*sizeof(float)));
    auto current=megafmm::create_plan(scale,policy);
    require(current->workspace_bytes()==0 &&
            current->lt_algorithm_config(nullptr,nullptr,beta==0 ? nullptr:source.as<float>(),output.as<float>(),nullptr)=="none",
            "packed scale plan unexpectedly uses packed operands/workspace");
    megafmm::run(*current,nullptr,nullptr,beta==0 ? nullptr:source.as<float>(),output.as<float>(),nullptr);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(actual.data(),output.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<c.size();++i)require(actual[i]==beta*c[i],"packed degenerate beta*C0 incorrect");
  }
  megafmm::Policy automatic=policy;automatic.algorithm=megafmm::Algorithm::Auto;
  auto automatic_plan=megafmm::create_plan(p,automatic);
  require(automatic_plan->algorithm_name()=="dense_cublaslt","packed candidate caused Auto to select uncertified SW");
  auto e=run_case<T>(p,megafmm::Algorithm::Auto,Pattern::Integers,false,true,1234567,variant,
                     false,size_t(8)<<30,"dense_cublaslt");
  require(e.nonfinite==0 && e.maxabs==0,"packed candidate changed Auto Dense result");
  std::cout << "PASS_PACKED_INPUT_POLICY dtype=" << megafmm::dtype_name(dtype)
            << " variant=" << variant << " overlap_null_overflow_checks=15 metadata_and_run=1"
            << " empty_cases=2 scale_cases=8 auto_dense=1\n";
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
  if(identity && variant!=15 && variant!=16 && variant!=19 && variant!=20 && variant!=21 && variant!=22 && variant!=23 && variant!=24 && variant!=25 && variant!=26 && variant!=27 && variant!=28 && variant!=29 && variant!=30 && variant!=31 && variant!=32 && variant!=33 && variant!=34 && variant!=35 && variant!=36 && variant!=38 && variant!=39 && variant!=40 && variant!=41 && variant!=42 && variant!=43 && variant!=44)continue;
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
    std::cout << (variant==44 ? "BEGIN_WGMMA_PSPLIT_MICROBATCH_EPILOGUE_REBIND dtype=":variant==42 ? "BEGIN_WGMMA_PSPLIT_RSA_BN16_EPILOGUE_REBIND dtype=":variant==43 ? "BEGIN_WGMMA_PSPLIT_SS_BN16_EPILOGUE_REBIND dtype=":variant==41 ? "BEGIN_WGMMA_PSPLIT_HYBRID_S_EPILOGUE_REBIND dtype=":variant==40 ? "BEGIN_WGMMA_PSPLIT_RSA_BK32_EPILOGUE_REBIND dtype=":variant==39 ? "BEGIN_WGMMA_PSPLIT_RSA_EPILOGUE_REBIND dtype=":variant==38 ? "BEGIN_WGMMA_PSPLIT_BK64_EPILOGUE_REBIND dtype=":"BEGIN_EPILOGUE_OUTPUT_REBIND dtype=") << megafmm::dtype_name(dtype)
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
    std::cout << (variant==44 ? "PASS_WGMMA_PSPLIT_MICROBATCH_EPILOGUE_REBIND dtype=":variant==42 ? "PASS_WGMMA_PSPLIT_RSA_BN16_EPILOGUE_REBIND dtype=":variant==43 ? "PASS_WGMMA_PSPLIT_SS_BN16_EPILOGUE_REBIND dtype=":variant==41 ? "PASS_WGMMA_PSPLIT_HYBRID_S_EPILOGUE_REBIND dtype=":variant==40 ? "PASS_WGMMA_PSPLIT_RSA_BK32_EPILOGUE_REBIND dtype=":variant==39 ? "PASS_WGMMA_PSPLIT_RSA_EPILOGUE_REBIND dtype=":variant==38 ? "PASS_WGMMA_PSPLIT_BK64_EPILOGUE_REBIND dtype=":"PASS_EPILOGUE_OUTPUT_REBIND dtype=") << megafmm::dtype_name(dtype)
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
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,9},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,10},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,11},
                     std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,12},
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
    if(algo==megafmm::Algorithm::WinogradHopper && (variant==9 || variant==10))
      require(plan->workspace_bytes()==fused_reserved_bytes(p),"fused input rebind is not P-only");
    if(algo==megafmm::Algorithm::WinogradHopper && (variant==11 || variant==12))
      check_packed_workspace(*plan,p,policy.workspace_limit);
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
  for(int variant:{-1,family==0 ? 2:family==1 ? 13:45}) {
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
template<class T> bool strict_numerics(megafmm::DType dtype,bool k128_only=false,bool pipeline_only=false,bool tma_only=false,bool wait1_only=false,bool three_stage_only=false,bool psplit_only=false,bool psplit_bk64_only=false,bool psplit_rsa_only=false,bool psplit_rsa_bk32_only=false,bool hybrid_s_only=false,int bn16_filter=0,bool microbatch_only=false) {
  bool pass=true;double floor=dtype==megafmm::DType::BF16 ? 1e-3:1e-4;
  struct Candidate { megafmm::Algorithm algorithm;int variant; };
  std::vector<Candidate> candidates={{megafmm::Algorithm::WinogradLt,0},{megafmm::Algorithm::WinogradLt,1}};
  for(int variant=0;variant<=8;++variant)candidates.push_back({megafmm::Algorithm::WinogradHopper,variant});
  candidates.push_back({megafmm::Algorithm::WinogradHopper,9});
  candidates.push_back({megafmm::Algorithm::WinogradHopper,10});
  for(int variant=0;variant<=23;++variant)candidates.push_back({megafmm::Algorithm::Megakernel,variant});
  candidates.push_back({megafmm::Algorithm::Megakernel,26});
  candidates.push_back({megafmm::Algorithm::Megakernel,27});
  candidates.push_back({megafmm::Algorithm::WinogradHopper,11});
  candidates.push_back({megafmm::Algorithm::WinogradHopper,12});
  candidates.push_back({megafmm::Algorithm::Megakernel,28});
  candidates.push_back({megafmm::Algorithm::Megakernel,29});
  candidates.push_back({megafmm::Algorithm::Megakernel,30});
  candidates.push_back({megafmm::Algorithm::Megakernel,31});
  candidates.push_back({megafmm::Algorithm::Megakernel,32});
  candidates.push_back({megafmm::Algorithm::Megakernel,33});
  candidates.push_back({megafmm::Algorithm::Megakernel,34});
  candidates.push_back({megafmm::Algorithm::Megakernel,35});
  candidates.push_back({megafmm::Algorithm::Megakernel,36});
  candidates.push_back({megafmm::Algorithm::Megakernel,38});
  candidates.push_back({megafmm::Algorithm::Megakernel,39});
  candidates.push_back({megafmm::Algorithm::Megakernel,40});
  candidates.push_back({megafmm::Algorithm::Megakernel,41});
  if(!k128_only && !pipeline_only && !tma_only && !wait1_only && !three_stage_only &&
     !psplit_only && !psplit_bk64_only && !psplit_rsa_only && !psplit_rsa_bk32_only && !hybrid_s_only && !microbatch_only)
    for(int variant:{42,43})if(psplit_bn16_route_enabled(variant))
      candidates.push_back({megafmm::Algorithm::Megakernel,variant});
  if(k128_only)candidates={{megafmm::Algorithm::Megakernel,24},{megafmm::Algorithm::Megakernel,25}};
  if(pipeline_only)candidates={{megafmm::Algorithm::Megakernel,28},{megafmm::Algorithm::Megakernel,29}};
  if(tma_only)candidates={{megafmm::Algorithm::Megakernel,30},{megafmm::Algorithm::Megakernel,31}};
  if(wait1_only)candidates={{megafmm::Algorithm::Megakernel,32},{megafmm::Algorithm::Megakernel,33}};
  if(three_stage_only)candidates={{megafmm::Algorithm::Megakernel,34},{megafmm::Algorithm::Megakernel,35}};
  if(psplit_only)candidates={{megafmm::Algorithm::Megakernel,36}};
  if(psplit_bk64_only)candidates={{megafmm::Algorithm::Megakernel,38}};
  if(psplit_rsa_only)candidates={{megafmm::Algorithm::Megakernel,39}};
  if(psplit_rsa_bk32_only)candidates={{megafmm::Algorithm::Megakernel,40}};
  if(hybrid_s_only)candidates={{megafmm::Algorithm::Megakernel,41}};
  if(!k128_only && !pipeline_only && !tma_only && !wait1_only && !three_stage_only &&
     !psplit_only && !psplit_bk64_only && !psplit_rsa_only && !psplit_rsa_bk32_only && !hybrid_s_only &&
     bn16_filter==0 && !microbatch_only && psplit_microbatch_route_enabled())
    candidates.push_back({megafmm::Algorithm::Megakernel,44});
  if(microbatch_only) {
    require(psplit_microbatch_route_enabled(),"requested microbatch numerical scope has no enabled routes");
    candidates={{megafmm::Algorithm::Megakernel,44}};
  }
  if(bn16_filter) {
    candidates.clear();
    for(int variant:{42,43})if((bn16_filter==-1 || bn16_filter==variant) && psplit_bn16_route_enabled(variant))
      candidates.push_back({megafmm::Algorithm::Megakernel,variant});
    require(!candidates.empty(),"requested BN16 numerical scope has no enabled routes");
  }
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
      auto dense=run_case<T>(p,megafmm::Algorithm::Dense,pat,false,certified,seed,0,!k128_only && !pipeline_only && !tma_only && !wait1_only && !three_stage_only && !psplit_only && !psplit_bk64_only && !psplit_rsa_only && !psplit_rsa_bk32_only && !hybrid_s_only && bn16_filter==0 && !microbatch_only);
      dense_ef.push_back(dense.ef);
      for(size_t candidate=0;candidate<candidates.size();++candidate) {
        const auto algo=candidates[candidate].algorithm;
        const int variant=candidates[candidate].variant;
        // Kernel numerical coverage does not require retuning seven leaves for
        // every input pattern. The original group retains tuned Dense; the K128
        // group shares an untuned Dense reference and verifies its actual route.
        const bool fused=algo==megafmm::Algorithm::WinogradHopper && variant>=9;
        const char* expected_route=fused ?
            (certified ? batched_name(algo,variant):"dense_cublaslt"):
            (k128_only || (algo==megafmm::Algorithm::Megakernel && (variant==26 || variant==27 || variant==28 || variant==29 || variant==30 || variant==31 || variant==32 || variant==33 || variant==34 || variant==35 || variant==36 || variant==38 || variant==39 || variant==40 || variant==41 || variant==42 || variant==43 || variant==44))) ?
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
    const std::string usage="usage: test_correctness [--strict-numerics | --strict-pipeline-only | --strict-tma-only | --strict-wait1-only | --strict-three-stage-only | --strict-psplit-only | --strict-psplit-bk64-only | --strict-psplit-rsa-only | --strict-psplit-rsa-bk32-only | --strict-hybrid-s-only | --strict-bn16-only | --strict-bn16-rsa-only | --strict-bn16-ss-only | --strict-microbatch-only] [--smoke-batched | --smoke-candidates [--mega-only] [--mega-variant 0..44 (37 and compiled-blocked routes disabled) | --hopper-variant 0..12]]";
    bool numeric=false,pipeline_numeric_only=false,tma_numeric_only=false,wait1_numeric_only=false,three_stage_numeric_only=false,psplit_numeric_only=false,psplit_bk64_numeric_only=false,psplit_rsa_numeric_only=false,psplit_rsa_bk32_numeric_only=false,hybrid_s_numeric_only=false,microbatch_numeric_only=false,candidates_only=false,mega_only=false,batched_only=false;
    int hopper_filter=-1,mega_filter=-1,bn16_numeric_filter=0;
    for(int i=1;i<argc;++i) {
      if(std::string(argv[i])=="--strict-numerics")numeric=true;
      else if(std::string(argv[i])=="--strict-pipeline-only")pipeline_numeric_only=true;
      else if(std::string(argv[i])=="--strict-tma-only")tma_numeric_only=true;
      else if(std::string(argv[i])=="--strict-wait1-only")wait1_numeric_only=true;
      else if(std::string(argv[i])=="--strict-three-stage-only")three_stage_numeric_only=true;
      else if(std::string(argv[i])=="--strict-psplit-only")psplit_numeric_only=true;
      else if(std::string(argv[i])=="--strict-psplit-bk64-only")psplit_bk64_numeric_only=true;
      else if(std::string(argv[i])=="--strict-psplit-rsa-only")psplit_rsa_numeric_only=true;
      else if(std::string(argv[i])=="--strict-psplit-rsa-bk32-only")psplit_rsa_bk32_numeric_only=true;
      else if(std::string(argv[i])=="--strict-hybrid-s-only")hybrid_s_numeric_only=true;
      else if(std::string(argv[i])=="--strict-microbatch-only")microbatch_numeric_only=true;
      else if(std::string(argv[i])=="--strict-bn16-only" || std::string(argv[i])=="--strict-bn16-rsa-only" || std::string(argv[i])=="--strict-bn16-ss-only") {
        require(bn16_numeric_filter==0,usage);
        bn16_numeric_filter=std::string(argv[i])=="--strict-bn16-only" ? -1:
                            std::string(argv[i])=="--strict-bn16-rsa-only" ? 42:43;
      }
      else if(std::string(argv[i])=="--smoke-batched")batched_only=true;
      else if(std::string(argv[i])=="--smoke-candidates")candidates_only=true;
      else if(std::string(argv[i])=="--mega-only")mega_only=true;
      else if(std::string(argv[i])=="--mega-variant") {
        require(i+1<argc,usage);
        std::string value=argv[++i];size_t consumed=0;
        mega_filter=std::stoi(value,&consumed);
        require(consumed==value.size() && mega_filter>=0 && mega_filter<=44,usage);
        require(mega_filter!=37,"Mega37 is disabled: asymmetric consumer register/spill gate failed before launch");
      }
      else if(std::string(argv[i])=="--hopper-variant") {
        require(i+1<argc,usage);
        std::string value=argv[++i];size_t consumed=0;
        hopper_filter=std::stoi(value,&consumed);
        require(consumed==value.size() && hopper_filter>=0 && hopper_filter<=12,usage);
      } else throw std::runtime_error(usage);
    }
    require((!mega_only && hopper_filter<0 && mega_filter<0) || candidates_only,
            "candidate filters require --smoke-candidates; "+usage);
    require(!mega_only || hopper_filter<0,"--mega-only and --hopper-variant cannot be combined; "+usage);
    require(mega_filter<0 || hopper_filter<0,"--mega-variant and --hopper-variant cannot be combined; "+usage);
    require(!batched_only || !candidates_only,"--smoke-batched and --smoke-candidates cannot be combined; "+usage);
    require(!pipeline_numeric_only || (!numeric && !tma_numeric_only && !wait1_numeric_only && !three_stage_numeric_only && !psplit_numeric_only && !psplit_bk64_numeric_only && !psplit_rsa_numeric_only && !psplit_rsa_bk32_numeric_only && !hybrid_s_numeric_only && bn16_numeric_filter==0 && !microbatch_numeric_only && !candidates_only && !batched_only),
            "--strict-pipeline-only is a separate numerical scope; "+usage);
    require(!tma_numeric_only || (!numeric && !pipeline_numeric_only && !wait1_numeric_only && !three_stage_numeric_only && !psplit_numeric_only && !psplit_bk64_numeric_only && !psplit_rsa_numeric_only && !psplit_rsa_bk32_numeric_only && !hybrid_s_numeric_only && bn16_numeric_filter==0 && !microbatch_numeric_only && !candidates_only && !batched_only),
            "--strict-tma-only is a separate numerical scope; "+usage);
    require(!wait1_numeric_only || (!numeric && !pipeline_numeric_only && !tma_numeric_only && !three_stage_numeric_only && !psplit_numeric_only && !psplit_bk64_numeric_only && !psplit_rsa_numeric_only && !psplit_rsa_bk32_numeric_only && !hybrid_s_numeric_only && bn16_numeric_filter==0 && !microbatch_numeric_only && !candidates_only && !batched_only),
            "--strict-wait1-only is a separate numerical scope; "+usage);
    require(!three_stage_numeric_only || (!numeric && !pipeline_numeric_only && !tma_numeric_only && !wait1_numeric_only && !psplit_numeric_only && !psplit_bk64_numeric_only && !psplit_rsa_numeric_only && !psplit_rsa_bk32_numeric_only && !hybrid_s_numeric_only && bn16_numeric_filter==0 && !microbatch_numeric_only && !candidates_only && !batched_only),
            "--strict-three-stage-only is a separate numerical scope; "+usage);
    require(!psplit_numeric_only || (!numeric && !pipeline_numeric_only && !tma_numeric_only && !wait1_numeric_only && !three_stage_numeric_only && !psplit_bk64_numeric_only && !psplit_rsa_numeric_only && !psplit_rsa_bk32_numeric_only && !hybrid_s_numeric_only && bn16_numeric_filter==0 && !microbatch_numeric_only && !candidates_only && !batched_only),
            "--strict-psplit-only is a separate numerical scope; "+usage);
    require(!psplit_bk64_numeric_only || (!numeric && !pipeline_numeric_only && !tma_numeric_only && !wait1_numeric_only && !three_stage_numeric_only && !psplit_numeric_only && !psplit_rsa_numeric_only && !psplit_rsa_bk32_numeric_only && !hybrid_s_numeric_only && bn16_numeric_filter==0 && !microbatch_numeric_only && !candidates_only && !batched_only),
            "--strict-psplit-bk64-only is a separate numerical scope; "+usage);
    require(!psplit_rsa_numeric_only || (!numeric && !pipeline_numeric_only && !tma_numeric_only && !wait1_numeric_only && !three_stage_numeric_only && !psplit_numeric_only && !psplit_bk64_numeric_only && !psplit_rsa_bk32_numeric_only && !hybrid_s_numeric_only && bn16_numeric_filter==0 && !microbatch_numeric_only && !candidates_only && !batched_only),
            "--strict-psplit-rsa-only is a separate numerical scope; "+usage);
    require(!psplit_rsa_bk32_numeric_only || (!numeric && !pipeline_numeric_only && !tma_numeric_only && !wait1_numeric_only && !three_stage_numeric_only && !psplit_numeric_only && !psplit_bk64_numeric_only && !psplit_rsa_numeric_only && !hybrid_s_numeric_only && bn16_numeric_filter==0 && !microbatch_numeric_only && !candidates_only && !batched_only),
            "--strict-psplit-rsa-bk32-only is a separate numerical scope; "+usage);
    require(!hybrid_s_numeric_only || (!numeric && !pipeline_numeric_only && !tma_numeric_only && !wait1_numeric_only && !three_stage_numeric_only && !psplit_numeric_only && !psplit_bk64_numeric_only && !psplit_rsa_numeric_only && !psplit_rsa_bk32_numeric_only && bn16_numeric_filter==0 && !microbatch_numeric_only && !candidates_only && !batched_only),
            "--strict-hybrid-s-only is a separate numerical scope; "+usage);
    require(!bn16_numeric_filter || (!numeric && !pipeline_numeric_only && !tma_numeric_only && !wait1_numeric_only && !three_stage_numeric_only && !psplit_numeric_only && !psplit_bk64_numeric_only && !psplit_rsa_numeric_only && !psplit_rsa_bk32_numeric_only && !hybrid_s_numeric_only && !microbatch_numeric_only && !candidates_only && !batched_only),
            "BN16 numerical flags are separate scopes; "+usage);
    require(!microbatch_numeric_only || (!numeric && !pipeline_numeric_only && !tma_numeric_only && !wait1_numeric_only && !three_stage_numeric_only && !psplit_numeric_only && !psplit_bk64_numeric_only && !psplit_rsa_numeric_only && !psplit_rsa_bk32_numeric_only && !hybrid_s_numeric_only && bn16_numeric_filter==0 && !candidates_only && !batched_only),
            "--strict-microbatch-only is a separate numerical scope; "+usage);
    if(microbatch_numeric_only) {
      bool bf16_pass=strict_numerics<__nv_bfloat16>(megafmm::DType::BF16,false,false,false,false,false,false,false,false,false,false,0,true);
      bool fp16_pass=strict_numerics<__half>(megafmm::DType::FP16,false,false,false,false,false,false,false,false,false,false,0,true);
      return bf16_pass && fp16_pass ? 0:2;
    }
    if(bn16_numeric_filter) {
      bool bf16_pass=strict_numerics<__nv_bfloat16>(megafmm::DType::BF16,false,false,false,false,false,false,false,false,false,false,bn16_numeric_filter);
      bool fp16_pass=strict_numerics<__half>(megafmm::DType::FP16,false,false,false,false,false,false,false,false,false,false,bn16_numeric_filter);
      return bf16_pass && fp16_pass ? 0:2;
    }
    if(hybrid_s_numeric_only) {
      bool bf16_pass=strict_numerics<__nv_bfloat16>(megafmm::DType::BF16,false,false,false,false,false,false,false,false,false,true);
      bool fp16_pass=strict_numerics<__half>(megafmm::DType::FP16,false,false,false,false,false,false,false,false,false,true);
      return bf16_pass && fp16_pass ? 0:2;
    }
    if(psplit_rsa_bk32_numeric_only) {
      bool bf16_pass=strict_numerics<__nv_bfloat16>(megafmm::DType::BF16,false,false,false,false,false,false,false,false,true);
      bool fp16_pass=strict_numerics<__half>(megafmm::DType::FP16,false,false,false,false,false,false,false,false,true);
      return bf16_pass && fp16_pass ? 0:2;
    }
    if(psplit_rsa_numeric_only) {
      bool bf16_pass=strict_numerics<__nv_bfloat16>(megafmm::DType::BF16,false,false,false,false,false,false,false,true);
      bool fp16_pass=strict_numerics<__half>(megafmm::DType::FP16,false,false,false,false,false,false,false,true);
      return bf16_pass && fp16_pass ? 0:2;
    }
    if(psplit_bk64_numeric_only) {
      bool bf16_pass=strict_numerics<__nv_bfloat16>(megafmm::DType::BF16,false,false,false,false,false,false,true);
      bool fp16_pass=strict_numerics<__half>(megafmm::DType::FP16,false,false,false,false,false,false,true);
      return bf16_pass && fp16_pass ? 0:2;
    }
    if(psplit_numeric_only) {
      bool bf16_pass=strict_numerics<__nv_bfloat16>(megafmm::DType::BF16,false,false,false,false,false,true);
      bool fp16_pass=strict_numerics<__half>(megafmm::DType::FP16,false,false,false,false,false,true);
      return bf16_pass && fp16_pass ? 0:2;
    }
    if(three_stage_numeric_only) {
      bool bf16_pass=strict_numerics<__nv_bfloat16>(megafmm::DType::BF16,false,false,false,false,true);
      bool fp16_pass=strict_numerics<__half>(megafmm::DType::FP16,false,false,false,false,true);
      return bf16_pass && fp16_pass ? 0:2;
    }
    if(wait1_numeric_only) {
      bool bf16_pass=strict_numerics<__nv_bfloat16>(megafmm::DType::BF16,false,false,false,true);
      bool fp16_pass=strict_numerics<__half>(megafmm::DType::FP16,false,false,false,true);
      return bf16_pass && fp16_pass ? 0:2;
    }
    if(tma_numeric_only) {
      bool bf16_pass=strict_numerics<__nv_bfloat16>(megafmm::DType::BF16,false,false,true);
      bool fp16_pass=strict_numerics<__half>(megafmm::DType::FP16,false,false,true);
      return bf16_pass && fp16_pass ? 0:2;
    }
    if(pipeline_numeric_only) {
      bool bf16_pass=strict_numerics<__nv_bfloat16>(megafmm::DType::BF16,false,true);
      bool fp16_pass=strict_numerics<__half>(megafmm::DType::FP16,false,true);
      return bf16_pass && fp16_pass ? 0:2;
    }
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
      if(mega_filter==44) {
        require(psplit_microbatch_route_enabled(),"Mega44 is disabled: compiled microbatch register/spill gate failed before launch");
        wgmma_scheduled_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_scheduled_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_scheduled_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_scheduled_native_bitwise_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_scheduled_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_scheduled_input_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_scheduled_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_scheduled_graph_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_scheduled_policy_suite(mega_filter);
        psplit_disabled37_policy_suite();
        std::cout << "PASS selected WGMMA microbatch integer DAG, BK64 stage8 K1024, partial-K fallback, sameatom N32 raw U vs39, host epilogue, current inputs, scalar outputs, Graph and policy tests\n";
        return 0;
      }
      if(mega_filter==42 || mega_filter==43) {
        require(psplit_bn16_route_enabled(mega_filter),"Mega"+std::to_string(mega_filter)+" is disabled: compiled BN16 register/spill gate failed before launch");
        wgmma_scheduled_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_scheduled_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_scheduled_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_scheduled_native_bitwise_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_scheduled_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_scheduled_input_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_scheduled_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_scheduled_graph_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_scheduled_policy_suite(mega_filter);
        for(int variant:{42,43})if(!psplit_bn16_route_enabled(variant))psplit_bn16_blocked_policy_suite(variant);
        psplit_disabled37_policy_suite();
        std::cout << "PASS selected WGMMA BN16 product-split integer DAG, stage16 K1024, partial-K fallback, explicitly typed sameatom/crossatom raw U references, host epilogue, current inputs, scalar outputs, Graph and policy tests\n";
        return 0;
      }
      if(mega_filter==41) {
        wgmma_psplit_hybrid_s_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_hybrid_s_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_hybrid_s_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_hybrid_s_native_bitwise_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_hybrid_s_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_hybrid_s_input_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_hybrid_s_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_hybrid_s_graph_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_hybrid_s_policy_suite(mega_filter);
        psplit_disabled37_policy_suite();
        std::cout << "PASS selected WGMMA product-split Hybrid S integer DAG, stage16 K1024, partial-K fallback, sameatom N32 raw U vs36, host epilogue, current inputs, scalar outputs, Graph and policy tests\n";
        return 0;
      }
      if(mega_filter==40) {
        wgmma_psplit_rsa_bk32_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_rsa_bk32_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_rsa_bk32_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_rsa_bk32_native_bitwise_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_rsa_bk32_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_rsa_bk32_input_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_rsa_bk32_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_rsa_bk32_graph_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_rsa_bk32_policy_suite(mega_filter);
        psplit_disabled37_policy_suite();
        std::cout << "PASS selected WGMMA product-split RS-A BK32 integer DAG, stage16 K1024, partial-K fallback, sameatom N32 raw U vs36, host epilogue, current inputs, scalar outputs, Graph and policy tests\n";
        return 0;
      }
      if(mega_filter==39) {
        wgmma_psplit_rsa_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_rsa_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_rsa_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_rsa_native_bitwise_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_rsa_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_rsa_input_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_rsa_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_rsa_graph_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_rsa_policy_suite(mega_filter);
        psplit_disabled37_policy_suite();
        std::cout << "PASS selected WGMMA product-split RS-A integer DAG, stage8 K1024, partial-K fallback, sameatom N32 raw U vs38, host epilogue, current inputs, scalar outputs, Graph and policy tests\n";
        return 0;
      }
      if(mega_filter==38) {
        wgmma_psplit_bk64_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_bk64_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_bk64_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_bk64_native_bitwise_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_bk64_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_bk64_input_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_bk64_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_bk64_graph_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_bk64_policy_suite(mega_filter);
        psplit_disabled37_policy_suite();
        std::cout << "PASS selected WGMMA product-split BK64 integer DAG, stage8 K1024, partial-K fallback, sameatom N32 raw U vs36, host epilogue, current inputs, scalar outputs, Graph and policy tests\n";
        return 0;
      }
      if(mega_filter==36) {
        wgmma_psplit_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_native_bitwise_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_input_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_psplit_graph_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_psplit_policy_suite(mega_filter);
        psplit_disabled37_policy_suite();
        std::cout << "PASS selected WGMMA product-split integer DAG, two-consumer tail, sameatom N32 raw U, host epilogue, current inputs, scalar outputs, Graph and policy tests\n";
        return 0;
      }
      if(mega_filter==34 || mega_filter==35) {
        wgmma_three_stage_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_three_stage_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_three_stage_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_three_stage_native_bitwise_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_three_stage_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_three_stage_input_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_three_stage_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_three_stage_graph_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_three_stage_policy_suite(mega_filter);
        std::cout << "PASS selected WGMMA three-stage WAIT1 integer DAG, mod3 phase wraps, sameatom raw U, host epilogue, current inputs, scalar outputs, Graph and policy tests\n";
        return 0;
      }
      if(mega_filter==32 || mega_filter==33) {
        wgmma_wait1_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_wait1_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_wait1_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_wait1_native_bitwise_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_wait1_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_wait1_input_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_wait1_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_wait1_graph_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_wait1_policy_suite(mega_filter);
        std::cout << "PASS selected WGMMA WAIT1 integer DAG, phase wraps, sameatom raw U, host epilogue, current inputs, scalar outputs, Graph and policy tests\n";
        return 0;
      }
      if(mega_filter==30 || mega_filter==31) {
        wgmma_tma_alias_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_tma_alias_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_tma_alias_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_tma_alias_native_bitwise_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_tma_alias_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_tma_alias_input_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_tma_alias_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_tma_alias_graph_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_tma_alias_policy_suite(mega_filter);
        std::cout << "PASS selected WGMMA TMA alias integer DAG, owners, raw U, epilogue, current maps, scalar outputs, Graph and policy tests\n";
        return 0;
      }
      if(mega_filter==28 || mega_filter==29) {
        wgmma_pipeline_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_pipeline_suite<__half>(megafmm::DType::FP16,mega_filter);
        if(mega_filter==29) {
          wgmma_pipeline_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16);
          wgmma_pipeline_native_bitwise_suite<__half>(megafmm::DType::FP16);
        }
        wgmma_pipeline_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_pipeline_input_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_pipeline_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,mega_filter);
        wgmma_pipeline_graph_suite<__half>(megafmm::DType::FP16,mega_filter);
        wgmma_pipeline_policy_suite(mega_filter);
        std::cout << "PASS selected WGMMA pipeline integer DAG, owners, scalar outputs, fallback, Graph and policy tests\n";
        return 0;
      }
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
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,8},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,9},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,10},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,11},
                           std::pair<megafmm::Algorithm,int>{megafmm::Algorithm::WinogradHopper,12}}) {
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
            const bool fused=candidate.first==megafmm::Algorithm::WinogradHopper && candidate.second>=9;
            const bool fused_fallback=fused && !(candidate.second<=10 ?
                fused_input_supported(p,candidate.second):packed_input_supported(p,candidate.second));
            require(fused_fallback || planned->algorithm_name().find("dense")==std::string::npos,
                    "supported candidate unexpectedly fell back to Dense");
            const char* route=fused_fallback ? "dense_cublaslt":
                              warp_mma ? warp_mma_name(candidate.second,p):
                              candidate.first==megafmm::Algorithm::WinogradHopper && candidate.second>=5 ?
                                batched_name(candidate.first,candidate.second):nullptr;
            auto e=run_case<T>(p,candidate.first,Pattern::Integers,alias,true,1234567,candidate.second,
                               !fused,size_t(8)<<30,route);
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
      if(hopper_filter==9 || hopper_filter==10) {
        fused_input_suite<__nv_bfloat16>(megafmm::DType::BF16,hopper_filter);
        fused_input_suite<__half>(megafmm::DType::FP16,hopper_filter);
        fused_input_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,hopper_filter);
        fused_input_native_bitwise_suite<__half>(megafmm::DType::FP16,hopper_filter);
        fused_input_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,hopper_filter);
        fused_input_output_rebind_suite<__half>(megafmm::DType::FP16,hopper_filter);
        fused_input_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,hopper_filter);
        fused_input_graph_suite<__half>(megafmm::DType::FP16,hopper_filter);
        rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,false,hopper_filter);
        rebind_suite<__half>(megafmm::DType::FP16,false,hopper_filter);
      }
      if(hopper_filter==11 || hopper_filter==12) {
        packed_input_suite<__nv_bfloat16>(megafmm::DType::BF16,hopper_filter);
        packed_input_suite<__half>(megafmm::DType::FP16,hopper_filter);
        packed_input_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,hopper_filter);
        packed_input_native_bitwise_suite<__half>(megafmm::DType::FP16,hopper_filter);
        packed_input_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,hopper_filter);
        packed_input_output_rebind_suite<__half>(megafmm::DType::FP16,hopper_filter);
        packed_input_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,hopper_filter);
        packed_input_graph_suite<__half>(megafmm::DType::FP16,hopper_filter);
        packed_input_policy_suite<__nv_bfloat16>(megafmm::DType::BF16,hopper_filter);
        packed_input_policy_suite<__half>(megafmm::DType::FP16,hopper_filter);
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
    for(int variant:{9,10}) {
      fused_input_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      fused_input_suite<__half>(megafmm::DType::FP16,variant);
      fused_input_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      fused_input_native_bitwise_suite<__half>(megafmm::DType::FP16,variant);
      fused_input_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      fused_input_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
      fused_input_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      fused_input_graph_suite<__half>(megafmm::DType::FP16,variant);
    }
    for(int variant:{11,12}) {
      packed_input_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      packed_input_suite<__half>(megafmm::DType::FP16,variant);
      packed_input_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      packed_input_native_bitwise_suite<__half>(megafmm::DType::FP16,variant);
      packed_input_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      packed_input_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
      packed_input_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      packed_input_graph_suite<__half>(megafmm::DType::FP16,variant);
      packed_input_policy_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      packed_input_policy_suite<__half>(megafmm::DType::FP16,variant);
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
    for(int variant:{28,29}) {
      wgmma_pipeline_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_pipeline_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_pipeline_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_pipeline_input_rebind_suite<__half>(megafmm::DType::FP16,variant);
      epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_pipeline_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_pipeline_graph_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_pipeline_policy_suite(variant);
    }
    wgmma_pipeline_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16);
    wgmma_pipeline_native_bitwise_suite<__half>(megafmm::DType::FP16);
    for(int variant:{30,31}) {
      wgmma_tma_alias_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_tma_alias_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_tma_alias_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_tma_alias_native_bitwise_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_tma_alias_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_tma_alias_input_rebind_suite<__half>(megafmm::DType::FP16,variant);
      epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_tma_alias_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_tma_alias_graph_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_tma_alias_policy_suite(variant);
    }
    for(int variant:{32,33}) {
      wgmma_wait1_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_wait1_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_wait1_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_wait1_native_bitwise_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_wait1_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_wait1_input_rebind_suite<__half>(megafmm::DType::FP16,variant);
      epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_wait1_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_wait1_graph_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_wait1_policy_suite(variant);
    }
    for(int variant:{34,35}) {
      wgmma_three_stage_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_three_stage_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_three_stage_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_three_stage_native_bitwise_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_three_stage_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_three_stage_input_rebind_suite<__half>(megafmm::DType::FP16,variant);
      epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_three_stage_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_three_stage_graph_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_three_stage_policy_suite(variant);
    }
    for(int variant:{36}) {
      wgmma_psplit_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_native_bitwise_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_input_rebind_suite<__half>(megafmm::DType::FP16,variant);
      epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_graph_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_policy_suite(variant);
      psplit_disabled37_policy_suite();
    }
    for(int variant:{38}) {
      wgmma_psplit_bk64_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_bk64_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_bk64_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_bk64_native_bitwise_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_bk64_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_bk64_input_rebind_suite<__half>(megafmm::DType::FP16,variant);
      epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_bk64_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_bk64_graph_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_bk64_policy_suite(variant);
    }
    for(int variant:{39}) {
      wgmma_psplit_rsa_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_rsa_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_rsa_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_rsa_native_bitwise_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_rsa_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_rsa_input_rebind_suite<__half>(megafmm::DType::FP16,variant);
      epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_rsa_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_rsa_graph_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_rsa_policy_suite(variant);
    }
    for(int variant:{40}) {
      wgmma_psplit_rsa_bk32_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_rsa_bk32_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_rsa_bk32_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_rsa_bk32_native_bitwise_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_rsa_bk32_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_rsa_bk32_input_rebind_suite<__half>(megafmm::DType::FP16,variant);
      epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_rsa_bk32_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_rsa_bk32_graph_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_rsa_bk32_policy_suite(variant);
    }
    for(int variant:{41}) {
      wgmma_psplit_hybrid_s_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_hybrid_s_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_hybrid_s_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_hybrid_s_native_bitwise_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_hybrid_s_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_hybrid_s_input_rebind_suite<__half>(megafmm::DType::FP16,variant);
      epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_hybrid_s_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_psplit_hybrid_s_graph_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_psplit_hybrid_s_policy_suite(variant);
    }
    for(int variant:{42,43}) {
      if(!psplit_bn16_route_enabled(variant)) {psplit_bn16_blocked_policy_suite(variant);continue;}
      wgmma_scheduled_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_scheduled_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_scheduled_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_scheduled_native_bitwise_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_scheduled_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_scheduled_input_rebind_suite<__half>(megafmm::DType::FP16,variant);
      epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_scheduled_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_scheduled_graph_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_scheduled_policy_suite(variant);
    }
    for(int variant:{44}) {
      if(!psplit_microbatch_route_enabled()) {psplit_microbatch_blocked_policy_suite();continue;}
      wgmma_scheduled_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_scheduled_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_scheduled_native_bitwise_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_scheduled_native_bitwise_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_scheduled_input_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_scheduled_input_rebind_suite<__half>(megafmm::DType::FP16,variant);
      epilogue_output_rebind_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      epilogue_output_rebind_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_scheduled_graph_suite<__nv_bfloat16>(megafmm::DType::BF16,variant);
      wgmma_scheduled_graph_suite<__half>(megafmm::DType::FP16,variant);
      wgmma_scheduled_policy_suite(variant);
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

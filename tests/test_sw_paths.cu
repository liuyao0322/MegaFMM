#include "megafmm/gemm.h"
#include "sw_fp32_batch.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
bool raw_b_route(int id);

void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
struct Buffer {
  void* data=nullptr;
  explicit Buffer(size_t bytes) {if(bytes)check(cudaMalloc(&data,bytes));}
  ~Buffer() {if(data)cudaFree(data);}
};
struct Stream {
  cudaStream_t value=nullptr;
  Stream() {check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
  ~Stream() {cudaStreamDestroy(value);}
};
template<class T>T rounded(float x);
template<>float rounded(float x) {return x;}
template<>__half rounded(float x) {return __float2half_rn(x);}
template<>__nv_bfloat16 rounded(float x) {return __float2bfloat16_rn(x);}

template<class T>void exercise(megafmm::DType dtype,int m,int n,int k,bool tuned,
                               bool zero_budget=false,bool layout=false,bool unaligned=false,float scale=1,bool packed=false,bool strided=false,bool inplace=false,int packed_id=0,bool force_fallback=false,int cooperative_cap=-1,bool coefficients=false,int map_change=0) {
  using namespace megafmm;
  Problem p;p.dtype=dtype;p.m=m;p.n=n;p.k=k;
  p.trans_a=p.trans_b=layout;
  p.lda=(layout?m:k)+((layout||strided)?3:0);p.ldb=(layout?k:n)+((layout||strided)?5:0);
  p.ldc=p.ldd=n+((layout||strided)?7:0);
  p.alpha=(layout||strided||coefficients)?0.5f:1;p.beta=(layout||strided||coefficients)?-0.25f:0;
  Policy policy;policy.algorithm=tuned?Algorithm::SWTuned:Algorithm::SWPipeline;
  policy.workspace_limit=zero_budget?0:size_t(8)<<30;
  policy.sw_tuning_allow_inaccurate=false;policy.tuning_input_sets=2;
  auto plan=create_plan(p,policy);
  require(plan->workspace_bytes()<=policy.workspace_limit,"workspace budget exceeded");
  if(zero_budget)require(plan->workspace_bytes()==0,"fallback allocated workspace");
  if(tuned) {
    const auto& candidates=plan->sw_tuning_results();
    require(!candidates.empty() && candidates.size()<=(dtype==DType::FP32?4u+detail::sw_fp32_batch::configurations.size():2u),"specialization pool grew");
    for(const auto& c:candidates)require(c.id==-1 || c.id==0 || c.id==9000016 || c.id==9000032 ||
      c.id==9100016 || c.id==9100032 ||
      (c.id>=9500000 && c.id<9500000+int(detail::sw_fp32_batch::configurations.size())),"size-specific candidate registered");
  }
  const size_t ac=size_t(layout?k:m)*p.lda,bc=size_t(layout?n:k)*p.ldb,dc=size_t(m)*p.ldd;
  std::vector<T>a(ac),b(bc);std::vector<float>initial(dc,-12345),host(dc+64);
  Buffer da((ac+1)*sizeof(T)),db((bc+1)*sizeof(T)),dd((dc+64)*4),c0(dc*4);
  const size_t workspace_bytes=packed?detail::sw_fp32_batch::workspace_size(p,packed_id):plan->workspace_bytes();
  if(packed)detail::sw_fp32_batch::setup(packed_id);
  Buffer w(workspace_bytes?workspace_bytes+512:0);
  T* ap=static_cast<T*>(da.data)+(unaligned?1:0);
  T* bp=static_cast<T*>(db.data)+(unaligned?1:0);
  float* output=static_cast<float*>(dd.data)+32;
  void* workspace=w.data?static_cast<char*>(w.data)+256:nullptr;
  Stream stream;
  detail::sw_fp32_batch::Maps maps;
  if(packed) {
    namespace batch=detail::sw_fp32_batch;
    Problem prepared_problem=p;
    if(map_change==1)prepared_problem.ldb+=4;
    if(map_change==2)prepared_problem.n-=8;
    if(map_change==3)prepared_problem.k-=8;
    maps=batch::prepare(prepared_problem,packed_id,workspace);
    if(cooperative_cap>=0)maps.cooperative_max_blocks=std::min(maps.cooperative_max_blocks,cooperative_cap);
    if(force_fallback){maps.cooperative_max_blocks=0;maps.raw_b_available=false;}
    if(batch::configurations[packed_id].strategy==batch::ExecutionStrategy::DirectBMaterialized || raw_b_route(packed_id)) {
      const bool route=batch::uses_direct_b(p,packed_id,reinterpret_cast<const float*>(ap),reinterpret_cast<const float*>(bp),maps);
      const bool expected=map_change==0 && !force_fallback && !unaligned && p.lda%4==0 && p.ldb%4==0 && k%8==0 && n%8==0 && (!raw_b_route(packed_id) || k%64==0) && !batch::uses_cooperative(p,packed_id,reinterpret_cast<const float*>(ap),reinterpret_cast<const float*>(bp),maps);
      require(route==expected,"incorrect direct B binding route");
      std::printf("ROUTE packed id=%d %dx%dx%d direct_b=%d forced=%d\n",packed_id,m,n,k,route,force_fallback);
    }
    if(batch::configurations[packed_id].strategy==batch::ExecutionStrategy::CooperativeFused) {
      const bool route=batch::uses_cooperative(p,packed_id,reinterpret_cast<const float*>(ap),reinterpret_cast<const float*>(bp),maps);
      if(force_fallback || unaligned || strided || k%8 || n%8 ||
         batch::configurations[packed_id].pack_execution==batch::PackExecution::SeparateKernel)
        require(!route,"ineligible binding used cooperative launch");
      else if(maps.cooperative_max_blocks>0) {
        const auto& cfg=batch::configurations[packed_id];
        const int tasks=((m+1)/2+cfg.bm-1)/cfg.bm*(((n+1)/2+cfg.bn-1)/cfg.bn);
        require(route==(cfg.persistent_tasks || tasks<=maps.cooperative_max_blocks),"incorrect cooperative capacity route");
      }
      std::printf("ROUTE packed id=%d %dx%dx%d cooperative=%d capacity=%d forced=%d\n",packed_id,m,n,k,route,maps.cooperative_max_blocks,force_fallback);
    }
  }
  for(int seed=1;seed<=3;++seed) {
    std::fill(a.begin(),a.end(),rounded<T>(0));
    for(int r=0;r<m;++r) {
      const int q=(r*13+seed*7)%k;
      a[layout?size_t(q)*p.lda+r:size_t(r)*p.lda+q]=rounded<T>((r+seed)%2?-scale:scale);
    }
    for(int q=0;q<k;++q)for(int col=0;col<n;++col)
      b[layout?size_t(col)*p.ldb+q:size_t(q)*p.ldb+col]=rounded<T>(scale*(int((q*17+col+seed)%7)-3));
    for(int r=0;r<m;++r)for(int col=0;col<n;++col)initial[size_t(r)*p.ldd+col]=float((r+col)%5-2);
    check(cudaMemcpyAsync(ap,a.data(),ac*sizeof(T),cudaMemcpyHostToDevice,stream.value));
    check(cudaMemcpyAsync(bp,b.data(),bc*sizeof(T),cudaMemcpyHostToDevice,stream.value));
    check(cudaMemcpyAsync(c0.data,initial.data(),dc*4,cudaMemcpyHostToDevice,stream.value));
    check(cudaMemsetAsync(dd.data,0x5a,(dc+64)*4,stream.value));
    check(cudaMemcpyAsync(output,initial.data(),dc*4,cudaMemcpyHostToDevice,stream.value));
    if(w.data)check(cudaMemsetAsync(w.data,0xa5,workspace_bytes+512,stream.value));
    if(tuned && seed==1) {
      bool rejected=false;
      try {run(*plan,ap,bp,nullptr,output,workspace,stream.value);}catch(const std::runtime_error&){rejected=true;}
      require(rejected,"untuned SW plan ran");
      tune(*plan,ap,bp,nullptr,output,workspace,stream.value);
      float best=INFINITY;int confirmed=0,selected=-1;
      for(const auto& c:plan->sw_tuning_results())if(!c.confirmation_ms.empty()) {
        require(c.numerical_pass && c.ranking_ms.size()==5 && c.confirmation_ms.size()==5,"incomplete strict tuning evidence");
        auto times=c.confirmation_ms;std::sort(times.begin(),times.end());best=std::min(best,times[2]);++confirmed;
      }
      require(confirmed>0 && confirmed<=4,"invalid finalist count");
      const auto metadata=plan->lt_algorithm_config(ap,bp,nullptr,output,workspace);
      const auto at=metadata.find("selected_id=");require(at!=std::string::npos,"missing tuning selection");
      selected=std::stoi(metadata.substr(at+12));
      for(const auto& c:plan->sw_tuning_results())if(c.id==selected) {
        auto times=c.confirmation_ms;std::sort(times.begin(),times.end());require(times.size()==5 && times[2]==best,"selection ignored confirmed median");
      }
    }
    if(packed)detail::sw_fp32_batch::run_prepared(p,packed_id,reinterpret_cast<const float*>(ap),reinterpret_cast<const float*>(bp),p.beta?(inplace?output:static_cast<float*>(c0.data)):nullptr,output,workspace,stream.value,maps);
    else run(*plan,ap,bp,layout?static_cast<float*>(c0.data):nullptr,output,workspace,stream.value);
    check(cudaMemcpyAsync(host.data(),dd.data,(dc+64)*4,cudaMemcpyDeviceToHost,stream.value));
    check(cudaStreamSynchronize(stream.value));
    const auto* bits=reinterpret_cast<const unsigned*>(host.data());
    for(int i=0;i<32;++i)require(bits[i]==0x5a5a5a5a && bits[32+dc+i]==0x5a5a5a5a,"output guard overwritten");
    for(int r=0;r<m;++r)for(int col=0;col<p.ldd;++col) {
      float expected=-12345;
      if(col<n) {
        const int q=(r*13+seed*7)%k;
        expected=p.alpha*((r+seed)%2?-scale:scale)*scale*(int((q*17+col+seed)%7)-3)+p.beta*initial[size_t(r)*p.ldd+col];
      }
      if(host[32+size_t(r)*p.ldd+col]!=expected) {
        std::fprintf(stderr,"mismatch %s %dx%dx%d tuned=%d zero=%d layout=%d at %d,%d: %g != %g\n",dtype_name(dtype),m,n,k,tuned,zero_budget,layout,r,col,host[32+size_t(r)*p.ldd+col],expected);
        throw std::runtime_error("fresh original-AB mismatch");
      }
    }
    if(w.data) {
      unsigned guards[128];
      check(cudaMemcpy(guards,w.data,256,cudaMemcpyDeviceToHost));
      check(cudaMemcpy(guards+64,static_cast<char*>(workspace)+workspace_bytes,256,cudaMemcpyDeviceToHost));
      for(unsigned x:guards)require(x==0xa5a5a5a5,"workspace guard overwritten");
    }
  }
  std::printf("PASS %s %dx%dx%d tuned=%d zero=%d layout=%d unaligned=%d packed=%d strided=%d inplace=%d fresh=3\n",dtype_name(dtype),m,n,k,tuned,zero_budget,layout,unaligned,packed,strided,inplace);
}

void failed_retune() {
  using namespace megafmm;
  Problem p;p.dtype=DType::FP32;p.m=33;p.n=65;p.k=129;
  Policy policy;policy.algorithm=Algorithm::SWTuned;policy.sw_tuning_allow_inaccurate=false;
  auto plan=create_plan(p,policy);Buffer a(p.m*p.k*4),b(p.k*p.n*4),d(p.m*p.n*4),w(plan->workspace_bytes());
  check(cudaMemset(a.data,0,p.m*p.k*4));check(cudaMemset(b.data,0,p.k*p.n*4));
  tune(*plan,a.data,b.data,nullptr,static_cast<float*>(d.data),w.data);
  float nan=std::numeric_limits<float>::quiet_NaN();check(cudaMemcpy(a.data,&nan,4,cudaMemcpyHostToDevice));
  bool rejected=false;try {tune(*plan,a.data,b.data,nullptr,static_cast<float*>(d.data),w.data);}catch(const std::runtime_error&){rejected=true;}
  require(rejected,"nonfinite retune accepted");
  rejected=false;try {run(*plan,a.data,b.data,nullptr,static_cast<float*>(d.data),w.data);}catch(const std::runtime_error&){rejected=true;}
  require(rejected,"failed retune retained stale winner");
  policy.sw_tuning_graph=true;rejected=false;
  try {auto invalid=create_plan(p,policy);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"graph tuning unexpectedly enabled");
  std::puts("PASS failed retune invalidates selection; graph policy rejected");
}

template<class T>void concurrent(megafmm::DType dtype,bool tuned,bool packed=false,int packed_id=0,bool force_raw=false) {
  using namespace megafmm;
  Problem p;p.dtype=dtype;p.m=65;p.n=80;p.k=force_raw?512:520;
  p.lda=p.k;p.ldb=p.ldc=p.ldd=p.n;
  Policy policy;policy.algorithm=tuned?Algorithm::SWTuned:Algorithm::SWPipeline;
  policy.sw_tuning_allow_inaccurate=false;
  auto plan=create_plan(p,policy);
  Buffer a(p.m*p.k*sizeof(T)),b0(p.k*p.n*sizeof(T)),b1(p.k*p.n*sizeof(T));
  const size_t bytes=packed?detail::sw_fp32_batch::workspace_size(p,packed_id):plan->workspace_bytes();
  Buffer d0(p.m*p.n*4),d1(p.m*p.n*4),w0(bytes),w1(bytes);
  detail::sw_fp32_batch::Maps maps0,maps1;
  if(packed) {
    detail::sw_fp32_batch::setup(packed_id);
    maps0=detail::sw_fp32_batch::prepare(p,packed_id,w0.data);
    maps1=detail::sw_fp32_batch::prepare(p,packed_id,w1.data);
    if(force_raw) {
      maps0.cooperative_max_blocks=maps1.cooperative_max_blocks=0;
      require(detail::sw_fp32_batch::uses_direct_b(p,packed_id,reinterpret_cast<const float*>(a.data),reinterpret_cast<const float*>(b0.data),maps0),"concurrency did not use raw B");
    }
    bool rejected=false;
    if(bytes) {
      try {detail::sw_fp32_batch::run_prepared(p,packed_id,reinterpret_cast<const float*>(a.data),reinterpret_cast<const float*>(b0.data),nullptr,static_cast<float*>(d0.data),w1.data,nullptr,maps0);}
      catch(const std::invalid_argument&) {rejected=true;}
      require(rejected,"packed accepted another binding workspace");
    }
    if(detail::sw_fp32_batch::configurations.size()>1) {
      rejected=false;
      const int other=(packed_id+1)%int(detail::sw_fp32_batch::configurations.size());
      try {detail::sw_fp32_batch::run_prepared(p,other,reinterpret_cast<const float*>(a.data),reinterpret_cast<const float*>(b0.data),nullptr,static_cast<float*>(d0.data),w0.data,nullptr,maps0);}
      catch(const std::invalid_argument&) {rejected=true;}
      require(rejected,"packed accepted another configuration tensor map");
    }
  }
  std::vector<T> av(p.m*p.k,rounded<T>(0)),bv(p.k*p.n,rounded<T>(1));
  for(int r=0;r<p.m;++r)av[size_t(r)*p.k+r]=rounded<T>(1);
  check(cudaMemcpy(a.data,av.data(),av.size()*sizeof(T),cudaMemcpyHostToDevice));
  check(cudaMemcpy(b0.data,bv.data(),bv.size()*sizeof(T),cudaMemcpyHostToDevice));
  if(tuned)tune(*plan,a.data,b0.data,nullptr,static_cast<float*>(d0.data),w0.data);
  std::fill(bv.begin(),bv.end(),rounded<T>(2));
  check(cudaMemcpy(b1.data,bv.data(),bv.size()*sizeof(T),cudaMemcpyHostToDevice));
  Stream s0,s1;
  std::vector<T> fresh0(bv.size()),fresh1(bv.size());
  for(int trial=0;trial<3;++trial) {
    std::fill(fresh0.begin(),fresh0.end(),rounded<T>(trial+1));
    std::fill(fresh1.begin(),fresh1.end(),rounded<T>(trial+4));
    check(cudaMemcpyAsync(b0.data,fresh0.data(),fresh0.size()*sizeof(T),cudaMemcpyHostToDevice,s0.value));
    check(cudaMemcpyAsync(b1.data,fresh1.data(),fresh1.size()*sizeof(T),cudaMemcpyHostToDevice,s1.value));
    if(packed) {
      detail::sw_fp32_batch::run_prepared(p,packed_id,reinterpret_cast<const float*>(a.data),reinterpret_cast<const float*>(b0.data),nullptr,static_cast<float*>(d0.data),w0.data,s0.value,maps0);
      detail::sw_fp32_batch::run_prepared(p,packed_id,reinterpret_cast<const float*>(a.data),reinterpret_cast<const float*>(b1.data),nullptr,static_cast<float*>(d1.data),w1.data,s1.value,maps1);
    } else {
      run(*plan,a.data,b0.data,nullptr,static_cast<float*>(d0.data),w0.data,s0.value);
      run(*plan,a.data,b1.data,nullptr,static_cast<float*>(d1.data),w1.data,s1.value);
    }
    check(cudaStreamSynchronize(s0.value));check(cudaStreamSynchronize(s1.value));
    std::vector<float> out(p.m*p.n);
    check(cudaMemcpy(out.data(),d0.data,out.size()*4,cudaMemcpyDeviceToHost));
    for(float x:out)require(x==trial+1,"concurrent first binding corrupted");
    check(cudaMemcpy(out.data(),d1.data,out.size()*4,cudaMemcpyDeviceToHost));
    for(float x:out)require(x==trial+4,"concurrent second binding corrupted");
  }
  std::printf("PASS concurrent shared plan %s tuned=%d packed=%d independent workspaces fresh=3\n",dtype_name(dtype),tuned,packed);
}

template<class T>void suite(megafmm::DType dtype) {
  for(auto shape:std::vector<std::vector<int>>{{1,1,1},{33,65,129},{65,80,520},{256,128,256},{512,512,512},{1024,1024,1024},{1025,1537,769}})
    exercise<T>(dtype,shape[0],shape[1],shape[2],false);
  for(auto shape:std::vector<std::vector<int>>{{1,1,1},{33,65,129},{512,512,512}})
    exercise<T>(dtype,shape[0],shape[1],shape[2],true);
  exercise<T>(dtype,33,65,129,false,true);
  exercise<T>(dtype,33,65,129,true,true);
  exercise<T>(dtype,35,49,67,false,false,true);
  exercise<T>(dtype,65,80,520,false,false,false,true);
}
bool raw_b_route(int id) {
  using namespace megafmm::detail::sw_fp32_batch;
  const auto& c=configurations.at(id);
  return c.fused_raw_b || c.producer_policy==ProducerPolicy::WarpTransformTmaB;
}
void packed_suite(int only_id=-1) {
  using namespace megafmm;
  for(const auto& c:detail::sw_fp32_batch::configurations) {
    if(only_id>=0 && c.id!=only_id)continue;
    std::printf("CONFIG packed id=%d %s\n",c.id,detail::sw_fp32_batch::name(c.id).c_str());
    const auto policy_name=detail::sw_fp32_batch::name(c.id);
    require(policy_name.find(";a_source=")!=std::string::npos &&
            policy_name.find(";pack_execution=")!=std::string::npos &&
            policy_name.find(";tail_policy=")!=std::string::npos &&
            policy_name.find(";producer_policy=")!=std::string::npos,"missing dataflow metadata");
    if(c.strategy==detail::sw_fp32_batch::ExecutionStrategy::CooperativeFused) {
      // Half-K=16 still has a padded-to-32 tail. Underflow in the final real
      // FMA can produce -0; trimming that tail must retain its conversion to +0.
      Problem p;p.dtype=DType::FP32;p.m=p.n=2;p.k=p.lda=32;
      p.ldb=p.ldc=p.ldd=2;
      std::vector<float> a(64,0),b(64,0);
      a[15]=a[31]=-1e-30f;b[30]=b[62]=1e-20f;
      Buffer da(256),db(256),d(16),w(detail::sw_fp32_batch::workspace_size(p,c.id));
      check(cudaMemcpy(da.data,a.data(),256,cudaMemcpyHostToDevice));
      check(cudaMemcpy(db.data,b.data(),256,cudaMemcpyHostToDevice));
      detail::sw_fp32_batch::setup(c.id);
      detail::sw_fp32_batch::run(p,c.id,static_cast<float*>(da.data),static_cast<float*>(db.data),nullptr,static_cast<float*>(d.data),w.data,nullptr);
      unsigned first=~0u;check(cudaMemcpy(&first,d.data,4,cudaMemcpyDeviceToHost));
      require(first==0,"guarded K tail changed signed-zero rounding");
      std::printf("PASS packed id=%d signed-zero K tail\n",c.id);
    }
    // Half-K=8,16,17,32,33 catches both BK choices, padding and zero-FMA tails.
    // 2051 rows exercises a partial M group after complete groups when GroupM > 1.
    for(auto shape:std::vector<std::vector<int>>{{1,1,1},{33,65,129},{65,80,520},{512,512,512},
        {1025,1537,769},{67,73,16},{67,73,32},{67,73,33},{67,73,64},{67,73,65},{2051,131,65}})
      exercise<float>(DType::FP32,shape[0],shape[1],shape[2],false,false,false,false,1,true,false,false,c.id);
    for(bool inplace:{false,true})
      exercise<float>(DType::FP32,35,49,67,false,false,false,true,1,true,true,inplace,c.id);
    if(c.strategy==detail::sw_fp32_batch::ExecutionStrategy::DirectBMaterialized || raw_b_route(c.id)) {
      exercise<float>(DType::FP32,512,512,512,false,false,false,false,1,true,false,false,c.id,true);
      exercise<float>(DType::FP32,65,80,raw_b_route(c.id)?512:520,false,false,false,false,1,true,false,false,c.id,false,raw_b_route(c.id)?0:-1);
      exercise<float>(DType::FP32,65,80,16384,false,false,false,false,1,true,false,false,c.id,false,raw_b_route(c.id)?0:-1);
      for(bool inplace:{false,true})
        exercise<float>(DType::FP32,65,80,raw_b_route(c.id)?512:520,false,false,false,false,1,true,false,inplace,c.id,false,raw_b_route(c.id)?0:-1,true);
      if(raw_b_route(c.id))exercise<float>(DType::FP32,65,80,520,false,false,false,false,1,true,false,false,c.id,false,0);
      // Padded maps remain usable, but raw descriptor's exact stride/dimensions
      // differ. Verify full packing fallback, including aligned changed ldb.
      for(int change:{1,2,3})
        exercise<float>(DType::FP32,65,80,raw_b_route(c.id)?576:544,false,false,false,false,1,true,false,false,c.id,false,raw_b_route(c.id)?0:-1,true,change);
    }
    if(c.strategy==detail::sw_fp32_batch::ExecutionStrategy::CooperativeFused) {
      // Force an unsupported cooperative capacity, then test a task grid
      // exceeding occupancy capacity through resident CTA iteration.
      exercise<float>(DType::FP32,512,512,512,false,false,false,false,1,true,false,false,c.id,true);
      exercise<float>(DType::FP32,1024,1024,1024,false,false,false,false,1,true,false,false,c.id);
      exercise<float>(DType::FP32,2048,2048,32,false,false,false,false,1,true,false,false,c.id);
      exercise<float>(DType::FP32,65,80,16384,false,false,false,false,1,true,false,false,c.id);
      if(c.persistent_tasks) {
        exercise<float>(DType::FP32,128,128,520,false,false,false,false,1,true,false,false,c.id,false,1);
        // Reuse a single CTA across one-, two- and three-stage tasks.
        for(int k:{8,96,160})
          exercise<float>(DType::FP32,96,128,k,false,false,false,false,1,true,false,false,c.id,false,1);
        // Large resident grid from the intermittent dense-input reproducer.
        exercise<float>(DType::FP32,4096,512,256,false,false,false,false,1,true,false,false,c.id);
      }

    }
    concurrent<float>(DType::FP32,false,true,c.id);
    if(raw_b_route(c.id))concurrent<float>(DType::FP32,false,true,c.id,true);
  }
  Problem budget_problem;budget_problem.dtype=DType::FP32;
  budget_problem.m=budget_problem.n=budget_problem.k=512;
  budget_problem.lda=budget_problem.ldb=budget_problem.ldc=budget_problem.ldd=512;
  Policy budget;budget.algorithm=Algorithm::SWTuned;
  const auto z=detail::sw_fp32_batch::shape(budget_problem);
  budget.workspace_limit=(z.a+z.b)*4;
  auto limited=create_plan(budget_problem,budget);
  bool found_fused=false;
  for(const auto& c:limited->sw_tuning_results())if(c.id>=9500000) {
    require(detail::sw_fp32_batch::configurations[c.id-9500000].strategy==detail::sw_fp32_batch::ExecutionStrategy::CooperativeFused,
      "materialized candidate exceeded fusion-only workspace budget");
    found_fused=true;
  }
  require(found_fused && limited->workspace_bytes()==budget.workspace_limit,"fusion-only workspace budget failed");
  for(int id:{-1,int(detail::sw_fp32_batch::configurations.size())}) {
    bool rejected=false;
    try {detail::sw_fp32_batch::setup(id);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"invalid packed template id accepted");
  }
}
}
int main(int argc,char** argv) {try {
  std::setvbuf(stdout,nullptr,_IOLBF,0);
  if(argc==3 && std::string(argv[1])=="--packed-smoke") {
    using namespace megafmm;
    const int id=std::stoi(argv[2]);
    require(id>=0 && id<int(detail::sw_fp32_batch::configurations.size()),"invalid packed smoke config");
    exercise<float>(DType::FP32,33,65,129,false,false,false,false,1,true,false,false,id);
    exercise<float>(DType::FP32,128,128,520,false,false,false,false,1,true,false,false,id,false,1);
    exercise<float>(DType::FP32,65,80,512,false,false,false,false,1,true,false,false,id,false,0);
    exercise<float>(DType::FP32,35,49,67,false,false,false,true,1,true,true,true,id);
    concurrent<float>(DType::FP32,false,true,id);
    return 0;
  }
  if(argc==2 && std::string(argv[1])=="--packed-only") {packed_suite();return 0;}
  if(argc==3 && std::string(argv[1])=="--packed-config") {
    const int id=std::stoi(argv[2]);
    require(id>=0 && id<int(megafmm::detail::sw_fp32_batch::configurations.size()),"invalid packed test config");
    packed_suite(id);return 0;
  }
  if(argc!=1)throw std::invalid_argument("usage: test_sw_paths [--packed-only | --packed-config ID | --packed-smoke ID]");
  suite<__nv_bfloat16>(megafmm::DType::BF16);
  suite<__half>(megafmm::DType::FP16);
  suite<float>(megafmm::DType::FP32);
  exercise<__nv_bfloat16>(megafmm::DType::BF16,256,128,256,true,false,false,false,1048576);
  packed_suite();
  for(bool tuned:{false,true}) {
    concurrent<__nv_bfloat16>(megafmm::DType::BF16,tuned);
    concurrent<__half>(megafmm::DType::FP16,tuned);
    concurrent<float>(megafmm::DType::FP32,tuned);
  }
  failed_retune();
  std::puts("PASS general SW paths, strict tuning, guards, native BF16 range, and fresh bindings");
  return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

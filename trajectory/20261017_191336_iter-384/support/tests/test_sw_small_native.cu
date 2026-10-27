// Reuse the independent integer oracle and fresh-input fixtures, not its main.
#define main original_sw_tuner_test_main
#include "test_sw_tuner.cu"
#undef main
#include "sw_small_native.h"
#include <cstring>
int main(){try{
 int device=0;cudaDeviceProp prop{};check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
 for(auto dtype:{DType::BF16,DType::FP16})for(auto dims:std::vector<std::vector<int>>{{128,256,512},{512,512,512},{512,1024,512},{1024,1024,512},{512,512,256},{512,512,1024},{8192,1024,512},{1024,8192,512},{2048,2048,1024}}){
  auto p=problem(dims[0],dims[1],dims[2],dtype);Stream stream;
  Buffer a(p.m*p.k*2),b(p.k*p.n*2),d((p.m*p.n+64)*4);
  Policy reference_policy;reference_policy.algorithm=Algorithm::Dense;
  auto dense=create_plan(p,reference_policy);Buffer dense_d(p.m*p.n*4),dense_ws(dense->workspace_bytes());
  for(int id=0;id<6;++id){
   if(id>=3 && p.k!=512)continue;
   auto z=sw_small_native::shape(p,{id,128,0});Buffer ws(z.bytes+256);
   check(cudaMemset(static_cast<char*>(ws.p)+z.bytes,0xa5,256));
   sw_small_native::Config config{id,128,0};auto r=sw_small_native::setup(p,config);
   candidate_context="small_native id="+std::to_string(id);require(r.resident>0,"nonresident small-native kernel");
   for(int seed:{1,2,3}){
    float scale=dtype==DType::BF16 && seed>1?1048576.0f:1.0f;
    if(dtype==DType::BF16)fixture<__nv_bfloat16>(p,a,b,seed,false,scale);else fixture<__half>(p,a,b,seed,false,scale);
    guard(p,d);
    if(seed==3){
     cudaGraph_t graph;cudaGraphExec_t exec;check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));
     sw_small_native::run(p,config,a.p,b.p,static_cast<float*>(d.p),ws.p,prop.multiProcessorCount,stream.s);
     check(cudaStreamEndCapture(stream.s,&graph));check(cudaGraphInstantiate(&exec,graph,0));
     // Change values after capture to detect input-dependent capture or caching.
     if(dtype==DType::BF16)fixture<__nv_bfloat16>(p,a,b,4,false,scale);else fixture<__half>(p,a,b,4,false,scale);
     check(cudaGraphLaunch(exec,stream.s));check(cudaStreamSynchronize(stream.s));verify(p,d,4,scale);cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
    }else{
     sw_small_native::run(p,config,a.p,b.p,static_cast<float*>(d.p),ws.p,prop.multiProcessorCount,stream.s);check(cudaStreamSynchronize(stream.s));verify(p,d,seed,scale);
    }
    unsigned char guard_bytes[256];check(cudaMemcpy(guard_bytes,static_cast<char*>(ws.p)+z.bytes,256,cudaMemcpyDeviceToHost));
    for(auto v:guard_bytes)require(v==0xa5,"small-native workspace overrun");
   }
   // Full-rank random integer data catches errors hidden by the separable
   // structured fixture. Reference FP64 AB uses only original input matrices.
   float scale=dtype==DType::BF16?1048576.0f:1.0f;
   std::mt19937 rng(773+id);std::uniform_int_distribution<int> integers(-2,2);
   auto random_values=[&](auto type){
    using T=decltype(type);std::vector<T> av(p.m*p.k),bv(p.k*p.n);
    for(auto& v:av)v=rounded<T>(float(integers(rng))*scale);
    for(auto& v:bv)v=rounded<T>(float(integers(rng))*scale);
    check(cudaMemcpy(a.p,av.data(),av.size()*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(b.p,bv.data(),bv.size()*2,cudaMemcpyHostToDevice));
   };
   if(dtype==DType::BF16)random_values(__nv_bfloat16{});else random_values(__half{});
   check(cudaDeviceSynchronize());guard(p,d);
   megafmm::run(*dense,a.p,b.p,nullptr,static_cast<float*>(dense_d.p),dense_ws.p,stream.s);
   auto original=detail::make_sw_reference(p,a.p,b.p,static_cast<float*>(dense_d.p),stream.s);
   sw_small_native::run(p,config,a.p,b.p,static_cast<float*>(d.p),ws.p,prop.multiProcessorCount,stream.s);
   auto error=detail::check_sw_reference(*original,static_cast<float*>(d.p),stream.s);
   require(error.relative_error==0,"random-integer small-native mismatch against original-input FP64 AB");
   std::vector<float> tail(64);check(cudaMemcpy(tail.data(),static_cast<float*>(d.p)+p.m*p.n,64*4,cudaMemcpyDeviceToHost));
   for(float v:tail)require(v==-12345,"random small-native output guard");
   unsigned char workspace_tail[256];check(cudaMemcpy(workspace_tail,static_cast<char*>(ws.p)+z.bytes,256,cudaMemcpyDeviceToHost));
   for(auto v:workspace_tail)require(v==0xa5,"random small-native workspace guard");
   // Same native tile and arithmetic, fresh fractional input: only wait/release
   // scheduling changes. Require exact output bits, not a relaxed error gate.
   if(dtype==DType::BF16)fixture<__nv_bfloat16>(p,a,b,847+id,true);else fixture<__half>(p,a,b,847+id,true);
   guard(p,d);Buffer expected(p.m*p.n*4);
   bool narrow=id!=5 && id!=7;
   detail::SWConfig matching_native{-99,narrow?16:32,id==2?64:128,id==2?4:8,128,0,narrow};
   Buffer native_ws(detail::sw_config_workspace(p,matching_native));
   detail::setup_sw_config(p,matching_native);
   detail::run_sw_config(p,matching_native,a.p,b.p,static_cast<float*>(expected.p),native_ws.p,stream.s);
   sw_small_native::run(p,config,a.p,b.p,static_cast<float*>(d.p),ws.p,prop.multiProcessorCount,stream.s);
   check(cudaStreamSynchronize(stream.s));
   std::vector<float> ev(p.m*p.n),actual(p.m*p.n+64);
   check(cudaMemcpy(ev.data(),expected.p,ev.size()*4,cudaMemcpyDeviceToHost));
   check(cudaMemcpy(actual.data(),d.p,actual.size()*4,cudaMemcpyDeviceToHost));
   require(std::memcmp(ev.data(),actual.data(),ev.size()*4)==0,"async fractional output differs from matching native tile");
   for(size_t i=ev.size();i<actual.size();++i)require(actual[i]==-12345,"fractional output guard");
   check(cudaMemcpy(workspace_tail,static_cast<char*>(ws.p)+z.bytes,256,cudaMemcpyDeviceToHost));
   for(auto v:workspace_tail)require(v==0xa5,"fractional workspace guard");
   printf("PASS small_native %dx%dx%d %s config=%d regs=%d shared=%zu resident=%d bytes=%zu fresh=5 graph+range+guards+integer_FP64+fractional_native_bitwise\n",dims[0],dims[1],dims[2],dtype_name(dtype),id,r.registers,r.shared,r.resident,z.bytes);fflush(stdout);
  }
 }
 return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}

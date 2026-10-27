// Reuse the independent integer oracle and fresh-input fixtures, not its main.
#define main original_sw_tuner_test_main
#include "test_sw_tuner.cu"
#undef main
#include "sw_seven_batch.h"
int main(){try{
 int device=0;cudaDeviceProp prop{};check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
 for(auto dtype:{DType::BF16,DType::FP16})for(auto dims:std::vector<std::vector<int>>{{1,3,1},{3,1,5},{129,257,33},{1024,1024,1024},{514,770,258},{512,1024,512},{12289,520,1032}}){
  auto p=problem(dims[0],dims[1],dims[2],dtype);auto z=sw_seven_batch::shape(p);Stream stream;
  Buffer a(p.m*p.k*2),b(p.k*p.n*2),d((p.m*p.n+64)*4),ws(z.bytes+256);
  check(cudaMemset(static_cast<char*>(ws.p)+z.bytes,0xa5,256));
  Policy reference_policy;reference_policy.algorithm=Algorithm::Dense;
  auto dense=create_plan(p,reference_policy);Buffer dense_d(p.m*p.n*4),dense_ws(dense->workspace_bytes());
  for(int id=0;id<4;++id){
   sw_seven_batch::Config config{id,512,0};auto r=sw_seven_batch::setup(p,config);
   candidate_context="seven_batch id="+std::to_string(id);require(r.resident>0,"nonresident seven-batch kernel");
   for(int seed:{1,2,3}){
    float scale=dtype==DType::BF16 && seed>1?1048576.0f:1.0f;
    if(dtype==DType::BF16)fixture<__nv_bfloat16>(p,a,b,seed,false,scale);else fixture<__half>(p,a,b,seed,false,scale);
    guard(p,d);
    if(seed==3){
     cudaGraph_t graph;cudaGraphExec_t exec;check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));
     sw_seven_batch::run(p,config,a.p,b.p,static_cast<float*>(d.p),ws.p,prop.multiProcessorCount,stream.s);
     check(cudaStreamEndCapture(stream.s,&graph));check(cudaGraphInstantiate(&exec,graph,0));
     // Change values after capture to detect input-dependent capture or caching.
     if(dtype==DType::BF16)fixture<__nv_bfloat16>(p,a,b,4,false,scale);else fixture<__half>(p,a,b,4,false,scale);
     check(cudaGraphLaunch(exec,stream.s));check(cudaStreamSynchronize(stream.s));verify(p,d,4,scale);cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
    }else{
     sw_seven_batch::run(p,config,a.p,b.p,static_cast<float*>(d.p),ws.p,prop.multiProcessorCount,stream.s);check(cudaStreamSynchronize(stream.s));verify(p,d,seed,scale);
    }
    unsigned char guard_bytes[256];check(cudaMemcpy(guard_bytes,static_cast<char*>(ws.p)+z.bytes,256,cudaMemcpyDeviceToHost));
    for(auto v:guard_bytes)require(v==0xa5,"seven-batch workspace overrun");
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
   sw_seven_batch::run(p,config,a.p,b.p,static_cast<float*>(d.p),ws.p,prop.multiProcessorCount,stream.s);
   auto error=detail::check_sw_reference(*original,static_cast<float*>(d.p),stream.s);
   require(error.relative_error==0,"random-integer seven-batch mismatch against original-input FP64 AB");
   std::vector<float> tail(64);check(cudaMemcpy(tail.data(),static_cast<float*>(d.p)+p.m*p.n,64*4,cudaMemcpyDeviceToHost));
   for(float v:tail)require(v==-12345,"random seven-batch output guard");
   unsigned char workspace_tail[256];check(cudaMemcpy(workspace_tail,static_cast<char*>(ws.p)+z.bytes,256,cudaMemcpyDeviceToHost));
   for(auto v:workspace_tail)require(v==0xa5,"random seven-batch workspace guard");
   printf("PASS seven_batch %dx%dx%d %s config=%d regs=%d shared=%zu resident=%d bytes=%zu fresh=4 graph+range+guards+full_rank_integer_FP64\n",dims[0],dims[1],dims[2],dtype_name(dtype),id,r.registers,r.shared,r.resident,z.bytes);fflush(stdout);
  }
 }
 return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}

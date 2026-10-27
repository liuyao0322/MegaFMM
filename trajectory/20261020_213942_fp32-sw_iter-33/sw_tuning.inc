// Included after run_internal: shares Plan/Lt machinery, never dispatches full AB as SW.
namespace {
struct SWTuningBuffer {
  void* data=nullptr;
  explicit SWTuningBuffer(size_t bytes){if(bytes)cuda_check(cudaMalloc(&data,bytes),"SW tuning allocation");}
  ~SWTuningBuffer(){if(data)cudaFree(data);}
};
struct SWTuningEvent {
  cudaEvent_t value=nullptr;
  SWTuningEvent(){cuda_check(cudaEventCreate(&value),"SW tuning event");}
  ~SWTuningEvent(){if(value)cudaEventDestroy(value);}
};
struct SWTuningGraphs {
  std::vector<cudaGraph_t> graphs;
  std::vector<cudaGraphExec_t> execs;
  ~SWTuningGraphs(){for(auto e:execs)cudaGraphExecDestroy(e);for(auto g:graphs)cudaGraphDestroy(g);}
};
float sw_median(std::vector<float> values) {
  if(values.empty())return INFINITY;
  std::sort(values.begin(),values.end());return values[values.size()/2];
}
void execute_sw_candidate(const Plan::Impl& plan,int i,const void* A,const void* B,float* D,void* ws,cudaStream_t stream) {
  const auto& c=plan.sw_configs.at(i);
  if(c.id==9600000 || c.id==9600001) {
    detail::sw_fp32_tiny::run(plan.problem,c.id-9600000,static_cast<const float*>(A),static_cast<const float*>(B),nullptr,D,stream);return;
  }
  if(c.id>=9500000 && c.id<9500000+int(detail::sw_fp32_batch::configurations.size())) {
    const int id=c.id-9500000;
    if(id<int(plan.sw_batch_maps.size()) && plan.sw_batch_maps[id].workspace==ws && plan.sw_batch_maps[id].id==id)
      detail::sw_fp32_batch::run_prepared(plan.problem,id,static_cast<const float*>(A),static_cast<const float*>(B),nullptr,D,ws,stream,plan.sw_batch_maps[id]);
    else detail::sw_fp32_batch::run(plan.problem,id,static_cast<const float*>(A),static_cast<const float*>(B),nullptr,D,ws,stream);
    return;
  }
  if(c.id>=9400000 && c.id<9400000+int(detail::sw_fp32_phase::configurations.size())) {
    detail::sw_fp32_phase::run(plan.problem,c.id-9400000,static_cast<const float*>(A),static_cast<const float*>(B),nullptr,D,stream);return;
  }
  if(c.id==9300000 || c.id==9300001) {
    detail::sw_fp32_tiled::small_run(plan.problem,c.id-9300000,static_cast<const float*>(A),static_cast<const float*>(B),nullptr,D,stream);return;
  }
  if(c.id>=9200000 && c.id<9200000+int(detail::sw_fp32_tiled::configurations.size())) {
    detail::sw_fp32_tiled::run(plan.problem,c.id-9200000,static_cast<const float*>(A),static_cast<const float*>(B),nullptr,D,stream);return;
  }
  if(c.id==9000016 || c.id==9000032 || c.id==9100016 || c.id==9100032) {
    detail::run_sw_fp32_config(plan.problem,static_cast<const float*>(A),static_cast<const float*>(B),nullptr,D,stream,c.bk,c.id>=9100000);return;
  }
  if(c.id>=8000000 && c.id<8100000){detail::sw_parameterized::run(plan.problem,c.id-8000000,A,B,D,ws,stream);return;}
  if(c.id>=700000 && c.id<700002)sw_frontier512::run(plan.problem,{c.id-700000,c.pack_cap,c.raster},A,B,D,ws,plan.sm_count,stream);
  else if(c.id>=600000 && c.id<600002)sw_cross_product::run(plan.problem,{c.id-600000,c.pack_cap,c.raster},A,B,D,ws,plan.sm_count,stream);
  else if(c.id>=500000 && c.id<500006)sw_small_native::run(plan.problem,{c.id-500000,c.pack_cap,c.raster},A,B,D,ws,plan.sm_count,stream);
  else if(c.id>=100000 && c.id<100004)
    sw_seven_batch::run(plan.problem,{c.id-100000,c.pack_cap,c.raster},A,B,D,ws,plan.sm_count,stream);
  else if(c.id==-2)detail::run_sw_pipeline(*plan.sw1024,A,B,D,ws,stream);
  else if(c.id==-3)detail::run_sw_hopper(plan.problem,A,B,D,ws,stream);
  else detail::run_sw_config(plan.problem,c,A,B,D,ws,stream);
}
void tune_sw(Plan::Impl& plan,const void* A,const void* B,float* D,void* ws,cudaStream_t stream) {
  // Failed retuning invalidates the previous choice. No stale winner survives a failed gate.
  plan.sw_selected=-1;plan.sw_tune_status="tuning";
  for(auto& r:plan.sw_results){r.status="unmeasured";r.numerical_pass=false;r.ranking_ms.clear();r.confirmation_ms.clear();r.relative_error=0;r.gate_limit=0;r.precision_errors.clear();r.precision_limits.clear();}
  struct OwnedStream {cudaStream_t value=nullptr;~OwnedStream(){if(value)cudaStreamDestroy(value);}} owned;
  if(plan.sw_policy.sw_tuning_graph && !stream) {
    cuda_check(cudaStreamSynchronize(nullptr),"SW default stream input readiness");
    cuda_check(cudaStreamCreateWithFlags(&owned.value,cudaStreamNonBlocking),"SW graph tuning stream");stream=owned.value;
  }
  const auto& p=plan.problem;
  // Shape/stride/address metadata only; every invocation still packs fresh data.
  plan.sw_batch_maps.clear();
  if(p.dtype==DType::FP32)for(const auto& config:plan.sw_configs)if(config.id>=9500000 && config.id<9500000+int(detail::sw_fp32_batch::configurations.size())) {
    const int id=config.id-9500000;
    if(detail::sw_fp32_batch::configurations[id].cutlass)continue;
    if(plan.sw_batch_maps.empty())plan.sw_batch_maps.resize(detail::sw_fp32_batch::configurations.size());
    plan.sw_batch_maps[id]=detail::sw_fp32_batch::prepare(p,id,ws);
  }
  if(reinterpret_cast<uintptr_t>(A)%16 || reinterpret_cast<uintptr_t>(B)%16)
    throw std::invalid_argument("SW tuner requires 16-byte aligned input bindings");
  const int sets=plan.strategy_input_sets;
  const size_t ab=size_t(p.m)*p.k*dtype_size(p.dtype),bb=size_t(p.k)*p.n*dtype_size(p.dtype),db=size_t(p.m)*p.n*4;
  std::vector<std::unique_ptr<SWTuningBuffer>> allocations;
  std::vector<const void*> as{A},bs{B};
  for(int j=1;j<sets;++j) {
    allocations.push_back(std::make_unique<SWTuningBuffer>(ab));as.push_back(allocations.back()->data);
    allocations.push_back(std::make_unique<SWTuningBuffer>(bb));bs.push_back(allocations.back()->data);
    cuda_check(cudaMemcpyAsync(const_cast<void*>(as.back()),A,ab,cudaMemcpyDeviceToDevice,stream),"SW rotating A");
    cuda_check(cudaMemcpyAsync(const_cast<void*>(bs.back()),B,bb,cudaMemcpyDeviceToDevice,stream),"SW rotating B");
  }
  // Dense is solely the independent error allowance reference. It is never a route in this search.
  SWTuningBuffer dense_output(db);
  cublasLtHandle_t handle=nullptr;
  lt_check(cublasLtCreate(&handle),"SW reference handle");
  struct HandleGuard {cublasLtHandle_t h;~HandleGuard(){cublasLtDestroy(h);}} handle_guard{handle};
  auto dense=make_lt_op(handle,p,plan.sw_policy.workspace_limit,32);
  SWTuningBuffer dense_workspace(dense->scratch_bytes());
  tune_lt(handle,*dense,A,B,nullptr,static_cast<float*>(dense_output.data),dense_workspace.data,stream);
  std::vector<std::shared_ptr<detail::SWReference>> references;
  for(int j=0;j<sets;++j) {
    lt_check(execute_lt(handle,*dense,bound_candidate(*dense,as[j],bs[j],nullptr,static_cast<float*>(dense_output.data)),as[j],bs[j],nullptr,static_cast<float*>(dense_output.data),dense_workspace.data,stream),"SW independent dense reference");
    references.push_back(detail::make_sw_reference(p,as[j],bs[j],static_cast<float*>(dense_output.data),stream));
  }
  std::vector<int> order;
  for(int i=0;i<int(plan.sw_results.size());++i)order.push_back(i);
  std::stable_sort(order.begin(),order.end(),[&](int a,int b){return plan.sw_results[a].model_score<plan.sw_results[b].model_score;});
  if(plan.sw_policy.sw_tuning_candidates>0 && int(order.size())>plan.sw_policy.sw_tuning_candidates)order.resize(plan.sw_policy.sw_tuning_candidates);
  for(auto& r:plan.sw_results)r.status="model_not_sampled";
  std::vector<int> pool;
  SWTuningEvent start,stop;
  // Captured graphs own only launch metadata. Every replay recomputes all SW transforms.
  SWTuningGraphs graphs;
  const bool graph=plan.sw_policy.sw_tuning_graph;
  std::vector<std::vector<cudaGraphExec_t>> routes(plan.sw_results.size());
  auto launch=[&](int i,int set){
    if(graph)cuda_check(cudaGraphLaunch(routes[i][set],stream),"SW tuning graph launch");
    else execute_sw_candidate(plan,i,as[set],bs[set],D,ws,stream);
  };
  for(int i:order) {
    auto& r=plan.sw_results[i];bool pass=true,finite=true;
    for(int j=0;j<sets;++j) {
      execute_sw_candidate(plan,i,as[j],bs[j],D,ws,stream);
      auto result=detail::check_sw_reference(*references[j],D,stream);
      r.precision_errors.push_back(result.relative_error);r.precision_limits.push_back(result.limit);
      pass&=result.pass;finite&=std::isfinite(result.relative_error);
      r.relative_error=std::max(r.relative_error,result.relative_error);r.gate_limit=std::max(r.gate_limit,result.limit);
    }
    r.numerical_pass=pass;
    if(!finite){r.status="nonfinite_rejected";continue;}
    if(!pass && !plan.sw_policy.sw_tuning_allow_inaccurate){r.status="precision_rejected";continue;}
    if(graph)for(int j=0;j<sets;++j) {
      cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"SW tuning capture begin");
      try {execute_sw_candidate(plan,i,as[j],bs[j],D,ws,stream);}
      catch(...){cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(stream,&abandoned);if(abandoned)cudaGraphDestroy(abandoned);throw;}
      cudaGraph_t g=nullptr;cuda_check(cudaStreamEndCapture(stream,&g),"SW tuning capture end");graphs.graphs.push_back(g);
      cudaGraphExec_t e=nullptr;cuda_check(cudaGraphInstantiate(&e,g,0),"SW tuning graph instantiate");graphs.execs.push_back(e);routes[i].push_back(e);
    }
    r.status=pass?"eligible":"eligible_error_reported";
    pool.push_back(i);
  }
  if(pool.empty()) {
    plan.sw_tune_status=plan.sw_policy.sw_tuning_allow_inaccurate?"no_finite_candidate":"no_precision_valid_candidate";
    throw std::runtime_error("SW tuning: "+plan.sw_tune_status+"; full-AB dense fallback is prohibited");
  }
  auto measure=[&](int i,int repetitions,int warm) {
    for(int w=0;w<warm;++w)launch(i,w%sets);
    cuda_check(cudaEventRecord(start.value,stream),"SW timing start");
    for(int j=0;j<repetitions;++j)launch(i,j%sets);
    cuda_check(cudaEventRecord(stop.value,stream),"SW timing stop");cuda_check(cudaEventSynchronize(stop.value),"SW timing wait");
    float ms=0;cuda_check(cudaEventElapsedTime(&ms,start.value,stop.value),"SW elapsed");
    if(!(ms>0) || !std::isfinite(ms))throw std::runtime_error("SW invalid timing sample");
    return ms/repetitions;
  };
  for(int round=0;round<5;++round)for(int j=0;j<int(pool.size());++j) {
    int i=pool[round%2?pool.size()-1-j:j];plan.sw_results[i].ranking_ms.push_back(measure(i,20,4));
  }
  std::stable_sort(pool.begin(),pool.end(),[&](int a,int b){return sw_median(plan.sw_results[a].ranking_ms)<sw_median(plan.sw_results[b].ranking_ms);});
  // Confirm several finalists; comparing only with cuBLAS would hide SW improvements below parity.
  pool.resize(std::min<size_t>(pool.size(),4));
  for(int round=0;round<5;++round)for(int j=0;j<int(pool.size());++j) {
    int i=pool[round%2?pool.size()-1-j:j];plan.sw_results[i].confirmation_ms.push_back(measure(i,100,20));
    plan.sw_results[i].status=plan.sw_results[i].numerical_pass?"confirmed":"confirmed_error_reported";
  }
  int selected=pool.front();
  for(int i:pool)if(sw_median(plan.sw_results[i].confirmation_ms)<sw_median(plan.sw_results[selected].confirmation_ms))selected=i;
  // Validate once more after sustained execution; tuning must not certify a corrupt replay.
  for(int j=0;j<sets;++j) {
    launch(selected,j);auto result=detail::check_sw_reference(*references[j],D,stream);
    if(!std::isfinite(result.relative_error) || (!plan.sw_policy.sw_tuning_allow_inaccurate && !result.pass))throw std::runtime_error("SW final validation failed");
  }
  plan.sw_selected=selected;
  plan.sw_tune_status=plan.sw_results[selected].numerical_pass?"precision_pass":"error_reported";
}
} // namespace

from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','c0df342:solution/sw_pipeline.cu'],text=True)
s=s.replace('uint64_t pack_ready;','uint64_t pack_ready,cluster_ready,cluster_done;')
pos=s.index('__device__ __forceinline__ int product_index')
s=s[:pos]+'''// Explicit cluster scope carries acquired global transform writes across CTAs.
__device__ void cluster_arrive(const uint64_t* ptr,int rank){
 uint32_t addr=cute::cast_smem_ptr_to_uint(ptr);
 asm volatile("{ .reg .b32 remote; mapa.shared::cluster.u32 remote,%0,%1; mbarrier.arrive.release.cluster.shared::cluster.b64 _,[remote]; }" :: "r"(addr),"r"(rank):"memory");
}
__device__ void cluster_wait(const uint64_t* ptr){
 uint32_t addr=cute::cast_smem_ptr_to_uint(ptr);
 asm volatile("{ .reg .pred done; AGAIN: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 done,[%0],0; @!done bra AGAIN; }" :: "r"(addr):"memory");
}

'''+s[pos:]
s=s.replace('Barrier::init(&s.pack_ready,608);','Barrier::init(&s.pack_ready,608);\n    Barrier::init(&s.cluster_ready,2);Barrier::init(&s.cluster_done,1);')
s=s.replace('  __syncthreads();','  cooperative_groups::this_cluster().sync();')
s=s.replace('        // CUDA12.4 cooperative launch', '''        int rank=blockIdx.x&1;
        cluster_arrive(&s.cluster_ready,0);
        if(rank==0){
        cluster_wait(&s.cluster_ready);
        // CUDA12.4 cooperative launch''')
s=s.replace('0x80000000u-127u','0x80000000u-63u')
needle='}while(!((old^current)&0x80000000u));'
s=s.replace(needle,needle+'''
        cluster_arrive(&s.cluster_done,0);
        cluster_arrive(&s.cluster_done,1);
        }
        cluster_wait(&s.cluster_done);''')
a=s.index('void run_sw_pipeline(')
s=s[:a]+'''template<class T> cudaLaunchConfig_t launch_config(cudaStream_t stream,cudaLaunchAttribute (&attrs)[2]){
  attrs[0].id=cudaLaunchAttributeCooperative;attrs[0].val.cooperative=1;
  attrs[1].id=cudaLaunchAttributeClusterDimension;attrs[1].val.clusterDim={2,1,1};
  cudaLaunchConfig_t config{};config.gridDim=dim3(16,8);config.blockDim=Threads;
  config.dynamicSmemBytes=sizeof(Shared<T>);config.stream=stream;config.attrs=attrs;config.numAttrs=2;
  return config;
}
void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream){
  CUtensorMap a{},b{},s{},t{};encode(a,A,state.dtype,true,false);encode(b,B,state.dtype,false,false);
  encode(s,workspace,state.dtype,true,true);encode(t,static_cast<char*>(workspace)+WorkspaceBytes/2,state.dtype,false,true);
  cudaLaunchAttribute attrs[2]{};
  if(state.dtype==DType::BF16){
    auto config=launch_config<cute::bfloat16_t>(stream,attrs);
    check(cudaLaunchKernelEx(&config,sw_cooperative_hilo<cute::bfloat16_t>,a,b,s,t,static_cast<const cute::bfloat16_t*>(A),static_cast<const cute::bfloat16_t*>(B),static_cast<cute::bfloat16_t*>(workspace),D),"SW cluster launch");
  }else{
    auto config=launch_config<cute::half_t>(stream,attrs);
    check(cudaLaunchKernelEx(&config,sw_cooperative_hilo<cute::half_t>,a,b,s,t,static_cast<const cute::half_t*>(A),static_cast<const cute::half_t*>(B),static_cast<cute::half_t*>(workspace),D),"SW cluster launch");
  }
}
} // namespace megafmm::detail
'''
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','c0df342:solution/plan.cpp'],text=True).replace('st=compact4planes_balanced_async_p1p2','st=compact4planes_balanced_async_p1p2;publication=cluster2_hierarchical')
Path('solution/plan.cpp').write_text(s)

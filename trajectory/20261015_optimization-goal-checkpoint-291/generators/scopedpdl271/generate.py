from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','18f9a6f:solution/sw_pipeline.cu'],text=True)
s=s.replace('  cudaTriggerProgrammaticLaunchCompletion();\n  for(int i=2*(blockIdx.x*512+threadIdx.x);i<Plane;i+=2*512*128){','''  auto flag=reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(packed)+WorkspaceBytes)+blockIdx.x*32;
  if(threadIdx.x==0){
    asm volatile("st.global.relaxed.gpu.u32 [%0],%1;" :: "l"(flag),"r"(0) : "memory");
    // Publish this invocation's zero before its CTA permits secondary launch.
    // Every primary CTA must trigger, so no secondary may see stale flags.
    __threadfence();
    cudaTriggerProgrammaticLaunchCompletion();
  }
  for(int local=2*threadIdx.x;local<32*64;local+=2*512){
    int i=(int(blockIdx.x)/8*32+local/64)*512+(int(blockIdx.x)%8)*64+local%64;''')
s=s.replace('  asm volatile("fence.proxy.async.global;" ::: "memory");\n}', '''  asm volatile("fence.proxy.async.global;" ::: "memory");
  __syncthreads();
  if(threadIdx.x==0)asm volatile("st.global.release.gpu.u32 [%0],%1;" :: "l"(flag),"r"(1) : "memory");
}''')
s=s.replace('      if(sequence==4)cudaGridDependencySynchronize();','''      if(sequence==4 || sequence==5){
        auto flags=reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(packed)+WorkspaceBytes);
        auto await_tile=[&](int tile){
          uint32_t ready;
          do{asm volatile("ld.global.acquire.gpu.u32 %0,[%1];":"=r"(ready):"l"(flags+tile*32):"memory");}while(!ready);
        };
        // Each primary produces all eight S/T planes for its32x64 input tile.
        // First transformed leaf q0/q1 covers every transformed region later used.
        for(int j=0;j<8;++j)await_tile((k0/32+j)*8+m0/64);
        for(int j=0;j<4;++j)await_tile((n0/32)*8+k0/64+j);
      }''')
s=s.replace('      // Raw P1/P2 can run while the primary transforms S/T. This wait\n      // guarantees completion and visibility even if no overlap is scheduled.','      // Per-tile acquire observes this invocation\'s release publication.')
assert 'cudaGridDependencySynchronize()' not in s
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','18f9a6f:solution/plan.cpp'],text=True).replace('impl->bytes=size_t(4)<<20;','impl->bytes=(size_t(4)<<20)+16384;').replace('workspace=4194304','workspace=4210688').replace('st=compact4planes_balanced_async_p1p2','st=compact4planes_tile32x64_async_p1p2').replace('publication=programmatic_dependent_launch_grid_dependency_wait','publication=pdl_prefix_fenced_zero_then_tile_release_acquire')
Path('solution/plan.cpp').write_text(s)

from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','18f9a6f:solution/sw_pipeline.cu'],text=True)
s=s.replace('  cudaTriggerProgrammaticLaunchCompletion();','''  if(threadIdx.x==0){
    auto flag=reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(packed)+WorkspaceBytes)+blockIdx.x*32;
    asm volatile("st.global.relaxed.gpu.u32 [%0],%1;" :: "l"(flag),"r"(0) : "memory");
    __threadfence();
    cudaTriggerProgrammaticLaunchCompletion();
  }''')
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','18f9a6f:solution/plan.cpp'],text=True).replace('impl->bytes=size_t(4)<<20;','impl->bytes=(size_t(4)<<20)+16384;').replace('workspace=4194304','workspace=4210688').replace('publication=programmatic_dependent_launch_grid_dependency_wait','publication=programmatic_dependent_launch_grid_dependency_wait;prefix=flag_zero_gpu_fence')
Path('solution/plan.cpp').write_text(s)

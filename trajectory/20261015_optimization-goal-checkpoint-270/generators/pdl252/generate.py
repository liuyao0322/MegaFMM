from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','62142fc:solution/sw_pipeline.cu'],text=True)
# Extract exactly the existing transform math into a separate primary kernel.
a=s.index('    auto av=reinterpret_cast<const uint32_t*>(A);');b=s.index('\n  }\n    asm volatile("fence.proxy.async.global;',a)
math=s[a:b]
pack='''template<class T> __global__ __launch_bounds__(512,1)
void sw_transform_pdl(const T* A,const T* B,T* packed){
  cudaTriggerProgrammaticLaunchCompletion();
  for(int i=2*(blockIdx.x*512+threadIdx.x);i<Plane;i+=2*512*128){
    int row=i/512,col=i%512,at=row*1024+col;
'''+math+'''
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
}
'''
a=s.index('template<class T> __global__');s=s[:a]+pack+s[a:]
s=s.replace('Threads=672','Threads=256').replace('  uint64_t pack_ready;','')
s=s.replace('    Barrier::init(&s.pack_ready,512);\n','')
a=s.index('      if(sequence==4){');b=s.index('      if(sequence>=Stages)',a)
s=s[:a]+'''      // Raw P1/P2 can run while the primary transforms S/T. This wait
      // guarantees completion and visibility even if no overlap is scheduled.
      if(sequence==4)cudaGridDependencySynchronize();
'''+s[b:]
a=s.index('  }else if((tid>=32 && tid<128) || tid>=256){');b=s.index('  }else if(tid>=128 && tid<256){',a)
s=s[:a]+s[b:]
a=s.index('  void* args[]={&a,&b,&s,&t,&A,&B,&workspace,&D};dim3 grid(16,8);')
s=s[:a]+'''  cudaLaunchAttribute attribute{};
  attribute.id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed=1;
  cudaLaunchConfig_t config{};config.gridDim=dim3(16,8);config.blockDim=dim3(Threads);
  config.dynamicSmemBytes=sizeof(Shared<cute::half_t>);config.stream=stream;
  config.attrs=&attribute;config.numAttrs=1;
  if(state.dtype==DType::BF16){
    sw_transform_pdl<cute::bfloat16_t><<<128,512,0,stream>>>(static_cast<const cute::bfloat16_t*>(A),static_cast<const cute::bfloat16_t*>(B),static_cast<cute::bfloat16_t*>(workspace));
    check(cudaPeekAtLastError(),"SW PDL transform launch");
    check(cudaLaunchKernelEx(&config,sw_cooperative_hilo<cute::bfloat16_t>,a,b,s,t,static_cast<const cute::bfloat16_t*>(A),static_cast<const cute::bfloat16_t*>(B),static_cast<cute::bfloat16_t*>(workspace),D),"SW PDL compute launch");
  }else{
    sw_transform_pdl<cute::half_t><<<128,512,0,stream>>>(static_cast<const cute::half_t*>(A),static_cast<const cute::half_t*>(B),static_cast<cute::half_t*>(workspace));
    check(cudaPeekAtLastError(),"SW PDL transform launch");
    check(cudaLaunchKernelEx(&config,sw_cooperative_hilo<cute::half_t>,a,b,s,t,static_cast<const cute::half_t*>(A),static_cast<const cute::half_t*>(B),static_cast<cute::half_t*>(workspace),D),"SW PDL compute launch");
  }
}
} // namespace megafmm::detail
'''
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','62142fc:solution/plan.cpp'],text=True).replace('1tma+4wgmma+16pack_only','1tma+4wgmma;transform_kernel=128x512').replace('publication=tma_leader_driver_workspace_barrier_cuda12.4','publication=programmatic_dependent_launch_grid_dependency_wait').replace('sw_cooperative;','sw_two_kernel_pdl;')
Path('solution/plan.cpp').write_text(s)

from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','13340d1:solution/sw_pipeline.cu'],text=True)
s=s.replace('  cudaTriggerProgrammaticLaunchCompletion();\n','').replace('      if(sequence==4)cudaGridDependencySynchronize();\n','')
a=s.index('  cudaLaunchAttribute attribute{};');b=s.index('  cudaLaunchConfig_t config{};',a);s=s[:a]+s[b:]
s=s.replace('  config.attrs=&attribute;config.numAttrs=1;\n','')
s=s.replace('      // Raw P1/P2 can run while the primary transforms S/T. This wait\n      // guarantees completion and visibility even if no overlap is scheduled.\n','')
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','13340d1:solution/plan.cpp'],text=True).replace('sw_two_kernel_pdl','sw_two_kernel_stream_order').replace('publication=programmatic_dependent_launch_grid_dependency_wait','publication=ordinary_stream_completion')
Path('solution/plan.cpp').write_text(s)

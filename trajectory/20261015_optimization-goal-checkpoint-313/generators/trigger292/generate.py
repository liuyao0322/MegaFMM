from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','1bdfb29:solution/sw_pipeline.cu'],text=True)
s=s.replace('  cudaTriggerProgrammaticLaunchCompletion();','  if(threadIdx.x==0)cudaTriggerProgrammaticLaunchCompletion();')
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1bdfb29:solution/plan.cpp'],text=True).replace('publication=programmatic_dependent_launch_grid_dependency_wait','publication=programmatic_dependent_launch_grid_dependency_wait;trigger=primary_thread0')
Path('solution/plan.cpp').write_text(p)

from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','13340d1:solution/sw_pipeline.cu'],text=True).replace('  cudaTriggerProgrammaticLaunchCompletion();','  if(threadIdx.x==0)cudaTriggerProgrammaticLaunchCompletion();')
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','13340d1:solution/plan.cpp'],text=True).replace('transform_kernel=128x512','transform_kernel=128x512;trigger=cta_leader')
Path('solution/plan.cpp').write_text(s)

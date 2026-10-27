from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','1bdfb29:solution/sw_pipeline.cu'],text=True)
s=s.replace('__launch_bounds__(512,1)','__launch_bounds__(256,1)').replace('2*(blockIdx.x*512+threadIdx.x)','2*(blockIdx.x*256+threadIdx.x)').replace('i+=2*512*128','i+=2*256*128').replace('<<<128,512,0,stream>>>','<<<128,256,0,stream>>>')
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1bdfb29:solution/plan.cpp'],text=True).replace('transform_kernel=128x512','transform_kernel=128x256')
Path('solution/plan.cpp').write_text(p)

from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','1bdfb29:solution/sw_pipeline.cu'],text=True).replace('i+=2*512*128','i+=2*512*256').replace('<<<128,512,0,stream>>>','<<<256,512,0,stream>>>')
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1bdfb29:solution/plan.cpp'],text=True).replace('transform_kernel=128x512','transform_kernel=256x512')
Path('solution/plan.cpp').write_text(p)

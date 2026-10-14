from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','13340d1:solution/sw_pipeline.cu'],text=True)
s=s.replace('2*512*128','2*512*64').replace('<<<128,512,0,stream>>>','<<<64,512,0,stream>>>')
assert s.count('<<<64,512,0,stream>>>')==2
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','13340d1:solution/plan.cpp'],text=True).replace('transform_kernel=128x512','transform_kernel=64x512')
Path('solution/plan.cpp').write_text(s)

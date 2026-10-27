from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','13340d1:solution/sw_pipeline.cu'],text=True)
s=s.replace('Plane=512*512,Threads=256','Plane=512*512,Threads=160').replace('GMMA::CLayout_64x32{}(tid-128,0)','GMMA::CLayout_64x32{}(tid,0)')
s=s.replace('if(tid==0){','if(tid==128){').replace('}else if(tid>=128 && tid<256){','}else if(tid<128){')
assert 'tid-128' not in s
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','13340d1:solution/plan.cpp'],text=True).replace('warp_roles=1tma+4wgmma','warp_roles=1tma+4wgmma;cta_threads=160;tc_warps=0to3;tma_warp=4')
Path('solution/plan.cpp').write_text(s)

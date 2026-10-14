from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','18f9a6f:solution/sw_pipeline.cu'],text=True)
a=s.index('void sw_transform_pdl(');b=s.index('template<class T> __global__ __launch_bounds__(Threads,1)',a)
v=s[a:b]
v=v.replace('2*(blockIdx.x*512+threadIdx.x)','4*(blockIdx.x*512+threadIdx.x)').replace('i+=2*512*128','i+=4*512*128')
v=v.replace('uint32_t','uint64_t').replace('at/2','at/4').replace('+512)/2','+512)/4').replace('+512*1024)/2','+512*1024)/4').replace('+512*1024+512)/2','+512*1024+512)/4')
v=v.replace('lane<2','lane<4').replace('(j*Plane+i)/2','(j*Plane+i)/4')
assert '/2]' not in v
s=s[:a]+v+s[b:]
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','18f9a6f:solution/plan.cpp'],text=True).replace('st=compact4planes_balanced_async_p1p2','st=compact4planes_balanced_async_p1p2;pack_io=64bit_4elements')
Path('solution/plan.cpp').write_text(s)

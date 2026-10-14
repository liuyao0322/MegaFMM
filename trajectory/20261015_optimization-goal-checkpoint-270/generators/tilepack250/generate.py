from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','62142fc:solution/sw_pipeline.cu'],text=True)
s=s.replace('int i=(blockIdx.y*gridDim.x+blockIdx.x)*(Plane/(16*8))+local;','int i=(blockIdx.x*32+local/64)*512+blockIdx.y*64+local%64;')
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','62142fc:solution/plan.cpp'],text=True).replace('st=compact4planes_balanced_async_p1p2','st=compact4planes_tile32x64_async_p1p2')
Path('solution/plan.cpp').write_text(s)

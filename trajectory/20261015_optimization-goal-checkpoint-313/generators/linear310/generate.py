from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','aba9ffd:solution/sw_pipeline.cu'],text=True)
s=s.replace('int r=i/BK,k=i%BK,off=lb(r,k);','int off=i;').replace('int r=i%BM,k=i/BM,off=la(r,k);','int off=i;')
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','aba9ffd:solution/plan.cpp'],text=True).replace('transform=fp32_pair_final_rn','transform=fp32_pair_final_rn_linear_physical_offsets')
Path('solution/plan.cpp').write_text(p)

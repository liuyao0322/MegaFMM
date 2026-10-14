from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','18f9a6f:solution/sw_pipeline.cu'],text=True).replace('int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;','int tid=threadIdx.x,m0=(blockIdx.y^(blockIdx.x&7))*BM,n0=blockIdx.x*BN;')
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','18f9a6f:solution/plan.cpp'],text=True).replace('operand_release=warp_leader','operand_release=warp_leader;cta_mapping=mxor_nlow3')
Path('solution/plan.cpp').write_text(s)

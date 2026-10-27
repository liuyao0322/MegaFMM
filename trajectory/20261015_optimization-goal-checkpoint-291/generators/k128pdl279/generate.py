from pathlib import Path
import subprocess
base=subprocess.check_output(['git','show','18f9a6f:solution/sw_pipeline.cu'],text=True)
g=Path('results/opt_goal/static218/generate.py').read_text();a=g.index("fn='''");b=g.index('s=s[:a]+fn+s[b:]')
g=g[a:b].replace('for q in range(2)','for q in range(4)').replace('seq=step*2+q;slot=seq%4;phase=seq//4%2','seq=step*4+q;slot=seq%8;phase=seq//8%2').replace('slot*3072','slot*1536').replace('for k in range(16)','for k in range(8)').replace('%%54,%%55','%%38,%%39').replace('(38+j)','(22+j)')
ctx={};exec(g,ctx);a=base.index('template<class T> __device__ __forceinline__ void tensor_chain');b=base.index('template<class T> __global__ __launch_bounds__(512',a)
s=base[:a]+ctx['fn']+base[b:];s=s.replace('BK=256,Stages=4','BK=128,Stages=8').replace('==196736','==196736').replace('if(sequence==4)','if(sequence==8)')
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','18f9a6f:solution/plan.cpp'],text=True).replace('operand_stages=4;bk=256','operand_stages=8;bk=128')
Path('solution/plan.cpp').write_text(p)
for f in ['internal.h','scale_output.cu']:Path('solution/'+f).write_bytes(subprocess.check_output(['git','show','18f9a6f:solution/'+f]))

from pathlib import Path
import subprocess
base=subprocess.check_output(['git','show','1bdfb29:solution/sw_pipeline.cu'],text=True)
g=Path('results/opt_goal/static218/generate.py').read_text();a=g.index("fn='''");b=g.index('s=s[:a]+fn+s[b:]');g=g[a:b]
g=g.replace('for q in range(2)','for q in range(3)').replace('seq=step*2+q;slot=seq%4;phase=seq//4%2','seq=step*3+q;slot=seq%5;phase=seq//5%2').replace('slot*3072','slot*2304')
# Operand constraints provide twelve K16 offsets; tail issues only eight.
g=g.replace('for k in range(16):\n    pred=', 'for k in range(8 if q==2 else 12):\n    pred=')
g=g.replace('for k in range(16):ins+=','for k in range(12):ins+=')
g=g.replace('%%54,%%55','%%46,%%47').replace('(38+j)','(30+j)')
ctx={};exec(g,ctx);a=base.index('template<class T> __device__ __forceinline__ void tensor_chain');b=base.index('template<class T> __global__ __launch_bounds__(512',a)
s=base[:a]+ctx['fn']+base[b:]
s=s.replace('BK=128,Stages=8','BK=192,Stages=5').replace('==196736','==184448').replace('q<512/BK','q<(512+BK-1)/BK').replace('sequence=step*(512/BK)+q','sequence=step*((512+BK-1)/BK)+q').replace('if(sequence==8)','if(sequence==6)')
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1bdfb29:solution/plan.cpp'],text=True).replace('smem=196736;operand_stages=8;bk=128','smem=184448;operand_stages=5;bk=192;tail_k=128;tail_mma_tiles=8')
Path('solution/plan.cpp').write_text(p)

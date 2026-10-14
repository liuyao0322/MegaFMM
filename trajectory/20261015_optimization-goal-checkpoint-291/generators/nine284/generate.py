from pathlib import Path
import subprocess
base=subprocess.check_output(['git','show','1722267:solution/sw_pipeline.cu'],text=True)
g=Path('results/opt_goal/k128pdl279/generate.py').read_text()
a=g.index("g=Path(");b=g.index("p=subprocess.check_output",a)
g=g[a:b].replace("s=s.replace('BK=256,Stages=4','BK=128,Stages=8')","s=s.replace('BK=256,Stages=4','BK=128,Stages=9')")
# Generator is derived from266, so use original base; adjust static scheduling.
base=subprocess.check_output(['git','show','18f9a6f:solution/sw_pipeline.cu'],text=True)
g=g.replace("'seq=step*4+q;slot=seq%8;phase=seq//8%2'","'seq=step*4+q;slot=seq%9;phase=seq//9%2'")
g=g.replace(".replace('==196736','==196736')",".replace('==196736','==221440')")
exec(g)
p=subprocess.check_output(['git','show','1722267:solution/plan.cpp'],text=True).replace('smem=196736;operand_stages=8','smem=221440;operand_stages=9')
Path('solution/plan.cpp').write_text(p)
assert 'Stages=9' in Path('solution/sw_pipeline.cu').read_text()

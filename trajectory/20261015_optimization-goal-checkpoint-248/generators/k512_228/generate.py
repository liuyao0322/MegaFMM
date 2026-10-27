from pathlib import Path
import subprocess
s=Path('results/opt_goal/static218/generate.py').read_text()
s=s.replace("s=subprocess.check_output(['git','show','1f1fb61:solution/sw_pipeline.cu'],text=True)","s=subprocess.check_output(['git','show','1f1fb61:solution/sw_pipeline.cu'],text=True)\ns=s.replace('BK=256','BK=512').replace('Stages=4','Stages=2').replace('sequence==4','sequence==2')")
s=s.replace('for q in range(2):','for q in range(1):').replace('seq=step*2+q;slot=seq%4;phase=seq//4%2','seq=step+q;slot=seq%2;phase=seq//2%2').replace('slot*3072','slot*6144').replace('for k in range(16):','for k in range(32):').replace('%%54,%%55','%%86,%%87').replace('(38+j)','(70+j)')
s=s.replace("Path('solution/sw_pipeline.cu').write_text(s)",'''# Split the MN-contiguous B tile along its outer K dimension, box max256.
s=s.replace('uint32_t(BM),uint32_t(is_a ? BM:BK)','uint32_t(BM),uint32_t(256)')
s=s.replace('SM90_TMA_LOAD_3D::copy(&tmap,&s.ready[slot],0,op,m0,k0,plane);', 'SM90_TMA_LOAD_3D::copy(&tmap,&s.ready[slot],0,op,m0,k0,plane);\\n        SM90_TMA_LOAD_3D::copy(&tmap,&s.ready[slot],0,op+BM*256,m0,k0+256,plane);')
s=s.replace('SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],0,op,m0+col,k0+row);', 'SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],0,op,m0+col,k0+row);\\n        SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],0,op+BM*256,m0+col,k0+row+256);')
Path('solution/sw_pipeline.cu').write_text(s)''')
s=s.replace("Path('solution/plan.cpp').write_text(s)","s=s.replace('operand_stages=4','operand_stages=2').replace('bk=256','bk=512;lhs_tma=split256')\nPath('solution/plan.cpp').write_text(s)")
exec(compile(s,'k512-generator','exec'))

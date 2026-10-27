from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','1f1fb61:solution/sw_pipeline.cu'],text=True)
s=s.replace('BK=256,Stages=4','BK=192,Stages=5').replace('==213120','==200832')
s=s.replace('template<class T,class TA,class TB>','template<class T,int ActiveK,class TA,class TB>').replace('kk<BK/16','kk<ActiveK/16')
s=s.replace('512/BK','((512+BK-1)/BK)').replace('if(sequence==4)','if(sequence==6)')
s=s.replace('mma_term<T>(ah,bh,acc);','if(q==2)mma_term<T,128>(ah,bh,acc);else mma_term<T,BK>(ah,bh,acc);')
Path('results/opt_goal/k192_216/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','1f1fb61:solution/plan.cpp'],text=True).replace('smem=213120','smem=200832').replace('operand_stages=4;bk=256','operand_stages=5;bk=192;tail_k=128')
Path('results/opt_goal/k192_216/plan.cpp').write_text(s)

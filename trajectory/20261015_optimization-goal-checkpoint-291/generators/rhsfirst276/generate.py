from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','18f9a6f:solution/sw_pipeline.cu'],text=True)
a=s.index('      // LHS is B_leaf^T');b=s.index('      // RHS is A_leaf^T',a);c=s.index('\n\n    }',b)
lhs=s[a:b];rhs=s[b:c];s=s[:a]+rhs+'\n'+lhs.rstrip()+s[c:]
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','18f9a6f:solution/plan.cpp'],text=True).replace('operand_release=warp_leader','operand_release=warp_leader;tma_order=rhs_then_lhs')
Path('solution/plan.cpp').write_text(s)

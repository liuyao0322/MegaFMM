from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','1bdfb29:solution/sw_pipeline.cu'],text=True)
s=s.replace('Barrier::init(&s.free[j],4)','Barrier::init(&s.free[j],1)').replace('"r"(tid&31)','"r"(tid-128)')
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1bdfb29:solution/plan.cpp'],text=True).replace('operand_release=warp_leader','operand_release=one_warpgroup_leader_after_wait0')
Path('solution/plan.cpp').write_text(p)

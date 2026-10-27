from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','1bdfb29:solution/sw_pipeline.cu'],text=True)
needle='  "wgmma.wait_group.sync.aligned 0;\\n"\n  "bar.warp.sync 0xffffffff;\\n"'
assert s.count(needle)==56
s=s.replace(needle,'  "wgmma.wait_group.sync.aligned 0;\\n"')
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1bdfb29:solution/plan.cpp'],text=True).replace('operand_release=warp_leader','operand_release=four_leaders_after_collective_wait0;postwait_warp_sync=off')
Path('solution/plan.cpp').write_text(p)

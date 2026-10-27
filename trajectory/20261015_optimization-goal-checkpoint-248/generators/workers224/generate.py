from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','c0df342:solution/sw_pipeline.cu'],text=True)
s=s.replace('Threads=768','Threads=672').replace('pack_ready,608','pack_ready,512').replace('local+=2*608','local+=2*512')
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','c0df342:solution/plan.cpp'],text=True).replace('4wgmma+19pack_only','4wgmma+16pack_only')
Path('solution/plan.cpp').write_text(s)

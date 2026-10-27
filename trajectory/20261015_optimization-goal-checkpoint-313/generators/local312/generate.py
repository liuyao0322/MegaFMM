from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','38adfbd:solution/sw_pipeline.cu'],text=True)
assert s.count('wait(&s.raw[slot],q%2);')==1
s=s.replace('wait(&s.raw[slot],q%2);','wait(&s.raw[slot],phase);')
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','38adfbd:solution/plan.cpp'],text=True).replace('publication=local_double_buffer_barriers','publication=local_double_buffer_barriers_correct_slot_phase')
Path('solution/plan.cpp').write_text(p)

from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','62142fc:solution/sw_pipeline.cu'],text=True)
s=s.replace('do{asm volatile("ld.acquire.gpu.u32','do{__nanosleep(64);asm volatile("ld.acquire.gpu.u32')
Path('solution/sw_pipeline.cu').write_text(s)
for name in ['internal.h','plan.cpp']:
 s=subprocess.check_output(['git','show','62142fc:solution/'+name],text=True)
 if name=='plan.cpp':s=s.replace('publication=tma_leader','publication_poll_backoff_ns=64;publication=tma_leader')
 Path('solution/'+name).write_text(s)

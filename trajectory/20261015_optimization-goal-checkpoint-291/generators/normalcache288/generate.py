from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','1bdfb29:solution/sw_pipeline.cu'],text=True)
import re
s,n=re.subn(r'(&s.ready\[slot\],)0,(op)',r'\1uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),\2',s);assert n==4,n
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1bdfb29:solution/plan.cpp'],text=True).replace('current_pointer_maps=4','current_pointer_maps=4;tma_cache_hint=evict_normal')
Path('solution/plan.cpp').write_text(p)

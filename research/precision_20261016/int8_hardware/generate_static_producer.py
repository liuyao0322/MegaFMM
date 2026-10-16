"""P57: add only a seven-way unroll request to frozen P56 TMA producer."""
from pathlib import Path
import argparse
h=Path(__file__).resolve().parent;ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=h/'p57');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:
 s=(h/'p56'/n).read_text()
 if n=='corrected.cuh':
  old=' if(tid==0){\n  for(int step=0;step<7;++step)for(int k=0;k<4;++k){';assert s.count(old)==1
  s=s.replace(old,' if(tid==0){\n  #pragma unroll 7\n  for(int step=0;step<7;++step)for(int k=0;k<4;++k){')
 (a.output/n).write_text(s)

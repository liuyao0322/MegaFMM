"""Isolate P56 from frozen P51: exactly one consumer unroll directive change."""
from pathlib import Path
import argparse
h=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=h/'p56');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:
 s=(h/'p51'/n).read_text()
 if n=='corrected.cuh':
  old=' #pragma unroll 1\n for(int step=0;step<7;++step)';assert s.count(old)==1
  s=s.replace(old,' #pragma unroll 7\n for(int step=0;step<7;++step)')
 (args.output/n).write_text(s)

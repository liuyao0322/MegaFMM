"""P50 spatial subdivision, from frozen P49; no numerical representation change."""
from pathlib import Path
r=Path(__file__).resolve().parent
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:
 s=(r/'p49'/name).read_text()
 if name=='pack.cuh':
  assert s.count('Rows=SideA?64:128,Cols=SideA?128:32')==1
  assert s.count('8*(tid%16)')==1
  s=s.replace('Rows=SideA?64:128,Cols=SideA?128:32','Rows=SideA?32:64,Cols=SideA?128:32')
  s=s.replace('8*(tid%16)','8*(tid%8)')
  s=s.replace('blockIdx.x/(SideA?32:64)','blockIdx.x/(SideA?64:128)')
  s=s.replace('int group=blockIdx.x/64;','int group=blockIdx.x/128;')
  s=s.replace('int group=(blockIdx.x-192)/32;','int group=(blockIdx.x-192)/64;')
 (r/name).write_text(s)

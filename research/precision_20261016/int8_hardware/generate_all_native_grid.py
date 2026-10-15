"""P51 smaller producer rows and complete cooperative grid, from frozen P50."""
from pathlib import Path
r=Path(__file__).resolve().parent
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:
 s=(r/'p50'/name).read_text()
 if name=='pack.cuh':
  assert s.count('Rows=SideA?32:64,Cols=SideA?128:32')==1
  assert s.count('8*(tid%8)')==1
  assert s.count('if(blockIdx.x<192)')==2
  s=s.replace('Rows=SideA?32:64,Cols=SideA?128:32','Rows=SideA?16:32,Cols=SideA?128:32')
  s=s.replace('8*(tid%8)','8*(tid%4)')
  s=s.replace('blockIdx.x/(SideA?64:128)','blockIdx.x/(SideA?128:256)')
  s=s.replace('if(blockIdx.x<192)','if(blockIdx.x<256)')
  s=s.replace('int group=blockIdx.x/128;','int group=blockIdx.x/256;')
  s=s.replace('int group=(blockIdx.x-192)/64;','int group=(blockIdx.x-256)/128;')
  s=s.replace('B occupies0..191, A192..287.192 is divisible by both tile counts,','B occupies0..255, A256..383.256 is divisible by both tile counts,')
  s=s.replace('All288CTAs','All384CTAs')
  s=s.replace('grid=288','grid=384').replace('active*sms<288','active*sms<384')
  s=s.replace('for288CTAs','for384CTAs').replace('dim3(288)','dim3(384)').replace('<<<288,256','<<<384,256')
 elif name=='compare.cu':
  assert s.count('<<<3*32,256')==2 and s.count('<<<3*64,256')==2
  s=s.replace('<<<3*32,256','<<<128,256').replace('<<<3*64,256','<<<256,256')
 (r/name).write_text(s)

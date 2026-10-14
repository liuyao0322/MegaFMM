from pathlib import Path
import re
p=Path('solution/sw_pipeline.cu');s=p.read_text()
s=s.replace('T a[4][ASize]','T a[5][ASize]').replace('T b[5][BSize]','T b[3][BSize]').replace('ready[4],free[4],bready[5],bfree[5]','ready[5],free[5],bready[3],bfree[3]')
s=s.replace('for(int j=0;j<Stages;++j){Barrier::init(&s.ready','for(int j=0;j<5;++j){Barrier::init(&s.ready').replace('for(int j=0;j<5;++j){Barrier::init(&s.bready','for(int j=0;j<3;++j){Barrier::init(&s.bready').replace('slots=is_a?4:5','slots=is_a?5:3')
for step in range(7):
 for q in range(2):
  seq=2*step+q
  pat=r'   "add.u32 rb,%%2,\d+;\\n"\n   "WAIT_%d_%d:.*?add.u32 fb,%%49,\d+;\\n"\n   "@leader mbarrier.arrive.shared::cta.b64 _,\[fb\];\\n"'%(step,q)
  def change(m):
   z=m.group(0)
   z=z.replace('add.u32 rb,%%2,%d;'%(8*(seq%4)),'add.u32 rb,%%2,%d;'%(8*(seq%5)))
   # phase associated with A and B labels is substituted independently
   z=z.replace('"WAIT_%d_%d:\\n"\n   "mbarrier.try_wait.parity.shared::cta.b64 done,[rb],%d,'%(step,q,(seq//4)%2),'"WAIT_%d_%d:\\n"\n   "mbarrier.try_wait.parity.shared::cta.b64 done,[rb],%d,'%(step,q,(seq//5)%2))
   z=z.replace('add.u32 rb,%%48,%d;'%(8*(seq%5)),'add.u32 rb,%%48,%d;'%(8*(seq%3)))
   z=z.replace('"WAITB_%d_%d:\\n"\n   "mbarrier.try_wait.parity.shared::cta.b64 done,[rb],%d,'%(step,q,(seq//5)%2),'"WAITB_%d_%d:\\n"\n   "mbarrier.try_wait.parity.shared::cta.b64 done,[rb],%d,'%(step,q,(seq//3)%2))
   z=z.replace('add.u64 ba,%%0,%d;'%(2048*(seq%4)),'add.u64 ba,%%0,%d;'%(2048*(seq%5))).replace('add.u64 bb,%%1,%d;'%(1024*(seq%5)),'add.u64 bb,%%1,%d;'%(1024*(seq%3)))
   z=z.replace('add.u32 fb,%%3,%d;'%(8*(seq%4)),'add.u32 fb,%%3,%d;'%(8*(seq%5))).replace('add.u32 fb,%%49,%d;'%(8*(seq%5)),'add.u32 fb,%%49,%d;'%(8*(seq%3)))
   return z
  s,n=re.subn(pat,change,s,flags=re.S);assert n==2,(seq,n)
p.write_text(s)
p=Path('solution/plan.cpp');s=p.read_text().replace('A4_B5','A5_B3');p.write_text(s)

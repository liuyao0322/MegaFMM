from pathlib import Path
import subprocess,re
s=subprocess.check_output(['git','show','1722267:solution/sw_pipeline.cu'],text=True)
pattern=r'  "wgmma.commit_group.sync.aligned;\\n"\n  "wgmma.wait_group.sync.aligned 0;\\n"\n  "bar.warp.sync 0xffffffff;\\n"\n  "add.u32 fb,%3,(\d+);\\n"\n  "@leader mbarrier.arrive.shared::cta.b64 _,\[fb\];\\n"'
count=0
def sub(m):
 global count
 q=count%4;count+=1;slot=int(m[1]);lines=['wgmma.commit_group.sync.aligned;']
 if q:
  lines+=['wgmma.wait_group.sync.aligned 1;','bar.warp.sync 0xffffffff;',f'add.u32 fb,%3,{slot-8};','@leader mbarrier.arrive.shared::cta.b64 _,[fb];']
 if q==3:
  lines+=['wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;',f'add.u32 fb,%3,{slot};','@leader mbarrier.arrive.shared::cta.b64 _,[fb];']
 return '\n'.join('  "'+v+'\\n"' for v in lines)
s=re.sub(pattern,sub,s);assert count==56,count
s=re.sub(r'(  "@!done bra WAIT_\d+_[123];\\n"\n)  "wgmma.fence.sync.aligned;\\n"\n',r'\1',s)
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1722267:solution/plan.cpp'],text=True).replace('bk=128','bk=128;issue=one_group_lookahead;retire=wait1_then_final_wait0')
Path('solution/plan.cpp').write_text(p)

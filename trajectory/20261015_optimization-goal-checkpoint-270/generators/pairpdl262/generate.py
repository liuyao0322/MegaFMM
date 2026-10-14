from pathlib import Path
import subprocess,re
s=subprocess.check_output(['git','show','13340d1:solution/sw_pipeline.cu'],text=True)
pattern=r'  "wgmma.commit_group.sync.aligned;\\n"\n  "wgmma.wait_group.sync.aligned 0;\\n"\n  "bar.warp.sync 0xffffffff;\\n"\n  "add.u32 fb,%3,(\d+);\\n"\n  "@leader mbarrier.arrive.shared::cta.b64 _,\[fb\];\\n"'
count=0
def sub(m):
 global count
 q=count%2;count+=1
 if q==0:return '  "wgmma.commit_group.sync.aligned;\\n"'
 slot=int(m[1])
 return '\n'.join('  "'+x+'\\n"' for x in ['wgmma.commit_group.sync.aligned;','wgmma.wait_group.sync.aligned 1;','bar.warp.sync 0xffffffff;',f'add.u32 fb,%3,{slot-8};','@leader mbarrier.arrive.shared::cta.b64 _,[fb];','wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;',f'add.u32 fb,%3,{slot};','@leader mbarrier.arrive.shared::cta.b64 _,[fb];'])
s=re.sub(pattern,sub,s);assert count==28,count
s=re.sub(r'(  "@!done bra WAIT_\d+_1;\\n"\n)  "wgmma.fence.sync.aligned;\\n"\n',r'\1',s)
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','13340d1:solution/plan.cpp'],text=True).replace('bk=256','bk=256;issue=two_K_groups;retire=wait1_then_wait0')
Path('solution/plan.cpp').write_text(s)

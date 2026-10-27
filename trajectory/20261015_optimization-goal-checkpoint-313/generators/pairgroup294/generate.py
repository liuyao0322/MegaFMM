from pathlib import Path
import subprocess,re
s=subprocess.check_output(['git','show','1bdfb29:solution/sw_pipeline.cu'],text=True)
pattern=r'  "wgmma.commit_group.sync.aligned;\\n"\n  "wgmma.wait_group.sync.aligned 0;\\n"\n  "bar.warp.sync 0xffffffff;\\n"\n  "add.u32 fb,%3,(\d+);\\n"\n  "@leader mbarrier.arrive.shared::cta.b64 _,\[fb\];\\n"'
count=0
def sub(m):
 global count
 q=count%2;count+=1
 if q!=1:return ''
 slot=int(m[1]);lines=['wgmma.commit_group.sync.aligned;','wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;']
 for j in range(2):lines += [f'add.u32 fb,%3,{slot-8+j*8};','@leader mbarrier.arrive.shared::cta.b64 _,[fb];']
 return '\n'.join('  "'+v+'\\n"' for v in lines)
s=re.sub(pattern,sub,s);assert count==56,count
s=re.sub(r'(  "@!done bra WAIT_\d+_[13];\\n"\n)  "wgmma.fence.sync.aligned;\\n"\n',r'\1',s)
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1bdfb29:solution/plan.cpp'],text=True).replace('operand_release=warp_leader','operand_release=two_slots_after_pair_wait0;mma_group=paired_2K128')
Path('solution/plan.cpp').write_text(p)

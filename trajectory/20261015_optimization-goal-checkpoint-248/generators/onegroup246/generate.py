from pathlib import Path
s=Path('results/opt_goal/static218/generate.py').read_text()
old="   lines+=['wgmma.commit_group.sync.aligned;','wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;','add.u32 fb,%%3,%d;'%(slot*8),'@leader mbarrier.arrive.shared::cta.b64 _,[fb];']"
new="""   if q==1:
    lines+=['wgmma.commit_group.sync.aligned;','wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;','add.u32 fb,%%3,%d;'%(((seq-1)%4)*8),'@leader mbarrier.arrive.shared::cta.b64 _,[fb];','add.u32 fb,%%3,%d;'%(slot*8),'@leader mbarrier.arrive.shared::cta.b64 _,[fb];']"""
assert old in s;s=s.replace(old,new).replace("'wgmma.fence.sync.aligned;'","('wgmma.fence.sync.aligned;' if q==0 else '')")
exec(compile(s,'onegroup246-generator','exec'))
p=Path('solution/sw_pipeline.cu');s=p.read_text().replace('Threads=768','Threads=672').replace('pack_ready,608','pack_ready,512').replace('local+=2*608','local+=2*512');p.write_text(s)
p=Path('solution/plan.cpp');s=p.read_text().replace('4wgmma+19pack_only','4wgmma+16pack_only').replace('bk=256','bk=256;issue=two_K_one_group;retire=pair_wait0');p.write_text(s)

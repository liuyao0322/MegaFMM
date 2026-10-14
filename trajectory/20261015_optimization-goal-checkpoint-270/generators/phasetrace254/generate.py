from pathlib import Path
p=Path('results/opt_goal/phasetrace254')
s=Path('results/opt_goal/batchtrace254/trace.cuh').read_text()
s=s.replace('int n0,float* D){','int n0,float* D,uint64_t* phases){')
s=s.replace('.reg .b64 ba,bb,da,db,dst;','.reg .b64 ba,bb,da,db,dst,ts;')
s=s.replace('"n"(int32_t(GMMA::Major::K)) : "memory"','"n"(int32_t(GMMA::Major::K)), "l"(phases) : "memory"')
def stamp(slot):return '  "@leader mov.u64 ts,%%globaltimer;\\n"\n  "@leader st.global.u64 [%56+'+str(slot*8)+'],ts;\\n"\n'
for block,(step,q) in enumerate([(0,0),(2,0),(6,1)]):
 label=f'  "WAIT_{step}_{q}:\\n"\n';s=s.replace(label,stamp(block*4)+label)
 branch=f'  "@!done bra WAIT_{step}_{q};\\n"\n';s=s.replace(branch,branch+stamp(block*4+1))
 # Each dtype has one occurrence; insert after the group's commit and wait.
 pos=0
 for dtype in range(2):
  a=s.index(branch,pos);b=s.index('  "wgmma.commit_group.sync.aligned;\\n"\n',a)
  after=b+len('  "wgmma.commit_group.sync.aligned;\\n"\n');s=s[:after]+stamp(block*4+2)+s[after:]
  b=s.index('  "wgmma.wait_group.sync.aligned 0;\\n"\n',after);after=b+len('  "wgmma.wait_group.sync.aligned 0;\\n"\n');s=s[:after]+stamp(block*4+3)+s[after:];pos=after+len(stamp(block*4+3))
s=s.replace('tensor_chain(s,tid,m0,n0,D);','tensor_chain(s,tid,m0,n0,D,trace+1024+(blockIdx.y*16+blockIdx.x)*64+((tid-128)/32)*16);')
assert s.count('"l"(phases)')==2
assert s.count('%%globaltimer')==25
(p/'trace.cuh').write_text(s)
(p/'base.cuh').write_text(Path('results/opt_goal/batchtrace254/base.cuh').read_text())
s=Path('results/opt_goal/batchtrace254/driver.cu').read_text()
s=s.replace('Batch=10;','Batch=10,TraceStride=1024+128*64;').replace('Batch*128*8*8','Batch*TraceStride*8').replace('traces(Batch*128*8)','traces(Batch*TraceStride)').replace('tr+j*128*8','tr+j*TraceStride').replace('traces[(invocation*128+c)*8+j]','traces[invocation*TraceStride+c*8+j]')
s=s.replace('puts("");}', 'for(int j=0;j<12;++j)printf(",%llu",(unsigned long long)traces[invocation*TraceStride+1024+c*64+j]);puts("");}')
s=s.replace('tc_start,tc_end");','tc_start,tc_end,raw_start,raw_ready,raw_issue,raw_done,st_start,st_ready,st_issue,st_done,last_start,last_ready,last_issue,last_done");')
(p/'driver.cu').write_text(s)
s=Path('results/opt_goal/batchtrace254/run.sh').read_text().replace('batchtrace254/','phasetrace254/')
(p/'run.sh').write_text(s)

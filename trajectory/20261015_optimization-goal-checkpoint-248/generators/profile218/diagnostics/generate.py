from pathlib import Path
import subprocess,re
out=Path('results/opt_goal/profile218/diagnostics/source')
s=subprocess.check_output(['git','show','c0df342:solution/sw_pipeline.cu'],text=True).split('struct SWState')[0]+'}\n'
(out/'baseline.cuh').write_text(s.replace('namespace megafmm::detail','namespace megafmm::diagnostic_base'))
t=s.replace('namespace megafmm::detail','namespace megafmm::diagnostic_trace')
t=t.replace('float* D){','float* D,Trace* traces,int sample){')
t=t.replace('tensor_chain(s,tid,m0,n0,D);','tensor_chain(s,tid,m0,n0,D,traces,sample);')
t=t.replace('.reg .b32 rb,fb; .reg .f32','.reg .b32 rb,fb; .reg .b64 tt0,tt1,tt2,tt3,tp; .reg .pred cap,writer; .reg .f32')
pattern=r'  "WAIT_(\d)_(\d):\\n".*?  "wgmma.wait_group.sync.aligned 0;\\n"'
def instrument(m):
 step,q=map(int,m.group(1,2));seq=step*2+q;v=m.group()
 prefix='  "setp.eq.u32 cap,%57,'+str(seq)+';\\n"\n  "@cap mov.u64 tt0,%%globaltimer;\\n"\n'
 needle='  "@!done bra WAIT_%d_%d;\\n"'%(step,q)
 assert needle in v
 v=v.replace(needle,needle+'\n  "@cap mov.u64 tt1,%%globaltimer;\\n"')
 needle='  "wgmma.wait_group.sync.aligned 0;\\n"'
 v=v.replace(needle,'  "@cap mov.u64 tt2,%%globaltimer;\\n"\n'+needle+'\n  "@cap mov.u64 tt3,%%globaltimer;\\n"')
 return prefix+v
t,n=re.subn(pattern,instrument,t,flags=re.S);assert n==28,n
needle='  "}\\n"\n  : :';assert t.count(needle)==2
code='''  "setp.eq.u32 writer,%58,0;\\n"
  "setp.lt.u32 cap,%57,14;\\n"
  "and.pred writer,writer,cap;\\n"
  "@writer st.global.v2.u64 [%56],{tt0,tt1};\\n"
  "add.u64 tp,%56,16;\\n"
  "@writer st.global.v2.u64 [tp],{tt2,tt3};\\n"
'''
t=t.replace(needle,code+needle)
t=t.replace('"n"(int32_t(GMMA::Major::K)) :','"n"(int32_t(GMMA::Major::K)), "l"(traces+blockIdx.y*gridDim.x+blockIdx.x), "r"(sample), "r"(tid-128) :')
t=t.replace('  if(tid==0){\n    for(int j=0;', '  uint64_t entry=(tid==0 && sample==14)?stamp():0;\n  if(tid==0){\n    for(int j=0;')
t=t.replace('        wait(&s.pack_ready,0);','        uint64_t a=sample==14?stamp():0;\n        wait(&s.pack_ready,0);\n        uint64_t b=sample==14?stamp():0;')
needle='}while(!((old^current)&0x80000000u));'
t=t.replace(needle,needle+'\n        if(sample==14)traces[blockIdx.y*gridDim.x+blockIdx.x]={entry,a,b,stamp()};')
(out/'instrumented.cuh').write_text(t)
s=Path('results/opt_goal/profile162/diagnostics/measure_template.cu').read_text()
a=s.index(' const void* funcs[]=');b=s.index('\n size_t shared=',a)
s=s[:a]+' const void* funcs[]={reinterpret_cast<void*>(diag::sw_cooperative_hilo<T>)};'+s[b:]
s=s.replace('role<6','role<1').replace('int samples=role<2?14:role<4?7:1;','int samples=15;')
(out/'measure.cu').write_text(s)

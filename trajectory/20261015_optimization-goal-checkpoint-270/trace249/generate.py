from pathlib import Path
import subprocess
p=Path('results/opt_goal/trace249')
s=subprocess.check_output(['git','show','492ba86:solution/sw_pipeline.cu'],text=True).replace('namespace megafmm::detail','namespace megafmm::base249')
(p/'base.cuh').write_text(s)
s=s.replace('namespace megafmm::base249','namespace megafmm::trace249')
s=s.replace('const T* A,const T* B,T* packed,float* D){','const T* A,const T* B,T* packed,float* D,uint64_t* trace){')
pos=s.index('template<class T> __global__');s=s[:pos]+'''__device__ __forceinline__ void stamp249(uint64_t* trace,int phase){uint64_t t;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t));trace[((blockIdx.y*16+blockIdx.x)*12)+phase]=t;}
'''+s[pos:]
s=s.replace('  if(tid==0){\n    for(int step','  if(tid==0){\n    stamp249(trace,0);\n    for(int step')
s=s.replace('        wait(&s.init_ready,0);','        stamp249(trace,1);\n        wait(&s.init_ready,0);\n        stamp249(trace,2);')
s=s.replace('        wait(&s.pack_ready,0);','        wait(&s.pack_ready,0);\n        stamp249(trace,3);')
s=s.replace('"l"(tf) : "memory");','"l"(tf) : "memory");\n        stamp249(trace,4);')
s=s.replace('}while(v!=8);','}while(v!=8);\n        stamp249(trace,5+q);')
s=s.replace('  }else if(tid==32){','  }else if(tid==32){\n    stamp249(trace,7);')
s=s.replace('    arrive(&s.init_ready);','    stamp249(trace,8);\n    arrive(&s.init_ready);')
s=s.replace('    tensor_chain(s,tid,m0,n0,D);','    if(tid==128)stamp249(trace,9);\n    tensor_chain(s,tid,m0,n0,D);\n    if(tid==128)stamp249(trace,10);')
s=s.replace('void* workspace,cudaStream_t stream){','void* workspace,cudaStream_t stream,uint64_t* trace){').replace('&workspace,&D};','&workspace,&D,&trace};')
(p/'trace.cuh').write_text(s)
s=Path('results/opt_goal/trace235/driver.cu').read_text().replace('base235','base249').replace('trace235','trace249').replace('12583168','4194560').replace('128*6','128*12').replace('traces[c*6','traces[c*12')
a=s.index('for(int c=0;c<128;++c)printf(');b=s.index('\n }}',a)
s=s[:a]+'''for(int c=0;c<128;++c){printf("%s,%d,%d,%d",name,trial,rep,c);for(int j=0;j<11;++j)printf(",%llu",(unsigned long long)traces[c*12+j]);puts("");}'''+s[b:]
s=s.replace('start,compute,fence,counter,finish,last','tma_start,raw_end,init_acquired,pack_acquired,published,q0_acquired,q1_acquired,coord_start,coord_end,tc_start,tc_end')
(p/'driver.cu').write_text(s)

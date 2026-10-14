p='results/opt_goal/profile187/diagnostics/';s=open('solution/sw_pipeline.cu').read()
s=s.replace(' int cta=p*32+tile,m0=blockIdx.y*BM,n0=blockIdx.x*BN;',''' int cta=p*32+tile,m0=blockIdx.y*BM,n0=blockIdx.x*BN;
 auto* trace=reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(packed)+(size_t(11)<<20)+4096)+cta*10;
 if(tid==0){for(int j=0;j<10;++j)trace[j]=0;stamp(trace,0);}''')
s=s.replace('template<class T> __global__','__device__ __forceinline__ void stamp(uint64_t* t,int i){uint64_t v;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(v));t[i]=v;}\ntemplate<class T> __global__',1)
s=s.replace('  arrive(&sh.pack_ready);','  if(worker==0)stamp(trace,1);\n  arrive(&sh.pack_ready);')
s=s.replace('    wait(&sh.pack_ready,0);','    stamp(trace,2);\n    wait(&sh.pack_ready,0);\n    stamp(trace,3);')
s=s.replace('    arrive(&sh.global_ready);','    stamp(trace,4);\n    arrive(&sh.global_ready);')
s=s.replace('  wait(&sh.global_ready,0);','  if(tid==128)stamp(trace,5);\n  wait(&sh.global_ready,0);\n  if(tid==128)stamp(trace,6);')
s=s.replace('  if(tid==128){\n   uint32_t old;','  if(tid==128){\n   stamp(trace,7);\n   uint32_t old;')
s=s.replace('  if(sh.last){','  if(tid==128)stamp(trace,8);\n  if(sh.last){')
s=s.replace('   }\n  }\n }\n}\nvoid check','   }\n   if(tid==128)stamp(trace,9);\n  }\n }\n}\nvoid check')
open(p+'instrumented.cuh','w').write(s)

from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','c0df342:solution/sw_pipeline.cu'],text=True)
a=s.index('  for(int local=2*worker;');b=s.index('    asm volatile("fence.proxy.async.global;',a)
old=s[a:b];body=old[old.index('    uint32_t words[16]={};'):old.rindex('\n  }')]
pre='''  uint32_t ap[2][4]={},bp[2][4]={};
  #pragma unroll
  for(int batch=0;batch<2;++batch){
    int local=2*worker+batch*1216;
    if(local<2048){
      int i=(blockIdx.y*gridDim.x+blockIdx.x)*2048+local;
      int at=(i/512)*1024+i%512;
      auto av=reinterpret_cast<const uint32_t*>(A);
      auto bv=reinterpret_cast<const uint32_t*>(B);
      ap[batch][0]=av[at/2];ap[batch][1]=av[(at+512)/2];ap[batch][2]=av[(at+512*1024)/2];ap[batch][3]=av[(at+512*1024+512)/2];
      bp[batch][0]=bv[at/2];bp[batch][1]=bv[(at+512)/2];bp[batch][2]=bv[(at+512*1024)/2];bp[batch][3]=bv[(at+512*1024+512)/2];
    }
  }
'''
pre+='  asm volatile("" : : '+','.join('"r"(%s[%d][%d])'%(v,i,j) for v in ['ap','bp'] for i in range(2) for j in range(4))+' : "memory");\n'
pre+='''  #pragma unroll
  for(int batch=0;batch<2;++batch){
    int local=2*worker+batch*1216;
    if(local<2048){
      int i=(blockIdx.y*gridDim.x+blockIdx.x)*2048+local;
      uint32_t abits[4]={ap[batch][0],ap[batch][1],ap[batch][2],ap[batch][3]};
      uint32_t bbits[4]={bp[batch][0],bp[batch][1],bp[batch][2],bp[batch][3]};
'''+body+'\n    }\n  }\n'
s=s[:a]+pre+s[b:]
Path('results/opt_goal/preload222/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','c0df342:solution/plan.cpp'],text=True).replace('st=compact4planes_balanced_async_p1p2','st=compact4planes_balanced_async_p1p2;load_overlap=two_input_batches')
Path('results/opt_goal/preload222/plan.cpp').write_text(s)

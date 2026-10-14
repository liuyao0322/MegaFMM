from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','62142fc:solution/sw_pipeline.cu'],text=True)
s=s.replace('Threads=672','Threads=704').replace('  uint64_t pack_ready;','  uint64_t pack_ready,init_ready;')
s=s.replace('Barrier::init(&s.pack_ready,512);','Barrier::init(&s.pack_ready,512);\n    Barrier::init(&s.init_ready,1);')
s=s.replace('int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;','''int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;
  // Invocation-local counters: S[row32][K256] count4; T[col64][K256] count8.
  auto* counters=reinterpret_cast<uint32_t*>(packed+8*Plane);''')
a=s.index('      if(sequence==4){');b=s.index('      if(sequence>=Stages)',a)
s=s[:a]+'''      if(sequence==4){
        // Initialization is globally ordered by a separate coordinator warp,
        // overlapped with raw P1/P2 and CUDA transformation, before any RED.
        wait(&s.init_ready,0);
        wait(&s.pack_ready,0);
        auto* sf=counters+2*blockIdx.x+blockIdx.y/4;
        auto* tf=counters+32+2*blockIdx.y+blockIdx.x/8;
        asm volatile("red.release.gpu.global.add.u32 [%0],1;" :: "l"(sf) : "memory");
        asm volatile("red.release.gpu.global.add.u32 [%0],1;" :: "l"(tf) : "memory");
      }
      if(sequence==4 || sequence==5){
        // Once acquired, the corresponding K-half remains immutable this call.
        auto* sf=counters+2*blockIdx.x+q;
        auto* tf=counters+32+2*blockIdx.y+q;
        uint32_t v;
        do{asm volatile("ld.acquire.gpu.u32 %0,[%1];":"=r"(v):"l"(sf):"memory");}while(v!=4);
        do{asm volatile("ld.acquire.gpu.u32 %0,[%1];":"=r"(v):"l"(tf):"memory");}while(v!=8);
      }
'''+s[b:]
old='''  }else if((tid>=32 && tid<128) || tid>=256){
    int worker=tid>=256 ? tid-160:tid-32;'''
new='''  }else if(tid==32){
    // Every counter has one initializer. No counter can receive a publication
    // until all initializers joined the per-invocation driver barrier.
    if(blockIdx.y%4==0)counters[2*blockIdx.x+blockIdx.y/4]=0;
    if(blockIdx.x%8==0)counters[32+2*blockIdx.y+blockIdx.x/8]=0;
    auto* barrier=&cooperative_groups::details::get_grid_workspace()->barrier;
    uint32_t increment=(blockIdx.x==0 && blockIdx.y==0)?0x80000000u-127u:1u;
    uint32_t old,current;
    asm volatile("atom.add.acq_rel.gpu.u32 %0,[%1],%2;":"=r"(old):"l"(barrier),"r"(increment):"memory");
    do{asm volatile("ld.acquire.gpu.u32 %0,[%1];":"=r"(current):"l"(barrier):"memory");}while(!((old^current)&0x80000000u));
    arrive(&s.init_ready);
  }else if((tid>=64 && tid<128) || tid>=256){
    int worker=tid>=256 ? tid-192:tid-64;'''
assert old in s;s=s.replace(old,new)
s=s.replace('int i=(blockIdx.y*gridDim.x+blockIdx.x)*(Plane/(16*8))+local;','int i=(blockIdx.x*32+local/64)*512+blockIdx.y*64+local%64;')
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','62142fc:solution/plan.cpp'],text=True)
s=s.replace('impl->bytes=size_t(4)<<20;','impl->bytes=(size_t(4)<<20)+256;').replace('1tma+4wgmma+16pack_only','1tma+1coordinator+4wgmma+16pack_only').replace('workspace=4194304','workspace=4194560').replace('publication=tma_leader_driver_workspace_barrier_cuda12.4','publication=early_coordinator_init_then_scoped_ST_counter')
Path('solution/plan.cpp').write_text(s)

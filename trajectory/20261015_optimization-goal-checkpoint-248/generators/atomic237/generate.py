from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','6308acc:solution/sw_pipeline.cu'],text=True)
s=s.replace('uint64_t pack_ready;\n  uint32_t last;','uint64_t pack_ready,output_ready;').replace('Barrier::init(&s.pack_ready,352);','Barrier::init(&s.pack_ready,352);\n    Barrier::init(&s.output_ready,1);')
s=s.replace('st.global.f32 [dst],x;','red.global.add.f32 [dst],x;')
s=s.replace('if(sequence==2)','if(sequence==0)')
s=s.replace('}while(!((old^current)&0x80000000u));','}while(!((old^current)&0x80000000u));\n        arrive(&s.output_ready);')
s=s.replace('    if(worker==0 && blockIdx.z==0)reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(packed)+(size_t(12)<<20))[blockIdx.y*8+blockIdx.x]=0;','''    unsigned cta=(blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x;
    for(unsigned local=unsigned(worker)*4;local<8192;local+=352*4)
      *reinterpret_cast<float4*>(D+cta*8192+local)=float4{0,0,0,0};''')
a=s.index('    auto* partial=reinterpret_cast<float*>(')
b=s.index('  // CTA shared storage survives',a)
s=s[:a]+'''    wait(&s.output_ready,0);
    tensor_chain(s,tid,m0,n0,D);
  }
'''+s[b:]
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','6308acc:solution/plan.cpp'],text=True).replace('impl->bytes=(size_t(12)<<20)+256;','impl->bytes=size_t(4)<<20;').replace('policy.workspace_limit<((size_t(12)<<20)+256)','policy.workspace_limit<(size_t(8)<<20)').replace('SW needs12MiB+256B workspace','SW needs8MiB workspace').replace('workspace=12583168','workspace=4194304').replace('merge=last_cta_partial_C','merge=atomic_two_partial_C;Dzero=per_call_before_publication')
Path('solution/plan.cpp').write_text(s)

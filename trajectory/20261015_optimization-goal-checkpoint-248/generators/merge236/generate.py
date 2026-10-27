from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','92569ff:solution/sw_pipeline.cu'],text=True)
s=s.replace('uint64_t pack_ready;','uint64_t pack_ready,merge_ready;').replace('Barrier::init(&s.pack_ready,352);','Barrier::init(&s.pack_ready,352);\n    Barrier::init(&s.merge_ready,1);')
a=s.index('    asm volatile("bar.sync 2,128;" ::: "memory");\n    if(s.last){')
b=s.index('  // CTA shared storage survives',a)
s=s[:a]+'''    if(tid==128)arrive(&s.merge_ready);
  }
  if(tid>=32){
    wait(&s.merge_ready,0);
    if(tid<416 && s.last){
      auto* partial=reinterpret_cast<float*>(reinterpret_cast<char*>(packed)+(size_t(4)<<20));
      #pragma unroll
      for(unsigned j=0;j<11;++j){
        unsigned element=unsigned(tid-32)*4+j*1536;
        if(element<16384){
          unsigned quadrant=element>>12,local=element&4095;
          unsigned at=(n0+(local>>6))*1024+m0+(local&63)+(quadrant>>1)*512*1024+(quadrant&1)*512;
          float4 x=*reinterpret_cast<const float4*>(partial+at);
          float4 y=*reinterpret_cast<const float4*>(partial+1024*1024+at);
          float4 z{__fadd_rn(x.x,y.x),__fadd_rn(x.y,y.y),__fadd_rn(x.z,y.z),__fadd_rn(x.w,y.w)};
          *reinterpret_cast<float4*>(D+at)=z;
        }
      }
    }
  }
'''+s[b:]
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','92569ff:solution/plan.cpp'],text=True).replace('merge=last_cta_partial_C_vector128','merge=last_cta_partial_C_vector128_workers384')
Path('solution/plan.cpp').write_text(s)

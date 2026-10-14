from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','6308acc:solution/sw_pipeline.cu'],text=True)
a=s.index('      #pragma unroll\n      for(int j=0;j<32;++j){',s.index('    if(s.last){'))
b=s.index('    }\n  }\n  // CTA shared',a)
s=s[:a]+'''      #pragma unroll
      for(unsigned j=0;j<32;++j){
        unsigned element=unsigned(tid-128)*4+j*512;
        unsigned quadrant=element>>12,local=element&4095;
        unsigned at=(n0+(local>>6))*1024+m0+(local&63)+(quadrant>>1)*512*1024+(quadrant&1)*512;
        float4 x=*reinterpret_cast<const float4*>(partial+at);
        float4 y=*reinterpret_cast<const float4*>(partial+1024*1024+at);
        float4 z{__fadd_rn(x.x,y.x),__fadd_rn(x.y,y.y),__fadd_rn(x.z,y.z),__fadd_rn(x.w,y.w)};
        *reinterpret_cast<float4*>(D+at)=z;
      }
'''+s[b:]
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','6308acc:solution/plan.cpp'],text=True).replace('merge=last_cta_partial_C','merge=last_cta_partial_C_vector128')
Path('solution/plan.cpp').write_text(s)

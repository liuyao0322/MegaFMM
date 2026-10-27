from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','d8e6261:solution/sw_pipeline.cu'],text=True)
a=s.index('    unsigned row=unsigned(worker)/BN');b=s.index('    float first[Chunks]',a)
s=s[:a]+'''    int mn0=GMMA::CLayout_64x32{}(worker,0);
    float* out=D+(m0+mn0%64)*1024+n0+mn0/64;
'''+s[b:]
a=s.index('      #pragma unroll\n      for(int j=0;j<Chunks;++j){');b=s.index('      arrive(&s.product_free',a)
s=s[:a]+'''      #pragma unroll
      for(int g=0;g<4;++g){
        float4 v=*reinterpret_cast<const float4*>(s.product[step%2]+g*512+worker*4);
        current[4*g]=v.x;current[4*g+1]=v.y;current[4*g+2]=v.z;current[4*g+3]=v.w;
      }
'''+s[b:]
a=s.index('      #pragma unroll\n      for(int j=0;j<Chunks;++j){',a);b=s.index('    }\n    }\n  }else if(tid>=128',a)
s=s[:a]+'''      #pragma unroll
      for(int pair=0;pair<8;++pair){
        int mn=GMMA::CLayout_64x32{}(0,2*pair);
        float2 result;
        #pragma unroll
        for(int c=0;c<2;++c){
          int j=2*pair+c;float value=0;
          if(step==0)first[j]=current[j];
          else if(step==1)value=__fadd_rn(first[j],current[j]);
          else if(step==2)first[j]=__fadd_rn(first[j],current[j]);
          else if(step==3)second[j]=__fadd_rn(first[j],current[j]);
          else if(step==4){value=__fadd_rn(second[j],current[j]);first[j]=__fadd_rn(first[j],current[j]);}
          else if(step==5)value=__fadd_rn(first[j],current[j]);
          else value=__fsub_rn(second[j],current[j]);
          if(c==0)result.x=value;else result.y=value;
        }
        if(step==1 || step>=4){
          int quadrant=step==1?0:step==4?512*1024+512:step==5?512:512*1024;
          *reinterpret_cast<float2*>(out+(mn%64)*1024+mn/64+quadrant)=result;
        }
      }
'''+s[b:]
a=s.index('      #pragma unroll\n      for(int j=0;j<16;++j){',s.index('  }else if(tid>=128'));b=s.index('      arrive(&s.product_ready',a)
s=s[:a]+'''      #pragma unroll
      for(int g=0;g<4;++g){
        *reinterpret_cast<float4*>(s.product[step%2]+g*512+(tid-128)*4)=make_float4(acc[4*g],acc[4*g+1],acc[4*g+2],acc[4*g+3]);
      }
'''+s[b:]
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','d8e6261:solution/plan.cpp'],text=True).replace('products=swizzled_shared','products=fragment_major_float4_shared')
s=s.replace('pack=fp32_pair_32bit_high_only','pack=fp32_pair_32bit_high_only;P_exchange=float4_fragment_major;consumer_layout=TC_fragment;output=float2')
Path('solution/plan.cpp').write_text(s)

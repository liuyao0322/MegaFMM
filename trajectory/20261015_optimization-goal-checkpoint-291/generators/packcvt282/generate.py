from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','1722267:solution/sw_pipeline.cu'],text=True)
a=s.index('void sw_transform_pdl(');b=s.index('template<class T> __global__ __launch_bounds__(Threads',a);v=s[a:b]
v=v.replace('uint32_t words[16]={};','float values[8][2];')
v=v.replace('''        T sh=T(sv[j]),th=T(tv[j]);
        words[2*j]|=uint32_t(sh.raw())<<(16*lane);
        words[8+2*j]|=uint32_t(th.raw())<<(16*lane);''','''        values[j][lane]=sv[j];
        values[4+j][lane]=tv[j];''')
v=v.replace('for(int j=0;j<8;++j)reinterpret_cast<uint32_t*>(packed)[(j*Plane+i)/2]=words[2*j];','''for(int j=0;j<8;++j){
      uint32_t word;
      if constexpr(std::is_same_v<T,cute::bfloat16_t>)
        asm("cvt.rn.bf16x2.f32 %0,%2,%1;" : "=r"(word) : "f"(values[j][0]),"f"(values[j][1]));
      else asm("cvt.rn.f16x2.f32 %0,%2,%1;" : "=r"(word) : "f"(values[j][0]),"f"(values[j][1]));
      reinterpret_cast<uint32_t*>(packed)[(j*Plane+i)/2]=word;
    }''')
assert 'words' not in v;s=s[:a]+v+s[b:];Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1722267:solution/plan.cpp'],text=True).replace('st=compact4planes_balanced_async_p1p2','st=compact4planes_balanced_async_p1p2;pack_conversion=fp32_to_pair_rn')
Path('solution/plan.cpp').write_text(p)

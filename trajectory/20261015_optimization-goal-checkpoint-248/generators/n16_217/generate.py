from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','1f1fb61:solution/sw_pipeline.cu'],text=True)
s=s.replace('BN=32,BK=256,Stages=4','BN=16,BK=128,Stages=5').replace('Threads=768','Threads=512').replace('==213120','==110720').replace('__launch_bounds__(Threads,1)','__launch_bounds__(Threads,2)')
s=s.replace('pack_ready,608','pack_ready,352').replace('2*608','2*352').replace('Plane/(16*8)','Plane/(32*8)').replace('0x80000000u-127u','0x80000000u-255u').replace('if(sequence==4)','if(sequence==8)').replace('dim3 grid(16,8)','dim3 grid(32,8)').replace('Count<128','Count<256').replace('requires128','requires256')
s=s.replace('Chunks=16','Chunks=8').replace('float (&d)[16]','float (&d)[8]').replace('float acc[16]','float acc[8]').replace('j<16;++j','j<8;++j').replace('CLayout_64x32','CLayout_64x16').replace('Shape<_32,Int<BK>>','Shape<_16,Int<BK>>').replace('Shape<_32,_16>','Shape<_16,_16>')
a=s.index('  using Op=');b=s.index('  // Build the canonical',a);s=s[:a]+s[b:]
a=s.index('    Op::fma(');b=s.index('\n',a)
asm='''    if constexpr(std::is_same_v<T,bfloat16_t>){
      asm volatile("{ .reg .pred p; setp.ne.b32 p,%10,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1,%11,%12; }" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]) : "l"(uint64_t(da)),"l"(uint64_t(db)),"r"(1),"n"(int32_t(GMMA::Major::MN)),"n"(int32_t(GMMA::Major::K)));
    }else{
      asm volatile("{ .reg .pred p; setp.ne.b32 p,%10,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1,%11,%12; }" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]) : "l"(uint64_t(da)),"l"(uint64_t(db)),"r"(1),"n"(int32_t(GMMA::Major::MN)),"n"(int32_t(GMMA::Major::K)));
    }'''
s=s[:a]+asm+s[b:]
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','1f1fb61:solution/plan.cpp'],text=True).replace('smem=213120','smem=110720').replace('operand_stages=4;bk=256','operand_stages=5;bk=128;ctas=256;resident_ctas_per_sm=2').replace('1tma+4simt+4wgmma+15pack_only','1tma+4simt+4wgmma+7pack_only').replace('leaf=Btrans_Atrans','leaf=Btrans_Atrans;mma=64x16x16')
Path('solution/plan.cpp').write_text(s)

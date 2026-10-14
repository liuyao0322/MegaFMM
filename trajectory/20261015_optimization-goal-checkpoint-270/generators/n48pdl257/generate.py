from pathlib import Path
s=Path('results/opt_goal/static218/generate.py').read_text()
s=s.replace("s=subprocess.check_output(['git','show','1f1fb61:solution/sw_pipeline.cu'],text=True)","s=subprocess.check_output(['git','show','1f1fb61:solution/sw_pipeline.cu'],text=True)\ns=s.replace('BN=32','BN=48').replace('grid(16,8)','grid(11,8)').replace('Count<128','Count<88').replace('Shape<_32,Int<BK>>','Shape<_48,Int<BK>>')")
s=s.replace('for j in range(16)','for j in range(24)').replace('d<16>,first<16>,second<16>','d<24>,first<24>,second<24>').replace('m64n32k16','m64n48k16').replace('CLayout_64x32','CLayout_64x48').replace('Shape<_32,Int<BK>>','Shape<_48,Int<BK>>').replace('Shape<_32,_16>','Shape<_48,_16>').replace('slot*3072','slot*3584').replace('%%54,%%55','%%62,%%63')
s=s.replace('leader,done;','leader,done,valid;')
s=s.replace("lines+=['st.global.f32 [dst],x;']","lines+=['setp.lt.u32 valid,%%64,%d;'%(512-((j&1)+8*(j//4))),'@valid st.global.f32 [dst],x;']")
s=s.replace(" ins+=['\"n\"(int32_t(GMMA::Major::MN))','\"n\"(int32_t(GMMA::Major::K))']"," ins+=['\"n\"(int32_t(GMMA::Major::MN))','\"n\"(int32_t(GMMA::Major::K))','\"r\"(n0+mn/64)']")
s=s.replace("s=s.replace('==213120','==196736')","s=s.replace('==213120','==229504')")
exec(compile(s,'n48-static-generator','exec'))
# Reuse the separated primary/secondary launch construction.
s=Path('results/opt_goal/pdl252/generate.py').read_text()
s=s.replace("s=subprocess.check_output(['git','show','62142fc:solution/sw_pipeline.cu'],text=True)","s=Path('solution/sw_pipeline.cu').read_text()")
s=s.replace("s.replace('Threads=672','Threads=256')","s.replace('Threads=768','Threads=256')")
s=s.replace("s=s.replace('    Barrier::init(&s.pack_ready,512);\\n','')","s=s.replace('    Barrier::init(&s.pack_ready,608);\\n','')")
s=s.replace('};dim3 grid(16,8);','};dim3 grid(11,8);').replace('config.gridDim=dim3(16,8)','config.gridDim=dim3(11,8)')
exec(compile(s,'n48-pdl-generator','exec'))
p=Path('solution/sw_pipeline.cu');s=p.read_text()
s=s.replace('  int blocks=0;','''  int blocks=0;
  if(p.dtype==DType::BF16){
    check(cudaFuncSetAttribute(sw_transform_pdl<cute::bfloat16_t>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"SW pack carveout");
    check(cudaFuncSetAttribute(sw_cooperative_hilo<cute::bfloat16_t>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"SW compute carveout");
  }else{
    check(cudaFuncSetAttribute(sw_transform_pdl<cute::half_t>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"SW pack carveout");
    check(cudaFuncSetAttribute(sw_cooperative_hilo<cute::half_t>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"SW compute carveout");
  }''')
p.write_text(s)
p=Path('solution/plan.cpp');s=p.read_text().replace('smem=196736','smem=229504').replace('bk=256','bk=256;tile=64x48;ctas=88;guard_edge_rows=1').replace('transform_kernel=128x512','transform_kernel=128x512;shared_carveout=100percent_both');p.write_text(s)

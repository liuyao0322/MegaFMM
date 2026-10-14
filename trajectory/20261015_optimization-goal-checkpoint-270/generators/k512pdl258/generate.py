from pathlib import Path
s=Path('results/opt_goal/pdl252/generate.py').read_text()
s=s.replace("['git','show','62142fc:solution/sw_pipeline.cu']","['git','show','e753a30:solution/sw_pipeline.cu']")
s=s.replace("s.replace('Threads=672','Threads=256')","s.replace('Threads=768','Threads=256')")
s=s.replace("s=s.replace('    Barrier::init(&s.pack_ready,512);\\n','')","s=s.replace('    Barrier::init(&s.pack_ready,608);\\n','')")
s=s.replace('if(sequence==4)','if(sequence==2)')
exec(compile(s,'k512pdl258-generator','exec'))
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
p=Path('solution/plan.cpp');s=p.read_text().replace('operand_stages=4','operand_stages=2').replace('bk=256','bk=512;lhs_tma=split256').replace('transform_kernel=128x512','transform_kernel=128x512;shared_carveout=100percent_both');p.write_text(s)

from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','8569e59:solution/sw_pipeline.cu'],text=True)
s=s.replace('  int blocks=0;','''  int blocks=0;
  if(p.dtype==DType::BF16){
    check(cudaFuncSetAttribute(sw_transform_pdl<cute::bfloat16_t>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"SW pack carveout");
    check(cudaFuncSetAttribute(sw_cooperative_hilo<cute::bfloat16_t>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"SW compute carveout");
  }else{
    check(cudaFuncSetAttribute(sw_transform_pdl<cute::half_t>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"SW pack carveout");
    check(cudaFuncSetAttribute(sw_cooperative_hilo<cute::half_t>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"SW compute carveout");
  }''')
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','8569e59:solution/plan.cpp'],text=True).replace('transform_kernel=128x512','transform_kernel=128x512;shared_carveout=100percent_both')
Path('solution/plan.cpp').write_text(s)

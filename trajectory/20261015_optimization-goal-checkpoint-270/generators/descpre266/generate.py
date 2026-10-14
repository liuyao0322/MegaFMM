from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','13340d1:solution/sw_pipeline.cu'],text=True)
s=s.replace('  int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;','''  int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;
  if(tid==32){
    asm volatile("prefetch.tensormap [%0]; prefetch.tensormap [%1]; prefetch.tensormap [%2]; prefetch.tensormap [%3];" :: "l"(&amap),"l"(&bmap),"l"(&smap),"l"(&tmap) : "memory");
  }''')
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','13340d1:solution/plan.cpp'],text=True).replace('current_pointer_maps=4','current_pointer_maps=4;descriptor_prefetch=warp1_before_init_barrier')
Path('solution/plan.cpp').write_text(s)

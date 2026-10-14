from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','1722267:solution/sw_pipeline.cu'],text=True)
needle='''    asm volatile("prefetch.tensormap [%0]; prefetch.tensormap [%1]; prefetch.tensormap [%2]; prefetch.tensormap [%3];" :: "l"(&amap),"l"(&bmap),"l"(&smap),"l"(&tmap) : "memory");'''
assert s.count(needle)==1
s=s.replace(needle,needle+'''
    // The raw operands are ready before the primary starts; prefetch data
    // while the other warp initializes the shared barrier state.
    for(int p=0;p<2;++p)for(int q=0;q<4;++q){
      int k0=q*BK;
      SM90_TMA_LOAD_2D::PREFETCH::copy(&bmap,m0,k0+(p==1?512:0));
      SM90_TMA_LOAD_3D::PREFETCH::copy(&amap,0,n0,(k0+(p==1?512:0))/64);
    }''')
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1722267:solution/plan.cpp'],text=True).replace('descriptor_prefetch=warp1_before_init_barrier','descriptor_prefetch=warp1_before_init_barrier;raw_data_prefetch=P1P2_warp1_before_init_barrier')
Path('solution/plan.cpp').write_text(p)

from pathlib import Path
import subprocess
base=subprocess.check_output(['git','show','18f9a6f:solution/sw_pipeline.cu'],text=True)
g=Path('results/opt_goal/static218/generate.py').read_text()
a=g.index("fn='''");b=g.index("s=s[:a]+fn+s[b:]")
g=g[a:b].replace('for j in range(16)','for j in range(8)').replace('d<16>,first<16>,second<16>','d<8>,first<8>,second<8>').replace('m64n32k16','m64n16k16').replace('CLayout_64x32','CLayout_64x16').replace('Shape<_32,','Shape<_16,').replace('Shape<_32,_16>','Shape<_16,_16>').replace('seq%4','seq%2').replace('seq//4%2','seq//2%2').replace('slot*3072','slot*2560').replace('%%54,%%55','%%46,%%47')
ctx={};exec(g,ctx)
a=base.index('template<class T> __device__ __forceinline__ void tensor_chain');b=base.index('template<class T> __global__ __launch_bounds__(512',a)
s=base[:a]+ctx['fn']+base[b:]
s=s.replace('BN=32,BK=256,Stages=4','BN=16,BK=256,Stages=2').replace('==196736','==82048').replace('Shape<_32,Int<BK>>','Shape<_16,Int<BK>>').replace('if(sequence==4)','if(sequence==4)').replace('config.gridDim=dim3(16,8)','config.gridDim=dim3(32,8)')
# Ordinary PDL launch need not have all 256 CTAs simultaneously resident;
# require two resident blocks per SM to test the intended occupancy.
s=s.replace('if(blocks*prop.multiProcessorCount<128)','if(blocks<2)').replace('SW cooperative grid requires128 resident CTA capacity','SW N16 requires two resident CTA capacity')
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','18f9a6f:solution/plan.cpp'],text=True).replace('smem=196736;operand_stages=4','smem=82048;operand_stages=2').replace('bk=256','bk=256;tile=64x16;ctas=256;resident_ctas_per_sm=2')
Path('solution/plan.cpp').write_text(p)
for f in ['internal.h','scale_output.cu']:
 Path('solution/'+f).write_bytes(subprocess.check_output(['git','show','18f9a6f:solution/'+f]))

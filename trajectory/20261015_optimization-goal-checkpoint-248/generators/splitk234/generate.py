from pathlib import Path
import subprocess
# Reuse corrected true-N64 generator with one K256 chunk per leaf.
g=Path('results/opt_goal/n64_227/generate.py').read_text()
g=g.replace("s=Path('results/opt_goal/static218/generate.py').read_text()", "s=Path('results/opt_goal/static218/generate.py').read_text()\ns=s.replace('for q in range(2):','for q in range(1):').replace('seq=step*2+q;','seq=step+q;')")
exec(compile(g,'splitk-base','exec'))
s=Path('solution/sw_pipeline.cu').read_text()
s=s.replace('uint64_t pack_ready;','uint64_t pack_ready;\n  uint32_t last;')
s=s.replace('q<512/BK','q<1').replace('step*(512/BK)+q','step+q').replace('k0=q*BK','k0=int(blockIdx.z)*BK+q*BK').replace('sequence==4','sequence==2')
s=s.replace('Plane/(8*8)','Plane/(16*8)').replace('(blockIdx.y*gridDim.x+blockIdx.x)*(Plane/(16*8))','((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*(Plane/(16*8))')
s=s.replace('blockIdx.x==0 && blockIdx.y==0','blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0').replace('80000000u-63u','80000000u-127u').replace('Count<64','Count<128').replace('grid(8,8)','grid(8,8,2)')
s=s.replace('    int worker=tid>=256 ? tid-160:tid-32;','''    int worker=tid>=256 ? tid-160:tid-32;
    if(worker==0 && blockIdx.z==0)reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(packed)+(size_t(12)<<20))[blockIdx.y*8+blockIdx.x]=0;''')
s=s.replace('    tensor_chain(s,tid,m0,n0,D);','''    auto* partial=reinterpret_cast<float*>(reinterpret_cast<char*>(packed)+(size_t(4)<<20));
    tensor_chain(s,tid,m0,n0,partial+blockIdx.z*1024*1024);
    __threadfence();
    asm volatile("bar.sync 2,128;" ::: "memory");
    if(tid==128){
      auto* counter=reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(packed)+(size_t(12)<<20))+blockIdx.y*8+blockIdx.x;
      uint32_t old;
      asm volatile("atom.add.acq_rel.gpu.u32 %0,[%1],1;":"=r"(old):"l"(counter):"memory");
      s.last=old==1;
    }
    asm volatile("bar.sync 2,128;" ::: "memory");
    if(s.last){
      #pragma unroll
      for(int j=0;j<32;++j){
        unsigned mn=GMMA::CLayout_64x64{}(unsigned(tid-128),j);
        unsigned index=(n0+mn/64)*1024+m0+mn%64;
        #pragma unroll
        for(int quadrant=0;quadrant<4;++quadrant){
          unsigned at=index+(quadrant/2)*512*1024+(quadrant%2)*512;
          D[at]=__fadd_rn(partial[at],partial[1024*1024+at]);
        }
      }
    }''')
Path('solution/sw_pipeline.cu').write_text(s)
s=Path('solution/plan.cpp').read_text().replace('impl->bytes=size_t(4)<<20;','impl->bytes=(size_t(12)<<20)+256;').replace('policy.workspace_limit<(size_t(8)<<20)','policy.workspace_limit<((size_t(12)<<20)+256)').replace('SW needs8MiB workspace','SW needs12MiB+256B workspace').replace('workspace=4194304','workspace=12583168').replace('compute_ctas=64','compute_ctas=128;split_k=2;merge=last_cta_partial_C')
Path('solution/plan.cpp').write_text(s)

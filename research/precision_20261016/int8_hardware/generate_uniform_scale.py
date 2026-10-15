"""P31 conditional common-scale accumulation, preserving frozen P30 fallback."""
from pathlib import Path

h=Path(__file__).resolve().parent
base=h/'p30'
s=(base/'corrected.cuh').read_text()
a=s.index('  }else{\n   #pragma unroll 1\n   for(int k=0;k<4;++k){')
b=s.index('  // Product/term',a)
old=s[a:b]
assert old.endswith('  }\n')
fallback=old[len('  }else{\n'):-len('  }\n')]
mma=old[old.index('    #pragma unroll\n    for(int inner=0;inner<4;++inner)'):old.index('    warpgroup_commit_batch();warpgroup_wait<0>();')]
mma=mma.replace('integer,inner?GMMA::ScaleOut::One:GMMA::ScaleOut::Zero',
                'integer,(k==0&&inner==0)?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One')
new='''  }else{
   int owner=(step-7)*4;
   int e0=__shfl_sync(0xffffffff,meta_e0,owner),e1=__shfl_sync(0xffffffff,meta_e1,owner);
   bool common=e0>=-126&&e0<=103&&e1>=-126&&e1<=103;
   #pragma unroll
   for(int kk=1;kk<4;++kk){
    int ek0=__shfl_sync(0xffffffff,meta_e0,owner+kk),ek1=__shfl_sync(0xffffffff,meta_e1,owner+kk);
    common=common&&(ek0==e0)&&(ek1==e1);
   }
   if(common){
    uint32_t integer[16]={};
    #pragma unroll
    for(int j=0;j<16;++j)warpgroup_fence_operand(integer[j]);
    warpgroup_arrive();
    #pragma unroll 1
    for(int k=0;k<4;++k){
     int seq=step*4+k,slot=seq%Stages;
     Barrier::wait(&s.ready[slot],seq/Stages%2);asm volatile("" ::: "memory");
     uint64_t offset=slot*24576/16;
''' + mma + '''     warpgroup_commit_batch();
     if(k>0){
      warpgroup_wait<1>();
      asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
      if((tid&31)==0)Barrier::arrive(&s.free[(seq-1)%Stages]);
     }
    }
    warpgroup_wait<0>();
    #pragma unroll
    for(int j=0;j<16;++j)warpgroup_fence_operand(integer[j]);
    float s0=__int_as_float((e0+127)<<23),s1=__int_as_float((e1+127)<<23);
    #pragma unroll
    for(int j=0;j<16;++j){
     int mn=GMMA::CLayout_64x32{}(tid-128,j);
     float x=__fmul_rn(float(static_cast<int32_t>(integer[j])),(mn%64)<32?s0:s1);
     d[j]=__fadd_rn(d[j],x);
    }
    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
    if((tid&31)==0)Barrier::arrive(&s.free[(step*4+3)%Stages]);
   }else{
''' + fallback + '''   }
  }
'''
s=s[:a]+new+s[b:]
s=s.replace('#include "megafmm/gemm.h"','#include "megafmm/gemm.h"\n#include <vector>\n#include <cstdio>')
s=s.replace('static_assert(2*128*127*127<4194304,"paired integer bound");',
'''static_assert(2*128*127*127<4194304,"paired integer bound");
static_assert(2*512*127*127<16777216,"common-scale exact FP32 integer bound");''')
# This helper is called ONLY during correctness, never from run/timed lambdas.
diagnostic='''
void diagnose_uniform_scales(const char* dtype,int pattern,int trial,Work w,cudaStream_t stream,int stream_id=0){
 std::vector<int> ea(9*32),eb(9*64);
 check(cudaMemcpyAsync(ea.data(),w.ea,ea.size()*sizeof(int),cudaMemcpyDeviceToHost,stream),"diagnostic A exponents");
 check(cudaMemcpyAsync(eb.data(),w.eb,eb.size()*sizeof(int),cudaMemcpyDeviceToHost,stream),"diagnostic B exponents");
 check(cudaStreamSynchronize(stream),"diagnostic metadata ready");
 int eligible=0,paired=0,exceptional=0;
 for(int by=0;by<8;++by)for(int bx=0;bx<16;++bx)for(int g=0;g<5;++g){
  int term=first_term(g),ap=qa_plane(term),bp=qb_plane(term);
  int ex[2][4];bool bounded=true,same=true;
  for(int half=0;half<2;++half)for(int k=0;k<4;++k){
   ex[half][k]=ea[ap*32+(bx*32/64)*4+k]+eb[bp*64+k*16+by*2+half];
   bounded=bounded&&ex[half][k]>=-126&&ex[half][k]<=103;
   if(k)same=same&&ex[half][k]==ex[half][0];
  }
  bool use=same&&bounded;eligible+=use;paired+=use&&g>=2;exceptional+=!bounded;
 }
 printf("SCALE_ELIGIBILITY dtype=%s pattern=%d trial=%d stream=%d groups=640 eligible=%d paired_eligible=%d fallback=%d exceptional=%d diagnostic_only=1\\n",
  dtype,pattern,trial,stream_id,eligible,paired,640-eligible,exceptional);
}
'''
pos=s.index('template<class T> __device__ __forceinline__ void float_mma')
s=s[:pos]+diagnostic+'\n'+s[pos:]
(h/'corrected.cuh').write_text(s)
for name in ['pack.cuh','pow2.cuh']:
    (h/name).write_text((base/name).read_text())
c=(base/'compare.cu').read_text()
anchor='  runs[0]();runs[3]();'
assert c.count(anchor)==1
c=c.replace(anchor,anchor+'\n  int8_research::diagnose_uniform_scales(dt,pat,trial,int8_research::Work(workspace.p),stream.s);')
(h/'compare.cu').write_text(c)
print('generated common-scale guarded consumer from frozen P30')

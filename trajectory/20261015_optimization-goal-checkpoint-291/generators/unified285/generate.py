from pathlib import Path
import subprocess
base=subprocess.check_output(['git','show','18f9a6f:solution/sw_pipeline.cu'],text=True)
g=Path('results/opt_goal/static218/generate.py').read_text();a=g.index("fn='''");b=g.index('s=s[:a]+fn+s[b:]');g=g[a:b]
g=g.replace('int n0,float* D){','int n0,float* D,const CUtensorMap& amap,const CUtensorMap& bmap,const CUtensorMap& smap,const CUtensorMap& tmap){')
g=g.replace('tid-128,0','tid,0')
g=g.replace('int tid','int tid',1)
g=g.replace("lines+=['wgmma.commit_group.sync.aligned;','wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;','add.u32 fb,%%3,%d;'%(slot*8),'@leader mbarrier.arrive.shared::cta.b64 _,[fb];']",'''lines+=['wgmma.commit_group.sync.aligned;','wgmma.wait_group.sync.aligned 0;','bar.sync 0,128;']
   if seq+4<14:
    future=seq+4;fp=[0,1,5,6,4,2,3][future//2];fk=(future%2)*256;fs=future%4
    lines+=['add.u32 rb,%%2,%d;'%(fs*8),'@leader mbarrier.arrive.expect_tx.shared::cta.b64 _,[rb],49152;','add.u32 fb,%%60,%d;'%(fs*49152)]
    if fp==3 or fp>=4:
     plane=3 if fp==3 else fp-4
     lines+=['@leader cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [fb],[%%59,{%%61,%d,%d}],[rb];'%(fk,plane)]
    else:
     col=512 if fp==2 else 0;row=512 if fp in [1,2] else 0
     lines+=['add.u32 xc,%%61,%d;'%col,'@leader cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [fb],[%%57,{xc,%d}],[rb];'%(fk+row)]
    lines+=['add.u32 fb,fb,32768;']
    if fp==2 or fp>=4:
     plane=3 if fp==2 else fp-4
     lines+=['@leader cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [fb],[%%58,{0,%%62,%d,%d}],[rb];'%(fk//64,plane)]
    else:
     col=512 if fp in [1,3] else 0;row=512 if fp==3 else 0
     lines+=['add.u32 xc,%%62,%d;'%row,'@leader cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [fb],[%%56,{0,xc,%d}],[rb];'%((fk+col)//64)]''')
g=g.replace('.reg .b32 rb,fb;','.reg .b32 rb,fb,xc;')
g=g.replace('"r"(tid&31)','"r"(tid)')
g=g.replace("ins+=['\"n\"(int32_t(GMMA::Major::MN))','\"n\"(int32_t(GMMA::Major::K))']","ins+=['\"n\"(int32_t(GMMA::Major::MN))','\"n\"(int32_t(GMMA::Major::K))','\"l\"(&amap)','\"l\"(&bmap)','\"l\"(&smap)','\"l\"(&tmap)','\"r\"(uint32_t(__cvta_generic_to_shared(s.operand)))','\"r\"(m0)','\"r\"(n0)']")
ctx={};exec(g,ctx)
a=base.index('template<class T> __device__ __forceinline__ void tensor_chain');b=base.index('template<class T> __global__ __launch_bounds__(512',a)
s=base[:a]+ctx['fn']+base[b:]
s=s.replace('Plane=512*512,Threads=256','Plane=512*512,Threads=128')
a=s.index('  if(tid==0){\n    for(int step=0;step<7;');b=s.index('  // CTA shared storage',a)
s=s[:a]+'''  if(tid==0){
    for(int step=0;step<2;++step)for(int q=0;q<2;++q){
      int sequence=2*step+q,k0=q*BK;
      TxBarrier::arrive_and_expect_tx(&s.ready[sequence],2*(ASize+BSize));
      T* op=s.operand[sequence];
      SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[sequence],0,op,m0,k0+(step==1?512:0));
      SM90_TMA_LOAD_3D::copy(&amap,&s.ready[sequence],0,op+ASize,0,n0,(k0+(step==1?512:0))/64);
    }
    cudaGridDependencySynchronize();
  }
  __syncthreads();
  tensor_chain(s,tid,m0,n0,D,amap,bmap,smap,tmap);
'''+s[b:]
assert s.count('cp.async.bulk.tensor.')==40,s.count('cp.async.bulk.tensor.')
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','18f9a6f:solution/plan.cpp'],text=True).replace('warp_roles=1tma+4wgmma','warp_roles=4wgmma_including_elected_tma;compute_threads=128').replace('operand_release=warp_leader','operand_release=cta_barrier_then_refill').replace('descriptor_prefetch=warp1_before_init_barrier','descriptor_prefetch=warp1_before_init_barrier;publication_wait=after_raw_preload_before_chain')
Path('solution/plan.cpp').write_text(p)

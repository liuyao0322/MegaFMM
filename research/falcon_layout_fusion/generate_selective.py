"""Build bounded selective S3/T3 fusion experiment from frozen iter331."""
from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','6d4167b:solution/sw_pipeline.cu'],text=True)
s=s.replace('  uint64_t ready[Stages],free[Stages];','  // Both TMA destinations start on the same1024-byte swizzle phase.\n  alignas(128) T difference[ASize+BSize];\n  uint64_t ready[Stages],free[Stages];\n  uint64_t difference_ready;')
s=s.replace('==196736,"shared budget"','==221440,"shared budget"')
s=s.replace('    for(int j=0;j<8;++j){\n      uint32_t word;', '    for(int j=0;j<8;++j){\n      if(j==2 || j==6)continue; // S3/T3 formed on-chip for P7.\n      uint32_t word;')
s=s.replace('    for(int j=0;j<Stages;++j){Barrier::init(&s.ready[j],1);Barrier::init(&s.free[j],4);}','    for(int j=0;j<Stages;++j){Barrier::init(&s.ready[j],1);Barrier::init(&s.free[j],4);}\n    Barrier::init(&s.difference_ready,1);')
old='''  if(tid==0){
    for(int step=0;step<7;++step)for(int q=0;q<512/BK;++q){
      int p=product_for_step(step);
      int sequence=step*(512/BK)+q,slot=sequence%Stages,k0=q*BK;
'''
new='''  if(tid<128){
    for(int step=0;step<7;++step)for(int q=0;q<512/BK;++q){
      int p=product_for_step(step);
      int sequence=step*(512/BK)+q,slot=sequence%Stages,k0=q*BK;
      if(p==6){
        T* op=s.operand[slot];
        if(tid==0){
          if(sequence>=Stages)wait(&s.free[slot],((sequence-Stages)/Stages)%2);
          TxBarrier::arrive_and_expect_tx(&s.difference_ready,4*(ASize+BSize));
          // Identical swizzled boxes allow elementwise physical subtraction:
          // LHS=T3^T=(B22-B12)^T, RHS=S3^T=(A11-A21)^T.
          SM90_TMA_LOAD_2D::copy(&bmap,&s.difference_ready,0,op,m0+512,k0+512);
          SM90_TMA_LOAD_2D::copy(&bmap,&s.difference_ready,0,s.difference,m0+512,k0);
          SM90_TMA_LOAD_3D::copy(&amap,&s.difference_ready,0,op+ASize,0,n0,k0/64);
          SM90_TMA_LOAD_3D::copy(&amap,&s.difference_ready,0,s.difference+ASize,0,n0+512,k0/64);
        }
        wait(&s.difference_ready,q%2);
        for(int i=tid;i<(ASize+BSize)/2;i+=128){
          uint32_t a=reinterpret_cast<uint32_t*>(op)[i];
          uint32_t b=reinterpret_cast<uint32_t*>(s.difference)[i],word;
          if constexpr(std::is_same_v<T,cute::half_t>){
            asm("sub.rn.f16x2 %0,%1,%2;" : "=r"(word) : "r"(a),"r"(b));
          }else{
            float x=__fsub_rn(float(T::bitcast(uint16_t(a))),float(T::bitcast(uint16_t(b))));
            float y=__fsub_rn(float(T::bitcast(uint16_t(a>>16))),float(T::bitcast(uint16_t(b>>16))));
            asm("cvt.rn.bf16x2.f32 %0,%2,%1;" : "=r"(word) : "f"(x),"f"(y));
          }
          reinterpret_cast<uint32_t*>(op)[i]=word;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("bar.sync 1,128;" ::: "memory");
        if(tid==0)arrive(&s.ready[slot]);
        continue;
      }
      if(tid!=0)continue;
'''
assert s.count(old)==1
s=s.replace(old,new)
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','6d4167b:solution/plan.cpp'],text=True)
s=s.replace('smem=196736','smem=221440').replace('st=compact4planes_balanced_async_p1p2','st=global6planes_onchip_S3T3_async_p1p2').replace('warp_roles=1tma+4wgmma','warp_roles=1tma+3diff+4wgmma')
Path('solution/plan.cpp').write_text(s)

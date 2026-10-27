from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','13340d1:solution/sw_pipeline.cu'],text=True)
s=s.replace('    cutlass::arch::fence_barrier_init();','''    cutlass::arch::fence_barrier_init();
    // The producer owns initialization; publish the first raw tile before the
    // CTA rendezvous. Consumers only inspect its barrier after the rendezvous.
    TxBarrier::arrive_and_expect_tx(&s.ready[0],2*(ASize+BSize));
    SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[0],0,s.operand[0],m0,0);
    SM90_TMA_LOAD_3D::copy(&amap,&s.ready[0],0,s.operand[0]+ASize,0,n0,0);''')
s=s.replace('      int sequence=step*(512/BK)+q,slot=sequence%Stages,k0=q*BK;','      int sequence=step*(512/BK)+q,slot=sequence%Stages,k0=q*BK;\n      if(sequence==0)continue;')
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','13340d1:solution/plan.cpp'],text=True).replace('operand_release=warp_leader','operand_release=warp_leader;first_raw_tma=before_init_rendezvous')
Path('solution/plan.cpp').write_text(s)

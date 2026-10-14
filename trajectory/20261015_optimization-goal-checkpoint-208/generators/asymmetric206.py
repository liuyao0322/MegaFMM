from pathlib import Path
s=Path('solution/sw_pipeline.cu').read_text()
s=s.replace('Plane=512*512,Threads=768','Plane=512*512,Threads=800').replace('  T operand[Stages][ASize+BSize];','  T a[4][ASize];\n  T b[5][BSize];').replace('uint64_t ready[Stages],free[Stages];','uint64_t ready[4],free[4],bready[5],bfree[5];').replace('uint64_t pack_ready;','uint64_t pack_ready,global_ready;').replace('==196736','==213248')
s=s.replace('make_smem_ptr(s.operand[0])','make_smem_ptr(s.a[0])').replace('make_smem_ptr(s.operand[0]+ASize)','make_smem_ptr(s.b[0])')
s=s.replace('    Barrier::init(&s.pack_ready,608);','    for(int j=0;j<5;++j){Barrier::init(&s.bready[j],1);Barrier::init(&s.bfree[j],4);}\n    Barrier::init(&s.global_ready,1);\n    Barrier::init(&s.pack_ready,608);')
# Add two independent producer barrier inputs at slots48/49, retaining all old operand numbers.
s=s.replace('"n"(int32_t(GMMA::Major::MN))\n', '"n"(int32_t(GMMA::Major::MN)), "r"(uint32_t(__cvta_generic_to_shared(s.bready))), "r"(uint32_t(__cvta_generic_to_shared(s.bfree)))\n')
# Transform each statically generated K stage independently in both dtype branches.
for step in range(7):
 for q in range(2):
  seq=step*2+q;slota=seq%4;slotb=seq%5;phaseb=(seq//5)%2
  marker='   "@!done bra WAIT_%d_%d;\\n"\n'%(step,q)
  addition=''.join('   "'+x+'\\n"\n' for x in ['add.u32 rb,%%48,%d;'%(8*slotb),'WAITB_%d_%d:'%(step,q),'mbarrier.try_wait.parity.shared::cta.b64 done,[rb],%d,0x989680;'%phaseb,'@!done bra WAITB_%d_%d;'%(step,q)])
  s=s.replace(marker,marker+addition)
  # Limit replacement to one K-stage segment, identified by label and next release.
  import re
  pat=r'("WAIT_%d_%d:.*?@leader mbarrier.arrive.shared::cta.b64 _,\[fb\];\\n"\n)'%(step,q)
  def change(m):
   z=m.group(1).replace('add.u64 ba,%%0,%d;'%(slota*3072),'add.u64 ba,%%0,%d;'%(slota*2048)).replace('add.u64 bb,%%1,%d;'%(slota*3072),'add.u64 bb,%%1,%d;'%(slotb*1024))
   return z+''.join('   "'+x+'\\n"\n' for x in ['add.u32 fb,%%49,%d;'%(slotb*8),'@leader mbarrier.arrive.shared::cta.b64 _,[fb];'])
  s,n=re.subn(pat,change,s,flags=re.S)
  assert n==2,(step,q,n)
# Replace producer branch in the global function only.
a=s.index('  if(tid==0){\n    for(int step=0;step<7;');b=s.index('  }else if((tid>=32 && tid<128) || tid>=256)',a)
prod='''  if(tid==0 || tid==32){
    bool is_a=tid==0;
    for(int step=0;step<7;++step)for(int q=0;q<2;++q){
      int p=product_for_step(step),sequence=2*step+q,k0=q*BK;
      int slots=is_a?4:5,slot=sequence%slots;
      if(sequence==4){
        if(is_a){
          wait(&s.pack_ready,0);
          auto* barrier=&cooperative_groups::details::get_grid_workspace()->barrier;
          uint32_t increment=(blockIdx.x==0 && blockIdx.y==0)?0x80000000u-127u:1u,old,current;
          asm volatile("atom.add.acq_rel.gpu.u32 %0,[%1],%2;":"=r"(old):"l"(barrier),"r"(increment):"memory");
          do{asm volatile("ld.acquire.gpu.u32 %0,[%1];":"=r"(current):"l"(barrier):"memory");}while(!((old^current)&0x80000000u));
          arrive(&s.global_ready);
        }else wait(&s.global_ready,0);
      }
      auto* free=is_a ? &s.free[slot]:&s.bfree[slot];
      auto* ready=is_a ? &s.ready[slot]:&s.bready[slot];
      if(sequence>=slots)wait(free,((sequence-slots)/slots)%2);
      TxBarrier::arrive_and_expect_tx(ready,2*(is_a?ASize:BSize));
      if(is_a){
        T* op=s.a[slot];
        if(p==2 || p>=4){
          int plane=p==2?3:p-4;SM90_TMA_LOAD_4D::copy(&smap,ready,0,op,0,m0,k0/64,plane);
        }else{
          int col=(p==1 || p==3)?512:0,row=p==3?512:0;SM90_TMA_LOAD_3D::copy(&amap,ready,0,op,0,m0+row,(k0+col)/64);
        }
      }else{
        T* op=s.b[slot];
        if(p==3 || p>=4){
          int plane=p==3?3:p-4;SM90_TMA_LOAD_3D::copy(&tmap,ready,0,op,n0,k0,plane);
        }else{
          int col=p==2?512:0,row=(p==1 || p==2)?512:0;SM90_TMA_LOAD_2D::copy(&bmap,ready,0,op,n0+col,k0+row);
        }
      }
    }
'''
s=s[:a]+prod+s[b:]
s=s.replace('((tid>=32 && tid<128) || tid>=256)','((tid>=64 && tid<128) || tid>=256)').replace('tid>=256 ? tid-160:tid-32','tid>=256 ? tid-192:tid-64')
Path('solution/sw_pipeline.cu').write_text(s)
p=Path('solution/plan.cpp');s=p.read_text().replace('1tma+4wgmma+19pack_only','2tma+4wgmma+19pack_only').replace('smem=196736','smem=213248').replace('operand_stages=4','operand_stages=A4_B5').replace('tc_full_static_ptx_chain','tc_static_ptx_asymmetric_A4_B5');p.write_text(s)

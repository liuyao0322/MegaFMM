from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','13340d1:solution/sw_pipeline.cu'],text=True)
s=s.replace(' int mn=GMMA::CLayout_64x32{}(tid-128,0);',''' uint32_t joint_free;
 asm volatile("mapa.shared::cluster.u32 %0,%1,0;":"=r"(joint_free):"r"(uint32_t(__cvta_generic_to_shared(s.free))));
 int mn=GMMA::CLayout_64x32{}(tid-128,0);''')
s=s.replace('"r"(uint32_t(__cvta_generic_to_shared(s.free)))','"r"(joint_free)')
# The replacement above must not change the input to its own initialization.
s=s.replace('"=r"(joint_free):"r"(joint_free)','"=r"(joint_free):"r"(uint32_t(__cvta_generic_to_shared(s.free))))') if False else s
# Repair the initialization expression explicitly.
s=s.replace('asm volatile("mapa.shared::cluster.u32 %0,%1,0;":"=r"(joint_free):"r"(joint_free));','asm volatile("mapa.shared::cluster.u32 %0,%1,0;":"=r"(joint_free):"r"(uint32_t(__cvta_generic_to_shared(s.free))));')
s=s.replace('@leader mbarrier.arrive.shared::cta.b64 _,[fb];','@leader mbarrier.arrive.release.cluster.shared::cluster.b64 _,[fb];')
s=s.replace('  if(tid==0){\n    for(int j=0;j<Stages;++j){Barrier::init(&s.ready[j],1);Barrier::init(&s.free[j],4);}','''  int rank=blockIdx.x&1;
  if(tid==0){
    for(int j=0;j<Stages;++j){Barrier::init(&s.ready[j],2);Barrier::init(&s.free[j],rank==0?8:1);}''')
s=s.replace('  __syncthreads();','  cooperative_groups::this_cluster().sync();')
s=s.replace('      if(sequence>=Stages)wait(&s.free[slot],((sequence-Stages)/Stages)%2);','''      if(sequence>=Stages){
        wait(&s.free[slot],((sequence-Stages)/Stages)%2);
        // Rank0 observes both sets of four consumer warp arrivals; it then
        // grants rank1 permission to reuse the same slot.
        if(rank==0)Barrier::arrive(&s.free[slot],1,1);
      }''')
s=s.replace('TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*(ASize+BSize));','TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*BSize);')
s=s.replace('      // LHS is B_leaf^T, M-contiguous original B data.','''      // One LHS load feeds both adjacent N tiles. Two arrivals prevent a
      // multicast completion from retiring a slot before the peer issues RHS.
      if(rank==0){
        TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*ASize);
        TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*ASize,1,1);
      // LHS is B_leaf^T, M-contiguous original B data.''')
s=s.replace('SM90_TMA_LOAD_3D::copy(&tmap,&s.ready[slot],0,op,','SM90_TMA_LOAD_MULTICAST_3D::copy(&tmap,&s.ready[slot],3,op,').replace('SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],0,op,','SM90_TMA_LOAD_MULTICAST_2D::copy(&bmap,&s.ready[slot],3,op,')
s=s.replace('      // RHS is A_leaf^T, K-contiguous original A data.','      }\n      // RHS is A_leaf^T, K-contiguous original A data.')
s=s.replace('  // CTA shared storage survives until all role threads return.','''  // No remote access may outlive the CTA owning the shared barrier.
  cooperative_groups::this_cluster().sync();''')
s=s.replace('  cudaLaunchAttribute attribute{};','  cudaLaunchAttribute attribute[2]{};').replace('  attribute.id=','  attribute[0].id=').replace('  attribute.val.','  attribute[0].val.')
s=s.replace('  config.attrs=&attribute;config.numAttrs=1;','''  attribute[1].id=cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim={2,1,1};
  config.attrs=attribute;config.numAttrs=2;''')
assert s.count('"r"(joint_free)')==2
assert s.count('mbarrier.arrive.release.cluster.shared::cluster')==28
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','13340d1:solution/plan.cpp'],text=True).replace('operand_release=warp_leader','operand_release=joint8_warp_leaders;cluster=2;lhs=multicast;slot_grant=rank0_to_rank1')
Path('solution/plan.cpp').write_text(s)

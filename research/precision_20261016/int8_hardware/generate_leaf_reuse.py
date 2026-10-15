"""P46 from frozen P45: reuse highs for main and all native residual dots."""
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p45'
s=(base/'corrected.cuh').read_text()
start=s.index('template<class T> __device__ __forceinline__ void consume')
end=s.index('template<class T> __global__ __launch_bounds__',start)
old=s[start:end]
# Reuse exactly the original seven-product streaming recombination branches.
rs=old.index('   if(p==0){');re=old.index('\n  }else{\n   int term=step-7;',rs)
recombine=old[rs:re]
consumer='''template<class T> __device__ __forceinline__ void consume(Shared& s,int tid,int m0,int n0,const int* ea,const int* eb,float* D){
 using namespace cute;
 constexpr auto la=tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,_128>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_32,_128>{});
 auto a=make_tensor(make_smem_ptr(reinterpret_cast<T*>(s.operand[0])),la);
 auto b=make_tensor(make_smem_ptr(reinterpret_cast<T*>(s.operand[0]+32768)),lb);
 auto da=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(a,Shape<_64,_16>{}));
 auto db=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(b,Shape<_32,_16>{}));
 float c[4][16]={};
 #pragma unroll 1
 for(int step=0;step<7;++step){
  int p=main_product(step);
  float d[16]={},r[16]={};
  #pragma unroll
  for(int j=0;j<16;++j){warpgroup_fence_operand(d[j]);warpgroup_fence_operand(r[j]);}
  warpgroup_arrive();
  #pragma unroll 1
  for(int k=0;k<4;++k){
   int seq=step*4+k,slot=seq%Stages;
   Barrier::wait(&s.ready[slot],seq/Stages%2);asm volatile("" ::: "memory");
   uint64_t offset=slot*49152/16;
   #pragma unroll
   for(int inner=0;inner<8;++inner)
    float_mma<T>(uint64_t(da(0,0,inner))+offset,uint64_t(db(0,0,inner))+offset,d,(k==0&&inner==0)?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One);
   if(p==2||p>=4){
    #pragma unroll
    for(int inner=0;inner<8;++inner)
     float_mma<T>(uint64_t(da(0,0,inner))+offset,uint64_t(db(0,0,inner))+offset+8192/16,r,(k==0&&inner==0)?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One);
   }
   if(p==3||p>=4){
    #pragma unroll
    for(int inner=0;inner<8;++inner)
     float_mma<T>(uint64_t(da(0,0,inner))+offset+16384/16,uint64_t(db(0,0,inner))+offset,r,(p==3&&k==0&&inner==0)?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One);
   }
   warpgroup_commit_batch();
   if(k>0){
    // Both fragments' preceding batch is complete before the slot is released.
    warpgroup_wait<1>();asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
    if((tid&31)==0)Barrier::arrive(&s.free[(seq-1)%Stages]);
   }
  }
  warpgroup_wait<0>();
  #pragma unroll
  for(int j=0;j<16;++j){warpgroup_fence_operand(d[j]);warpgroup_fence_operand(r[j]);}
  asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
  if((tid&31)==0)Barrier::arrive(&s.free[(step*4+3)%Stages]);
  if(p>=2){
   constexpr float inverse=(std::is_same_v<T,cute::half_t> ? (1.f/2048.f) : (1.f/256.f));
   #pragma unroll
   for(int j=0;j<16;++j)d[j]=__fadd_rn(d[j],__fmul_rn(r[j],inverse));
  }
RECOMBINE
 }
 #pragma unroll
 for(int j=0;j<16;++j){
  int mn=GMMA::CLayout_64x32{}(tid-128,j),at=(n0+mn/64)*1024+m0+mn%64;
  D[at]=c[0][j];D[at+512]=c[1][j];D[at+512*1024]=c[2][j];D[at+512*1024+512]=c[3][j];
 }
}

'''.replace('RECOMBINE',recombine)
s=s[:start]+consumer+s[end:]
start=s.index('  for(int step=0;step<12;++step)');end=s.index('\n }else if(tid>=128)',start)
producer='''  for(int step=0;step<7;++step)for(int k=0;k<4;++k){
   int p=main_product(step),low=p<4?10:p+3;
   int seq=step*4+k,slot=seq%Stages;
   if(seq>=Stages){Barrier::wait(&s.free[slot],((seq-Stages)/Stages)%2);asm volatile("" ::: "memory");}
   int bytes=p<2?24576:p==2?32768:p==3?40960:49152;
   TxBarrier::arrive_and_expect_tx(&s.ready[slot],bytes);
   SM90_TMA_LOAD_3D::copy(&high_b,&s.ready[slot],0,s.operand[slot],m0,k*128,p);
   SM90_TMA_LOAD_4D::copy(&high_a,&s.ready[slot],0,s.operand[slot]+32768,0,n0,k*2,p);
   if(p==2||p>=4)
    SM90_TMA_LOAD_4D::copy(&high_a,&s.ready[slot],0,s.operand[slot]+40960,0,n0,k*2,low);
   if(p==3||p>=4)
    SM90_TMA_LOAD_3D::copy(&high_b,&s.ready[slot],0,s.operand[slot]+16384,m0,k*128,low);
  }'''
s=s[:start]+producer+s[end:]
(h/'corrected.cuh').write_text(s)
for name in ('pack.cuh','compare.cu','pow2.cuh'):(h/name).write_bytes((base/name).read_bytes())
print('P46: seven corrected leaves, all eight residual dots, two fragments, 28 K batches, 88 TMA commands/CTA')

// Prepared P53 body: one full-K512 native corrected leaf per CTA, N32 retained.
template<class T> __device__ __forceinline__ void consume(Shared& s,int tid,int m0,int n0,int p,float* leaf){
 using namespace cute;
 constexpr auto la=tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,_128>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_32,_128>{});
 auto a=make_tensor(make_smem_ptr(reinterpret_cast<T*>(s.operand[0])),la);
 auto b=make_tensor(make_smem_ptr(reinterpret_cast<T*>(s.operand[0]+32768)),lb);
 auto da=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(a,Shape<_64,_16>{}));
 auto db=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(b,Shape<_32,_16>{}));
 float d[16]={},r[16]={};
 #pragma unroll
 for(int j=0;j<16;++j){warpgroup_fence_operand(d[j]);warpgroup_fence_operand(r[j]);}
 warpgroup_arrive();
 #pragma unroll 1
 for(int k=0;k<4;++k){
  int slot=k%ConsumerStages;
  Barrier::wait(&s.ready[slot],k/ConsumerStages%2);asm volatile("" ::: "memory");
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
   warpgroup_wait<1>();asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
   if((tid&31)==0)Barrier::arrive(&s.free[(k-1)%ConsumerStages]);
  }
 }
 warpgroup_wait<0>();
 #pragma unroll
 for(int j=0;j<16;++j){warpgroup_fence_operand(d[j]);warpgroup_fence_operand(r[j]);}
 asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
 if((tid&31)==0)Barrier::arrive(&s.free[3%ConsumerStages]);
 if(p>=2){
  constexpr float inverse=(std::is_same_v<T,cute::half_t> ? (1.f/2048.f) : (1.f/256.f));
  #pragma unroll
  for(int j=0;j<16;++j)d[j]=__fadd_rn(d[j],__fmul_rn(r[j],inverse));
 }
 #pragma unroll
 for(int j=0;j<16;++j){
  int mn=GMMA::CLayout_64x32{}(tid-128,j),at=(n0+mn/64)*512+m0+mn%64;
  leaf[p*Plane+at]=d[j];
 }
}
template<class T> __global__ __launch_bounds__(256,1)
void corrected(const __grid_constant__ CUtensorMap high_a,const __grid_constant__ CUtensorMap high_b,
 const int* ea,const int* eb,float* leaf){
 using namespace cute;
 extern __shared__ __align__(128) unsigned char buffer[];
 auto& s=*reinterpret_cast<Shared*>(buffer);
 int tid=threadIdx.x,m0=blockIdx.y*64,n0=blockIdx.x*32,p=blockIdx.z;
 if(tid==0){for(int j=0;j<ConsumerStages;++j){Barrier::init(&s.ready[j],1);Barrier::init(&s.free[j],4);}cutlass::arch::fence_barrier_init();}
 __syncthreads();
 if(tid==0){
  for(int k=0;k<4;++k){
   int low=p<4?10:p+3,slot=k%ConsumerStages;
   if(k>=ConsumerStages){Barrier::wait(&s.free[slot],((k-ConsumerStages)/ConsumerStages)%2);asm volatile("" ::: "memory");}
   int bytes=p<2?24576:p==2?32768:p==3?40960:49152;
   TxBarrier::arrive_and_expect_tx(&s.ready[slot],bytes);
   SM90_TMA_LOAD_3D::copy(&high_b,&s.ready[slot],0,s.operand[slot],m0,k*128,p);
   SM90_TMA_LOAD_4D::copy(&high_a,&s.ready[slot],0,s.operand[slot]+32768,0,n0,k*2,p);
   if(p==2||p>=4)SM90_TMA_LOAD_4D::copy(&high_a,&s.ready[slot],0,s.operand[slot]+40960,0,n0,k*2,low);
   if(p==3||p>=4)SM90_TMA_LOAD_3D::copy(&high_b,&s.ready[slot],0,s.operand[slot]+16384,m0,k*128,low);
  }
 }else if(tid>=128)consume<T>(s,tid,m0,n0,p,leaf);
}
__global__ void recombine(const float* __restrict__ leaf,float* __restrict__ D){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=Plane)return;
 float p0=leaf[i],p1=leaf[Plane+i],p5=leaf[5*Plane+i],p6=leaf[6*Plane+i];
 float p4=leaf[4*Plane+i],p2=leaf[2*Plane+i],p3=leaf[3*Plane+i];
 float t=__fadd_rn(p0,p5),u=__fadd_rn(t,p6);
 int at=(i/512)*1024+i%512;
 D[at]=__fadd_rn(p0,p1);D[at+512]=__fadd_rn(__fadd_rn(t,p4),p2);
 D[at+512*1024]=__fsub_rn(u,p3);D[at+512*1024+512]=__fadd_rn(u,p4);
}


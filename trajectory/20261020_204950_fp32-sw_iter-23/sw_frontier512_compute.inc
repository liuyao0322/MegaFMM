template<class T,int N,int K,int Stages,bool Short=false,int Specialize=0> __global__ __launch_bounds__(256,1)
void compute(const __grid_constant__ CUtensorMap amap,const __grid_constant__ CUtensorMap bmap,
             const __grid_constant__ CUtensorMap smap,const __grid_constant__ CUtensorMap tmap,
             int mh,int nh,int kh,float* D,int M,int NN,int packed,int raster=-1) {
  using namespace cute;
  static_assert(Specialize==0 || (Short && K==256));
  if constexpr(Specialize>0)kh=K;
  if constexpr(Specialize>1)raster=0;
  extern __shared__ __align__(128) unsigned char bytes[];
  auto& s=*reinterpret_cast<Shared<T,N,K,Stages>*>(bytes);
  int tid=threadIdx.x,m0=blockIdx.y*64,n0=blockIdx.x*N,kt=Short?1:kh/K;
  {
    if(raster>0 || (raster<0 && kh>=1024)) {
      int group_size=raster>0?raster:(kh>=8192?2:4),gx=gridDim.x,gy=gridDim.y;
      int pid=blockIdx.x+blockIdx.y*gx,group=pid/(group_size*gy),first=group*group_size;
      int count=min(group_size,gx-first),within=pid%(group_size*gy);
      m0=(within/count)*64;n0=(first+within%count)*N;
    }
  }
  if(tid==32) asm volatile("prefetch.tensormap [%0]; prefetch.tensormap [%1]; prefetch.tensormap [%2]; prefetch.tensormap [%3];"::"l"(&amap),"l"(&bmap),"l"(&smap),"l"(&tmap):"memory");
  if(tid==0) {
    for(int i=0;i<Stages;++i){Barrier::init(&s.ready[i],1);Barrier::init(&s.free[i],4);}
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  if(tid==0) {
    // Long-K row groups reuse A planes while B streams through L2.
    const uint64_t a_hint=kh>=8192?uint64_t(TMA::CacheHintSm90::EVICT_LAST):0;
    const uint64_t b_hint=kh>=8192?uint64_t(TMA::CacheHintSm90::EVICT_FIRST):0;
    if constexpr(Short) {
    #pragma unroll
    for(int step=0;step<7;++step)
    #pragma unroll
    for(int q=0;q<1;++q) {
      int p=product_for_step(step),seq=step*kt+q,slot=seq%Stages,k0=q*K;
      // All transformed global operands are complete before their first TMA.
      if(seq==2*kt)cudaGridDependencySynchronize();
      if(seq>=Stages)wait_free(&s.free[slot],((seq-Stages)/Stages)%2);
      TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*(64+N)*K);
      T* dst=s.operand[slot];
      if(p==3 || p>=4) {
        SM90_TMA_LOAD_3D::copy(&tmap,&s.ready[slot],b_hint,dst,m0,k0,p==3?3:p-4);
      } else {
        int col=p==2?nh:0,row=p==1 || p==2?kh:0;
        SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],b_hint,dst,m0+col,k0+row);
      }
      if(p==2 || p>=4) {
        SM90_TMA_LOAD_4D::copy(&smap,&s.ready[slot],a_hint,dst+64*K,0,n0,k0/64,p==2?3:p-4);
      } else {
        int col=p==1 || p==3?kh:0,row=p==3?mh:0;
        SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],a_hint,dst+64*K,0,n0+row,(k0+col)/64);
      }
    }
    } else {
    for(int step=0;step<7;++step)for(int q=0;q<kt;++q) {
      int p=product_for_step(step),seq=step*kt+q,slot=seq%Stages,k0=q*K;
      // All transformed global operands are complete before their first TMA.
      if(seq==2*kt)cudaGridDependencySynchronize();
      if(seq>=Stages)wait_free(&s.free[slot],((seq-Stages)/Stages)%2);
      TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*(64+N)*K);
      T* dst=s.operand[slot];
      if(p==3 || p>=4) {
        SM90_TMA_LOAD_3D::copy(&tmap,&s.ready[slot],b_hint,dst,m0,k0,p==3?3:p-4);
      } else {
        int col=p==2?nh:0,row=p==1 || p==2?kh:0;
        SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],b_hint,dst,m0+col,k0+row);
      }
      if(p==2 || p>=4) {
        SM90_TMA_LOAD_4D::copy(&smap,&s.ready[slot],a_hint,dst+64*K,0,n0,k0/64,p==2?3:p-4);
      } else {
        int col=p==1 || p==3?kh:0,row=p==3?mh:0;
        SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],a_hint,dst+64*K,0,n0+row,(k0+col)/64);
      }
    }
    }
  } else if(tid>=128) {
    consume_8_4(s,tid,m0,n0,mh,nh,kt,D);
  }
}

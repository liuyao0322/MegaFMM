// One CTA owns all seven products of one logical SW output tile.
// Original quadrants are guarded independently; no padded TMA crosses a half.
template<class T> __global__ __launch_bounds__(128)
void small_sw(const T* A,const T* B,float* D,int M,int N,int K) {
  using namespace cute;
  constexpr int W=8,BK=64,STRIDE=(64+W)*BK;
  extern __shared__ __align__(128) unsigned char storage[];
  T* sm=reinterpret_cast<T*>(storage);
  constexpr auto la=tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
  constexpr auto lb=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<Int<W>,Int<BK>>{});
  int mh=(M+1)/2,nh=(N+1)/2,kh=(K+1)/2,tid=threadIdx.x;
  int r0=blockIdx.x*W,c0=blockIdx.y*64;
  float d[7][W/2]={};
  for(int q=0;q<kh;q+=BK) {
    for(int i=tid;i<W*BK;i+=128) {
      int r=r0+i/BK,k=q+i%BK;
      float x[4]={};
      if(r<mh && k<kh) {
        if(r<M && k<K)x[0]=float(A[int64_t(r)*K+k]);
        if(r<M && k+kh<K)x[1]=float(A[int64_t(r)*K+k+kh]);
        if(r+mh<M && k<K)x[2]=float(A[int64_t(r+mh)*K+k]);
        if(r+mh<M && k+kh<K)x[3]=float(A[int64_t(r+mh)*K+k+kh]);
      }
      float s1=__fadd_rn(x[2],x[3]),s2=__fsub_rn(s1,x[0]);
      float v[7]={x[0],x[1],__fsub_rn(x[1],s2),x[3],s1,s2,__fsub_rn(x[0],x[2])};
      #pragma unroll
      for(int p=0;p<7;++p) { auto out=make_tensor(make_smem_ptr(sm+p*STRIDE+64*BK),lb);out(i/BK,i%BK)=T(v[p]); }
    }
    for(int i=tid;i<64*BK;i+=128) {
      int c=c0+i%64,k=q+i/64;
      float x[4]={};
      if(c<nh && k<kh) {
        if(k<K && c<N)x[0]=float(B[int64_t(k)*N+c]);
        if(k<K && c+nh<N)x[1]=float(B[int64_t(k)*N+c+nh]);
        if(k+kh<K && c<N)x[2]=float(B[int64_t(k+kh)*N+c]);
        if(k+kh<K && c+nh<N)x[3]=float(B[int64_t(k+kh)*N+c+nh]);
      }
      float t1=__fsub_rn(x[1],x[0]),t2=__fsub_rn(x[3],t1);
      float v[7]={x[0],x[2],x[3],__fsub_rn(t2,x[2]),t1,t2,__fsub_rn(x[3],x[1])};
      #pragma unroll
      for(int p=0;p<7;++p) { auto out=make_tensor(make_smem_ptr(sm+p*STRIDE),la);out(i%64,i/64)=T(v[p]); }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    #pragma unroll
    for(int p=0;p<7;++p) {
      #pragma unroll
      for(int j=0;j<W/2;++j)warpgroup_fence_operand(d[p][j]);
    }
    warpgroup_arrive();
    #pragma unroll
    for(int p=0;p<7;++p) {
      auto a=make_tensor(make_smem_ptr(sm+p*STRIDE),la);
      auto b=make_tensor(make_smem_ptr(sm+p*STRIDE+64*BK),lb);
      auto da=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(a,Shape<_64,_16>{}));
      auto db=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(b,Shape<Int<W>,_16>{}));
      #pragma unroll
      for(int j=0;j<BK/16;++j) {
        if constexpr(std::is_same_v<T,bfloat16_t>)
          SM90_64x8x16_F32BF16BF16_SS<GMMA::Major::MN,GMMA::Major::K>::fma(da(0,0,j),db(0,0,j),d[p][0],d[p][1],d[p][2],d[p][3]);
        else SM90_64x8x16_F32F16F16_SS<GMMA::Major::MN,GMMA::Major::K>::fma(da(0,0,j),db(0,0,j),d[p][0],d[p][1],d[p][2],d[p][3]);
      }
      warpgroup_commit_batch();
    }
    warpgroup_wait<0>();
    #pragma unroll
    for(int p=0;p<7;++p) {
      #pragma unroll
      for(int j=0;j<W/2;++j)warpgroup_fence_operand(d[p][j]);
    }
    __syncthreads();
  }
  #pragma unroll
  for(int j=0;j<W/2;++j) {
    int mn=GMMA::CLayout_64x8{}(tid,j),r=r0+mn/64,c=c0+mn%64;
    float u2=__fadd_rn(d[0][j],d[5][j]),u3=__fadd_rn(u2,d[6][j]),u4=__fadd_rn(u2,d[4][j]);
    if(r<mh && c<nh) {
      if(r<M && c<N)D[int64_t(r)*N+c]=__fadd_rn(d[0][j],d[1][j]);
      if(r<M && c+nh<N)D[int64_t(r)*N+c+nh]=__fadd_rn(u4,d[2][j]);
      if(r+mh<M && c<N)D[int64_t(r+mh)*N+c]=__fsub_rn(u3,d[3][j]);
      if(r+mh<M && c+nh<N)D[int64_t(r+mh)*N+c+nh]=__fadd_rn(u3,d[4][j]);
    }
  }
}
bool use_small(const Problem& p) {return p.m<=1536 && p.n<=8192 && p.k<=1536;}

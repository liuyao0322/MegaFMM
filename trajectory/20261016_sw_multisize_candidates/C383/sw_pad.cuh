struct PadShape {int mh,nh,kh;size_t sa,sb,bytes;};
PadShape pad_shape(const Problem& p) {
  PadShape a{};a.mh=int(((p.m+1)/2+15)/16*16);a.nh=int(((p.n+1)/2+63)/64*64);a.kh=int(((p.k+1)/2+127)/128*128);
  a.sa=size_t(a.mh)*a.kh;a.sb=size_t(a.kh)*a.nh;a.bytes=16*(a.sa+a.sb);return a;
}
bool padded_shape(const Problem& p){return p.m%256 || p.n%128 || p.k%256;}
template<class T> __global__ __launch_bounds__(256)
void pack_padded(const T* A,const T* B,T* workspace,int M,int N,int K,int mh,int nh,int kh) {
  cudaTriggerProgrammaticLaunchCompletion();
  int lm=(M+1)/2,ln=(N+1)/2,lk=(K+1)/2;
  int64_t sa=int64_t(mh)*kh,sb=int64_t(kh)*nh;
  T* rawA=workspace+4*(sa+sb);T* rawB=rawA+4*sa;
  for(int side=0;side<2;++side) {
    int cols=side?nh:kh,halfrows=side?kh:mh;
    int64_t plane=side?sb:sa;
    T* raw=side?rawB:rawA;T* trans=workspace+(side?4*sa:0);
    for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<plane;i+=int64_t(gridDim.x)*256) {
      int r=int(i/cols),c=int(i%cols);float x[4]={};
      if(side==0) {
        if(r<lm && c<lk) {
          x[0]=float(A[int64_t(r)*K+c]);
          if(c+lk<K)x[1]=float(A[int64_t(r)*K+c+lk]);
          if(r+lm<M)x[2]=float(A[int64_t(r+lm)*K+c]);
          if(r+lm<M && c+lk<K)x[3]=float(A[int64_t(r+lm)*K+c+lk]);
        }
      } else if(r<lk && c<ln) {
        x[0]=float(B[int64_t(r)*N+c]);
        if(c+ln<N)x[1]=float(B[int64_t(r)*N+c+ln]);
        if(r+lk<K)x[2]=float(B[int64_t(r+lk)*N+c]);
        if(r+lk<K && c+ln<N)x[3]=float(B[int64_t(r+lk)*N+c+ln]);
      }
      int64_t at=int64_t(r)*(2*cols)+c;
      raw[at]=T(x[0]);raw[at+cols]=T(x[1]);raw[at+int64_t(halfrows)*2*cols]=T(x[2]);raw[at+int64_t(halfrows)*2*cols+cols]=T(x[3]);
      float v[4];
      if(side==0){v[0]=__fadd_rn(x[2],x[3]);v[1]=__fsub_rn(v[0],x[0]);v[2]=__fsub_rn(x[0],x[2]);v[3]=__fsub_rn(x[1],v[1]);}
      else{v[0]=__fsub_rn(x[1],x[0]);v[1]=__fsub_rn(x[3],v[0]);v[2]=__fsub_rn(x[3],x[1]);v[3]=__fsub_rn(v[1],x[2]);}
      #pragma unroll
      for(int j=0;j<4;++j)trans[j*plane+i]=T(v[j]);
    }
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

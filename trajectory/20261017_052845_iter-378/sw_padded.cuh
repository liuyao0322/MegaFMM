struct PadShape {int mh,nh,kh;size_t sa,sb,bytes;};
int padded_width(const Problem& p){return p.m>768?64:16;}
PadShape pad_shape(const Problem& p,int width=0,int bk=128) {
  if(!width)width=padded_width(p);
  PadShape a{};a.mh=int(((p.m+1)/2+width-1)/width*width);a.nh=int(((p.n+1)/2+63)/64*64);a.kh=int(((p.k+1)/2+bk-1)/bk*bk);
  a.sa=size_t(a.mh)*a.kh;a.sb=size_t(a.kh)*a.nh;a.bytes=16*(a.sa+a.sb);return a;
}
bool padded_shape(const Problem& p){return p.m%256 || p.n%128 || p.k%256;}
template<class T,class Input=T> __global__ __launch_bounds__(256)
void pack_padded(const Input* A,const Input* B,T* workspace,int M,int N,int K,int mh,int nh,int kh) {
  cudaTriggerProgrammaticLaunchCompletion();
  int lm=(M+1)/2,ln=(N+1)/2,lk=(K+1)/2;
  int64_t sa=int64_t(mh)*kh,sb=int64_t(kh)*nh;
  T* rawA=workspace+4*(sa+sb);T* rawB=rawA+4*sa;
  for(int side=0;side<2;++side) {
    int cols=side?nh:kh,halfrows=side?kh:mh;
    int64_t plane=side?sb:sa;
    T* raw=side?rawB:rawA;T* trans=workspace+(side?4*sa:0);
    for(int64_t i=2*(int64_t(blockIdx.x)*256+threadIdx.x);i<plane;i+=2*int64_t(gridDim.x)*256) {
      int r=int(i/cols),c=int(i%cols);float x[4][2]={},v[4][2];
      #pragma unroll
      for(int l=0;l<2;++l) {
        int col=c+l;
        if(side==0) {
          if(r<lm && col<lk) {
            x[0][l]=float(A[int64_t(r)*K+col]);
            if(col+lk<K)x[1][l]=float(A[int64_t(r)*K+col+lk]);
            if(r+lm<M)x[2][l]=float(A[int64_t(r+lm)*K+col]);
            if(r+lm<M && col+lk<K)x[3][l]=float(A[int64_t(r+lm)*K+col+lk]);
          }
          v[0][l]=__fadd_rn(x[2][l],x[3][l]);v[1][l]=__fsub_rn(v[0][l],x[0][l]);v[2][l]=__fsub_rn(x[0][l],x[2][l]);v[3][l]=__fsub_rn(x[1][l],v[1][l]);
        } else {
          if(r<lk && col<ln) {
            x[0][l]=float(B[int64_t(r)*N+col]);
            if(col+ln<N)x[1][l]=float(B[int64_t(r)*N+col+ln]);
            if(r+lk<K)x[2][l]=float(B[int64_t(r+lk)*N+col]);
            if(r+lk<K && col+ln<N)x[3][l]=float(B[int64_t(r+lk)*N+col+ln]);
          }
          v[0][l]=__fsub_rn(x[1][l],x[0][l]);v[1][l]=__fsub_rn(x[3][l],v[0][l]);v[2][l]=__fsub_rn(x[3][l],x[1][l]);v[3][l]=__fsub_rn(v[1][l],x[2][l]);
        }
      }
      int64_t at=int64_t(r)*2*cols+c;
      reinterpret_cast<uint32_t*>(raw)[at/2]=pair<T>(x[0][0],x[0][1]);
      reinterpret_cast<uint32_t*>(raw)[(at+cols)/2]=pair<T>(x[1][0],x[1][1]);
      reinterpret_cast<uint32_t*>(raw)[(at+int64_t(halfrows)*2*cols)/2]=pair<T>(x[2][0],x[2][1]);
      reinterpret_cast<uint32_t*>(raw)[(at+int64_t(halfrows)*2*cols+cols)/2]=pair<T>(x[3][0],x[3][1]);
      #pragma unroll
      for(int j=0;j<4;++j)reinterpret_cast<uint32_t*>(trans)[(j*plane+i)/2]=pair<T>(v[j][0],v[j][1]);
    }
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

// One warp evaluates all seven 1x1 logical SW leaf products. Volatile shared
// publication preserves the seven lane results even though only D11 is live.
template<class T> __global__ void scalar_seven(const T* A,const T* B,float* D) {
  __shared__ volatile float product[7];
  int t=threadIdx.x;
  if(t<7) {
    float a=float(*A),b=float(*B);
    float av=(t==0 || t==2 || t==6)?a:t==5?-a:0.0f;
    float bv=(t==0 || t==3 || t==5)?b:t==4?-b:0.0f;
    product[t]=__fmul_rn(av,bv);
  }
  __syncthreads();
  if(t==0)D[0]=__fadd_rn(product[0],product[1]);
}
bool scalar_shape(const Problem& p){return p.m==1 && p.n==1 && p.k==1;}

"""P39: reduce both transformed high-plane maxima before quantizing either."""
from pathlib import Path
h=Path(__file__).resolve().parent
base=h/'p33'
s=(base/'pack.cuh').read_text()
old='float maxima[8];int exponent;int8_t transpose'
new='float maxima[8];int exponent;float maxima_second[8];int exponent_second;int8_t transpose'
assert s.count(old)==1;s=s.replace(old,new)
start=s.index('template<class T,bool SideA,int NativePlane,int QP>')
end=s.index('template<class T,bool SideA,bool Balance,int Group>',start)
original=s[start:end]
quant=original[original.index(' auto quantize='):original.rindex('\n}')]
# Reuse the exact quantizer, fallback, transpose and retirement barrier.
helper='''template<class T,bool SideA,int QP> __device__ __forceinline__
void pair_quant_store(int which,int eh,int8_t* q,
 PairPackShared<SideA>& s,int r0,int c0){
 constexpr int Rows=SideA?64:128,Cols=SideA?128:32,Size=Rows*Cols,PT=Size/256;
 constexpr int Bits=(std::is_same_v<T,cute::half_t> ? 11 : 8);
 int tid=threadIdx.x,el=eh-Bits;
'''+quant+'''
}

template<class T,bool SideA,int NP0,int QP0,int NP1,int QP1> __device__ __forceinline__
void pair_store_both(T* high,int8_t* q,int* exp,
 PairPackShared<SideA>& s,int tile,int r0,int c0){
 constexpr int Rows=SideA?64:128,Cols=SideA?128:32,PT=Rows*Cols/256;
 constexpr int Tiles=(512/Rows)*(512/Cols),Bits=(std::is_same_v<T,cute::half_t> ? 11 : 8);
 int tid=threadIdx.x;float hi0=0,hi1=0;
 #pragma unroll 4
 for(int j=0;j<PT;++j){
  int i=tid+256*j,at=(r0+i/Cols)*512+c0+i%Cols;
  T h0=T(s.values[0][i]),h1=T(s.values[1][i]);
  high[NP0*Plane+at]=h0;high[NP1*Plane+at]=h1;
  hi0=fmaxf(hi0,fabsf(float(h0)));hi1=fmaxf(hi1,fabsf(float(h1)));
 }
 for(int step=16;step;step/=2){
  hi0=fmaxf(hi0,__shfl_xor_sync(0xffffffff,hi0,step));
  hi1=fmaxf(hi1,__shfl_xor_sync(0xffffffff,hi1,step));
 }
 if((tid&31)==0){s.maxima[tid/32]=hi0;s.maxima_second[tid/32]=hi1;}
 __syncthreads();
 if(tid==0){
  float peak0=0,peak1=0;
  for(int w=0;w<8;++w){peak0=fmaxf(peak0,s.maxima[w]);peak1=fmaxf(peak1,s.maxima_second[w]);}
  int e0=peak0>0?int(ceilf(log2f(peak0/127.f))):0;
  int e1=peak1>0?int(ceilf(log2f(peak1/127.f))):0;
  s.exponent=e0;s.exponent_second=e1;
  exp[QP0*Tiles+tile]=e0;exp[(QP0+4)*Tiles+tile]=e0-Bits;
  exp[QP1*Tiles+tile]=e1;exp[(QP1+4)*Tiles+tile]=e1-Bits;
 }
 __syncthreads();
 pair_quant_store<T,SideA,QP0>(0,s.exponent,q,s,r0,c0);
 pair_quant_store<T,SideA,QP1>(1,s.exponent_second,q,s,r0,c0);
}

'''
s=s[:end]+helper+s[end:]
old='if constexpr(Group==0){pair_store<T,SideA,4,0>(0,high,q,exp,s,tile,r0,c0);pair_store<T,SideA,5,1>(1,high,q,exp,s,tile,r0,c0);}'
new='if constexpr(Group==0)pair_store_both<T,SideA,4,0,5,1>(high,q,exp,s,tile,r0,c0);'
assert s.count(old)==1;s=s.replace(old,new)
old='else if constexpr(Group==1){pair_store<T,SideA,6,2>(0,high,q,exp,s,tile,r0,c0);pair_store<T,SideA,(SideA?2:3),3>(1,high,q,exp,s,tile,r0,c0);}'
new='else if constexpr(Group==1)pair_store_both<T,SideA,6,2,(SideA?2:3),3>(high,q,exp,s,tile,r0,c0);'
assert s.count(old)==1;s=s.replace(old,new)
(h/'pack.cuh').write_text(s)
for name in ['corrected.cuh','compare.cu','pow2.cuh']:(h/name).write_text((base/name).read_text())
print('generated P39 dual maxima, original quantizer and transpose, P33 consumer')

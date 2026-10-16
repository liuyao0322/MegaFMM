"""Isolate P60 paired storage from retained P56; leave working reference intact."""
from pathlib import Path
import hashlib,json
h=Path(__file__).resolve().parent;repo=h.parents[2]
out=h/'p60';out.mkdir(exist_ok=True)
s=(h/'p56/pack.cuh').read_text()
s=s.replace('#include <type_traits>','#include <type_traits>\n#include <cutlass/numeric_conversion.h>')
start=s.index('// Native corrections need no tile maxima')
end=s.index('\ntemplate<class T,bool SideA,bool Balance> __global__',start)
s=s[:start]+r'''
// P60 pairs storage/conversion only; each component keeps scalar FP32 RN math.
template<class T> __device__ __forceinline__ cutlass::Array<T,2>
paired_native_convert(cutlass::Array<float,2> const& v){
 return cutlass::NumericArrayConverter<T,float,2,cutlass::FloatRoundStyle::round_to_nearest>{}(v);
}
template<class T,bool Aligned> __device__ __forceinline__ cutlass::Array<float,2>
paired_native_load(const T* p){
 cutlass::Array<T,2> native;
 if constexpr(Aligned){
  uint32_t word=*reinterpret_cast<const uint32_t*>(p);
  native=*reinterpret_cast<const cutlass::Array<T,2>*>(&word);
 }else{native[0]=p[0];native[1]=p[1];}
 return {float(native[0]),float(native[1])};
}
template<class T,int NativePlane,int QP> __device__ __forceinline__
void paired_native_store(cutlass::Array<float,2> const& v,int at,T* high,T* q){
 auto native=paired_native_convert<T>(v);
 *reinterpret_cast<uint32_t*>(high+NativePlane*Plane+at)=*reinterpret_cast<const uint32_t*>(&native);
 if constexpr(QP<4){
  constexpr float scale=(std::is_same_v<T,cute::half_t> ? 2048.f : 256.f);
  cutlass::Array<float,2> residual;
  #pragma unroll
  for(int k=0;k<2;++k)residual[k]=__fmul_rn(__fsub_rn(v[k],float(native[k])),scale);
  auto low=paired_native_convert<T>(residual);
  *reinterpret_cast<uint32_t*>(q+QP*Plane+at)=*reinterpret_cast<const uint32_t*>(&low);
 }
}

template<class T,bool SideA,bool Balance,int Group> __device__ __forceinline__
void pack_pair(const T* __restrict__ X,T* __restrict__ high,T* __restrict__ q,int* __restrict__ exp,
 const int* __restrict__ inside,PairPackShared<SideA>& s){
 if constexpr(Group!=0)return;
 constexpr int Rows=SideA?16:32,Cols=SideA?128:32,PT=Rows*Cols/512,Tiles=(512/Rows)*(512/Cols);
 int tile=blockIdx.x%Tiles,tid=threadIdx.x,lane=tid%32,warp=tid/32;
 int r0=tile/(512/Cols)*Rows,c0=tile%(512/Cols)*Cols;
 auto load=[&](auto fast,auto aligned,cutlass::Array<float,2> sf0,cutlass::Array<float,2> sf1){
  #pragma unroll 4
  for(int j=0;j<PT;++j){
   int i=2*(tid+256*j),r=r0+i/Cols,c=c0+i%Cols,at=r*1024+c;
   auto a=paired_native_load<T,decltype(aligned)::value>(X+at);
   auto b=paired_native_load<T,decltype(aligned)::value>(X+at+512);
   auto cc=paired_native_load<T,decltype(aligned)::value>(X+at+512*1024);
   auto d=paired_native_load<T,decltype(aligned)::value>(X+at+512*1024+512);
   float bf0=0,bf1=0;
   if constexpr(Balance&&decltype(fast)::value&&!SideA){
    int source=lane/16+2*j;
    bf0=__shfl_sync(0xffffffff,sf0[0],source);bf1=__shfl_sync(0xffffffff,sf1[0],source);
   }
   #pragma unroll
   for(int k=0;k<2;++k){
    if constexpr(Balance){
     if constexpr(decltype(fast)::value){
      float f0=SideA?sf0[k]:bf0,f1=SideA?sf1[k]:bf1;
      a[k]=__fmul_rn(a[k],f0);d[k]=__fmul_rn(d[k],f1);
      if constexpr(SideA){b[k]=__fmul_rn(b[k],f1);cc[k]=__fmul_rn(cc[k],f0);}
      else{b[k]=__fmul_rn(b[k],f0);cc[k]=__fmul_rn(cc[k],f1);}
     }else{
      int e0=SideA?-inside[c+k]:inside[r],e1=SideA?-inside[c+k+512]:inside[r+512];
      a[k]=pair_pow2(a[k],e0);d[k]=pair_pow2(d[k],e1);
      if constexpr(SideA){b[k]=pair_pow2(b[k],e1);cc[k]=pair_pow2(cc[k],e0);}
      else{b[k]=pair_pow2(b[k],e0);cc[k]=pair_pow2(cc[k],e1);}
     }
    }
   }
   int target=r*512+c;
   paired_native_store<T,0,8>(a,target,high,q);
   paired_native_store<T,1,8>(SideA?b:cc,target,high,q);
   paired_native_store<T,(SideA?3:2),8>(d,target,high,q);
   cutlass::Array<float,2> v1,v2,v3,v4;
   #pragma unroll
   for(int k=0;k<2;++k){
    if constexpr(SideA){
     v1[k]=__fadd_rn(cc[k],d[k]);v2[k]=__fsub_rn(v1[k],a[k]);
     v3[k]=__fsub_rn(a[k],cc[k]);v4[k]=__fsub_rn(b[k],v2[k]);
    }else{
     v1[k]=__fsub_rn(b[k],a[k]);v2[k]=__fsub_rn(d[k],v1[k]);
     v3[k]=__fsub_rn(d[k],b[k]);v4[k]=__fsub_rn(v2[k],cc[k]);
    }
   }
   paired_native_store<T,4,0>(v1,target,high,q);paired_native_store<T,5,1>(v2,target,high,q);
   paired_native_store<T,6,2>(v3,target,high,q);paired_native_store<T,(SideA?2:3),3>(v4,target,high,q);
  }
 };
 cutlass::Array<float,2> sf0{0,0},sf1{0,0};bool normal=true;
 if constexpr(Balance){
  #pragma unroll
  for(int k=0;k<(SideA?2:1);++k){
   int pos=SideA?c0+2*(tid%64)+k:r0+2*warp+(lane%2)+16*((lane/2)%2);
   int e0=SideA?-inside[pos]:inside[pos],e1=SideA?-inside[pos+512]:inside[pos+512];
   normal=normal&&e0>=-126&&e0<=127&&e1>=-126&&e1<=127;
   if(e0>=-126&&e0<=127&&e1>=-126&&e1<=127){sf0[k]=__int_as_float((e0+127)<<23);sf1[k]=__int_as_float((e1+127)<<23);}
  }
  if constexpr(!SideA)normal=__all_sync(0xffffffff,normal);
 }
 auto choose_alignment=[&](auto fast){
  if((reinterpret_cast<uintptr_t>(X)&3u)==0)load(fast,std::true_type{},sf0,sf1);
  else load(fast,std::false_type{},sf0,sf1);
 };
 if constexpr(Balance){if(normal)choose_alignment(std::true_type{});else choose_alignment(std::false_type{});}
 else choose_alignment(std::false_type{});
 asm volatile("fence.proxy.async.global;" ::: "memory");
}
''' +s[end:]
# Original direct_native_store remains available to historical diagnostic consumers.
old=(h/'p56/pack.cuh').read_text()
a=old.index('template<class T,int NativePlane,int QP>');b=old.index('\ntemplate<class T,bool SideA,bool Balance,int Group>',a)
s=s.replace('// P60 pairs storage/conversion only;',old[a:b]+'\n// P60 pairs storage/conversion only;')
(out/'pack.cuh').write_text(s)
for name in ['corrected.cuh','compare.cu','pow2.cuh']:(out/name).write_bytes((h/'p56'/name).read_bytes())
assert s[s.index('template<class T,bool SideA,bool Balance> __global__'):]==old[old.index('template<class T,bool SideA,bool Balance> __global__'):]
files=[Path(__file__),h/'p60_proposal_manifest.json']+list(out.iterdir())
(h/'p60_source_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print('P60 sources generated; original cooperative reduction/launch and working P56 unchanged')

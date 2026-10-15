"""P47: remove obsolete FP32 transform staging, keep P46 resource requests.

Generate only; execution/acceptance belongs to the new candidate. The first
control deliberately retains the old 65536B shared request to separate direct
stores from a later resource-capacity experiment.
"""
import argparse
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p46'
ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=h);args=ap.parse_args()
out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
s=(base/'pack.cuh').read_text()
start=s.index('template<class T,bool SideA,int NativePlane,int QP>')
end=s.index('template<class T,bool SideA,bool Balance,int Group>',start)
s=s[:start]+'''// Native corrections need no tile maxima or cross-thread transform exchange.
template<class T,int NativePlane,int QP> __device__ __forceinline__
void direct_native_store(float v,int at,T* high,T* q){
 T native=T(v);high[NativePlane*Plane+at]=native;
 if constexpr(QP<4){
  constexpr float scale=(std::is_same_v<T,cute::half_t> ? 2048.f : 256.f);
  q[QP*Plane+at]=T(__fmul_rn(__fsub_rn(v,float(native)),scale));
 }
}

'''+s[end:]
old='''   if constexpr(Group==2){
    high[r*512+c]=T(a);high[Plane+r*512+c]=T(SideA?b:cc);s.values[0][i]=d;
   }else if constexpr(SideA){
    float s1=__fadd_rn(cc,d),s2=__fsub_rn(s1,a);
    if constexpr(Group==0){s.values[0][i]=s1;s.values[1][i]=s2;}else{s.values[0][i]=__fsub_rn(a,cc);s.values[1][i]=__fsub_rn(b,s2);}
   }else{
    float t1=__fsub_rn(b,a),t2=__fsub_rn(d,t1);
    if constexpr(Group==0){s.values[0][i]=t1;s.values[1][i]=t2;}else{s.values[0][i]=__fsub_rn(d,b);s.values[1][i]=__fsub_rn(t2,cc);}
   }'''
new='''   int target=r*512+c;
   if constexpr(Group==2){
    direct_native_store<T,0,8>(a,target,high,q);
    direct_native_store<T,1,8>(SideA?b:cc,target,high,q);
    direct_native_store<T,(SideA?3:2),8>(d,target,high,q);
   }else if constexpr(SideA){
    float s1=__fadd_rn(cc,d),s2=__fsub_rn(s1,a);
    if constexpr(Group==0){
     direct_native_store<T,4,0>(s1,target,high,q);direct_native_store<T,5,1>(s2,target,high,q);
    }else{
     direct_native_store<T,6,2>(__fsub_rn(a,cc),target,high,q);direct_native_store<T,2,3>(__fsub_rn(b,s2),target,high,q);
    }
   }else{
    float t1=__fsub_rn(b,a),t2=__fsub_rn(d,t1);
    if constexpr(Group==0){
     direct_native_store<T,4,0>(t1,target,high,q);direct_native_store<T,5,1>(t2,target,high,q);
    }else{
     direct_native_store<T,6,2>(__fsub_rn(d,b),target,high,q);direct_native_store<T,3,3>(__fsub_rn(t2,cc),target,high,q);
    }
   }'''
assert s.count(old)==1;s=s.replace(old,new)
old=''' __syncthreads();
 if constexpr(Group==0)pair_store_both<T,SideA,4,0,5,1>(high,q,exp,s,tile,r0,c0);
 else if constexpr(Group==1)pair_store_both<T,SideA,6,2,(SideA?2:3),3>(high,q,exp,s,tile,r0,c0);
 else pair_store<T,SideA,(SideA?3:2),8>(0,high,q,exp,s,tile,r0,c0);
'''
assert s.count(old)==1;s=s.replace(old,'')
s=s.replace('// Paired FP32 transforms stay shared. Native correction B is MN-major and\n// correction A is K-major, so neither side needs a physical transpose.',
 '''// P47 keeps the historical shared allocation for an isolated staging control.
// pack_pair writes native highs/lows directly; only inside maxima use scratch.''')
(out/'pack.cuh').write_text(s)
for name in ('corrected.cuh','compare.cu','pow2.cuh'):(out/name).write_bytes((base/name).read_bytes())
print('P47 generated: direct native high/low stores, no FP32 staging/barrier; shared requests and P46 consumer unchanged')

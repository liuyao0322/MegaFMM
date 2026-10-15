"""N2: all native leaves once per K64, from the immutable N1 source contract."""
from pathlib import Path
r=Path(__file__).resolve().parent;out=r/'n2';out.mkdir(exist_ok=True)
s=(r/'prototype.cuh').read_text().replace('#include "mma.cuh"','#include "../mma.cuh"').replace('#include "inside_reference.cuh"','#include "../inside_reference.cuh"')
s=s.replace('LeafBytes=24576,Slots=2','LeafBytes=135168,Slots=1').replace('sizeof(Shared)==147584','sizeof(Shared)==184448')
a=s.index('template<class T,bool SideA> __device__ __forceinline__ void make_leaf');b=s.index('template<class T> __device__ __forceinline__ void consume(',a)
s=s[:a]+'''template<class T,int P,int L,int Count> __device__ __forceinline__ void store_native(float v,T* target,int at){
 T high=T(v);target[P*Count+at]=high;
 if constexpr(L<4){
  constexpr float scale=(std::is_same_v<T,cute::half_t> ? 2048.f : 256.f);
  target[(7+L)*Count+at]=T(__fmul_rn(__fsub_rn(v,float(high)),scale));
 }
}
template<class T,bool SideA> __device__ __forceinline__ void make_all(Shared& s,int k,const int* inside){
 using namespace cute;
 constexpr auto layout=[](){if constexpr(SideA)return tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_32,_64>{});else return tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,_64>{});}();
 constexpr int Count=SideA?2048:4096;
 const T* raw=reinterpret_cast<const T*>(s.raw[0]+(SideA?0:16384));
 T* target=reinterpret_cast<T*>(s.leaf[0]+(SideA?90112:0));
 #pragma unroll 4
 for(int i=threadIdx.x;i<Count;i+=128){
  int row=SideA?i/64:i%64,ki=SideA?i%64:i/64,at=as_position_independent_swizzle_layout(layout)(make_coord(row,ki));
  int e0=inside[k*64+ki],e1=inside[k*64+ki+512];if constexpr(SideA){e0=-e0;e1=-e1;}
  float a=pow2(float(raw[at]),e0),d=pow2(float(raw[3*Count+at]),e1);
  float b=pow2(float(raw[Count+at]),SideA?e1:e0),c=pow2(float(raw[2*Count+at]),SideA?e0:e1);
  store_native<T,0,8,Count>(a,target,at);store_native<T,1,8,Count>(SideA?b:c,target,at);
  if constexpr(SideA){
   float s1=__fadd_rn(c,d),s2=__fsub_rn(s1,a);
   store_native<T,3,8,Count>(d,target,at);store_native<T,4,0,Count>(s1,target,at);
   store_native<T,5,1,Count>(s2,target,at);store_native<T,6,2,Count>(__fsub_rn(a,c),target,at);
   store_native<T,2,3,Count>(__fsub_rn(b,s2),target,at);
  }else{
   float t1=__fsub_rn(b,a),t2=__fsub_rn(d,t1);
   store_native<T,2,8,Count>(d,target,at);store_native<T,4,0,Count>(t1,target,at);
   store_native<T,5,1,Count>(t2,target,at);store_native<T,6,2,Count>(__fsub_rn(d,b),target,at);
   store_native<T,3,3,Count>(__fsub_rn(t2,c),target,at);
  }
 }
}
template<class T> __device__ __forceinline__ void produce(Shared& s,const CUtensorMap* a,const CUtensorMap* b,const int* inside,int m0,int n0){
 if(threadIdx.x==0)load_raw(s,a,b,0,m0,n0);
 #pragma unroll 1
 for(int k=0;k<8;++k){
  Barrier::wait(&s.raw_ready[0],k%2);asm volatile("" ::: "memory");
  if(k){Barrier::wait(&s.leaf_free[0],(k-1)%2);asm volatile("" ::: "memory");}
  make_all<T,false>(s,k,inside);make_all<T,true>(s,k,inside);
  proxy_fence();Barrier::arrive(&s.leaf_ready[0]);
  // All raw reads complete before the issuer overwrites the single raw slot.
  proxy_fence();producer_sync();
  if(threadIdx.x==0&&k+1<8)load_raw(s,a,b,k+1,m0,n0);
 }
}

'''+s[b:]
s=s.replace('s.leaf[0]+16384','s.leaf[0]+90112')
s=s.replace('int p=product(seq%7),slot=seq%Slots;','int p=product(seq%7);')
s=s.replace('Barrier::wait(&s.leaf_ready[slot],(seq/Slots)%2);asm volatile("" ::: "memory");','if(seq%7==0)Barrier::wait(&s.leaf_ready[0],(seq/7)%2);asm volatile("" ::: "memory");')
s=s.replace('warpgroup_arrive();uint64_t offset=slot*LeafBytes/16;','warpgroup_arrive();uint64_t ao=p*8192/16,bo=p*4096/16;')
s=s.replace('uint64_t(da(0,0,inner))+offset+8192/16','uint64_t(da(0,0,inner))+low_plane(p)*8192/16')
s=s.replace('uint64_t(db(0,0,inner))+offset+4096/16','uint64_t(db(0,0,inner))+low_plane(p)*4096/16')
s=s.replace('uint64_t(da(0,0,inner))+offset','uint64_t(da(0,0,inner))+ao').replace('uint64_t(db(0,0,inner))+offset','uint64_t(db(0,0,inner))+bo')
s=s.replace('if((threadIdx.x&31)==0)Barrier::arrive(&s.leaf_free[slot]);','if(seq%7==6&&(threadIdx.x&31)==0)Barrier::arrive(&s.leaf_free[0]);')
s=s.replace('int p=product(seq%7),k=seq/7,slot=seq%Slots;','int p=product(seq%7),k=seq/7;')
s=s.replace('s.leaf[slot]+16384','s.leaf[0]+90112+p*4096').replace('s.leaf[slot]+20480','s.leaf[0]+90112+low_plane(p)*4096')
s=s.replace('reinterpret_cast<const uint16_t*>(s.leaf[slot]);','reinterpret_cast<const uint16_t*>(s.leaf[0]+p*8192);').replace('s.leaf[slot]+8192','s.leaf[0]+low_plane(p)*8192')
assert 'slot' not in s[s.index('void consume('):s.index('template<class T,bool Probe> __global__')]
(out/'prototype.cuh').write_text(s)
for name in ['compare.cu','probe.cu']:
 s=(r/name).read_text().replace('../../../tests/','../../../../tests/').replace('../direct_b128_w1.cuh','../../direct_b128_w1.cuh').replace('../int8_hardware/p47/pack.cuh','../../int8_hardware/p47/pack.cuh')
 if name=='probe.cu':s=s.replace('103001+trial*359+pattern*67','104003+trial*367+pattern*71')
 (out/name).write_text(s)
for name in ['build.sh','build_probe.sh']:
 s=(r/name).read_text().replace('/../../.."','/../../../.."').replace('root=research/precision_20261016/fused_native','root=research/precision_20261016/fused_native/n2')
 if name=='build_probe.sh':s=s.replace('\nnvcc ', '\nmkdir -p \"$root/bin\"\nnvcc ')
 (out/name).write_text(s)

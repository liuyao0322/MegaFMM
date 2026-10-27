from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','62142fc:solution/sw_pipeline.cu'],text=True)
pos=s.index('template<class T> __device__ __forceinline__ void tensor_chain')
fn='''template<class T,bool IsB> __device__ __forceinline__ void transform_pack(const T* input,T* packed,unsigned worker){
 for(unsigned local=2*worker;local<2048;local+=512){
  unsigned i=(blockIdx.y*gridDim.x+blockIdx.x)*2048+local;
  unsigned at=(i/512)*1024+i%512;
  auto v=reinterpret_cast<const uint32_t*>(input);
  uint32_t bits[4]={v[at/2],v[(at+512)/2],v[(at+512*1024)/2],v[(at+512*1024+512)/2]},words[4]={};
  #pragma unroll
  for(int lane=0;lane<2;++lane){
   float v11=float(T::bitcast(uint16_t(bits[0]>>(16*lane)))),v12=float(T::bitcast(uint16_t(bits[1]>>(16*lane))));
   float v21=float(T::bitcast(uint16_t(bits[2]>>(16*lane)))),v22=float(T::bitcast(uint16_t(bits[3]>>(16*lane))));
   float values[4];
   if constexpr(IsB){float t1=__fsub_rn(v12,v11),t2=__fsub_rn(v22,t1);values[0]=t1;values[1]=t2;values[2]=__fsub_rn(v22,v12);values[3]=__fsub_rn(t2,v21);}
   else{float s1=__fadd_rn(v21,v22),s2=__fsub_rn(s1,v11);values[0]=s1;values[1]=s2;values[2]=__fsub_rn(v11,v21);values[3]=__fsub_rn(v12,s2);}
   #pragma unroll
   for(int j=0;j<4;++j){T value=T(values[j]);words[j]|=uint32_t(value.raw())<<(16*lane);}
  }
  #pragma unroll
  for(int j=0;j<4;++j)reinterpret_cast<uint32_t*>(packed)[(((IsB?4:0)+j)*Plane+i)/2]=words[j];
 }
}

'''
s=s[:pos]+fn+s[pos:]
a=s.index('  for(int local=2*worker;');b=s.index('    asm volatile("fence.proxy.async.global;',a)
s=s[:a]+'''    if(worker<256)transform_pack<T,false>(A,packed,worker);
    else transform_pack<T,true>(B,packed,worker-256);
'''+s[b:]
Path('solution/sw_pipeline.cu').write_text(s)
for name in ['internal.h','plan.cpp']:
 s=subprocess.check_output(['git','show','62142fc:solution/'+name],text=True)
 if name=='plan.cpp':s=s.replace('st=compact4planes_balanced_async_p1p2','st=compact4planes_split256S_256T_async_p1p2')
 Path('solution/'+name).write_text(s)

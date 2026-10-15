"""P29 controlled launch/barrier fusion from frozen, validated P27.

Duplicate the exact reduction and pack bodies, changing only logical indexing
and phase ordering. Keep the original diagnostic kernels and native consumer.
"""
from pathlib import Path

h = Path(__file__).resolve().parent
base = h / 'p27'
s = (base / 'pack.cuh').read_text()
s = s.replace('#include <cuda_runtime.h>', '#include <cuda_runtime.h>\n#include <cooperative_groups.h>')
partial = s.split('void inside_partial(const T* A,const T* B,float* partial){', 1)[1].split('\n}', 1)[0]
partial = partial.replace('int slab=blockIdx.x,chunk=blockIdx.y,', 'int slab=blockIdx.x%8,chunk=blockIdx.x/8,')
partial = partial.replace('__shared__ float aa[256],bb[128];', 'float* aa=reinterpret_cast<float*>(scratch_buffer);float* bb=aa+256;')
finalize = s.split('void inside_finalize(const float* partial,int* exponent){', 1)[1].split('\n}', 1)[0]
finalize = finalize.replace('exponent[k]', 'inside[k]')
joint_start = s.index('void pack_joint(')
joint = s[joint_start:].split(' extern __shared__ __align__(128) unsigned char scratch_buffer[];', 1)[1].split('\n}\n', 1)[0]
cooperative = '''
// All288CTAs participate in both grid barriers. Scratch is reused only after
// the grid has published partial maxima and then finalized all1024 exponents.
template<class T> __global__ __launch_bounds__(256,1)
void preprocess_cooperative(const T* __restrict__ A,const T* __restrict__ B,
 T* __restrict__ ha,T* __restrict__ hb,int8_t* __restrict__ qa,int8_t* __restrict__ qb,
 int* __restrict__ ea,int* __restrict__ eb,int* __restrict__ inside,float* __restrict__ partial){
 constexpr bool Balance=true;
 extern __shared__ __align__(128) unsigned char scratch_buffer[];
 auto grid=cooperative_groups::this_grid();
 if(blockIdx.x<256){
''' + partial + '''
 }
 grid.sync();
 if(blockIdx.x<128){
''' + finalize + '''
 }
 grid.sync();
''' + joint + '''
}

template<class T> void init_cooperative_preprocess(){
 constexpr size_t smem=sizeof(PairPackShared<true>);
 check(cudaFuncSetAttribute(preprocess_cooperative<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem),"cooperative shared");
 int device=0,supported=0,sms=0,active=0;
 check(cudaGetDevice(&device),"cooperative device");
 check(cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,device),"cooperative support");
 check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"cooperative SM count");
 check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,preprocess_cooperative<T>,256,smem),"cooperative occupancy");
 printf("COOPERATIVE_INIT dtype=%s supported=%d sm_count=%d active_blocks_per_sm=%d grid=288 shared_bytes=%zu\\n",
  (std::is_same_v<T,cute::half_t> ? "fp16" : "bf16"),supported,sms,active,smem);
 if(!supported||active*sms<288)throw std::runtime_error("Cooperative preprocess requires resident capacity for288CTAs");
}
'''
position = s.index('template<class T,bool Balance> void preprocess(')
s = s[:position] + cooperative + '\n' + s[position:]
old = ''' if constexpr(Balance){inside_partial<<<dim3(8,32),256,0,stream>>>(A,B,w.partial);inside_finalize<<<128,256,0,stream>>>(w.partial,w.inside);}
 pack_joint<T,Balance><<<288,256,sizeof(PairPackShared<true>),stream>>>(
  A,B,static_cast<T*>(w.ha),static_cast<T*>(w.hb),w.qa,w.qb,w.ea,w.eb,w.inside);'''
new = ''' if constexpr(Balance){
  T* ha=static_cast<T*>(w.ha);T* hb=static_cast<T*>(w.hb);
  void* args[]={&A,&B,&ha,&hb,&w.qa,&w.qb,&w.ea,&w.eb,&w.inside,&w.partial};
  check(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(preprocess_cooperative<T>),dim3(288),dim3(256),args,sizeof(PairPackShared<true>),stream),"cooperative preprocess launch");
 }else{
  pack_joint<T,false><<<288,256,sizeof(PairPackShared<true>),stream>>>(
   A,B,static_cast<T*>(w.ha),static_cast<T*>(w.hb),w.qa,w.qb,w.ea,w.eb,w.inside);
 }'''
assert s.count(old) == 1
s = s.replace(old,new)
(h / 'pack.cuh').write_text(s)
c = (base / 'corrected.cuh').read_text()
c = c.replace('void init(){', 'void init(){\n init_cooperative_preprocess<cute::half_t>();init_cooperative_preprocess<cute::bfloat16_t>();')
(h / 'corrected.cuh').write_text(c)
for name in ['compare.cu', 'pow2.cuh']:
    (h / name).write_text((base / name).read_text())
print('generated cooperative preprocess from frozen P27')

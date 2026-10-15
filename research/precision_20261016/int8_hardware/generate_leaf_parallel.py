"""Prepared P53 generator, from frozen P51; apply only as a new candidate."""
from pathlib import Path
r=Path(__file__).resolve().parent
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:
 s=(r/'p51'/name).read_text()
 if name=='pack.cuh':
  lines=s.splitlines(keepends=True);found=0
  for i,line in enumerate(lines):
   if line.startswith('constexpr size_t WorkBytes='):
    lines[i]=line.replace('WorkBytes','BaseWorkBytes')+'constexpr size_t LeafBytes=7*Plane*sizeof(float),WorkBytes=BaseWorkBytes+LeafBytes;\n';found+=1
  assert found==1;s=''.join(lines)
 elif name=='corrected.cuh':
  prefix=s[:s.index('template<class T> __device__ __forceinline__ void consume')]
  prefix=prefix.replace('namespace int8_research {','namespace int8_research {\nconstexpr int ConsumerStages=2;',1)
  prefix=prefix.replace('operand[Stages][49152];uint64_t ready[Stages],free[Stages]','operand[ConsumerStages][49152];uint64_t ready[ConsumerStages],free[ConsumerStages]')
  prefix=prefix.replace('sizeof(Shared)==196736','sizeof(Shared)==98432')
  tail=s[s.index('void encode('):]
  tail=tail.replace('Work w(workspace);CUtensorMap ha,hb;','Work w(workspace);CUtensorMap ha,hb;float* leaf=reinterpret_cast<float*>(static_cast<char*>(workspace)+BaseWorkBytes);')
  assert tail.count('<<<dim3(16,8),256,sizeof(Shared),stream>>>(ha,hb,w.ea,w.eb,D);')==2
  tail=tail.replace('<<<dim3(16,8),256,sizeof(Shared),stream>>>(ha,hb,w.ea,w.eb,D);','<<<dim3(16,8,7),256,sizeof(Shared),stream>>>(ha,hb,w.ea,w.eb,leaf);')
  tail=tail.replace(' check(cudaPeekAtLastError(),"INT8 corrected launch");',' check(cudaPeekAtLastError(),"native leaf launch");\n recombine<<<1024,256,0,stream>>>(leaf,D);\n check(cudaPeekAtLastError(),"native recombine launch");')
  helper='''template<class T> void log_leaf_capacity(){
 int device=0,sms=0,active=0;
 check(cudaGetDevice(&device),"leaf device");
 check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"leaf SM count");
 check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,corrected<T>,256,sizeof(Shared)),"leaf occupancy");
 printf("LEAF_CONSUMER_INIT dtype=%s sm_count=%d active_blocks_per_sm=%d grid=896 shared_bytes=%zu\\n",
  (std::is_same_v<T,cute::half_t> ? "fp16" : "bf16"),sms,active,sizeof(Shared));
 if(active<=0)throw std::runtime_error("No resident leaf CTA capacity");
}
'''
  tail=tail.replace('void init(){',helper+'void init(){',1)
  for dtype,t in [('BF','bfloat16_t'),('FP','half_t')]:
   old=f'check(cudaFuncSetAttribute(corrected<cute::{t}>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"INT8 {dtype} carveout");'
   assert old in tail;tail=tail.replace(old,old+f'\n   log_leaf_capacity<cute::{t}>();')
  s=prefix+(r/'leaf_parallel_consumer.inc').read_text()+tail
 (r/name).write_text(s)

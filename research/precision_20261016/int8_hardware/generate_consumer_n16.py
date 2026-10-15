"""Prepared P52 consumer-only geometry/ring generator, from frozen P51."""
from pathlib import Path
r=Path(__file__).resolve().parent
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:
 s=(r/'p51'/name).read_text()
 if name=='corrected.cuh':
  s=s.replace('Stages','ConsumerStages')
  s=s.replace('namespace int8_research {','namespace int8_research {\nconstexpr int ConsumerStages=2;',1)
  changes={
   'operand[ConsumerStages][49152]':'operand[ConsumerStages][40960]',
   'sizeof(Shared)==196736':'sizeof(Shared)==82048',
   'SM90_64x32x16_':'SM90_64x16x16_',
   'Shape<_32,_128>':'Shape<_16,_128>',
   'Shape<_32,_16>':'Shape<_16,_16>',
   'float c[4][16]':'float c[4][8]',
   'float d[16]={},r[16]={}':'float d[8]={},r[8]={}',
   'j<16':'j<8',
   'slot*49152/16':'slot*40960/16',
   'uint64_t(db(0,0,inner))+offset+8192/16':'uint64_t(db(0,0,inner))+offset+4096/16',
   'CLayout_64x32':'CLayout_64x16',
   'n0=blockIdx.x*32':'n0=blockIdx.x*16',
   'int bytes=p<2?24576:p==2?32768:p==3?40960:49152;':'int bytes=p<2?20480:p==2?24576:p==3?36864:40960;',
   's.operand[slot]+40960':'s.operand[slot]+36864',
   'box[1]=32;':'box[1]=16;',
   'dim3(16,8)':'dim3(32,8)',
  }
  for old,new in changes.items():
   assert old in s,old
   s=s.replace(old,new)
  old=','.join(f'd[{j}]' for j in range(16))+',scale'
  new=','.join(f'd[{j}]' for j in range(8))+',scale'
  assert s.count(old)==2
  s=s.replace(old,new)
  helper='''template<class T> void log_consumer_capacity(){
 int active=0,device=0,sms=0;
 check(cudaGetDevice(&device),"consumer device");
 check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"consumer SM count");
 check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,corrected<T>,256,sizeof(Shared)),"consumer occupancy");
 printf("CONSUMER_INIT dtype=%s sm_count=%d active_blocks_per_sm=%d grid=256 shared_bytes=%zu\\n",
  (std::is_same_v<T,cute::half_t>?"fp16":"bf16"),sms,active,sizeof(Shared));
 if(active<=0)throw std::runtime_error("No resident consumer CTA capacity");
}
'''
  s=s.replace('void init(){',helper+'void init(){',1)
  for dtype,t in [('BF','bfloat16_t'),('FP','half_t')]:
   old=f'check(cudaFuncSetAttribute(corrected<cute::{t}>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"INT8 {dtype} carveout");'
   assert old in s
   s=s.replace(old,old+f'\n   log_consumer_capacity<cute::{t}>();')
 (r/name).write_text(s)

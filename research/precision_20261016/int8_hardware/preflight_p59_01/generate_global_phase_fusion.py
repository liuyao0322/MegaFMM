"""Isolate P59: P58 preparation helper, grid publication, P56 consumer."""
from pathlib import Path
import hashlib,json
h=Path(__file__).resolve().parent;repo=h.parents[2];out=h/'p59';out.mkdir(exist_ok=True)
pack=(h/'p58/pack.cuh').read_text()
start=pack.index('// All128CTAs participate')
end=pack.index('\ntemplate<class T> void init_cooperative_preprocess',start)
kernel=pack[start:end];body=kernel[kernel.index(' constexpr bool Balance=true;'):]
body=body.replace(' extern __shared__ __align__(128) unsigned char scratch_buffer[];\n','')
signature='''template<class T> __device__ __forceinline__
void prepare_resident(const T* __restrict__ A,const T* __restrict__ B,
 T* __restrict__ ha,T* __restrict__ hb,T* __restrict__ qa,T* __restrict__ qb,
 int* __restrict__ ea,int* __restrict__ eb,int* __restrict__ inside,float* __restrict__ partial,
 unsigned char* scratch_buffer){
'''
helper=signature+body
wrapper=kernel[:kernel.index(' constexpr bool Balance=true;')]+''' extern __shared__ __align__(128) unsigned char scratch_buffer[];
 prepare_resident<T>(A,B,ha,hb,qa,qb,ea,eb,inside,partial,scratch_buffer);
}
'''
pack=pack[:start]+helper+'\n'+wrapper+pack[end:]
c=(h/'p56/corrected.cuh').read_text()
a=c.index('template<class T> __global__ __launch_bounds__(256,1)\nvoid corrected(')
b=c.index('\nvoid encode(',a)
fused=c[a:b].replace('void corrected(','void fused_global(')
fused=fused.replace(' const int* ea,const int* eb,float* D){',' const T* A,const T* B,Work w,float* D){')
needle=' auto& s=*reinterpret_cast<Shared*>(buffer);'
assert fused.count(needle)==1
fused=fused.replace(needle,''' prepare_resident<T>(A,B,static_cast<T*>(w.ha),static_cast<T*>(w.hb),
  static_cast<T*>(w.qa),static_cast<T*>(w.qb),w.ea,w.eb,w.inside,w.partial,buffer);
 cooperative_groups::this_grid().sync();
'''+needle)
fused=fused.replace('m0=blockIdx.y*64,n0=blockIdx.x*32','m0=(blockIdx.x/16)*64,n0=(blockIdx.x%16)*32')
fused=fused.replace('consume<T>(s,tid,m0,n0,ea,eb,D)','consume<T>(s,tid,m0,n0,w.ea,w.eb,D)')
c=c[:b]+'\n'+fused+c[b:]
i=c.index('void init(){')
init='''template<class T> void init_fused_global(){
 check(cudaFuncSetAttribute(fused_global<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)),"fused global shared");
 check(cudaFuncSetAttribute(fused_global<T>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"fused global carveout");
 int device=0,supported=0,sms=0,active=0;cudaFuncAttributes attr{};
 check(cudaGetDevice(&device),"fused device");
 check(cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,device),"fused support");
 check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"fused SM count");
 check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fused_global<T>,256,sizeof(Shared)),"fused occupancy");
 check(cudaFuncGetAttributes(&attr,fused_global<T>),"fused attributes");
 printf("FUSED_INIT dtype=%s supported=%d sm_count=%d active_blocks_per_sm=%d grid=128 shared_bytes=%zu regs=%d\\n",
  (std::is_same_v<T,cute::half_t>?"fp16":"bf16"),supported,sms,active,sizeof(Shared),attr.numRegs);
 if(!supported||active*sms<128)throw std::runtime_error("Fused global requires actual resident capacity128");
}
template<class T> void launch_fused_global(CUtensorMap& ha,CUtensorMap& hb,const T* A,const T* B,Work w,float* D,cudaStream_t stream){
 void* args[]={&ha,&hb,&A,&B,&w,&D};
 check(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(fused_global<T>),dim3(128),dim3(256),args,sizeof(Shared),stream),"fused global launch");
}
'''
c=c[:i]+init+c[i:]
c=c.replace('void init(){','void init(){\n init_fused_global<cute::half_t>();init_fused_global<cute::bfloat16_t>();')
needle=' encode(ha,w.ha,dt,true);encode(hb,w.hb,dt,false);'
assert c.count(needle)==1
c=c.replace(needle,needle+'''
 if(balance){
  if(dt==megafmm::DType::BF16)launch_fused_global<cute::bfloat16_t>(ha,hb,static_cast<const cute::bfloat16_t*>(A),static_cast<const cute::bfloat16_t*>(B),w,D,stream);
  else launch_fused_global<cute::half_t>(ha,hb,static_cast<const cute::half_t*>(A),static_cast<const cute::half_t*>(B),w,D,stream);
  check(cudaPeekAtLastError(),"fused corrected launch");return;
 }
''')
(out/'pack.cuh').write_text(pack);(out/'corrected.cuh').write_text(c)
for name in ['compare.cu','pow2.cuh']:(out/name).write_bytes((h/'p56'/name).read_bytes())
def consumer(text):
 a=text.index('template<class T> __device__ __forceinline__ void consume')
 return text[a:text.index('template<class T> __global__ __launch_bounds__(256,1)',a)]
assert consumer(c)==consumer((h/'p56/corrected.cuh').read_text())
def producer(text):
 a=text.index(' if(tid==0){\n  for(int step=0;step<7;')
 return text[a:text.index('}else if(tid>=128)consume',a)+1]
assert producer(fused)==producer((h/'p56/corrected.cuh').read_text())
files=[Path(__file__),h/'p59_proposal_manifest.json']+[out/n for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']]
(h/'p59_source_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print('P59 isolated generation complete; working P56 unchanged')

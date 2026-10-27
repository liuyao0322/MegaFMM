from pathlib import Path
import subprocess
s=Path('/tmp/megafmm_leaf189.cu').read_text()
s=s.replace('WorkspaceBytes=size_t(12)<<20','WorkspaceBytes=PackedBytes')
s=s.replace('  uint32_t last;\n','')
s=s.replace('p=blockIdx.z,tile=blockIdx.y*8+blockIdx.x','p=blockIdx.x%8,tile=blockIdx.y*8+blockIdx.x/8')
s=s.replace('int cta=p*32+tile,m0=blockIdx.y*BM,n0=blockIdx.x*BN;','int cta=tile*8+p,m0=blockIdx.y*BM,n0=(blockIdx.x/8)*BN;')
a=s.index(' auto* products=');b=s.index(' if(tid==0)',a);s=s[:a]+s[b:]
s=s.replace('  if(p==0 && worker==0)counts[tile]=0;\n','').replace('2*62464','2*73728')
s=s.replace('if(tid==0){\n  // Raw','if(tid==0){\n  // Raw')
s=s.replace('0x80000000u-223u','0x80000000u-255u')
s=s.replace('   if(q==512/BK)break;','   if(q==512/BK || p==7)break;')
s=s.replace('}else if(tid>=128 && tid<384){','}else if(p<7 && tid>=128 && tid<384){')
a=s.index('  #pragma unroll\n  for(int j=0;j<32;j+=2)',s.index('wait(&sh.global_ready,0)'))
b=s.index('  if(sh.last){',a)
# Replace global P stores/counter with shared P stores, after joint operand retirement.
s=s[:a]+'''  asm volatile("bar.sync 3,256;" ::: "memory");
  #pragma unroll
  for(int j=0;j<32;j+=2){
   int mn=GMMA::CLayout_64x64{}(lane,j),row=wg*64+mn%64,col=mn/64;
   int index=row*64+(col^((row&3)<<3));
   auto* local=reinterpret_cast<float*>(sh.operand);
   reinterpret_cast<float2*>(local+index)[0]=make_float2(acc[j],acc[j+1]);
  }
 }
 __syncthreads();
 auto cluster=cooperative_groups::this_cluster();
 cluster.sync();
 if(p==7 && tid>=128 && tid<384){
  float* peers[7];
  #pragma unroll
  for(int z=0;z<7;++z)peers[z]=cluster.map_shared_rank(reinterpret_cast<float*>(sh.operand),z);
''' +s[b+len('  if(sh.last){'):]
# Epilogue index now cluster-local shared swizzle.
s=s.replace('    int index=row*512+col;','    int localrow=row-m0,localcol=col-n0;\n    int index=localrow*64+(localcol^((localrow&3)<<3));')
s=s.replace('reinterpret_cast<const float2*>(products+z*Plane+index)[0]','reinterpret_cast<const float2*>(peers[z]+index)[0]')
# Existing tail ends two if scopes (last and TC); now only rank7 if, append final cluster sync.
a=s.index('\nvoid check(')
s=s[:a].replace('   }\n  }\n }\n}\n','   }\n }\n cluster.sync();\n}\n')+s[a:]
s=s.replace('Count<224','Count<256').replace('requires224','requires256')
a=s.index('void run_sw_pipeline(')
s=s[:a]+'''template<class T> cudaLaunchConfig_t launch_config(cudaStream_t stream,cudaLaunchAttribute (&attrs)[2]){
 attrs[0].id=cudaLaunchAttributeCooperative;attrs[0].val.cooperative=1;
 attrs[1].id=cudaLaunchAttributeClusterDimension;attrs[1].val.clusterDim={8,1,1};
 cudaLaunchConfig_t config{};config.gridDim=dim3(64,4);config.blockDim=Threads;
 config.dynamicSmemBytes=sizeof(Shared<T>);config.stream=stream;config.attrs=attrs;config.numAttrs=2;
 return config;
}
void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream){
 CUtensorMap a{},b{},s{},t{};encode(a,A,state.dtype,true,false);encode(b,B,state.dtype,false,false);
 encode(s,workspace,state.dtype,true,true);encode(t,static_cast<char*>(workspace)+PackedBytes/2,state.dtype,false,true);
 cudaLaunchAttribute attrs[2]{};
 if(state.dtype==DType::BF16){
  auto config=launch_config<cute::bfloat16_t>(stream,attrs);
  check(cudaLaunchKernelEx(&config,sw_cooperative_hilo<cute::bfloat16_t>,a,b,s,t,static_cast<const cute::bfloat16_t*>(A),static_cast<const cute::bfloat16_t*>(B),static_cast<cute::bfloat16_t*>(workspace),D),"SW cluster launch");
 }else{
  auto config=launch_config<cute::half_t>(stream,attrs);
  check(cudaLaunchKernelEx(&config,sw_cooperative_hilo<cute::half_t>,a,b,s,t,static_cast<const cute::half_t*>(A),static_cast<const cute::half_t*>(B),static_cast<cute::half_t*>(workspace),D),"SW cluster launch");
 }
}
} // namespace megafmm::detail
'''
Path('solution/sw_pipeline.cu').write_text(s)
plan=subprocess.check_output(['git','show','50baf06:solution/plan.cpp'],text=True)
plan=plan.replace('size_t(12)','size_t(4)').replace('workspace=12582912','workspace=4194304').replace('global_leaf_products_last_cta','cluster8_leaf_dsm').replace('global_leaf','cluster8_dsm_leaf')
Path('solution/plan.cpp').write_text(plan)

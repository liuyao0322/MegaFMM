from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','55c0bbf:solution/sw_pipeline.cu'],text=True)
a=s.index('template<class T,class TA,class TB>');b=s.index('template<class T> __global__',a)
fn='''template<class T,int Group> __device__ __forceinline__ void tensor_chain(Shared<T>& s,int tid){
 using namespace cute;
 constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_MN_SW64_Atom<T>{},Shape<_32,Int<BK>>{});
 auto at=make_tensor(make_smem_ptr(s.operand[0]),la);
 auto bt=make_tensor(make_smem_ptr(s.operand[0]+ASize),lb);
 auto da_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(at,Shape<_64,_16>{}));
 auto db_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(bt,Shape<_32,_16>{}));
'''
for group in range(2):
 fn+=' if constexpr(Group==%d){\n'%group
 for dtype,cond in [('bf16','if constexpr(std::is_same_v<T,bfloat16_t>)'),('f16','else')]:
  lines=['{ .reg .pred p,leader,done; .reg .b64 ba,bb,da,db; .reg .b32 rb,fb,sp; .reg .f32 d<16>;', 'setp.ne.u32 p,1,0;','setp.eq.u32 leader,%7,0;']
  for step in range(group,7,2):
   lines+=['mov.f32 d%d,0f00000000;'%j for j in range(16)]
   for q in range(2):
    seq=step*2+q;slot=seq%4;phase=seq//4%2
    label='WAIT_%d_%d'%(step,q)
    lines+=['add.u32 rb,%%2,%d;'%(slot*8),label+':','mbarrier.try_wait.parity.shared::cta.b64 done,[rb],%d,0x989680;'%phase,'@!done bra '+label+';','wgmma.fence.sync.aligned;','add.u64 ba,%%0,%d;'%(slot*3072),'add.u64 bb,%%1,%d;'%(slot*3072)]
    for k in range(16):
     lines+=['add.u64 da,ba,%%%d;'%(8+2*k),'add.u64 db,bb,%%%d;'%(9+2*k),'wgmma.mma_async.sync.aligned.m64n32k16.f32.%s.%s {%s},da,db,p,1,1,%%40,%%41;'%(dtype,dtype,','.join('d%d'%j for j in range(16)))]
    lines+=['wgmma.commit_group.sync.aligned;','wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;','add.u32 fb,%%3,%d;'%(slot*8),'@leader mbarrier.arrive.shared::cta.b64 _,[fb];']
   if step>=2:
    lines+=['PRODUCT_FREE_%d:'%step,'mbarrier.try_wait.parity.shared::cta.b64 done,[%%6],%d,0x989680;'%((step-2)//2%2),'@!done bra PRODUCT_FREE_%d;'%step]
   for g in range(4):
    lines+=['add.u32 sp,%%4,%d;'%(g*2048),'st.shared.v4.f32 [sp],{'+','.join('d%d'%(g*4+j) for j in range(4))+'};']
   lines+=['mbarrier.arrive.shared::cta.b64 _,[%5];']
  lines+=['}']
  ins=['"l"(uint64_t(da_tiles(0,0,0)))','"l"(uint64_t(db_tiles(0,0,0)))','"r"(uint32_t(__cvta_generic_to_shared(s.ready)))','"r"(uint32_t(__cvta_generic_to_shared(s.free)))','"r"(uint32_t(__cvta_generic_to_shared(s.product[Group]+((tid-128)%128)*4)))','"r"(uint32_t(__cvta_generic_to_shared(s.product_ready+Group)))','"r"(uint32_t(__cvta_generic_to_shared(s.product_free+Group)))','"r"(tid&31)']
  for k in range(16):ins+=['"n"(uint32_t(da_tiles.layout()(0,0,Int<%d>{})))'%k,'"n"(uint32_t(db_tiles.layout()(0,0,Int<%d>{})))'%k]
  ins+=['"n"(int32_t(GMMA::Major::K))','"n"(int32_t(GMMA::Major::MN))']
  fn+=' '+cond+'{asm volatile(\n'+''.join('  "'+l+'\\n"\n' for l in lines)+'  : : '+', '.join(ins)+' : "memory");}\n'
 fn+=' }\n'
fn+='}\n'
s=s[:a]+fn+s[b:]
a=s.index('  }else if(tid>=128 && tid<256)');b=s.index('  // CTA shared storage',a)
s=s[:a]+'''  }else if(tid>=128 && tid<384){
    if(tid<256)tensor_chain<T,0>(s,tid);else tensor_chain<T,1>(s,tid);
  }
'''+s[b:]
s=s.replace('Threads=768','Threads=896').replace('(tid>=32 && tid<128) || tid>=256','(tid>=32 && tid<128) || tid>=384').replace('tid>=256 ? tid-160:tid-32','tid>=384 ? tid-288:tid-32')
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','d8e6261:solution/plan.cpp'],text=True).replace('1tma+4simt+4wgmma+15pack_only','1tma+4simt+8wgmma+15pack_only').replace('product_layout=row_xor_0_3_4','product_layout=fragment_major_float4;tc=two_static_ptx_chains;consumer_layout=TC_fragment;output=float2')
Path('solution/plan.cpp').write_text(s)

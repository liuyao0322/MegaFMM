from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','1f1fb61:solution/sw_pipeline.cu'],text=True)
a=s.index('template<class T,class TA,class TB>');b=s.index('template<class T> __global__',a)
fn='''template<class T> __device__ __forceinline__ void tensor_chain(Shared<T>& s,int tid,int m0,int n0,float* D){
 using namespace cute;
 constexpr auto la=tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_32,Int<BK>>{});
 auto a=make_tensor(make_smem_ptr(s.operand[0]),la);
 auto b=make_tensor(make_smem_ptr(s.operand[0]+ASize),lb);
 auto da_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(a,Shape<_64,_16>{}));
 auto db_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(b,Shape<_32,_16>{}));
 int mn=GMMA::CLayout_64x32{}(tid-128,0);
 float* out=D+(n0+mn/64)*1024+m0+mn%64;
'''
for dtype,cond in [('bf16','if constexpr(std::is_same_v<T,bfloat16_t>)'),('f16','else')]:
 lines=['{ .reg .pred p,noacc,leader,done; .reg .b64 ba,bb,da,db,dst;', '.reg .b32 rb,fb; .reg .f32 d<16>,first<16>,second<16>,x;', 'setp.ne.u32 p,1,0;','setp.eq.u32 noacc,1,0;','setp.eq.u32 leader,%4,0;']
 for step in range(7):
  for q in range(4):
   seq=step*4+q;slot=seq%9;phase=seq//9%2
   lines+=['add.u32 rb,%%2,%d;'%(slot*8),'WAIT_%d_%d:'%(step,q),'mbarrier.try_wait.parity.shared::cta.b64 done,[rb],%d,0x989680;'%phase,'@!done bra WAIT_%d_%d;'%(step,q),'wgmma.fence.sync.aligned;','add.u64 ba,%%0,%d;'%(slot*1536),'add.u64 bb,%%1,%d;'%(slot*1536)]
   for k in range(8):
    pred='noacc' if q==0 and k==0 else 'p'
    lines+=['add.u64 da,ba,%%%d;'%(6+2*k),'add.u64 db,bb,%%%d;'%(7+2*k),'wgmma.mma_async.sync.aligned.m64n32k16.f32.%s.%s {%s},da,db,%s,1,1,%%38,%%39;'%(dtype,dtype,','.join('d%d'%j for j in range(16)),pred)]
   lines+=['wgmma.commit_group.sync.aligned;','wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;','add.u32 fb,%%3,%d;'%(slot*8),'@leader mbarrier.arrive.shared::cta.b64 _,[fb];']
  if step==0:lines+=['mov.f32 first%d,d%d;'%(j,j) for j in range(16)]
  elif step==2:lines+=['add.rn.f32 first%d,first%d,d%d;'%(j,j,j) for j in range(16)]
  elif step==3:lines+=['add.rn.f32 second%d,first%d,d%d;'%(j,j,j) for j in range(16)]
  else:
   base={1:0,4:2099200,5:2048,6:2097152}[step]
   for j in range(16):
    if step in [1,5]:lines+=['add.rn.f32 x,first%d,d%d;'%(j,j)]
    elif step==4:lines+=['add.rn.f32 x,second%d,d%d;'%(j,j),'add.rn.f32 first%d,first%d,d%d;'%(j,j,j)]
    else:lines+=['sub.rn.f32 x,second%d,d%d;'%(j,j)]
    lines+=['add.u64 dst,%%5,%%%d;'%(22+j)]
    if base:lines+=['add.u64 dst,dst,%d;'%base]
    lines+=['st.global.f32 [dst],x;']
 lines+=['}']
 ins=['"l"(uint64_t(da_tiles(0,0,0)))','"l"(uint64_t(db_tiles(0,0,0)))','"r"(uint32_t(__cvta_generic_to_shared(s.ready)))','"r"(uint32_t(__cvta_generic_to_shared(s.free)))','"r"(tid&31)','"l"(out)']
 for k in range(8):ins+=['"n"(uint32_t(da_tiles.layout()(0,0,Int<%d>{})))'%k,'"n"(uint32_t(db_tiles.layout()(0,0,Int<%d>{})))'%k]
 for j in range(16):
  mn='int(GMMA::CLayout_64x32{}(0,Int<%d>{}))'%j
  ins+=['"n"(4*(%s%%64+(%s/64)*1024))'%(mn,mn)]
 ins+=['"n"(int32_t(GMMA::Major::MN))','"n"(int32_t(GMMA::Major::K))']
 fn+=' '+cond+'{asm volatile(\n'+''.join('  "'+v+'\\n"\n' for v in lines)+'  : : '+', '.join(ins)+' : "memory");}\n'
fn+='}\n'
s=s[:a]+fn+s[b:]
a=s.index('    if(worker<128){');b=s.index('  // CTA shared storage',a)
s=s[:a]+'''  }else if(tid>=128 && tid<256){
    tensor_chain(s,tid,m0,n0,D);
  }
'''+s[b:]
a=s.index('  // Row-major P');b=s.index('  uint64_t pack_ready;',a)
s=s[:a]+'  uint64_t ready[Stages],free[Stages];\n'+s[b:]
s=s.replace('BK=256,Stages=4','BK=128,Stages=9').replace('==213120','==221440').replace('if(sequence==4)','if(sequence==8)')
s=s.replace('    for(int j=0;j<2;++j){Barrier::init(&s.product_ready[j],128);Barrier::init(&s.product_free[j],128);}\n','')
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','1f1fb61:solution/plan.cpp'],text=True).replace('smem=213120','smem=221440').replace('operand_stages=4;bk=256','operand_stages=9;bk=128').replace('1tma+4simt+4wgmma+15pack_only','1tma+4wgmma+19pack_only').replace('recombine=streaming_fixed_chunk_offsets','recombine=private_static_ptx;accum_init=first_mma_overwrite').replace('product_slots=2','product_slots=0').replace('product_layout=column_xor_3_4','product_layout=registers')
Path('solution/plan.cpp').write_text(s)

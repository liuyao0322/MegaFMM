from pathlib import Path
s=Path('solution/sw_pipeline.cu').read_text()
a=s.index('template<class T> __device__ __forceinline__ void mma_product');b=s.index('template<class T> __global__',a)
fn='''template<class T> __device__ __forceinline__ void tensor_chain(Shared<T>& s,int tid,int m0,int n0,float* D){
 using namespace cute;
 constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_MN_SW64_Atom<T>{},Shape<_32,Int<BK>>{});
 auto a=make_tensor(make_smem_ptr(s.operand[0]),la);
 auto b=make_tensor(make_smem_ptr(s.operand[0]+ASize),lb);
 auto da_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(a,Shape<_64,_16>{}));
 auto db_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(b,Shape<_32,_16>{}));
 int mn=GMMA::CLayout_64x32{}(tid-128,0);
 float* out=D+(m0+mn%64)*1024+n0+mn/64;
'''
for dtype,cond in [('bf16','if constexpr(std::is_same_v<T,bfloat16_t>)'),('f16','else')]:
 lines=['{ .reg .pred p,leader,done; .reg .b64 ba,bb,da,db,dst;', '.reg .b32 rb,fb; .reg .f32 d<16>,e<16>,first<16>,second<16>,x,y;', 'setp.ne.u32 p,1,0;','setp.eq.u32 leader,%4,0;']
 def release(seq):
  lines.extend(['bar.warp.sync 0xffffffff;','add.u32 fb,%%3,%d;'%(8*(seq%4)),'@leader mbarrier.arrive.shared::cta.b64 _,[fb];'])
 def recombine(step,acc):
  if step==0:lines.extend(['mov.f32 first%d,%s%d;'%(j,acc,j) for j in range(16)])
  elif step==2:lines.extend(['add.rn.f32 first%d,first%d,%s%d;'%(j,j,acc,j) for j in range(16)])
  elif step==3:lines.extend(['add.rn.f32 second%d,first%d,%s%d;'%(j,j,acc,j) for j in range(16)])
  else:
   base={1:0,4:2099200,5:2048,6:2097152}[step]
   for j in range(8):
    for c,t in enumerate(['x','y']):
     v=2*j+c
     if step in [1,5]:lines.append('add.rn.f32 %s,first%d,%s%d;'%(t,v,acc,v))
     elif step==4:lines.extend(['add.rn.f32 %s,second%d,%s%d;'%(t,v,acc,v),'add.rn.f32 first%d,first%d,%s%d;'%(v,v,acc,v)])
     else:lines.append('sub.rn.f32 %s,second%d,%s%d;'%(t,v,acc,v))
    lines.append('add.u64 dst,%%5,%%%d;'%(38+j))
    if base:lines.append('add.u64 dst,dst,%d;'%base)
    lines.append('st.global.v2.f32 [dst],{x,y};')
 for step in range(7):
  acc='e' if step%2 else 'd'
  lines.extend(['mov.f32 %s%d,0f00000000;'%(acc,j) for j in range(16)])
  for q in range(2):
   seq=step*2+q;slot=seq%4;phase=seq//4%2
   lines.extend(['add.u32 rb,%%2,%d;'%(slot*8),'WAIT_%d_%d:'%(step,q),'mbarrier.try_wait.parity.shared::cta.b64 done,[rb],%d,0x989680;'%phase,'@!done bra WAIT_%d_%d;'%(step,q),'wgmma.fence.sync.aligned;','add.u64 ba,%%0,%d;'%(slot*3072),'add.u64 bb,%%1,%d;'%(slot*3072)])
   for k in range(16):
    lines.extend(['add.u64 da,ba,%%%d;'%(6+2*k),'add.u64 db,bb,%%%d;'%(7+2*k),'wgmma.mma_async.sync.aligned.m64n32k16.f32.%s.%s {%s}, da, db, p, 1, 1, %%46, %%47;'%(dtype,dtype,', '.join('%s%d'%(acc,j) for j in range(16)))])
   lines.append('wgmma.commit_group.sync.aligned;')
   if seq==0:
    lines.append('wgmma.wait_group.sync.aligned 0;');release(0)
   elif step>0:
    lines.append('wgmma.wait_group.sync.aligned 1;');release(seq-1)
    if q==0:recombine(step-1,'d' if step%2 else 'e')
 lines.append('wgmma.wait_group.sync.aligned 0;');release(13);recombine(6,'d')
 lines+=['}']
 fn+=' '+cond+'{\n  asm volatile(\n'+''.join('   "'+x+'\\n"\n' for x in lines)
 ins=['"l"(uint64_t(da_tiles(0,0,0)))','"l"(uint64_t(db_tiles(0,0,0)))','"r"(uint32_t(__cvta_generic_to_shared(s.ready)))','"r"(uint32_t(__cvta_generic_to_shared(s.free)))','"r"(tid&31)','"l"(out)']
 for k in range(16):ins+=['"n"(uint32_t(da_tiles.layout()(0,0,Int<%d>{})))'%k,'"n"(uint32_t(db_tiles.layout()(0,0,Int<%d>{})))'%k]
 for j in range(8):
  mn='int(GMMA::CLayout_64x32{}(0,Int<%d>{}))'%(2*j)
  ins+=['"n"(4*((%s%%64)*1024+%s/64))'%(mn,mn)]
 ins+=['"n"(int32_t(GMMA::Major::K))','"n"(int32_t(GMMA::Major::MN))']
 fn+='   :\n   : '+', '.join(ins)+'\n   : "memory");\n }\n'
fn+='}\n\n';s=s[:a]+fn+s[b:]
a=s.index('  }else if(tid>=128 && tid<256)');b=s.index('  // CTA shared storage',a)
s=s[:a]+'''  }else if(tid>=128 && tid<256){
    tensor_chain(s,tid,m0,n0,D);
  }
'''+s[b:]
Path('solution/sw_pipeline.cu').write_text(s)
p=Path('solution/plan.cpp');s=p.read_text().replace('tc_register_private_ptx_accumulators','tc_static_ptx_pingpong_wait1');p.write_text(s)

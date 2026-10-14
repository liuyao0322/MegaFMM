from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','6308acc:solution/sw_pipeline.cu'],text=True)
a=s.index('template<class T> __device__ __forceinline__ void tensor_chain');b=s.index('template<class T> __global__',a)
fn=s[a:s.index(' if constexpr',a)]
fn=fn.replace('float* D){','float* D,uint32_t* flags){')
for dtype,cond in [('bf16','if constexpr(std::is_same_v<T,bfloat16_t>)'),('f16','else')]:
 lines=['{ .reg .pred p,noacc,leader,flagleader,firstk,done; .reg .b64 ba,bb,da,db,dst,fptr;', '.reg .b32 rb,fb,flag; .reg .f32 d<32>,first<32>,second<32>,x,y;', 'setp.ne.u32 p,1,0;','setp.eq.u32 noacc,1,0;','setp.eq.u32 leader,%4,0;','setp.eq.u32 flagleader,%75,0;','setp.eq.u32 firstk,%74,0;']
 for step in range(7):
  seq=step;slot=seq%3;phase=seq//3%2
  lines+=['add.u32 rb,%%2,%d;'%(slot*8),'WAIT_%d:'%step,'mbarrier.try_wait.parity.shared::cta.b64 done,[rb],%d,0x989680;'%phase,'@!done bra WAIT_%d;'%step,'wgmma.fence.sync.aligned;','add.u64 ba,%%0,%d;'%(slot*4096),'add.u64 bb,%%1,%d;'%(slot*4096)]
  for k in range(16):
   pred='noacc' if k==0 else 'p'
   lines+=['add.u64 da,ba,%%%d;'%(6+2*k),'add.u64 db,bb,%%%d;'%(7+2*k),'wgmma.mma_async.sync.aligned.m64n64k16.f32.%s.%s {%s},da,db,%s,1,1,%%70,%%71;'%(dtype,dtype,','.join('d%d'%j for j in range(32)),pred)]
  lines+=['wgmma.commit_group.sync.aligned;','wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;','add.u32 fb,%%3,%d;'%(slot*8),'@leader mbarrier.arrive.shared::cta.b64 _,[fb];']
  if step==0:lines+=['mov.f32 first%d,d%d;'%(j,j) for j in range(32)]
  elif step==2:lines+=['add.rn.f32 first%d,first%d,d%d;'%(j,j,j) for j in range(32)]
  elif step==3:lines+=['add.rn.f32 second%d,first%d,d%d;'%(j,j,j) for j in range(32)]
  else:
   base,quad={1:(0,0),4:(2099200,3),5:(2048,1),6:(2097152,2)}[step]
   if step==1:lines+=['OUTPUT_READY:','mbarrier.try_wait.parity.shared::cta.b64 done,[%72],0,0x989680;','@!done bra OUTPUT_READY;']
   lines+=['add.u64 fptr,%%73,%d;'%(quad*4),'@firstk bra NO_WAIT_%d;'%step,'@!leader bra ACQUIRED_%d;'%step,'FLAG_WAIT_%d:'%step,'ld.acquire.gpu.u32 flag,[fptr];','setp.eq.u32 done,flag,1;','@!done bra FLAG_WAIT_%d;'%step,'ACQUIRED_%d:'%step,'bar.warp.sync 0xffffffff;','NO_WAIT_%d:'%step]
   for j in range(32):
    if step in [1,5]:lines+=['add.rn.f32 x,first%d,d%d;'%(j,j)]
    elif step==4:lines+=['add.rn.f32 x,second%d,d%d;'%(j,j),'add.rn.f32 first%d,first%d,d%d;'%(j,j,j)]
    else:lines+=['sub.rn.f32 x,second%d,d%d;'%(j,j)]
    lines+=['add.u64 dst,%%5,%%%d;'%(38+j)]
    if base:lines+=['add.u64 dst,dst,%d;'%base]
    lines+=['@firstk st.global.f32 [dst],x;','@!firstk ld.global.f32 y,[dst];','@!firstk add.rn.f32 x,y,x;','@!firstk st.global.f32 [dst],x;']
   lines+=['@!firstk bra NEXT_%d;'%step,'bar.sync 3,128;','@flagleader st.release.gpu.u32 [fptr],1;','NEXT_%d:'%step]
 lines+=['}']
 ins=['"l"(uint64_t(da_tiles(0,0,0)))','"l"(uint64_t(db_tiles(0,0,0)))','"r"(uint32_t(__cvta_generic_to_shared(s.ready)))','"r"(uint32_t(__cvta_generic_to_shared(s.free)))','"r"(tid&31)','"l"(out)']
 for k in range(16):ins+=['"n"(uint32_t(da_tiles.layout()(0,0,Int<%d>{})))'%k,'"n"(uint32_t(db_tiles.layout()(0,0,Int<%d>{})))'%k]
 for j in range(32):
  mn='int(GMMA::CLayout_64x64{}(0,Int<%d>{}))'%j
  ins+=['"n"(4*(%s%%64+(%s/64)*1024))'%(mn,mn)]
 ins+=['"n"(int32_t(GMMA::Major::MN))','"n"(int32_t(GMMA::Major::K))','"r"(uint32_t(__cvta_generic_to_shared(&s.output_ready)))','"l"(flags)','"r"(blockIdx.z)','"r"(tid-128)']
 fn+=' '+cond+'{asm volatile(\n'+''.join('  "'+v+'\\n"\n' for v in lines)+'  : : '+', '.join(ins)+' : "memory");}\n'
fn+='}\n'
s=s[:a]+fn+s[b:]
s=s.replace('uint64_t pack_ready;\n  uint32_t last;','uint64_t pack_ready,output_ready;').replace('Barrier::init(&s.pack_ready,352);','Barrier::init(&s.pack_ready,352);\n    Barrier::init(&s.output_ready,1);')
s=s.replace('}while(!((old^current)&0x80000000u));','}while(!((old^current)&0x80000000u));\n        arrive(&s.output_ready);')
s=s.replace('    if(worker==0 && blockIdx.z==0)reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(packed)+(size_t(12)<<20))[blockIdx.y*8+blockIdx.x]=0;','''    if(worker==0 && blockIdx.z==0){
      auto* flags=reinterpret_cast<uint4*>(reinterpret_cast<char*>(packed)+(size_t(4)<<20));
      flags[blockIdx.y*8+blockIdx.x]=uint4{0,0,0,0};
    }''')
a=s.index('    auto* partial=reinterpret_cast<float*>(');b=s.index('  // CTA shared storage survives',a)
s=s[:a]+'''    auto* flags=reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(packed)+(size_t(4)<<20))+(blockIdx.y*8+blockIdx.x)*4;
    tensor_chain(s,tid,m0,n0,D,flags);
  }
'''+s[b:]
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','6308acc:solution/plan.cpp'],text=True).replace('(size_t(12)<<20)+256','(size_t(4)<<20)+1024').replace('SW needs12MiB+256B workspace','SW needs4MiB+1024B workspace').replace('workspace=12583168','workspace=4195328').replace('merge=last_cta_partial_C','merge=ordered_quadrant_handoff')
Path('solution/plan.cpp').write_text(s)

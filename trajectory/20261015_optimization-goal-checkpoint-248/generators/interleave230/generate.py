from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','62142fc:solution/sw_pipeline.cu'],text=True)
a=s.index('template<class T> __device__ __forceinline__ void tensor_chain');b=s.index('template<class T> __global__',a)
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
schedule=[(pair*2+side,q) for pair in range(3) for q in range(2) for side in range(2)]+[(6,0),(6,1)]
def recombine(step):
 reg='a' if step%2==0 else 'b';v=[]
 if step==0:return ['mov.f32 first%d,%s%d;'%(j,reg,j) for j in range(16)]
 if step==2:return ['add.rn.f32 first%d,first%d,%s%d;'%(j,j,reg,j) for j in range(16)]
 if step==3:return ['add.rn.f32 second%d,first%d,%s%d;'%(j,j,reg,j) for j in range(16)]
 base={1:0,4:2099200,5:2048,6:2097152}[step]
 for j in range(16):
  if step in [1,5]:v+=['add.rn.f32 x,first%d,%s%d;'%(j,reg,j)]
  elif step==4:v+=['add.rn.f32 x,second%d,%s%d;'%(j,reg,j),'add.rn.f32 first%d,first%d,%s%d;'%(j,j,reg,j)]
  else:v+=['sub.rn.f32 x,second%d,%s%d;'%(j,reg,j)]
  v+=['add.u64 dst,%%5,%%%d;'%(38+j)]
  if base:v+=['add.u64 dst,dst,%d;'%base]
  v+=['st.global.f32 [dst],x;']
 return v
for dtype,cond in [('bf16','if constexpr(std::is_same_v<T,bfloat16_t>)'),('f16','else')]:
 lines=['{ .reg .pred p,noacc,leader,done; .reg .b64 ba,bb,da,db,dst;', '.reg .b32 rb,fb; .reg .f32 a<16>,b<16>,first<16>,second<16>,x;', 'setp.ne.u32 p,1,0;','setp.eq.u32 noacc,1,0;','setp.eq.u32 leader,%4,0;']
 for seq,(step,q) in enumerate(schedule):
  slot=seq%4;phase=seq//4%2;reg='a' if step%2==0 else 'b'
  lines+=['add.u32 rb,%%2,%d;'%(slot*8),'WAIT_%d:'%seq,'mbarrier.try_wait.parity.shared::cta.b64 done,[rb],%d,0x989680;'%phase,'@!done bra WAIT_%d;'%seq,'wgmma.fence.sync.aligned;','add.u64 ba,%%0,%d;'%(slot*3072),'add.u64 bb,%%1,%d;'%(slot*3072)]
  for k in range(16):
   pred='noacc' if q==0 and k==0 else 'p'
   lines+=['add.u64 da,ba,%%%d;'%(6+2*k),'add.u64 db,bb,%%%d;'%(7+2*k),'wgmma.mma_async.sync.aligned.m64n32k16.f32.%s.%s {%s},da,db,%s,1,1,%%54,%%55;'%(dtype,dtype,','.join(reg+str(j) for j in range(16)),pred)]
  lines+=['wgmma.commit_group.sync.aligned;']
  if seq:
   prev=seq-1;ps,pq=schedule[prev]
   lines+=['wgmma.wait_group.sync.aligned 1;','bar.warp.sync 0xffffffff;','add.u32 fb,%%3,%d;'%((prev%4)*8),'@leader mbarrier.arrive.shared::cta.b64 _,[fb];']
   if pq==1:lines+=recombine(ps)
 lines+=['wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;','add.u32 fb,%3,8;','@leader mbarrier.arrive.shared::cta.b64 _,[fb];']+recombine(6)+['}']
 ins=['"l"(uint64_t(da_tiles(0,0,0)))','"l"(uint64_t(db_tiles(0,0,0)))','"r"(uint32_t(__cvta_generic_to_shared(s.ready)))','"r"(uint32_t(__cvta_generic_to_shared(s.free)))','"r"(tid&31)','"l"(out)']
 for k in range(16):ins+=['"n"(uint32_t(da_tiles.layout()(0,0,Int<%d>{})))'%k,'"n"(uint32_t(db_tiles.layout()(0,0,Int<%d>{})))'%k]
 for j in range(16):
  mn='int(GMMA::CLayout_64x32{}(0,Int<%d>{}))'%j
  ins+=['"n"(4*(%s%%64+(%s/64)*1024))'%(mn,mn)]
 ins+=['"n"(int32_t(GMMA::Major::MN))','"n"(int32_t(GMMA::Major::K))']
 fn+=' '+cond+'{asm volatile(\n'+''.join('  "'+v+'\\n"\n' for v in lines)+'  : : '+', '.join(ins)+' : "memory");}\n'
fn+='}\n'
s=s[:a]+fn+s[b:]
s=s.replace('for(int step=0;step<7;++step)for(int q=0;q<512/BK;++q){\n      int p=product_for_step(step);\n      int sequence=step*(512/BK)+q,slot=sequence%Stages,k0=q*BK;', '''for(int sequence=0;sequence<14;++sequence){
      int step=sequence<12 ? (sequence/4)*2+(sequence&1) : 6;
      int q=sequence<12 ? (sequence%4)/2 : sequence-12;
      int p=product_for_step(step);
      int slot=sequence%Stages,k0=q*BK;''')
Path('solution/sw_pipeline.cu').write_text(s)
for name in ['internal.h','plan.cpp']:
 s=subprocess.check_output(['git','show','62142fc:solution/'+name],text=True)
 if name=='plan.cpp':s=s.replace('accum_init=first_mma_overwrite','accum_init=first_mma_overwrite;pipeline=dual_acc_pair_interleave_wait1')
 Path('solution/'+name).write_text(s)

from pathlib import Path
import subprocess
base=subprocess.check_output(['git','show','1bdfb29:solution/sw_pipeline.cu'],text=True)
s=base.replace('BN=32,BK=128,Stages=8','BN=64,BK=128,Stages=5')
s=s.replace('  uint64_t ready[Stages],free[Stages];','  float u[2][4096];\n  uint64_t ready[Stages],free[Stages],uready[2];')
a=s.index('template<class T> __device__ __forceinline__ void tensor_chain');b=s.index('template<class T> __global__ __launch_bounds__(512',a)
fn='''template<class T,int Rank> __device__ __forceinline__ void tensor_chain(Shared<T>& s,int tid,int m0,int n0,float* D){
 using namespace cute;
 constexpr auto la=tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
 auto a=make_tensor(make_smem_ptr(s.operand[0]),la);
 auto b=make_tensor(make_smem_ptr(s.operand[0]+ASize),lb);
 auto da_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(a,Shape<_64,_16>{}));
 auto db_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(b,Shape<_64,_16>{}));
 int mn=GMMA::CLayout_64x64{}(tid-128,0);
 float* out=D+(n0+mn/64)*1024+m0+mn%64;
 uint32_t ubase=uint32_t(__cvta_generic_to_shared(s.u));
 uint32_t ur=uint32_t(__cvta_generic_to_shared(s.uready));
 if constexpr(Rank==0)asm("mapa.shared::cluster.u32 %0,%1,1;":"=r"(ur):"r"(ur));
 else asm("mapa.shared::cluster.u32 %0,%1,0;":"=r"(ubase):"r"(ubase));
 ubase+=(tid-128)*16;
'''
for rank in range(2):
 for dtype,cond in [('bf16','if constexpr(std::is_same_v<T,bfloat16_t>)'),('f16','if constexpr(!std::is_same_v<T,bfloat16_t>)')]:
  lines=['{ .reg .pred p,noacc,leader,done; .reg .b64 ba,bb,da,db,dst;', '.reg .b32 rb,fb,ua,ur; .reg .f32 d<32>,first<32>,second<32>,x;', 'setp.ne.u32 p,1,0;','setp.eq.u32 noacc,1,0;','setp.eq.u32 leader,%4,0;']
  def store_output(which,base):
   for j in range(32):
    lines.append(f'{which}.rn.f32 x,'+('second' if base in [2099200,2097152] else 'first')+f'{j},d{j};')
    lines.append(f'add.u64 dst,%5,%{22+j};')
    if base:lines.append(f'add.u64 dst,dst,{base};')
    lines.append('st.global.f32 [dst],x;')
  def publish(which,array):
   for g in range(8):
    lines.append(f'add.u32 ua,%56,{array*16384+g*2048};')
    lines.append('st.shared.v4.f32 [ua],{'+','.join(f'{which}{g*4+j}' for j in range(4))+'};')
   lines.extend(['bar.warp.sync 0xffffffff;',f'add.u32 ur,%57,{array*8};','@leader mbarrier.arrive.release.cluster.shared::cluster.b64 _,[ur];'])
  for step in range(4 if rank==0 else 3):
   for q in range(4):
    seq=step*4+q;slot=seq%5;phase=seq//5%2
    lines += [f'add.u32 rb,%2,{slot*8};',f'WAIT_{step}_{q}:',f'mbarrier.try_wait.parity.shared::cta.b64 done,[rb],{phase},0x989680;',f'@!done bra WAIT_{step}_{q};','wgmma.fence.sync.aligned;',f'add.u64 ba,%0,{slot*2048};',f'add.u64 bb,%1,{slot*2048};']
    for k in range(8):
     pred='noacc' if q==0 and k==0 else 'p'
     lines.extend([f'add.u64 da,ba,%{6+2*k};',f'add.u64 db,bb,%{7+2*k};',f'wgmma.mma_async.sync.aligned.m64n64k16.f32.{dtype}.{dtype} {{'+','.join(f'd{j}' for j in range(32))+f'}},da,db,{pred},1,1,%54,%55;'])
    lines+=['wgmma.commit_group.sync.aligned;','wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;',f'add.u32 fb,%3,{slot*8};','@leader mbarrier.arrive.shared::cta.b64 _,[fb];']
   if rank==0:
    if step==0:lines.extend(f'mov.f32 first{j},d{j};' for j in range(32))
    elif step==1:store_output('add',0)
    elif step==2:
     lines.extend(f'add.rn.f32 first{j},first{j},d{j};' for j in range(32));publish('first',0)
    else:
     lines.extend(f'add.rn.f32 second{j},first{j},d{j};' for j in range(32));publish('second',1)
   else:
    if step==0:
     for i in range(2):
      lines.extend([f'add.u32 ur,%57,{i*8};',f'WAIT_U{i}:','mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 done,[ur],0,0x989680;',f'@!done bra WAIT_U{i};'])
      for g in range(8):
       lines.append(f'add.u32 ua,%56,{i*16384+g*2048};')
       lines.append('ld.shared::cluster.v4.f32 {'+','.join(('first' if i==0 else 'second')+str(g*4+j) for j in range(4))+'},[ua];')
     store_output('add',2099200)
     lines.extend(f'add.rn.f32 first{j},first{j},d{j};' for j in range(32))
    elif step==1:store_output('add',2048)
    else:store_output('sub',2097152)
  lines+=['}']
  ins=['"l"(uint64_t(da_tiles(0,0,0)))','"l"(uint64_t(db_tiles(0,0,0)))','"r"(uint32_t(__cvta_generic_to_shared(s.ready)))','"r"(uint32_t(__cvta_generic_to_shared(s.free)))','"r"(tid&31)','"l"(out)']
  for k in range(8):ins+=['"n"(uint32_t(da_tiles.layout()(0,0,Int<%d>{})))'%k,'"n"(uint32_t(db_tiles.layout()(0,0,Int<%d>{})))'%k]
  for j in range(32):
   mn='int(GMMA::CLayout_64x64{}(0,Int<%d>{}))'%j
   ins+=['"n"(4*(%s%%64+(%s/64)*1024))'%(mn,mn)]
  ins+=['"n"(int32_t(GMMA::Major::MN))','"n"(int32_t(GMMA::Major::K))','"r"(ubase)','"r"(ur)']
  fn+=f' if constexpr(Rank=={rank}){{ '+cond+'{asm volatile(\n'+''.join('  "'+v+'\\n"\n' for v in lines)+'  : : '+', '.join(ins)+' : "memory");}}\n'
fn+='}\n'
s=s[:a]+fn+s[b:]
a=s.index('void sw_cooperative_hilo(');b=s.index('void check(',a);v=s[a:b]
v=v.replace('cutlass::arch::fence_barrier_init();','for(int j=0;j<2;++j)Barrier::init(&s.uready[j],4);\n    cutlass::arch::fence_barrier_init();')
v=v.replace('__syncthreads();','cooperative_groups::this_cluster().sync();')
v=v.replace('Shape<_32,Int<BK>>','Shape<_64,Int<BK>>')
v=v.replace('for(int step=0;step<7;++step)','for(int step=0;step<(blockIdx.z==0 ? 4:3);++step)')
v=v.replace('int p=product_for_step(step);','int p=blockIdx.z==0 ? (step==0?0:step==1?1:step==2?5:6) : (step==0?4:step==1?2:3);')
v=v.replace('if(sequence==8)','if(sequence==(blockIdx.z==0 ? 8:0))')
v=v.replace('tensor_chain(s,tid,m0,n0,D);','if(blockIdx.z==0)tensor_chain<T,0>(s,tid,m0,n0,D);\n    else tensor_chain<T,1>(s,tid,m0,n0,D);')
v=v.replace('  // CTA shared storage survives until all role threads return.','  cooperative_groups::this_cluster().sync(); // Preserve remotely read U storage lifetime.')
s=s[:a]+v+s[b:]
s=s.replace('cudaLaunchAttribute attribute{};','cudaLaunchAttribute attributes[2]{};\n  auto& attribute=attributes[0];\n  attributes[1].id=cudaLaunchAttributeClusterDimension;\n  attributes[1].val.clusterDim={1,1,2};')
s=s.replace('config.gridDim=dim3(16,8)','config.gridDim=dim3(8,8,2)').replace('config.attrs=&attribute;config.numAttrs=1','config.attrs=attributes;config.numAttrs=2')
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1bdfb29:solution/plan.cpp'],text=True).replace('operand_stages=8;bk=128','operand_stages=5;bk=128;cluster=1x1x2;leaf_split=4plus3;tile=64x64;u_exchange=dsm_vec4')
Path('solution/plan.cpp').write_text(p)
for f in ['internal.h','scale_output.cu']:Path('solution/'+f).write_bytes(subprocess.check_output(['git','show','1bdfb29:solution/'+f]))

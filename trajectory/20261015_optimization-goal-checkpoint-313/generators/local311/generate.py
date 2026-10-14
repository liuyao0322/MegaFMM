from pathlib import Path
import subprocess
base=subprocess.check_output(['git','show','1bdfb29:solution/sw_pipeline.cu'],text=True)
head=base[:base.index('constexpr int BM=')]
s=head+'''constexpr int BM=64,BN=32,BK=64,Threads=384,ASize=BM*BK,BSize=BN*BK;
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
template<class T>struct Operands{T b[7][ASize];T a[7][BSize];};
template<class T>struct alignas(128) Shared{Operands<T> operand[2];uint64_t raw[2],pack[2],free[2];};
static_assert(sizeof(Shared<cute::half_t>)==172160,"shared budget");
__device__ __forceinline__ void wait(const uint64_t* b,int phase){Barrier::wait(b,phase);asm volatile("" ::: "memory");}
__device__ __forceinline__ void arrive(const uint64_t* b){asm volatile("" ::: "memory");Barrier::arrive(b);}
template<class T>__device__ __forceinline__ uint32_t pair(float x,float y){
 uint32_t z;
 if constexpr(std::is_same_v<T,cute::bfloat16_t>)asm("cvt.rn.bf16x2.f32 %0,%2,%1;":"=r"(z):"f"(x),"f"(y));
 else asm("cvt.rn.f16x2.f32 %0,%2,%1;":"=r"(z):"f"(x),"f"(y));
 return z;
}
template<class T>__device__ __forceinline__ void transform(Operands<T>& s,int worker){
 using namespace cute;
 constexpr auto la=tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_32,Int<BK>>{});
 for(int i=worker*2;i<BSize;i+=512){
  int off=i;
  uint32_t words[4];
  #pragma unroll
  for(int p=0;p<4;++p)words[p]=*reinterpret_cast<const uint32_t*>(s.a[p]+off);
  float values[4][2];
  #pragma unroll
  for(int j=0;j<2;++j){
   float a11=float(T::bitcast(uint16_t(words[0]>>(j*16)))),a12=float(T::bitcast(uint16_t(words[1]>>(j*16))));
   float a21=float(T::bitcast(uint16_t(words[2]>>(j*16)))),a22=float(T::bitcast(uint16_t(words[3]>>(j*16))));
   float s1=__fadd_rn(a21,a22),s2=__fsub_rn(s1,a11);
   values[0][j]=s1;values[1][j]=s2;values[2][j]=__fsub_rn(a11,a21);values[3][j]=__fsub_rn(a12,s2);
  }
  #pragma unroll
  for(int p=0;p<4;++p)*reinterpret_cast<uint32_t*>(s.a[p==0?2:p+3]+off)=pair<T>(values[p][0],values[p][1]);
 }
 for(int i=worker*2;i<ASize;i+=512){
  int off=i;
  uint32_t words[4];
  #pragma unroll
  for(int p=0;p<4;++p)words[p]=*reinterpret_cast<const uint32_t*>(s.b[p]+off);
  float values[4][2];
  #pragma unroll
  for(int j=0;j<2;++j){
   float b11=float(T::bitcast(uint16_t(words[0]>>(j*16)))),b12=float(T::bitcast(uint16_t(words[1]>>(j*16))));
   float b21=float(T::bitcast(uint16_t(words[2]>>(j*16)))),b22=float(T::bitcast(uint16_t(words[3]>>(j*16))));
   float t1=__fsub_rn(b12,b11),t2=__fsub_rn(b22,t1);
   values[0][j]=t1;values[1][j]=t2;values[2][j]=__fsub_rn(b22,b12);values[3][j]=__fsub_rn(t2,b21);
  }
  #pragma unroll
  for(int p=0;p<4;++p)*reinterpret_cast<uint32_t*>(s.b[p==0?1:p+3]+off)=pair<T>(values[p][0],values[p][1]);
 }
}
template<class T>__device__ __forceinline__ void tensor_chain(Shared<T>& s,int tid,int m0,int n0,float* D){
 using namespace cute;
 constexpr auto la=tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_32,Int<BK>>{});
 auto a=make_tensor(make_smem_ptr(s.operand[0].b[0]),la);
 auto b=make_tensor(make_smem_ptr(s.operand[0].a[0]),lb);
 auto da_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(a,Shape<_64,_16>{}));
 auto db_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(b,Shape<_32,_16>{}));
 int mn=GMMA::CLayout_64x32{}(tid,0);
 float* out=D+(n0+mn/64)*1024+m0+mn%64;
'''
for dtype,cond in [('bf16','if constexpr(std::is_same_v<T,bfloat16_t>)'),('f16','else')]:
 lines=['{ .reg .pred p,acc,leader,done,last; .reg .b64 ba,bb,da,db,dst;','.reg .b32 rb,fb,q,phase; .reg .f32 '+','.join(f'p{i}{j}' for i in range(7) for j in range(16))+',x,y,z;','setp.ne.u32 p,1,0;','setp.eq.u32 leader,%4,0;','mov.u32 q,0;','LOOP_K:','and.b32 phase,q,1;','WAIT_PACK:','mbarrier.try_wait.parity.shared::cta.b64 done,[%2],phase,0x989680;','@!done bra WAIT_PACK;','setp.ne.u32 acc,q,0;','wgmma.fence.sync.aligned;']
 # Natural product order here, each has its own persistent accumulator.
 for p,bp,ap in [(0,0,0),(1,2,1),(2,3,6),(3,6,3),(4,1,2),(5,4,4),(6,5,5)]:
  lines += [f'add.u64 ba,%0,{bp*512};',f'add.u64 bb,%1,{ap*256};']
  for k in range(4):
   lines += [f'add.u64 da,ba,%{6+2*k};',f'add.u64 db,bb,%{7+2*k};',f'wgmma.mma_async.sync.aligned.m64n32k16.f32.{dtype}.{dtype} {{'+','.join(f'p{p}{j}' for j in range(16))+f'}},da,db,'+('acc' if k==0 else 'p')+',1,1,%30,%31;']
  lines+=['wgmma.commit_group.sync.aligned;']
 lines+=['wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;','@leader mbarrier.arrive.shared::cta.b64 _,[%3];','add.u32 q,q,1;','setp.lt.u32 last,q,4;','@last bra LOOP_K;']
 body=lines[lines.index('wgmma.fence.sync.aligned;')+1:lines.index('wgmma.wait_group.sync.aligned 0;')]
 lines=lines[:lines.index('mov.u32 q,0;')]
 lines[0]=lines[0].replace('p,acc,','p,acc,noacc,')
 lines.append('setp.eq.u32 noacc,1,0;')
 for qidx in range(8):
  slot=qidx%2;phase=qidx//2%2
  lines.extend([f'add.u32 rb,%2,{slot*8};',f'WAIT_PACK_{qidx}:',f'mbarrier.try_wait.parity.shared::cta.b64 done,[rb],{phase},0x989680;',f'@!done bra WAIT_PACK_{qidx};','wgmma.fence.sync.aligned;'])
  import re
  for x in body:
   x=x.replace(',acc,',',noacc,' if qidx==0 else ',p,')
   x=re.sub(r'(add.u64 b[ab],%[01],)([0-9]+);',lambda m:m[1]+str(int(m[2])+slot*5376)+';',x)
   lines.append(x)
  lines.extend(['wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;',f'add.u32 fb,%3,{slot*8};','@leader mbarrier.arrive.shared::cta.b64 _,[fb];'])
 for j in range(16):
  # original C11=P1+P2;U2=P1+P6;U3=U2+P7;U4=U2+P5
  for outoffset,ops in [(0,[f'add.rn.f32 x,p0{j},p1{j};']),(2099200,[f'add.rn.f32 y,p0{j},p5{j};',f'add.rn.f32 z,y,p6{j};',f'add.rn.f32 x,z,p4{j};']),(2048,[f'add.rn.f32 y,y,p4{j};',f'add.rn.f32 x,y,p2{j};']),(2097152,[f'sub.rn.f32 x,z,p3{j};'])]:
   lines+=ops+[f'add.u64 dst,%5,%{14+j};']
   if outoffset:lines.append(f'add.u64 dst,dst,{outoffset};')
   lines.append('st.global.f32 [dst],x;')
 lines+=['}']
 ins=['"l"(uint64_t(da_tiles(0,0,0)))','"l"(uint64_t(db_tiles(0,0,0)))','"r"(uint32_t(__cvta_generic_to_shared(s.pack)))','"r"(uint32_t(__cvta_generic_to_shared(s.free)))','"r"(tid&31)','"l"(out)']
 for k in range(4):ins+=['"n"(uint32_t(da_tiles.layout()(0,0,Int<%d>{})))'%k,'"n"(uint32_t(db_tiles.layout()(0,0,Int<%d>{})))'%k]
 for j in range(16):
  mn='int(GMMA::CLayout_64x32{}(0,Int<%d>{}))'%j
  ins+=['"n"(4*(%s%%64+(%s/64)*1024))'%(mn,mn)]
 ins+=['"n"(int32_t(GMMA::Major::MN))','"n"(int32_t(GMMA::Major::K))']
 s+=' '+cond+'{asm volatile(\n'+''.join('  "'+v+'\\n"\n' for v in lines)+'  : : '+', '.join(ins)+' : "memory");}\n'
s+='''}
template<class T>__global__ __launch_bounds__(Threads,1)
void sw_cooperative_hilo(const __grid_constant__ CUtensorMap amap,const __grid_constant__ CUtensorMap bmap,float* D){
 using namespace cute;
 extern __shared__ __align__(128) unsigned char buffer[];
 auto& s=*reinterpret_cast<Shared<T>*>(buffer);
 int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;
 if(tid==128){
  for(int i=0;i<2;++i){Barrier::init(&s.raw[i],1);Barrier::init(&s.pack[i],256);Barrier::init(&s.free[i],4);}
  cutlass::arch::fence_barrier_init();
 }
 __syncthreads();
 if(tid<128)tensor_chain(s,tid,m0,n0,D);
 else{
  for(int q=0;q<8;++q){
   int slot=q%2,phase=q/2%2;
   auto& op=s.operand[slot];
   if(q>=2)wait(&s.free[slot],(q/2-1)%2);
   if(tid==128){
    TxBarrier::arrive_and_expect_tx(&s.raw[slot],8*(ASize+BSize));
    #pragma unroll
    for(int p=0;p<4;++p){
     SM90_TMA_LOAD_2D::copy(&bmap,&s.raw[slot],0,op.b[p],m0+(p%2)*512,q*BK+(p/2)*512);
     SM90_TMA_LOAD_3D::copy(&amap,&s.raw[slot],0,op.a[p],0,n0+(p/2)*512,(q*BK+(p%2)*512)/64);
    }
   }
   wait(&s.raw[slot],q%2);
   transform(op,tid-128);
   asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
   arrive(&s.pack[slot]);
  }
 }
}
'''
a=base.index('void check(');b=base.index('} // namespace',a)
s+=base[a:b]+'} // namespace\nstruct SWState{DType dtype;};\n'
a=base.index('std::shared_ptr<SWState> make_sw_pipeline');b=base.index('void run_sw_pipeline',a);v=base[a:b]
# Primary kernel no longer exists, and standalone compute does not require cooperative launch.
v='\n'.join(x for x in v.split('\n') if 'sw_transform_pdl<' not in x)
s+=v
s+='''void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream){
 CUtensorMap a{},b{};encode(a,A,state.dtype,true,false);encode(b,B,state.dtype,false,false);
 if(state.dtype==DType::BF16)
  sw_cooperative_hilo<cute::bfloat16_t><<<dim3(16,8),Threads,sizeof(Shared<cute::bfloat16_t>),stream>>>(a,b,D);
 else sw_cooperative_hilo<cute::half_t><<<dim3(16,8),Threads,sizeof(Shared<cute::half_t>),stream>>>(a,b,D);
 check(cudaPeekAtLastError(),"SW local transform launch");
}
} // namespace megafmm::detail
'''
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1bdfb29:solution/plan.cpp'],text=True)
a=p.index('  if (impl_->sw) return "');b=p.index('\n',a)
p=p[:a]+'''  if (impl_->sw) return "sw_local_shared_transform;threads=384;warp_roles=8cuda+4wgmma;tma_leader=128;smem=172160;bk=64;operand_stages=2;accumulators=7x16;transform=fp32_pair_final_rn;leaf=Btrans_Atrans;publication=local_double_buffer_barriers;workspace_reserved=4194304;workspace_used=0;current_pointer_maps=2;leaf_terms=7;residual_correction=off;alpha=1;beta=0";'''+p[b:]
Path('solution/plan.cpp').write_text(p)

from pathlib import Path
import subprocess
old=subprocess.check_output(['git','show',subprocess.check_output(['git','log','--format=%H','--grep=iter 153','-1'],text=True).strip()+':solution/sw_pipeline.cu'],text=True)
start='''#include "internal.h"
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <stdexcept>
#include <type_traits>
namespace megafmm::detail {
namespace {
constexpr int BM=64,BN=32,BK=256,ASize=BM*BK,BSize=BK*BN,Elements=BM*BN;
constexpr int Threads=768,Workers=608;
constexpr size_t WorkspaceBytes=size_t(7)*512*512*4;
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
template<class T>struct alignas(128) Shared{
 T raw[4*(ASize+BSize)];
 T extra_b[BSize];
 float product4[Elements];
 uint64_t ready,free,brequest,bready,arequest,aready;
};
static_assert(sizeof(Shared<cute::half_t>)==221312,"shared budget");
__device__ void wait(const uint64_t* p,int phase){Barrier::wait(p,phase);asm volatile("" ::: "memory");}
__device__ void arrive(const uint64_t* p){asm volatile("" ::: "memory");Barrier::arrive(p);}
'''
a=old.index('template<class T,bool IsA,class Layout>');b=old.index('template<class T,class LA,class LB>',a)
transform=old[a:b].replace('w+=224','w+=Workers').replace('*reinterpret_cast<uint32_t*>(extra+idx)=bits[3];','if constexpr(!IsA)*reinterpret_cast<uint32_t*>(extra+idx)=bits[3];')
fn='''template<class T>__device__ __forceinline__ void tensor_chain(Shared<T>& s,int tid,int m0,int n0,float* W,float* D){
 using namespace cute;
 constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_MN_SW64_Atom<T>{},Shape<_32,Int<BK>>{});
 auto at=make_tensor(make_smem_ptr(s.raw),la);
 auto bt=make_tensor(make_smem_ptr(s.raw+4*ASize),lb);
 auto et=make_tensor(make_smem_ptr(s.extra_b),lb);
 auto da_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(at,Shape<_64,_16>{}));
 auto db_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(bt,Shape<_32,_16>{}));
 auto de_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(et,Shape<_32,_16>{}));
 int lane=tid-128,cta=blockIdx.y*gridDim.x+blockIdx.x;
 int mn=GMMA::CLayout_64x32{}(lane,0);
 float* out=D+(m0+mn%64)*1024+n0+mn/64;
 float* partial=W+cta*7*Elements+lane*4;
'''
for dtype,cond in [('bf16','if constexpr(std::is_same_v<T,bfloat16_t>)'),('f16','else')]:
 lines=['{ .reg .pred leader,done,p; .reg .b64 ba,bb,da,db,gaddr,od;', '.reg .b32 rb,fb,sp3,sp4; .reg .f32 d<16>,first<16>,second<16>,x,y,t0,t1;', 'setp.eq.u32 leader,%5,0;','setp.ne.u32 p,1,0;', 'mad.lo.u32 sp3,%4,16,%3;','add.u32 sp3,sp3,196608;','mad.lo.u32 sp4,%4,16,%3;','add.u32 sp4,sp4,212992;']
 def waitbar(field,phase,label):
  lines.extend(['add.u32 rb,%%3,%d;'%field,label+':','mbarrier.try_wait.parity.shared::cta.b64 done,[rb],%d,0x989680;'%phase,'@!done bra '+label+';'])
 def request(field):
  lines.extend(['bar.warp.sync 0xffffffff;','add.u32 fb,%%3,%d;'%field,'@leader mbarrier.arrive.shared::cta.b64 _,[fb];'])
 def pair_store(pair,quadrant):
  lines.append('add.u64 od,%%7,%%%d;'%(40+pair))
  if quadrant:lines.append('add.u64 od,od,%d;'%quadrant)
  lines.append('st.global.v2.f32 [od],{x,y};')
 def vecshared(base,load=False):
  for g in range(4):
   lines.append('add.u32 fb,%s,%d;'%(base,g*2048))
   lines.append('st.shared.v4.f32 [fb],{'+','.join('d%d'%(g*4+j) for j in range(4))+'};')
 for q in range(2):
  waitbar(221184,q,'RAW_%d'%q)
  for index,pid in enumerate([0,1,3,2,5,6,4]):
   if index==2:request(221200);waitbar(221208,q,'BTRANS_%d'%q)
   if index==3:request(221216);waitbar(221224,q,'ATRANS_%d'%q)
   # First K stores partial leaves; second K starts at the exact same FP32 values.
   if q==0:lines.extend(['mov.f32 d%d,0f00000000;'%j for j in range(16)])
   else:
    for g in range(4):
     lines.append('add.u64 gaddr,%%6,%d;'%(pid*8192+g*2048))
     lines.append('ld.global.v4.f32 {'+','.join('d%d'%(g*4+j) for j in range(4))+'},[gaddr];')
   ap={0:0,1:1,3:3,2:3,5:1,6:2,4:0}[pid]
   bp={0:0,1:2,3:3,2:0,5:1,6:2,4:0}[pid]
   lines.extend(['wgmma.fence.sync.aligned;','add.u64 ba,%%0,%d;'%(ap*2048)])
   if pid==2:lines.append('mov.b64 bb,%2;')
   else:lines.append('add.u64 bb,%%1,%d;'%(bp*1024))
   for k in range(16):
    lines.extend(['add.u64 da,ba,%%%d;'%(8+2*k),'add.u64 db,bb,%%%d;'%(9+2*k),'wgmma.mma_async.sync.aligned.m64n32k16.f32.%s.%s {%s},da,db,p,1,1,%%48,%%49;'%(dtype,dtype,','.join('d%d'%j for j in range(16)))])
   lines.extend(['wgmma.commit_group.sync.aligned;','wgmma.wait_group.sync.aligned 0;'])
   if q==0:
    for g in range(4):
     lines.append('add.u64 gaddr,%%6,%d;'%(pid*8192+g*2048))
     lines.append('st.global.v4.f32 [gaddr],{'+','.join('d%d'%(g*4+j) for j in range(4))+'};')
   elif pid==0:lines.extend(['mov.f32 first%d,d%d;'%(j,j) for j in range(16)])
   elif pid==1:
    for pair in range(8):
     lines.extend(['add.rn.f32 x,first%d,d%d;'%(2*pair,2*pair),'add.rn.f32 y,first%d,d%d;'%(2*pair+1,2*pair+1)]);pair_store(pair,0)
   elif pid==3:vecshared('sp4')
   elif pid==2:vecshared('sp3')
   elif pid==5:lines.extend(['add.rn.f32 first%d,first%d,d%d;'%(j,j,j) for j in range(16)])
   elif pid==6:lines.extend(['add.rn.f32 second%d,first%d,d%d;'%(j,j,j) for j in range(16)])
   else:
    for pair in range(8):
     j=pair*2
     lines.extend(['add.rn.f32 x,second%d,d%d;'%(j,j),'add.rn.f32 y,second%d,d%d;'%(j+1,j+1)])
     pair_store(pair,2099200)
     lines.extend(['add.rn.f32 first%d,first%d,d%d;'%(j,j,j),'add.rn.f32 first%d,first%d,d%d;'%(j+1,j+1,j+1),'add.u32 fb,sp3,%d;'%((j//4)*2048+(j%4)*4),'ld.shared.v2.f32 {t0,t1},[fb];','add.rn.f32 x,first%d,t0;'%j,'add.rn.f32 y,first%d,t1;'%(j+1)])
     pair_store(pair,2048)
     lines.extend(['add.u32 fb,sp4,%d;'%((j//4)*2048+(j%4)*4),'ld.shared.v2.f32 {t0,t1},[fb];','sub.rn.f32 x,second%d,t0;'%j,'sub.rn.f32 y,second%d,t1;'%(j+1)])
     pair_store(pair,2097152)
  request(221192)
 lines.append('}')
 fn+=' '+cond+'{\n  asm volatile(\n'+''.join('   "'+v+'\\n"\n' for v in lines)
 ins=['"l"(uint64_t(da_tiles(0,0,0)))','"l"(uint64_t(db_tiles(0,0,0)))','"l"(uint64_t(de_tiles(0,0,0)))','"r"(uint32_t(__cvta_generic_to_shared(&s)))','"r"(lane)','"r"(tid&31)','"l"(partial)','"l"(out)']
 for k in range(16):ins.extend(['"n"(uint32_t(da_tiles.layout()(0,0,Int<%d>{})))'%k,'"n"(uint32_t(db_tiles.layout()(0,0,Int<%d>{})))'%k])
 for pair in range(8):
  mn='int(GMMA::CLayout_64x32{}(0,Int<%d>{}))'%(2*pair)
  ins.append('"n"(4*((%s%%64)*1024+%s/64))'%(mn,mn))
 ins.extend(['"n"(int32_t(GMMA::Major::K))','"n"(int32_t(GMMA::Major::MN))'])
 fn+='   :\n   : '+', '.join(ins)+'\n   : "memory");\n }\n'
fn+='}\n'
kernel='''template<class T>__global__ __launch_bounds__(Threads,1)
void sw_cooperative_hilo(const __grid_constant__ CUtensorMap amap,const __grid_constant__ CUtensorMap bmap,float* W,float* D){
 using namespace cute;
 extern __shared__ __align__(128) unsigned char buffer[];
 auto& s=*reinterpret_cast<Shared<T>*>(buffer);
 int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;
 if(tid==0){
  Barrier::init(&s.ready,1);Barrier::init(&s.free,4);
  Barrier::init(&s.brequest,4);Barrier::init(&s.bready,Workers);
  Barrier::init(&s.arequest,4);Barrier::init(&s.aready,Workers);
  cutlass::arch::fence_barrier_init();
 }
 __syncthreads();
 constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_MN_SW64_Atom<T>{},Shape<_32,Int<BK>>{});
 if(tid==0){
  for(int q=0;q<2;++q){
   if(q)wait(&s.free,0);
   TxBarrier::arrive_and_expect_tx(&s.ready,2*4*(ASize+BSize));
   #pragma unroll
   for(int j=0;j<4;++j){
    SM90_TMA_LOAD_3D::copy(&amap,&s.ready,0,s.raw+j*ASize,0,m0+(j/2)*512,(q*BK+(j%2)*512)/64);
    SM90_TMA_LOAD_2D::copy(&bmap,&s.ready,0,s.raw+4*ASize+j*BSize,n0+(j%2)*512,q*BK+(j/2)*512);
   }
  }
 }else if(tid>=128 && tid<256){
  tensor_chain(s,tid,m0,n0,W,D);
 }else if((tid>=32 && tid<128)||tid>=256){
  int worker=tid>=256?tid-160:tid-32;
  for(int q=0;q<2;++q){
   wait(&s.brequest,q);
   transform_shared<T,false>(s.raw+4*ASize,s.extra_b,lb,worker);
   asm volatile("fence.proxy.async.shared::cta;" ::: "memory");arrive(&s.bready);
   wait(&s.arequest,q);
   transform_shared<T,true>(s.raw,static_cast<T*>(nullptr),la,worker);
   asm volatile("fence.proxy.async.shared::cta;" ::: "memory");arrive(&s.aready);
  }
 }
}
'''
host=old[old.index('void check(cudaError_t'):]
host=host.replace('if(blocks*prop.multiProcessorCount<128)throw std::invalid_argument("SW cooperative grid requires128 resident CTA capacity");','if(blocks<1)throw std::invalid_argument("SW needs one resident CTA");')
a=host.index('void run_sw_pipeline(')
host=host[:a]+'''void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream){
 CUtensorMap a{},b{};encode(a,A,state.dtype,true,false);encode(b,B,state.dtype,false,false);
 dim3 grid(16,8);
 if(state.dtype==DType::BF16)sw_cooperative_hilo<cute::bfloat16_t><<<grid,Threads,sizeof(Shared<cute::bfloat16_t>),stream>>>(a,b,static_cast<float*>(workspace),D);
 else sw_cooperative_hilo<cute::half_t><<<grid,Threads,sizeof(Shared<cute::half_t>),stream>>>(a,b,static_cast<float*>(workspace),D);
 check(cudaGetLastError(),"SW local FP32 partial-leaf launch");
}
} // namespace megafmm::detail
'''
Path('solution/sw_pipeline.cu').write_text(start+transform+fn+kernel+host)
p=Path('solution/plan.cpp');s=p.read_text().replace('size_t(4)<<20','size_t(7)<<20').replace('SW needs4MiB','SW needs7MiB')
a=s.index('"sw_cooperative;');b=s.index('"',a+1)
s=s[:a]+'"sw_local_shared;threads=768;warp_roles=1tma_4tc_19transform;smem=221312;bk=256;raw_slots=1;workspace=7340032;leaf_terms=7;residual_correction=off;partial_fp32_leaves=workspace;transform=local_inplace_B_then_A;recombine=static_ptx_original_order;grid_publication=none;current_pointer_maps=2"'+s[b+1:]
p.write_text(s)

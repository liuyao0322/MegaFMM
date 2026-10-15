// Pre-MMA only: TMA bytes/layout/two-slot reuse and output-coordinate coverage.
#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#include "p52/corrected.cuh"

using namespace int8_research;
struct alignas(128) ProbeShared {
 uint8_t before[1024];Shared core;uint8_t after[128];
};
static_assert(sizeof(ProbeShared)==83200);
__global__ void synthetic_planes(uint16_t* a,uint16_t* b,int seed,int trial){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<11*Plane;i+=gridDim.x*blockDim.x){
  unsigned x=trial==0?unsigned(i):trial==1?unsigned(i>>16):unsigned(hash(uint64_t(i)^uint64_t(seed))>>32);
  unsigned y=trial==0?unsigned(i):trial==1?unsigned(i>>16):unsigned(hash(uint64_t(i)^uint64_t(seed)^0x632be59bd9b4e019ULL)>>32);
  a[i]=uint16_t(x+unsigned(seed));b[i]=uint16_t(y+unsigned(seed)*3u+17u);
 }
}
__global__ void init_coverage(unsigned* c){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Count+256;i+=gridDim.x*blockDim.x)c[i]=(i<128||i>=Count+128)?1234567u:0u;
}
template<class T,bool SideA> __device__ __forceinline__ unsigned compare_operand(
 const uint8_t* source,const uint16_t* global,int plane,int m0,int n0,int k0,int lane){
 using namespace cute;
 constexpr auto layout=[](){
  if constexpr(SideA)return tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_16,_128>{});
  else return tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,_128>{});
 }();
 const uint16_t* data=reinterpret_cast<const uint16_t*>(source);unsigned errors=0;
 for(int i=lane;i<(SideA?2048:8192);i+=128){
  int row=SideA?i/128:i%64,k=SideA?i%128:i/64;
  int at=as_position_independent_swizzle_layout(layout)(make_coord(row,k));
  int expected=plane*Plane+(SideA?(n0+row)*512+k0+k:(k0+k)*512+m0+row);
  errors+=data[at]!=global[expected];
 }
 return errors;
}
template<class T> __global__ __launch_bounds__(256,1)
void layout_probe(const __grid_constant__ CUtensorMap high_a,const __grid_constant__ CUtensorMap high_b,
 const uint16_t* expected_a,const uint16_t* expected_b,unsigned* error,unsigned* coverage){
 using namespace cute;
 extern __shared__ __align__(128) unsigned char buffer[];
 auto& outer=*reinterpret_cast<ProbeShared*>(buffer);auto& s=outer.core;
 int tid=threadIdx.x,m0=blockIdx.y*64,n0=blockIdx.x*16;
 for(int i=tid;i<1024;i+=256)outer.before[i]=0x3c;
 if(tid<128)outer.after[tid]=0xc3;
 if(tid==0){for(int j=0;j<ConsumerStages;++j){Barrier::init(&s.ready[j],1);Barrier::init(&s.free[j],4);}cutlass::arch::fence_barrier_init();}
 __syncthreads();
 // PRODUCER_BEGIN: verbatim P52 corrected kernel producer, audited separately.
 if(tid==0){
  for(int step=0;step<7;++step)for(int k=0;k<4;++k){
   int p=main_product(step),low=p<4?10:p+3;
   int seq=step*4+k,slot=seq%ConsumerStages;
   if(seq>=ConsumerStages){Barrier::wait(&s.free[slot],((seq-ConsumerStages)/ConsumerStages)%2);asm volatile("" ::: "memory");}
   int bytes=p<2?20480:p==2?24576:p==3?36864:40960;
   TxBarrier::arrive_and_expect_tx(&s.ready[slot],bytes);
   SM90_TMA_LOAD_3D::copy(&high_b,&s.ready[slot],0,s.operand[slot],m0,k*128,p);
   SM90_TMA_LOAD_4D::copy(&high_a,&s.ready[slot],0,s.operand[slot]+32768,0,n0,k*2,p);
   if(p==2||p>=4)
    SM90_TMA_LOAD_4D::copy(&high_a,&s.ready[slot],0,s.operand[slot]+36864,0,n0,k*2,low);
   if(p==3||p>=4)
    SM90_TMA_LOAD_3D::copy(&high_b,&s.ready[slot],0,s.operand[slot]+16384,m0,k*128,low);
  }
 }
 // PRODUCER_END
 else if(tid>=128){
  unsigned ae=0,be=0;
  for(int step=0;step<7;++step)for(int k=0;k<4;++k){
   int p=main_product(step),low=p<4?10:p+3,seq=step*4+k,slot=seq%ConsumerStages;
   Barrier::wait(&s.ready[slot],seq/ConsumerStages%2);asm volatile("" ::: "memory");
   be+=compare_operand<T,false>(s.operand[slot],expected_b,p,m0,n0,k*128,tid-128);
   ae+=compare_operand<T,true>(s.operand[slot]+32768,expected_a,p,m0,n0,k*128,tid-128);
   if(p==2||p>=4)ae+=compare_operand<T,true>(s.operand[slot]+36864,expected_a,low,m0,n0,k*128,tid-128);
   if(p==3||p>=4)be+=compare_operand<T,false>(s.operand[slot]+16384,expected_b,low,m0,n0,k*128,tid-128);
   // Generic reads have completed; no asynchronous MMA is executed in this probe.
   asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
   if((tid&31)==0)Barrier::arrive(&s.free[slot]);
  }
  if(ae)atomicAdd(error,ae);if(be)atomicAdd(error+1,be);
  for(int j=0;j<8;++j){
   int mn=GMMA::CLayout_64x16{}(tid-128,j),at=(n0+mn/64)*1024+m0+mn%64;
   atomicAdd(coverage+128+at,1u);atomicAdd(coverage+128+at+512,1u);
   atomicAdd(coverage+128+at+512*1024,1u);atomicAdd(coverage+128+at+512*1024+512,1u);
  }
 }
 __syncthreads();
 unsigned guards=0;for(int i=tid;i<1024;i+=256)guards+=outer.before[i]!=0x3c;
 if(tid<128)guards+=outer.after[tid]!=0xc3;if(guards)atomicAdd(error+2,guards);
}
template<class T> int probe(const char* dtype,megafmm::DType dt){
 Stream stream;Buffer workspace(WorkBytes+512),errors(3*sizeof(unsigned)),coverage((Count+256)*sizeof(unsigned));
 Work work(static_cast<char*>(workspace.p)+256);int failures=0;
 check(cudaFuncSetAttribute(layout_probe<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(ProbeShared)));
 for(int trial=0;trial<3;++trial){
  int seed=114067+433*trial+(dt==megafmm::DType::BF16?0:227);
  check(cudaMemsetAsync(workspace.p,0x6d,WorkBytes+512,stream.s));
  synthetic_planes<<<512,256,0,stream.s>>>((uint16_t*)work.ha,(uint16_t*)work.hb,seed,trial);
  init_coverage<<<512,256,0,stream.s>>>(coverage.as<unsigned>());
  check(cudaMemsetAsync(errors.p,0,3*sizeof(unsigned),stream.s));
  CUtensorMap ha,hb;encode(ha,work.ha,dt,true);encode(hb,work.hb,dt,false);
  layout_probe<T><<<dim3(32,8),256,sizeof(ProbeShared),stream.s>>>(ha,hb,(uint16_t*)work.ha,(uint16_t*)work.hb,errors.as<unsigned>(),coverage.as<unsigned>());
  check(cudaGetLastError());
  unsigned result[3];std::vector<unsigned> counts(Count+256);uint8_t guards[512];
  check(cudaMemcpyAsync(result,errors.p,sizeof(result),cudaMemcpyDeviceToHost,stream.s));
  check(cudaMemcpyAsync(counts.data(),coverage.p,counts.size()*sizeof(unsigned),cudaMemcpyDeviceToHost,stream.s));
  check(cudaMemcpyAsync(guards,workspace.p,256,cudaMemcpyDeviceToHost,stream.s));
  check(cudaMemcpyAsync(guards+256,(char*)workspace.p+256+WorkBytes,256,cudaMemcpyDeviceToHost,stream.s));
  check(cudaStreamSynchronize(stream.s));
  int coverage_errors=0,guard_errors=0;for(int i=0;i<Count+256;++i)coverage_errors+=counts[i]!=((i<128||i>=Count+128)?1234567u:1u);
  for(auto x:guards)guard_errors+=x!=0x6d;
  bool pass=result[0]==0&&result[1]==0&&result[2]==0&&coverage_errors==0&&guard_errors==0;failures+=!pass;
  printf("CONSUMER_LAYOUT dtype=%s trial=%d seed=%d ctas=256 sequences=28 slots=2 native_comparisons=115343360 a_errors=%u b_errors=%u shared_guard_errors=%u coverage_errors=%d workspace_guard_errors=%d pass=%d\n",dtype,trial,seed,result[0],result[1],result[2],coverage_errors,guard_errors,pass);fflush(stdout);
 }
 return failures;
}
int main(){try{
 // Registers the candidate kernels and queries capacity; never launches corrected.
 int8_research::init();
 return (probe<cute::bfloat16_t>("bf16",megafmm::DType::BF16)+probe<cute::half_t>("fp16",megafmm::DType::FP16))?2:0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}

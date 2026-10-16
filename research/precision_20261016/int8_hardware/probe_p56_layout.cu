// Own new-seed full-grid pre-MMA P56 native TMA/ring/layout probe.
#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#include "p56/corrected.cuh"
using namespace int8_research;
struct alignas(128) ProbeShared56 {uint8_t before[1024];Shared core;uint8_t after[128];};
static_assert(sizeof(ProbeShared56)==197888);
__global__ void synthetic56(uint16_t* ha,uint16_t* hb,int seed,int trial){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<11*Plane;i+=gridDim.x*blockDim.x){
  unsigned x=trial==0?unsigned(i):trial==1?unsigned(i>>16):unsigned(hash(uint64_t(i)^uint64_t(seed))>>32);
  unsigned y=trial==0?unsigned(i):trial==1?unsigned(i>>16):unsigned(hash(uint64_t(i)^uint64_t(seed)^0x632be59bd9b4e019ULL)>>32);
  ha[i]=uint16_t(x+seed);hb[i]=uint16_t(y+3u*seed+17u);
 }
}
__global__ void coverage_init56(unsigned* x){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Plane+256;i+=gridDim.x*blockDim.x)x[i]=(i<128||i>=Plane+128)?1234567u:0u;}
template<class T,bool LogicalA> __device__ unsigned native56(const uint8_t* s,const uint16_t* expected,int p,int n0,int m0,int k0,int lane){
 using namespace cute;
 constexpr auto layout=[](){if constexpr(LogicalA)return tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_32,_128>{});else return tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,_128>{});}();
 unsigned errors=0;
 for(int i=lane;i<(LogicalA?4096:8192);i+=128){
  int r=LogicalA?i/128:i%64,k=LogicalA?i%128:i/64;
  int at=as_position_independent_swizzle_layout(layout)(make_coord(r,k));
  int src=p*Plane+(LogicalA?(n0+r)*512+k0+k:(k0+k)*512+m0+r);
  errors+=reinterpret_cast<const uint16_t*>(s)[at]!=expected[src];
 }
 return errors;
}
template<class T> __global__ __launch_bounds__(256,1)
void layout56(const __grid_constant__ CUtensorMap high_a,const __grid_constant__ CUtensorMap high_b,
 const uint16_t* expected_a,const uint16_t* expected_b,unsigned* error,unsigned* coverage){
 using namespace cute;
 extern __shared__ __align__(128) unsigned char buffer[];
 auto& outer=*reinterpret_cast<ProbeShared56*>(buffer);auto& s=outer.core;
 int tid=threadIdx.x,m0=blockIdx.y*64,n0=blockIdx.x*32;
 for(int i=tid;i<1024;i+=256)outer.before[i]=0x3c;if(tid<128)outer.after[tid]=0xc3;
 if(tid==0){for(int j=0;j<Stages;++j){Barrier::init(&s.ready[j],1);Barrier::init(&s.free[j],4);}cutlass::arch::fence_barrier_init();}
 __syncthreads();
 // PRODUCER_BEGIN
 if(tid==0){
  for(int step=0;step<7;++step)for(int k=0;k<4;++k){
   int p=main_product(step),low=p<4?10:p+3;
   int seq=step*4+k,slot=seq%Stages;
   if(seq>=Stages){Barrier::wait(&s.free[slot],((seq-Stages)/Stages)%2);asm volatile("" ::: "memory");}
   int bytes=p<2?24576:p==2?32768:p==3?40960:49152;
   TxBarrier::arrive_and_expect_tx(&s.ready[slot],bytes);
   SM90_TMA_LOAD_3D::copy(&high_b,&s.ready[slot],0,s.operand[slot],m0,k*128,p);
   SM90_TMA_LOAD_4D::copy(&high_a,&s.ready[slot],0,s.operand[slot]+32768,0,n0,k*2,p);
   if(p==2||p>=4)
    SM90_TMA_LOAD_4D::copy(&high_a,&s.ready[slot],0,s.operand[slot]+40960,0,n0,k*2,low);
   if(p==3||p>=4)
    SM90_TMA_LOAD_3D::copy(&high_b,&s.ready[slot],0,s.operand[slot]+16384,m0,k*128,low);
  }
 }
 // PRODUCER_END
 else if(tid>=128){
  unsigned ne=0;const int order[7]={0,1,5,6,4,2,3};
  for(int step=0;step<7;++step)for(int k=0;k<4;++k){
   int p=order[step],seq=step*4+k,slot=seq%Stages;
   Barrier::wait(&s.ready[slot],seq/Stages%2);asm volatile("" ::: "memory");
   ne+=native56<T,false>(s.operand[slot],expected_b,p,n0,m0,k*128,tid-128);
   ne+=native56<T,true>(s.operand[slot]+32768,expected_a,p,n0,m0,k*128,tid-128);
   if(p==2||p>=4)ne+=native56<T,true>(s.operand[slot]+40960,expected_a,p==2?10:p+3,n0,m0,k*128,tid-128);
   if(p==3||p>=4)ne+=native56<T,false>(s.operand[slot]+16384,expected_b,p==3?10:p+3,n0,m0,k*128,tid-128);
   // Generic reads only; not a WGMMA wait-order certificate.
   asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");if((tid&31)==0)Barrier::arrive(&s.free[slot]);
  }
  for(int j=0;j<16;++j){int mn=GMMA::CLayout_64x32{}(tid-128,j);atomicAdd(coverage+128+(n0+mn/64)*512+m0+mn%64,1u);}
  if(ne)atomicAdd(error,ne);
 }
 __syncthreads();unsigned ge=0;for(int i=tid;i<1024;i+=256)ge+=outer.before[i]!=0x3c;
 if(tid<128)ge+=outer.after[tid]!=0xc3;if(ge)atomicAdd(error+1,ge);
}
template<class T> int probe56(const char* dtype,megafmm::DType dt){
 Stream stream;Buffer work(WorkBytes+512),error(2*sizeof(unsigned)),coverage((Plane+256)*4);Work w((char*)work.p+256);
 check(cudaFuncSetAttribute(layout56<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(ProbeShared56)));
 cudaFuncAttributes prep{},consumer{};int active=0,device=0;cudaDeviceProp prop{};
 check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
 check(cudaFuncGetAttributes(&prep,preprocess_cooperative_columns<T>));check(cudaFuncGetAttributes(&consumer,corrected<T>));
 check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,corrected<T>,256,sizeof(Shared)));require(active>=1,"candidate consumer capacity");
 printf("STATIC_RESOURCE dtype=%s prep_regs=%d consumer_regs=%d shared_bytes=%zu active_blocks_per_sm=%d sm_count=%d\n",dtype,prep.numRegs,consumer.numRegs,sizeof(Shared),active,prop.multiProcessorCount);fflush(stdout);
 int failures=0;
 for(int trial=0;trial<3;++trial){
  int seed=128171+491*trial+(dt==megafmm::DType::BF16?0:269);
  check(cudaMemsetAsync(work.p,0x6d,WorkBytes+512,stream.s));check(cudaMemsetAsync(error.p,0,8,stream.s));
  synthetic56<<<512,256,0,stream.s>>>((uint16_t*)w.ha,(uint16_t*)w.hb,seed,trial);
  coverage_init56<<<512,256,0,stream.s>>>(coverage.as<unsigned>());
  CUtensorMap ha,hb;encode(ha,w.ha,dt,true);encode(hb,w.hb,dt,false);
  layout56<T><<<dim3(16,8),256,sizeof(ProbeShared56),stream.s>>>(ha,hb,(uint16_t*)w.ha,(uint16_t*)w.hb,error.as<unsigned>(),coverage.as<unsigned>());check(cudaGetLastError());
  unsigned result[2];std::vector<unsigned> cov(Plane+256);uint8_t guards[512];
  check(cudaMemcpyAsync(result,error.p,8,cudaMemcpyDeviceToHost,stream.s));check(cudaMemcpyAsync(cov.data(),coverage.p,cov.size()*4,cudaMemcpyDeviceToHost,stream.s));
  check(cudaMemcpyAsync(guards,work.p,256,cudaMemcpyDeviceToHost,stream.s));check(cudaMemcpyAsync(guards+256,(char*)work.p+256+WorkBytes,256,cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
  unsigned ce=0,ge=0;for(int i=0;i<Plane+256;++i)ce+=cov[i]!=((i<128||i>=Plane+128)?1234567u:1u);for(auto x:guards)ge+=x!=0x6d;
  bool pass=result[0]==0&&result[1]==0&&ce==0&&ge==0;failures+=!pass;
  printf("STATIC_LAYOUT dtype=%s trial=%d seed=%d ctas=128 sequences=28 slots=4 native_comparisons=69206016 coordinates=262144 native_errors=%u shared_guard_errors=%u coverage_errors=%u workspace_guard_errors=%u pass=%d\n",dtype,trial,seed,result[0],result[1],ce,ge,pass);fflush(stdout);
 }
 return failures;
}
int main(){try{int8_research::init();return probe56<cute::bfloat16_t>("bf16",megafmm::DType::BF16)+probe56<cute::half_t>("fp16",megafmm::DType::FP16)?2:0;}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}

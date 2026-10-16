// Own new-seed full-grid pre-MMA P59 native TMA/ring/layout probe.
#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#define int8_research p56_reference
#include "p56/pack.cuh"
#undef int8_research
#include "p59/corrected.cuh"
using namespace int8_research;
struct alignas(128) ProbeShared59 {uint8_t before[1024];Shared core;uint8_t after[128];};
static_assert(sizeof(ProbeShared59)==197888);
template<class T> __global__ void varied_fill(T* a,T* b,double* ad,double* bd,int pattern,int seed){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Count;i+=gridDim.x*blockDim.x){
  int r=i/N,c=i%N;
  int ai=pattern==2?r*N+c%512:i,bi=pattern==2?(r%512)*N+c:i;
  float x=float(hash(uint64_t(ai)^seed)>>40)*(2.f/16777216.f)-1;
  float y=float(hash(uint64_t(bi)^seed^0x632be59bd9b4e019ULL)>>40)*(2.f/16777216.f)-1;
  if(pattern==0){
   constexpr bool bf=std::is_same_v<T,__nv_bfloat16>;
   x=ldexpf(x,bf?60:10);y=ldexpf(y,bf?-60:-10);
  }
  if(pattern==1){x=ldexpf(x,r<512?6:-6);y=ldexpf(y,c<512?-6:6);}
  if(pattern==2){if(c>=512)x=-x+1.f/256;}
  if(pattern==3){x=ldexpf(x,c<512?0:-12);y=ldexpf(y,r<512?-12:0);}
  if(pattern==4){x=ldexpf(x,int(hash(i+seed)%17)-8);y=ldexpf(y,int(hash(i+seed+1)%17)-8);}
  if(pattern==5){x=(r==c)?1:0;y=y*2;}
  if(pattern==6){bool af=((c%2)==1)^(c>=512),bf=((r%2)==1)^(r>=512);x=ldexpf(x,af?-12:0);y=ldexpf(y,bf?0:-12);}
  // Different K128 restoration exponents force the original fallback.
  if(pattern==7){x=ldexpf(x,4*((c%512)/128)-6);y=ldexpf(y,4*((r%512)/128)-6);}
  // BF16 restoration exponents below the common-path guard; FP16 stays representable.
  if(pattern==8){constexpr bool bf=std::is_same_v<T,__nv_bfloat16>;x=ldexpf(x,bf?-60:-10);y=ldexpf(y,bf?-60:-10);}
  if(pattern==9||pattern==10){
   constexpr bool bf=std::is_same_v<T,__nv_bfloat16>;
   int hi=bf?127:15,lo=bf?-132:-24;
   x=ldexpf(x,pattern==9?hi:lo);y=ldexpf(y,pattern==9?lo:hi);
  }
  if(pattern==11){
   constexpr bool bf=std::is_same_v<T,__nv_bfloat16>;
   x=ldexpf(x,(c%64)<32?(bf?127:15):0);
   y=ldexpf(y,(r%64)<32?(bf?-132:-24):0);
  }
  T xx=rounded<T>(x),yy=rounded<T>(y);
  a[i]=xx;b[i]=yy;ad[i]=float(xx);bd[i]=float(yy);
 }
}

__global__ void coverage_init59(unsigned* x){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Plane+256;i+=gridDim.x*blockDim.x)x[i]=(i<128||i>=Plane+128)?1234567u:0u;}
template<class T,bool LogicalA> __device__ unsigned native59(const uint8_t* s,const uint16_t* expected,int p,int n0,int m0,int k0,int lane){
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
void layout59(const __grid_constant__ CUtensorMap high_a,const __grid_constant__ CUtensorMap high_b,
 const T* input_a,const T* input_b,Work w,const uint16_t* expected_a,const uint16_t* expected_b,unsigned* error,unsigned* coverage){
 using namespace cute;
 extern __shared__ __align__(128) unsigned char buffer[];
 auto& outer=*reinterpret_cast<ProbeShared59*>(buffer);auto& s=outer.core;
 int tid=threadIdx.x,m0=(blockIdx.x/16)*64,n0=(blockIdx.x%16)*32;
 for(int i=tid;i<1024;i+=256)outer.before[i]=0x3c;if(tid<128)outer.after[tid]=0xc3;
 prepare_resident<T>(input_a,input_b,static_cast<T*>(w.ha),static_cast<T*>(w.hb),
  static_cast<T*>(w.qa),static_cast<T*>(w.qb),w.ea,w.eb,w.inside,w.partial,reinterpret_cast<unsigned char*>(&s));
 cooperative_groups::this_grid().sync();
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
   ne+=native59<T,false>(s.operand[slot],expected_b,p,n0,m0,k*128,tid-128);
   ne+=native59<T,true>(s.operand[slot]+32768,expected_a,p,n0,m0,k*128,tid-128);
   if(p==2||p>=4)ne+=native59<T,true>(s.operand[slot]+40960,expected_a,p==2?10:p+3,n0,m0,k*128,tid-128);
   if(p==3||p>=4)ne+=native59<T,false>(s.operand[slot]+16384,expected_b,p==3?10:p+3,n0,m0,k*128,tid-128);
   // Generic reads only; not a WGMMA wait-order certificate.
   asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");if((tid&31)==0)Barrier::arrive(&s.free[slot]);
  }
  for(int j=0;j<16;++j){int mn=GMMA::CLayout_64x32{}(tid-128,j);atomicAdd(coverage+128+(n0+mn/64)*512+m0+mn%64,1u);}
  if(ne)atomicAdd(error,ne);
 }
 __syncthreads();unsigned ge=0;for(int i=tid;i<1024;i+=256)ge+=outer.before[i]!=0x3c;
 if(tid<128)ge+=outer.after[tid]!=0xc3;if(ge)atomicAdd(error+1,ge);
}
__global__ void compare_inside59(const int* expected,const int* actual,unsigned* errors){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<1024&&expected[i]!=actual[i])atomicAdd(errors,1u);
}
template<class Input,class T> int probe59(const char* dtype,megafmm::DType dt){
 Stream stream;Buffer work(WorkBytes+512),error(3*sizeof(unsigned)),coverage((Plane+256)*4);
 Buffer a(Count*2),b(Count*2),ad(Count*8),bd(Count*8),reference_work(p56_reference::WorkBytes);
 Work w((char*)work.p+256);p56_reference::Work expected(reference_work.p);
 p56_reference::init_cooperative_preprocess<T>();
 check(cudaFuncSetAttribute(layout59<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(ProbeShared59)));
 cudaFuncAttributes prep{},consumer{};int active=0,device=0;cudaDeviceProp prop{};
 check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
 check(cudaFuncGetAttributes(&prep,preprocess_cooperative_columns<T>));check(cudaFuncGetAttributes(&consumer,fused_global<T>));
 check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fused_global<T>,256,sizeof(Shared)));require(active>=1,"candidate fused capacity");
 printf("STATIC_RESOURCE dtype=%s prep_regs=%d fused_regs=%d shared_bytes=%zu active_blocks_per_sm=%d sm_count=%d\n",dtype,prep.numRegs,consumer.numRegs,sizeof(Shared),active,prop.multiProcessorCount);fflush(stdout);
 int probe_active=0;
 check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&probe_active,layout59<T>,256,sizeof(ProbeShared59)));
 require(probe_active*prop.multiProcessorCount>=128,"layout probe cooperative capacity");
 printf("FUSED_PROBE_CAPACITY dtype=%s active_blocks_per_sm=%d sm_count=%d grid=128 shared_bytes=%zu\n",dtype,probe_active,prop.multiProcessorCount,sizeof(ProbeShared59));
 int failures=0;
 for(int trial=0;trial<3;++trial){
  int pattern=trial==0?0:trial==1?2:11;int seed=134227+523*trial+(dt==megafmm::DType::BF16?0:283);
  check(cudaMemsetAsync(work.p,0x6d,WorkBytes+512,stream.s));check(cudaMemsetAsync(error.p,0,12,stream.s));
  varied_fill<<<512,256,0,stream.s>>>(a.as<Input>(),b.as<Input>(),ad.as<double>(),bd.as<double>(),pattern,seed);
  p56_reference::preprocess<T,true>(a.as<T>(),b.as<T>(),expected,stream.s);
  coverage_init59<<<512,256,0,stream.s>>>(coverage.as<unsigned>());
  CUtensorMap ha,hb;encode(ha,w.ha,dt,true);encode(hb,w.hb,dt,false);
  const T* input_a=a.as<T>();const T* input_b=b.as<T>();
  auto expected_a=(const uint16_t*)expected.ha;auto expected_b=(const uint16_t*)expected.hb;
  auto error_ptr=error.as<unsigned>();auto coverage_ptr=coverage.as<unsigned>();
  void* args[]={&ha,&hb,&input_a,&input_b,&w,&expected_a,&expected_b,&error_ptr,&coverage_ptr};
  check(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(layout59<T>),dim3(128),dim3(256),args,sizeof(ProbeShared59),stream.s));
  compare_inside59<<<4,256,0,stream.s>>>(expected.inside,w.inside,error.as<unsigned>()+2);check(cudaGetLastError());
  unsigned result[3];std::vector<unsigned> cov(Plane+256);uint8_t guards[512];
  check(cudaMemcpyAsync(result,error.p,12,cudaMemcpyDeviceToHost,stream.s));check(cudaMemcpyAsync(cov.data(),coverage.p,cov.size()*4,cudaMemcpyDeviceToHost,stream.s));
  check(cudaMemcpyAsync(guards,work.p,256,cudaMemcpyDeviceToHost,stream.s));check(cudaMemcpyAsync(guards+256,(char*)work.p+256+WorkBytes,256,cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
  unsigned ce=0,ge=0;for(int i=0;i<Plane+256;++i)ce+=cov[i]!=((i<128||i>=Plane+128)?1234567u:1u);for(auto x:guards)ge+=x!=0x6d;
  bool pass=result[0]==0&&result[1]==0&&result[2]==0&&ce==0&&ge==0;failures+=!pass;
  printf("STATIC_LAYOUT dtype=%s trial=%d seed=%d pattern=%d ctas=128 sequences=28 slots=4 native_comparisons=69206016 coordinates=262144 native_errors=%u shared_guard_errors=%u coverage_errors=%u workspace_guard_errors=%u inside_exponents=1024 inside_errors=%u pass=%d\n",dtype,trial,seed,pattern,result[0],result[1],ce,ge,result[2],pass);fflush(stdout);
 }
 return failures;
}
int main(){try{int8_research::init();return probe59<__nv_bfloat16,cute::bfloat16_t>("bf16",megafmm::DType::BF16)+probe59<__half,cute::half_t>("fp16",megafmm::DType::FP16)?2:0;}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}

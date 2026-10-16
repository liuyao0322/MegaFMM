// Own pre-MMA native+integer TMA/layout/ring and fine metadata checks.
#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#include "p55/corrected.cuh"
using namespace int8_research;
struct alignas(128) ProbeShared55 {uint8_t before[1024];Shared core;uint8_t after[128];};
static_assert(sizeof(ProbeShared55)==198400);
__global__ void synthetic55(uint16_t* ha,uint16_t* hb,uint8_t* qa,uint8_t* qb,int* ea,int* eb,int seed,int trial,int bits){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<9*Plane;i+=gridDim.x*blockDim.x){
  unsigned x=trial==0?unsigned(i):trial==1?unsigned(i>>16):unsigned(hash(uint64_t(i)^uint64_t(seed))>>32);
  unsigned y=trial==0?unsigned(i):trial==1?unsigned(i>>16):unsigned(hash(uint64_t(i)^uint64_t(seed)^0x632be59bd9b4e019ULL)>>32);
  if(i<7*Plane){ha[i]=uint16_t(x+seed);hb[i]=uint16_t(y+3u*seed+17u);}
  // Middle-byte pattern adds index information outside the native uint16 range.
  qa[i]=uint8_t((trial==1?unsigned(i>>8):x)+5u*seed);qb[i]=uint8_t((trial==1?unsigned(i>>8):y)+7u*seed+23u);
 }
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<9*64;i+=gridDim.x*blockDim.x){
  int p=i/64,g=i%64,base=p<4?p:p<8?p-4:4;
  ea[i]=int(hash(uint64_t(base*64+g)^uint64_t(seed))%81)-50-(p>=4&&p<8?bits:0);
  eb[i]=int(hash(uint64_t(base*64+g)^uint64_t(seed)^0x632be59bd9b4e019ULL)%81)-50-(p>=4&&p<8?bits:0);
 }
}
__global__ void coverage_init55(unsigned* x){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Plane+256;i+=gridDim.x*blockDim.x)x[i]=(i<128||i>=Plane+128)?1234567u:0u;}
template<class T,bool LogicalA> __device__ unsigned native55(const uint8_t* s,const uint16_t* expected,int p,int n0,int m0,int k0,int lane){
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
template<bool LogicalA> __device__ unsigned integer55(const uint8_t* s,const uint8_t* expected,int p,int n0,int m0,int k0,int lane){
 using namespace cute;
 constexpr auto layout=[](){if constexpr(LogicalA)return tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{},Shape<_32,_128>{});else return tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{},Shape<_64,_128>{});}();
 unsigned errors=0;
 for(int i=lane;i<(LogicalA?4096:8192);i+=128){
  int r=i/128,k=i%128;
  int at=as_position_independent_swizzle_layout(layout)(make_coord(r,k));
  int src=p*Plane+(LogicalA?n0+r:m0+r)*512+k0+k;errors+=s[at]!=expected[src];
 }
 return errors;
}
template<class T> __global__ __launch_bounds__(256,1)
void layout55(const __grid_constant__ CUtensorMap high_a,const __grid_constant__ CUtensorMap high_b,
 const __grid_constant__ CUtensorMap quant_a,const __grid_constant__ CUtensorMap quant_b,
 const uint16_t* expected_a,const uint16_t* expected_b,const uint8_t* expected_qa,const uint8_t* expected_qb,
 const int* ea,const int* eb,unsigned* error,unsigned* coverage){
 using namespace cute;
 extern __shared__ __align__(128) unsigned char buffer[];
 auto& outer=*reinterpret_cast<ProbeShared55*>(buffer);auto& s=outer.core;
 int tid=threadIdx.x,m0=blockIdx.y*64,n0=blockIdx.x*32;
 for(int i=tid;i<1024;i+=256)outer.before[i]=0x3c;if(tid<128)outer.after[tid]=0xc3;
 for(int i=tid;i<36;i+=256)s.ae[i/4][i%4]=ea[(i/4)*64+n0/8+i%4];
 for(int i=tid;i<72;i+=256)s.be[i/8][i%8]=eb[(i/8)*64+m0/8+i%8];
 if(tid==0){for(int j=0;j<Stages;++j){Barrier::init(&s.ready[j],1);Barrier::init(&s.free[j],4);}cutlass::arch::fence_barrier_init();}
 __syncthreads();
 // PRODUCER_BEGIN: verbatim frozen P55 corrected producer; audited before execution.
 if(tid==0){
  for(int step=0;step<12;++step)for(int k=0;k<4;++k){
   int seq=step*4+k,slot=seq%Stages;
   if(seq>=Stages){Barrier::wait(&s.free[slot],((seq-Stages)/Stages)%2);asm volatile("" ::: "memory");}
   TxBarrier::arrive_and_expect_tx(&s.ready[slot],(step<7||step>=9)?24576:12288);
   if(step<7){
    int p=main_product(step);
    SM90_TMA_LOAD_3D::copy(&high_b,&s.ready[slot],0,s.operand[slot],m0,k*128,p);
    SM90_TMA_LOAD_4D::copy(&high_a,&s.ready[slot],0,s.operand[slot]+16384,0,n0,k*2,p);
   }else{
    int term=first_term(step-7);
    SM90_TMA_LOAD_3D::copy(&quant_b,&s.ready[slot],0,s.operand[slot],k*128,m0,qb_plane(term));
    SM90_TMA_LOAD_3D::copy(&quant_a,&s.ready[slot],0,s.operand[slot]+16384,k*128,n0,qa_plane(term));
    if(step>=9){
     SM90_TMA_LOAD_3D::copy(&quant_b,&s.ready[slot],0,s.operand[slot]+8192,k*128,m0,qb_plane(term+1));
     SM90_TMA_LOAD_3D::copy(&quant_a,&s.ready[slot],0,s.operand[slot]+20480,k*128,n0,qa_plane(term+1));
    }
   }
  }
 }
 // PRODUCER_END
 else if(tid>=128){
  unsigned ne=0,qe=0,me=0;
  const int ap[8]={7,8,4,0,5,1,6,2},bp[8]={8,7,0,4,1,5,2,6};
  const int order[7]={0,1,5,6,4,2,3};
  for(int step=0;step<12;++step)for(int k=0;k<4;++k){
   int seq=step*4+k,slot=seq%Stages;
   Barrier::wait(&s.ready[slot],seq/Stages%2);asm volatile("" ::: "memory");
   if(step<7){
    ne+=native55<T,false>(s.operand[slot],expected_b,order[step],n0,m0,k*128,tid-128);
    ne+=native55<T,true>(s.operand[slot]+16384,expected_a,order[step],n0,m0,k*128,tid-128);
   }else{
    int t=step==7?0:step==8?1:2+2*(step-9);
    qe+=integer55<false>(s.operand[slot],expected_qb,bp[t],n0,m0,k*128,tid-128);
    qe+=integer55<true>(s.operand[slot]+16384,expected_qa,ap[t],n0,m0,k*128,tid-128);
    if(step>=9){
     qe+=integer55<false>(s.operand[slot]+8192,expected_qb,bp[t+1],n0,m0,k*128,tid-128);
     qe+=integer55<true>(s.operand[slot]+20480,expected_qa,ap[t+1],n0,m0,k*128,tid-128);
    }
   }
   // Generic reads retire the current slot. No asynchronous MMA in this probe.
   asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");if((tid&31)==0)Barrier::arrive(&s.free[slot]);
  }
  for(int i=tid-128;i<108;i+=128){
   if(i<36)me+=s.ae[i/4][i%4]!=ea[(i/4)*64+(n0+(i%4)*8)/8];
   else{int j=i-36;me+=s.be[j/8][j%8]!=eb[(j/8)*64+(m0+(j%8)*8)/8];}
  }
  for(int j=0;j<16;++j){
   int mn=GMMA::CLayout_64x32{}(tid-128,j),r=mn/64,c=mn%64;
   atomicAdd(coverage+128+(n0+r)*512+m0+c,1u);
   for(int g=0;g<5;++g){
    int t=g<2?g:2+2*(g-2);
    int e=s.ae[ap[t]][r/8]+s.be[bp[t]][c/8];
    me+=e!=ea[ap[t]*64+(n0+r)/8]+eb[bp[t]*64+(m0+c)/8];
    if(g>=2)me+=e!=s.ae[ap[t+1]][r/8]+s.be[bp[t+1]][c/8];
   }
  }
  if(ne)atomicAdd(error,ne);if(qe)atomicAdd(error+1,qe);if(me)atomicAdd(error+2,me);
 }
 __syncthreads();unsigned ge=0;for(int i=tid;i<1024;i+=256)ge+=outer.before[i]!=0x3c;
 if(tid<128)ge+=outer.after[tid]!=0xc3;if(ge)atomicAdd(error+3,ge);
}
template<class T> int probe55(const char* dtype,megafmm::DType dt){
 Stream stream;Buffer work(WorkBytes+512),error(4*sizeof(unsigned)),coverage((Plane+256)*4);Work w((char*)work.p+256);
 check(cudaFuncSetAttribute(layout55<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(ProbeShared55)));
 int failures=0;
 for(int trial=0;trial<3;++trial){
  int seed=126157+479*trial+(dt==megafmm::DType::BF16?0:257);
  check(cudaMemsetAsync(work.p,0x6d,WorkBytes+512,stream.s));check(cudaMemsetAsync(error.p,0,16,stream.s));
  synthetic55<<<512,256,0,stream.s>>>((uint16_t*)w.ha,(uint16_t*)w.hb,(uint8_t*)w.qa,(uint8_t*)w.qb,w.ea,w.eb,seed,trial,dt==megafmm::DType::BF16?8:11);
  coverage_init55<<<512,256,0,stream.s>>>(coverage.as<unsigned>());
  CUtensorMap ha,hb,qa,qb;encode(ha,w.ha,dt,true,false);encode(hb,w.hb,dt,false,false);encode(qa,w.qa,dt,true,true);encode(qb,w.qb,dt,false,true);
  layout55<T><<<dim3(16,8),256,sizeof(ProbeShared55),stream.s>>>(ha,hb,qa,qb,(uint16_t*)w.ha,(uint16_t*)w.hb,(uint8_t*)w.qa,(uint8_t*)w.qb,w.ea,w.eb,error.as<unsigned>(),coverage.as<unsigned>());check(cudaGetLastError());
  unsigned result[4];std::vector<unsigned> cov(Plane+256);uint8_t guards[512];
  check(cudaMemcpyAsync(result,error.p,16,cudaMemcpyDeviceToHost,stream.s));check(cudaMemcpyAsync(cov.data(),coverage.p,cov.size()*4,cudaMemcpyDeviceToHost,stream.s));
  check(cudaMemcpyAsync(guards,work.p,256,cudaMemcpyDeviceToHost,stream.s));check(cudaMemcpyAsync(guards+256,(char*)work.p+256+WorkBytes,256,cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
  unsigned ce=0,ge=0;for(int i=0;i<Plane+256;++i)ce+=cov[i]!=((i<128||i>=Plane+128)?1234567u:1u);for(auto x:guards)ge+=x!=0x6d;
  bool pass=result[0]==0&&result[1]==0&&result[2]==0&&result[3]==0&&ce==0&&ge==0;failures+=!pass;
  printf("FINE_LAYOUT dtype=%s trial=%d seed=%d ctas=128 sequences=48 slots=8 native_comparisons=44040192 integer_comparisons=50331648 coordinates=262144 native_errors=%u integer_errors=%u metadata_errors=%u shared_guard_errors=%u coverage_errors=%u workspace_guard_errors=%u pass=%d\n",dtype,trial,seed,result[0],result[1],result[2],result[3],ce,ge,pass);fflush(stdout);
 }
 return failures;
}
int main(){try{int8_research::init();return probe55<cute::bfloat16_t>("bf16",megafmm::DType::BF16)+probe55<cute::half_t>("fp16",megafmm::DType::FP16)?2:0;}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}

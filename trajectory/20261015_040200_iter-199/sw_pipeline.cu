#include "internal.h"
#include <cuda.h>
#include <cooperative_groups.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <stdexcept>
#include <type_traits>

namespace megafmm::detail {
namespace {
constexpr int BM=128,BN=64,BK=128,Stages=2,ASingle=64*BK,ASize=2*ASingle,BSize=BK*BN,Threads=384,Plane=512*512;
constexpr size_t PackedBytes=size_t(4)<<20,WorkspaceBytes=PackedBytes;
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
template<class T> struct alignas(128) Shared{
  T operand[Stages][ASize+BSize];
  uint64_t ready[Stages],free[Stages],pack_ready,global_ready;
};
static_assert(sizeof(Shared<cute::half_t>)==98432,"shared budget");
__device__ void wait(const uint64_t* b,int phase){Barrier::wait(b,phase);asm volatile("" ::: "memory");}
__device__ void arrive(const uint64_t* b){asm volatile("" ::: "memory");Barrier::arrive(b);}
template<class T,class TA,class TB>
__device__ void mma_term(TA a,TB b,float (&d)[32]) {
  using namespace cute;
  using Op=std::conditional_t<std::is_same_v<T,bfloat16_t>,
      SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>,
      SM90_64x64x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::MN>>;
  // Build the canonical descriptor once; CuTe preserves the swizzled K offsets
  // in descriptor units instead of rebuilding address bitfields for each MMA.
  auto da_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(
      tiled_divide(a,Shape<_64,_16>{}));
  auto db_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(
      tiled_divide(b,Shape<_64,_16>{}));
  auto da_base=da_tiles(0,0,0);
  auto db_base=db_tiles(0,0,0);
  #pragma unroll
  for(int kk=0;kk<BK/16;++kk){
    // All operands fit below 256 KiB of shared memory. K offsets cannot carry
    // out of the 14-bit start-address field (16-byte units), so keep the high
    // descriptor word invariant rather than generating a 64-bit add/carry.
    auto da=da_base,db=db_base;
    da.reg32_[0]+=uint32_t(da_tiles.layout()(0,0,kk));
    db.reg32_[0]+=uint32_t(db_tiles.layout()(0,0,kk));
    Op::fma(da,db,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],d[16],d[17],d[18],d[19],d[20],d[21],d[22],d[23],d[24],d[25],d[26],d[27],d[28],d[29],d[30],d[31]);
  }
}

template<class T> __global__ __launch_bounds__(Threads,2)
void sw_cooperative_hilo(const __grid_constant__ CUtensorMap amap,
 const __grid_constant__ CUtensorMap bmap,const __grid_constant__ CUtensorMap smap,
 const __grid_constant__ CUtensorMap tmap,const T* A,const T* B,T* packed,float* D){
 using namespace cute;
 extern __shared__ __align__(128) unsigned char buffer[];
 auto& sh=*reinterpret_cast<Shared<T>*>(buffer);
 int tid=threadIdx.x,p=blockIdx.x%8,tile=blockIdx.y*8+blockIdx.x/8;
 int cta=tile*8+p,m0=blockIdx.y*BM,n0=(blockIdx.x/8)*BN;
 if(tid==0){
  for(int j=0;j<Stages;++j){Barrier::init(&sh.ready[j],1);Barrier::init(&sh.free[j],8);}
  Barrier::init(&sh.pack_ready,p<2 ? 96:352);Barrier::init(&sh.global_ready,1);
  cutlass::arch::fence_barrier_init();
 }
 __syncthreads();
 if((tid>=32 && tid<128)||(p>=2 && tid>=128 && tid<384)){
  int worker=tid-32;
  int global_worker=p<2 ? (p*32+tile)*96+worker : 2*32*96+((p-2)*32+tile)*352+worker;
  for(int i=2*global_worker;i<Plane;i+=2*73728){
   int row=i/512,col=i%512,at=row*1024+col;
    auto av=reinterpret_cast<const uint32_t*>(A);
    auto bv=reinterpret_cast<const uint32_t*>(B);
    uint32_t abits[4]={av[at/2],av[(at+512)/2],av[(at+512*1024)/2],av[(at+512*1024+512)/2]};
    uint32_t bbits[4]={bv[at/2],bv[(at+512)/2],bv[(at+512*1024)/2],bv[(at+512*1024+512)/2]};
    uint32_t words[16]={};
    #pragma unroll
    for(int lane=0;lane<2;++lane){
      float a11=float(T::bitcast(uint16_t(abits[0]>>(16*lane)))),a12=float(T::bitcast(uint16_t(abits[1]>>(16*lane))));
      float a21=float(T::bitcast(uint16_t(abits[2]>>(16*lane)))),a22=float(T::bitcast(uint16_t(abits[3]>>(16*lane))));
      float b11=float(T::bitcast(uint16_t(bbits[0]>>(16*lane)))),b12=float(T::bitcast(uint16_t(bbits[1]>>(16*lane))));
      float b21=float(T::bitcast(uint16_t(bbits[2]>>(16*lane)))),b22=float(T::bitcast(uint16_t(bbits[3]>>(16*lane))));
      float s1=__fadd_rn(a21,a22),s2=__fsub_rn(s1,a11);
      float t1=__fsub_rn(b12,b11),t2=__fsub_rn(b22,t1);
      float sv[4]={s1,s2,__fsub_rn(a11,a21),__fsub_rn(a12,s2)};
      float tv[4]={t1,t2,__fsub_rn(b22,b12),__fsub_rn(t2,b21)};
      #pragma unroll
      for(int j=0;j<4;++j){
        T sh=T(sv[j]),th=T(tv[j]);
        words[2*j]|=uint32_t(sh.raw())<<(16*lane);
        words[8+2*j]|=uint32_t(th.raw())<<(16*lane);
      }
    }
    #pragma unroll
    for(int j=0;j<8;++j)reinterpret_cast<uint32_t*>(packed)[(j*Plane+i)/2]=words[2*j];

  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
  arrive(&sh.pack_ready);
 }
 if(tid==0){
  // Raw P1/P2 may execute while CUDA warps pack; all publication waits remain per invocation.
  for(int q=0;q<=512/BK;++q){
   if(q==(p<2 ? 512/BK:0)){
    wait(&sh.pack_ready,0);
    auto* barrier=&cooperative_groups::details::get_grid_workspace()->barrier;
    uint32_t inc=cta==0 ? 0x80000000u-255u:1u,old,now;
    asm volatile("atom.add.acq_rel.gpu.u32 %0,[%1],%2;":"=r"(old):"l"(barrier),"r"(inc):"memory");
    do{asm volatile("ld.acquire.gpu.u32 %0,[%1];":"=r"(now):"l"(barrier):"memory");}while(!((old^now)&0x80000000u));
    arrive(&sh.global_ready);
   }
   if(q==512/BK || p==7)break;
   int slot=q%Stages,k0=q*BK;
   if(q>=Stages)wait(&sh.free[slot],((q-Stages)/Stages)%2);
   TxBarrier::arrive_and_expect_tx(&sh.ready[slot],2*(ASize+BSize));
   T* op=sh.operand[slot];
   #pragma unroll
   for(int w=0;w<2;++w){
    if(p==2 || p>=4){
     int plane=p==2 ? 3:p-4;
     SM90_TMA_LOAD_4D::copy(&smap,&sh.ready[slot],0,op+w*ASingle,0,m0+w*64,k0/64,plane);
    }else{
     int col=(p==1||p==3)?512:0,row=p==3?512:0;
     SM90_TMA_LOAD_3D::copy(&amap,&sh.ready[slot],0,op+w*ASingle,0,m0+w*64+row,(k0+col)/64);
    }
   }
   if(p==3 || p>=4){
    int plane=p==3 ? 3:p-4;
    SM90_TMA_LOAD_3D::copy(&tmap,&sh.ready[slot],0,op+ASize,n0,k0,plane);
   }else{
    int col=p==2?512:0,row=(p==1||p==2)?512:0;
    SM90_TMA_LOAD_2D::copy(&bmap,&sh.ready[slot],0,op+ASize,n0+col,k0+row);
   }
  }
 }else if(p<7 && tid>=128 && tid<384){
  int wg=(tid-128)/128,lane=(tid-128)%128;
  constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
  constexpr auto lb=tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
  float acc[32]={};
  #pragma unroll 1
  for(int q=0;q<512/BK;++q){
   int slot=q%Stages;
   wait(&sh.ready[slot],(q/Stages)%2);
   auto ah=make_tensor(make_smem_ptr(sh.operand[slot]+wg*ASingle),la);
   auto bh=make_tensor(make_smem_ptr(sh.operand[slot]+ASize),lb);
   warpgroup_arrive();mma_term<T>(ah,bh,acc);warpgroup_commit_batch();warpgroup_wait<0>();
   __syncwarp();if((tid&31)==0)arrive(&sh.free[slot]);
  }
  wait(&sh.global_ready,0);
  asm volatile("bar.sync 3,256;" ::: "memory");
  #pragma unroll
  for(int j=0;j<32;j+=2){
   int mn=GMMA::CLayout_64x64{}(lane,j),row=wg*64+mn%64,col=mn/64;
   int index=row*64+(col^((row&3)<<3));
   auto* local=reinterpret_cast<float*>(sh.operand);
   reinterpret_cast<float2*>(local+index)[0]=make_float2(acc[j],acc[j+1]);
  }
 }
 __syncthreads();
 auto cluster=cooperative_groups::this_cluster();
 cluster.sync();
 if(p==7 && tid>=128 && tid<384){
  float* peers[7];
  #pragma unroll
  for(int z=0;z<7;++z)peers[z]=cluster.map_shared_rank(reinterpret_cast<float*>(sh.operand),z);

   #pragma unroll 1
   for(int j=0;j<16;++j){
    int worker=tid-128;
    int row=m0+worker/32+8*j,col=n0+2*(worker%32);
    int localrow=row-m0,localcol=col-n0;
    int index=localrow*64+(localcol^((localrow&3)<<3));
    float2 v[7];
    #pragma unroll
    for(int z=0;z<7;++z)v[z]=reinterpret_cast<const float2*>(peers[z]+index)[0];
    float2 out[4];
    #pragma unroll
    for(int c=0;c<2;++c){
     float a=c?v[0].y:v[0].x,b=c?v[1].y:v[1].x;
     float p3=c?v[2].y:v[2].x,p4=c?v[3].y:v[3].x,p5=c?v[4].y:v[4].x;
     float p6=c?v[5].y:v[5].x,p7=c?v[6].y:v[6].x;
     float u2=__fadd_rn(a,p6),u3=__fadd_rn(u2,p7);
     float x[4]={__fadd_rn(a,b),__fadd_rn(__fadd_rn(u2,p5),p3),__fsub_rn(u3,p4),__fadd_rn(u3,p5)};
     #pragma unroll
     for(int z=0;z<4;++z){if(c)out[z].y=x[z];else out[z].x=x[z];}
    }
    reinterpret_cast<float2*>(D+row*1024+col)[0]=out[0];
    reinterpret_cast<float2*>(D+row*1024+col+512)[0]=out[1];
    reinterpret_cast<float2*>(D+(row+512)*1024+col)[0]=out[2];
    reinterpret_cast<float2*>(D+(row+512)*1024+col+512)[0]=out[3];
   }
 }
 cluster.sync();
}
void check(cudaError_t e,const char* where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
void encode(CUtensorMap& map,const void* ptr,DType dtype,bool is_a,bool transformed){
  if(reinterpret_cast<uintptr_t>(ptr)%16)throw std::invalid_argument("SW TMA input must be16-byte aligned");
  if(is_a){
    // Expose K/64 as an outer tensor dimension. The shared box still packs
    // (64 K elements, BM rows, BK/64 chunks), matching the existing GMMA layout.
    uint64_t dims[4]={64,transformed ? 512ULL:1024ULL,transformed ? 8ULL:16ULL,4};
    uint64_t strides[3]={transformed ? 1024ULL:2048ULL,128,512ULL*512*2};
    uint32_t box[4]={64,64,BK/64,1},element_strides[4]={1,1,1,1};
    auto result=cuTensorMapEncodeTiled(&map,dtype==DType::BF16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        transformed ? 4:3,const_cast<void*>(ptr),dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(result!=CUDA_SUCCESS)throw std::runtime_error("SW A tensor map status "+std::to_string(int(result)));
    return;
  }
  uint64_t dims[3]={transformed ? 512ULL:1024ULL,transformed ? 512ULL:1024ULL,4};
  uint64_t strides[2]={dims[0]*2,dims[0]*dims[1]*2};
  uint32_t box[3]={uint32_t(is_a ? 64:BN),uint32_t(is_a ? BM:BK),1},element_strides[3]={1,1,1};
  auto result=cuTensorMapEncodeTiled(&map,dtype==DType::BF16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
      transformed ? 3:2,const_cast<void*>(ptr),dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,
      is_a ? CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(result!=CUDA_SUCCESS)throw std::runtime_error("SW tensor map status "+std::to_string(int(result)));
}
} // namespace
struct SWState{DType dtype;};
std::shared_ptr<SWState> make_sw_pipeline(const Problem& p){
  if(p.m!=1024 || p.n!=1024 || p.k!=1024 || p.trans_a || p.trans_b || p.lda!=1024 || p.ldb!=1024 || p.ldd!=1024 || p.alpha!=1 || p.beta!=0)
    throw std::invalid_argument("SW requires contiguous non-transposed1024^3 alpha1 beta0");
  int device;cudaDeviceProp prop{};check(cudaGetDevice(&device),"SW device");check(cudaGetDeviceProperties(&prop,device),"SW properties");
  if(prop.major!=9 || prop.minor!=0 || !prop.cooperativeLaunch)throw std::invalid_argument("SW needs cooperative Hopper launch");
  int blocks=0;
  if(p.dtype==DType::BF16){
    check(cudaFuncSetAttribute(sw_cooperative_hilo<cute::bfloat16_t>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<cute::bfloat16_t>)),"SW shared opt-in");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,sw_cooperative_hilo<cute::bfloat16_t>,Threads,sizeof(Shared<cute::bfloat16_t>)),"SW occupancy capacity");
  }else{
    check(cudaFuncSetAttribute(sw_cooperative_hilo<cute::half_t>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<cute::half_t>)),"SW shared opt-in");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,sw_cooperative_hilo<cute::half_t>,Threads,sizeof(Shared<cute::half_t>)),"SW occupancy capacity");
  }
  if(blocks*prop.multiProcessorCount<256)throw std::invalid_argument("SW cooperative grid requires256 resident CTA capacity");
  return std::make_shared<SWState>(SWState{p.dtype});
}
template<class T> cudaLaunchConfig_t launch_config(cudaStream_t stream,cudaLaunchAttribute (&attrs)[2]){
 attrs[0].id=cudaLaunchAttributeCooperative;attrs[0].val.cooperative=1;
 attrs[1].id=cudaLaunchAttributeClusterDimension;attrs[1].val.clusterDim={8,1,1};
 cudaLaunchConfig_t config{};config.gridDim=dim3(64,4);config.blockDim=Threads;
 config.dynamicSmemBytes=sizeof(Shared<T>);config.stream=stream;config.attrs=attrs;config.numAttrs=2;
 return config;
}
void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream){
 CUtensorMap a{},b{},s{},t{};encode(a,A,state.dtype,true,false);encode(b,B,state.dtype,false,false);
 encode(s,workspace,state.dtype,true,true);encode(t,static_cast<char*>(workspace)+PackedBytes/2,state.dtype,false,true);
 cudaLaunchAttribute attrs[2]{};
 if(state.dtype==DType::BF16){
  auto config=launch_config<cute::bfloat16_t>(stream,attrs);
  check(cudaLaunchKernelEx(&config,sw_cooperative_hilo<cute::bfloat16_t>,a,b,s,t,static_cast<const cute::bfloat16_t*>(A),static_cast<const cute::bfloat16_t*>(B),static_cast<cute::bfloat16_t*>(workspace),D),"SW cluster launch");
 }else{
  auto config=launch_config<cute::half_t>(stream,attrs);
  check(cudaLaunchKernelEx(&config,sw_cooperative_hilo<cute::half_t>,a,b,s,t,static_cast<const cute::half_t*>(A),static_cast<const cute::half_t*>(B),static_cast<cute::half_t*>(workspace),D),"SW cluster launch");
 }
}
} // namespace megafmm::detail

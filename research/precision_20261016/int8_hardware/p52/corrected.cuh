#pragma once
#include "pack.cuh"
#include "megafmm/gemm.h"
#include <vector>
#include <cstdio>

namespace int8_research {
constexpr int ConsumerStages=2;
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
struct alignas(128) Shared {
 uint8_t operand[ConsumerStages][40960];uint64_t ready[ConsumerStages],free[ConsumerStages];
};
template<class T> struct QuantType {using type=T;};
static_assert(sizeof(Shared)==82048);
__host__ __device__ constexpr int main_product(int step){return step<2?step:step==2?5:step==3?6:step==4?4:step==5?2:3;}
__host__ __device__ constexpr int qa_plane(int term){return term<2?(term==0?10:3):(term-2)/2+((term&1)?4:7);}
__host__ __device__ constexpr int qb_plane(int term){return term<2?(term==0?2:10):(term-2)/2+((term&1)?7:4);}

// Five groups retain all eight original cross terms, in native T with scaled low.
__host__ __device__ constexpr int first_term(int group){return group<2?group:2+2*(group-2);}
// No adaptive quantization exponents exist in this representation. This host-only
// diagnostic stays outside every timing/capture arm and never reads old metadata.
void diagnose_uniform_scales(const char* dtype,int pattern,int trial,Work w,cudaStream_t stream,int stream_id=0){
 printf("NATIVE_CORRECTION dtype=%s pattern=%d trial=%d stream=%d low_scale_power=%d diagnostic_only=1\n",dtype,pattern,trial,stream_id,std::string(dtype)=="fp16"?11:8);
}

template<class T> __device__ __forceinline__ void float_mma(uint64_t a,uint64_t b,float* d,cute::GMMA::ScaleOut scale){
 using namespace cute;
 if constexpr(std::is_same_v<T,cute::half_t>)
  SM90_64x16x16_F32F16F16_SS<GMMA::Major::MN,GMMA::Major::K>::fma(a,b,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],scale);
 else
  SM90_64x16x16_F32BF16BF16_SS<GMMA::Major::MN,GMMA::Major::K>::fma(a,b,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],scale);
}
template<class T> __device__ __forceinline__ void consume(Shared& s,int tid,int m0,int n0,const int* ea,const int* eb,float* D){
 using namespace cute;
 constexpr auto la=tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,_128>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_16,_128>{});
 auto a=make_tensor(make_smem_ptr(reinterpret_cast<T*>(s.operand[0])),la);
 auto b=make_tensor(make_smem_ptr(reinterpret_cast<T*>(s.operand[0]+32768)),lb);
 auto da=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(a,Shape<_64,_16>{}));
 auto db=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(b,Shape<_16,_16>{}));
 float c[4][8]={};
 #pragma unroll 1
 for(int step=0;step<7;++step){
  int p=main_product(step);
  float d[8]={},r[8]={};
  #pragma unroll
  for(int j=0;j<8;++j){warpgroup_fence_operand(d[j]);warpgroup_fence_operand(r[j]);}
  warpgroup_arrive();
  #pragma unroll 1
  for(int k=0;k<4;++k){
   int seq=step*4+k,slot=seq%ConsumerStages;
   Barrier::wait(&s.ready[slot],seq/ConsumerStages%2);asm volatile("" ::: "memory");
   uint64_t offset=slot*40960/16;
   #pragma unroll
   for(int inner=0;inner<8;++inner)
    float_mma<T>(uint64_t(da(0,0,inner))+offset,uint64_t(db(0,0,inner))+offset,d,(k==0&&inner==0)?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One);
   if(p==2||p>=4){
    #pragma unroll
    for(int inner=0;inner<8;++inner)
     float_mma<T>(uint64_t(da(0,0,inner))+offset,uint64_t(db(0,0,inner))+offset+4096/16,r,(k==0&&inner==0)?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One);
   }
   if(p==3||p>=4){
    #pragma unroll
    for(int inner=0;inner<8;++inner)
     float_mma<T>(uint64_t(da(0,0,inner))+offset+16384/16,uint64_t(db(0,0,inner))+offset,r,(p==3&&k==0&&inner==0)?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One);
   }
   warpgroup_commit_batch();
   if(k>0){
    // Both fragments' preceding batch is complete before the slot is released.
    warpgroup_wait<1>();asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
    if((tid&31)==0)Barrier::arrive(&s.free[(seq-1)%ConsumerStages]);
   }
  }
  warpgroup_wait<0>();
  #pragma unroll
  for(int j=0;j<8;++j){warpgroup_fence_operand(d[j]);warpgroup_fence_operand(r[j]);}
  asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
  if((tid&31)==0)Barrier::arrive(&s.free[(step*4+3)%ConsumerStages]);
  if(p>=2){
   constexpr float inverse=(std::is_same_v<T,cute::half_t> ? (1.f/2048.f) : (1.f/256.f));
   #pragma unroll
   for(int j=0;j<8;++j)d[j]=__fadd_rn(d[j],__fmul_rn(r[j],inverse));
  }
   if(p==0){
    #pragma unroll
    for(int j=0;j<8;++j){c[0][j]=c[1][j]=c[2][j]=c[3][j]=d[j];}
   }
   else if(p==1){
    #pragma unroll
    for(int j=0;j<8;++j){c[0][j]=__fadd_rn(c[0][j],d[j]);}
   }
   else if(p==5){
    #pragma unroll
    for(int j=0;j<8;++j){c[1][j]=__fadd_rn(c[1][j],d[j]);c[2][j]=__fadd_rn(c[2][j],d[j]);c[3][j]=__fadd_rn(c[3][j],d[j]);}
   }
   else if(p==6){
    #pragma unroll
    for(int j=0;j<8;++j){c[2][j]=__fadd_rn(c[2][j],d[j]);c[3][j]=__fadd_rn(c[3][j],d[j]);}
   }
   else if(p==4){
    #pragma unroll
    for(int j=0;j<8;++j){c[1][j]=__fadd_rn(c[1][j],d[j]);c[3][j]=__fadd_rn(c[3][j],d[j]);}
   }
   else if(p==2){
    #pragma unroll
    for(int j=0;j<8;++j){c[1][j]=__fadd_rn(c[1][j],d[j]);}
   }
   else{
    #pragma unroll
    for(int j=0;j<8;++j){c[2][j]=__fsub_rn(c[2][j],d[j]);}
   }
 }
 #pragma unroll
 for(int j=0;j<8;++j){
  int mn=GMMA::CLayout_64x16{}(tid-128,j),at=(n0+mn/64)*1024+m0+mn%64;
  D[at]=c[0][j];D[at+512]=c[1][j];D[at+512*1024]=c[2][j];D[at+512*1024+512]=c[3][j];
 }
}

template<class T> __global__ __launch_bounds__(256,1)
void corrected(const __grid_constant__ CUtensorMap high_a,const __grid_constant__ CUtensorMap high_b,
 const int* ea,const int* eb,float* D){
 using namespace cute;
 extern __shared__ __align__(128) unsigned char buffer[];
 auto& s=*reinterpret_cast<Shared*>(buffer);
 int tid=threadIdx.x,m0=blockIdx.y*64,n0=blockIdx.x*16;
 if(tid==0){for(int j=0;j<ConsumerStages;++j){Barrier::init(&s.ready[j],1);Barrier::init(&s.free[j],4);}cutlass::arch::fence_barrier_init();}
 __syncthreads();
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
 }else if(tid>=128)consume<T>(s,tid,m0,n0,ea,eb,D);
}

void encode(CUtensorMap& map,const void* ptr,megafmm::DType dt,bool side_a){
 uint64_t dims[4]={512,512,11,1},strides[3]={1024,uint64_t(Plane)*2,0};
 uint32_t box[4]={64,128,1,1},steps[4]={1,1,1,1};
 int rank=3;
 if(side_a){rank=4;dims[0]=64;dims[1]=512;dims[2]=8;dims[3]=11;strides[0]=1024;strides[1]=128;strides[2]=Plane*2;box[1]=16;box[2]=2;}
 CUtensorMapDataType type=dt==megafmm::DType::BF16?CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
 auto e=cuTensorMapEncodeTiled(&map,type,rank,const_cast<void*>(ptr),dims,strides,box,steps,
  CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 if(e!=CUDA_SUCCESS)throw std::runtime_error("native correction tensor map "+std::to_string(int(e)));
}
template<class T,bool Balance> void init_shared_packer(){
 check(cudaFuncSetAttribute(pack_joint<T,Balance>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(PairPackShared<true>)),"joint pack dynamic shared");
 check(cudaFuncSetAttribute(pack<T,true,Balance>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(PairPackShared<true>)),"pair A dynamic shared");
 check(cudaFuncSetAttribute(pack<T,false,Balance>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(PairPackShared<false>)),"pair B dynamic shared");
}
template<class T> void log_consumer_capacity(){
 int active=0,device=0,sms=0;
 check(cudaGetDevice(&device),"consumer device");
 check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"consumer SM count");
 check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,corrected<T>,256,sizeof(Shared)),"consumer occupancy");
 printf("CONSUMER_INIT dtype=%s sm_count=%d active_blocks_per_sm=%d grid=256 shared_bytes=%zu\n",
  (std::is_same_v<T,cute::half_t> ? "fp16" : "bf16"),sms,active,sizeof(Shared));
 if(active<=0)throw std::runtime_error("No resident consumer CTA capacity");
}
void init(){
 init_cooperative_preprocess<cute::half_t>();init_cooperative_preprocess<cute::bfloat16_t>();
 init_shared_packer<cute::half_t,true>();init_shared_packer<cute::half_t,false>();
 init_shared_packer<cute::bfloat16_t,true>();init_shared_packer<cute::bfloat16_t,false>();
 for(auto dtype:{megafmm::DType::BF16,megafmm::DType::FP16}){
  if(dtype==megafmm::DType::BF16){
   check(cudaFuncSetAttribute(corrected<cute::bfloat16_t>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)),"INT8 BF shared");
   check(cudaFuncSetAttribute(corrected<cute::bfloat16_t>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"INT8 BF carveout");
   log_consumer_capacity<cute::bfloat16_t>();
  }else{
   check(cudaFuncSetAttribute(corrected<cute::half_t>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)),"INT8 FP shared");
   check(cudaFuncSetAttribute(corrected<cute::half_t>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"INT8 FP carveout");
   log_consumer_capacity<cute::half_t>();
  }
 }
}
void run(megafmm::DType dt,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream,bool balance=true){
 Work w(workspace);CUtensorMap ha,hb;
 encode(ha,w.ha,dt,true);encode(hb,w.hb,dt,false);
 if(dt==megafmm::DType::BF16){
  if(balance)preprocess<cute::bfloat16_t,true>(static_cast<const cute::bfloat16_t*>(A),static_cast<const cute::bfloat16_t*>(B),w,stream);
  else preprocess<cute::bfloat16_t,false>(static_cast<const cute::bfloat16_t*>(A),static_cast<const cute::bfloat16_t*>(B),w,stream);
  corrected<cute::bfloat16_t><<<dim3(32,8),256,sizeof(Shared),stream>>>(ha,hb,w.ea,w.eb,D);
 }else{
  if(balance)preprocess<cute::half_t,true>(static_cast<const cute::half_t*>(A),static_cast<const cute::half_t*>(B),w,stream);
  else preprocess<cute::half_t,false>(static_cast<const cute::half_t*>(A),static_cast<const cute::half_t*>(B),w,stream);
  corrected<cute::half_t><<<dim3(32,8),256,sizeof(Shared),stream>>>(ha,hb,w.ea,w.eb,D);
 }
 check(cudaPeekAtLastError(),"INT8 corrected launch");
}
} // namespace int8_research

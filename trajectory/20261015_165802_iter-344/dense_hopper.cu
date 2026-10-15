#include "internal.h"
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace megafmm::detail {
namespace {
using namespace cute;
template<class T,int M,int N,int ClusterM,int Schedule=0,int Stages=0,int K=64,bool DirectStore=false>
struct Candidate {
  using Tile=Shape<Int<M>,Int<N>,Int<K>>;
  using Cluster=Shape<Int<ClusterM>,_1,_1>;
  using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,float,float,
      float,cutlass::layout::RowMajor,4,float,cutlass::layout::RowMajor,4,
      cute::conditional_t<DirectStore,cutlass::epilogue::NoSmemWarpSpecialized,
        cute::conditional_t<Schedule==0,cutlass::epilogue::TmaWarpSpecializedCooperative,
                            cutlass::epilogue::TmaWarpSpecialized>>>::CollectiveOp;
  using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
      T,cutlass::layout::RowMajor,8,T,cutlass::layout::RowMajor,8,float,Tile,Cluster,
      cute::conditional_t<Stages==0,
        cutlass::gemm::collective::StageCountAutoCarveout<int(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::collective::StageCount<Stages>>,
      cute::conditional_t<Schedule==0,cutlass::gemm::KernelTmaWarpSpecializedCooperative,
        cute::conditional_t<Schedule==1,cutlass::gemm::KernelTmaWarpSpecialized,
                            cutlass::gemm::KernelTmaWarpSpecializedPingpong>>>::CollectiveOp;
  using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue>;
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  static bool run(const Problem& p,const void* A,const void* B,const float* C,float* D,
                  void* workspace,int sm_count,cudaStream_t stream,int operation,int swizzle=1,int raster=0) {
    if(operation==2) {
      auto e=cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,Kernel::SharedStorageSize);
      if(e==cudaSuccess) e=cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
          cudaFuncAttributePreferredSharedMemoryCarveout,100);
      if(e!=cudaSuccess) throw std::runtime_error(std::string("Hopper setup: ")+cudaGetErrorString(e));
      return true;
    }
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {int(p.m),int(p.n),int(p.k),1},
      {static_cast<const T*>(A),typename Kernel::StrideA{p.lda,Int<1>{},int64_t(0)},
       static_cast<const T*>(B),typename Kernel::StrideB{Int<1>{},p.ldb,int64_t(0)}},
      {{},p.beta==0 ? D:C,typename Kernel::StrideC{p.beta==0 ? p.ldd:p.ldc,Int<1>{},int64_t(0)},
       D,typename Kernel::StrideD{p.ldd,Int<1>{},int64_t(0)}},
      {0,sm_count}
    };
    args.scheduler.max_swizzle_size=swizzle;
    using Raster=typename Kernel::TileScheduler::RasterOrderOptions;
    if(raster) args.scheduler.raster_order=raster==1 ? Raster::AlongM:Raster::AlongN;
    args.epilogue.thread.alpha=p.alpha;args.epilogue.thread.beta=p.beta;
    if(Gemm::can_implement(args)!=cutlass::Status::kSuccess) return false;
    // The persistent scheduler has zero workspace. No counters or values are
    // retained between invocations. Hardware information is captured at setup.
    if(Gemm::get_workspace_size(args)!=0) return false;
    if(operation==0) return true;
    auto params=Kernel::to_underlying_arguments(args,workspace);
    auto status=Gemm::run(params,stream);
    if(status!=cutlass::Status::kSuccess)
      throw std::runtime_error(std::string("Hopper dense: ")+cutlassGetStatusString(status));
    return true;
  }
};
template<class T>
bool dispatch(int id,const Problem& p,const void* A,const void* B,const float* C,float* D,
              void* workspace,int sm_count,cudaStream_t stream,int operation) {
  switch(id) {
    case 0:return Candidate<T,128,64,1>::run(p,A,B,C,D,workspace,sm_count,stream,operation);
    case 1:return Candidate<T,128,128,1>::run(p,A,B,C,D,workspace,sm_count,stream,operation);
    case 2:return Candidate<T,128,256,1>::run(p,A,B,C,D,workspace,sm_count,stream,operation);
    case 3:return Candidate<T,128,128,2>::run(p,A,B,C,D,workspace,sm_count,stream,operation);
    case 9:return Candidate<T,128,128,1>::run(p,A,B,C,D,workspace,sm_count,stream,operation,4);
    case 10:return Candidate<T,128,128,1>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 11:return Candidate<T,128,256,1>::run(p,A,B,C,D,workspace,sm_count,stream,operation,4);
    case 12:return Candidate<T,128,256,1>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 13:return Candidate<T,128,128,2>::run(p,A,B,C,D,workspace,sm_count,stream,operation,4);
    case 14:return Candidate<T,128,128,2>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 15:return Candidate<T,64,32,1,1,3>::run(p,A,B,C,D,workspace,sm_count,stream,operation);
    case 16:return Candidate<T,64,64,1,1,3>::run(p,A,B,C,D,workspace,sm_count,stream,operation);
    case 17:return Candidate<T,128,128,1,2,3>::run(p,A,B,C,D,workspace,sm_count,stream,operation);
    case 18:return Candidate<T,256,128,1>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 19:return Candidate<T,128,256,2>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 20:return Candidate<T,128,256,1,2,3>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 21:return Candidate<T,128,256,1>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8,2);
    case 22:return Candidate<T,128,128,2>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8,2);
    case 23:return Candidate<T,256,128,1>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8,1);
    case 26:return Candidate<T,64,32,1,1,3,128>::run(p,A,B,C,D,workspace,sm_count,stream,operation,1);
    case 27:return Candidate<T,64,64,1,1,3,128>::run(p,A,B,C,D,workspace,sm_count,stream,operation,1);
    case 28:return Candidate<T,64,64,1,1,2,256>::run(p,A,B,C,D,workspace,sm_count,stream,operation,1);
    case 29:return Candidate<T,128,64,1,0,0,128>::run(p,A,B,C,D,workspace,sm_count,stream,operation,1);
    case 30:return Candidate<T,128,128,1,0,0,128>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 31:return Candidate<T,128,256,1,0,0,128,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 32:return Candidate<T,128,128,1,0,0,128,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    default:return false;
  }
}
template<class T> __global__ void small_gemm(Problem p,const T* A,const T* B,const float* C,float* D) {
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<p.m*p.n;i+=int64_t(gridDim.x)*blockDim.x) {
    int64_t r=i/p.n,c=i%p.n;
    float value=0;
    for(int64_t q=0;q<p.k;++q) {
      float a=float(A[p.trans_a ? q*p.lda+r:r*p.lda+q]);
      float b=float(B[p.trans_b ? c*p.ldb+q:q*p.ldb+c]);
      value=fmaf(a,b,value);
    }
    value=p.alpha*value;
    if(p.beta!=0) value=fmaf(p.beta,C[r*p.ldc+c],value);
    D[r*p.ldd+c]=value;
  }
}
}
const char* hopper_candidate_name(int id) {
  static const char* names[]={"128x64x64_c1","128x128x64_c1","128x256x64_c1","128x128x64_c2","simt_small","pad_128x64x64_c1","pad_128x128x64_c1","pad_128x256x64_c1","pad_128x128x64_c2",
      "128x128x64_c1_sw4","128x128x64_c1_sw8","128x256x64_c1_sw4",
      "128x256x64_c1_sw8","128x128x64_c2_sw4","128x128x64_c2_sw8",
      "64x32x64_basic_s3","64x64x64_basic_s3","128x128x64_pingpong_s3",
      "256x128x64_c1_sw8","128x256x64_c2_sw8","128x256x64_pingpong_s3_sw8",
      "128x256x64_c1_sw8_rasterN","128x128x64_c2_sw8_rasterN","256x128x64_c1_sw8_rasterM","wmma_32x64x32","wmma_32x64x64","64x32x128_basic_s3","64x64x128_basic_s3","64x64x256_basic_s2","128x64x128_c1","128x128x128_c1_sw8","128x256x128_direct_sw8","128x128x128_direct_sw8"};
  return id>=0 && id<HopperCandidateCount ? names[id]:"invalid";
}
bool hopper_dense(int id,const Problem& p,const void* A,const void* B,const float* C,float* D,
                  void* workspace,int sm_count,cudaStream_t stream,int operation) {
  if(id==24 || id==25) return wmma_dense(id==24?32:64,p,A,B,C,D,stream,operation);
  if((id==15 || id==16 || id==17 || id==20) && (p.m>2048 || p.n>2048 || p.k>2048) && operation!=2) return false;
  if(id>=5 && id<=8)
    return padded_hopper(id-5,p,A,B,C,D,workspace,sm_count,stream,operation);
  if(id==4) {
    if(p.m*p.n>4096 || p.k>256) return false;
    if(operation!=1) return true;
    int blocks=int((p.m*p.n+127)/128);
    if(p.dtype==DType::BF16)
      small_gemm<<<blocks,128,0,stream>>>(p,static_cast<const __nv_bfloat16*>(A),static_cast<const __nv_bfloat16*>(B),C,D);
    else small_gemm<<<blocks,128,0,stream>>>(p,static_cast<const __half*>(A),static_cast<const __half*>(B),C,D);
    auto e=cudaPeekAtLastError();
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    return true;
  }
  if(operation==2) {
    if(p.dtype==DType::BF16) return dispatch<cutlass::bfloat16_t>(id,p,A,B,C,D,workspace,sm_count,stream,operation);
    return dispatch<cutlass::half_t>(id,p,A,B,C,D,workspace,sm_count,stream,operation);
  }
  if(p.trans_a || p.trans_b || p.lda%8 || p.ldb%8 || p.ldd%4 ||
      (p.beta!=0 && p.ldc%4) || p.k%8 || p.n%8) return false;
  if(reinterpret_cast<uintptr_t>(A)%16 || reinterpret_cast<uintptr_t>(B)%16 ||
      reinterpret_cast<uintptr_t>(D)%16 || (p.beta!=0 && reinterpret_cast<uintptr_t>(C)%16)) return false;
  if(p.dtype==DType::BF16)
    return dispatch<cutlass::bfloat16_t>(id,p,A,B,C,D,workspace,sm_count,stream,operation);
  return dispatch<cutlass::half_t>(id,p,A,B,C,D,workspace,sm_count,stream,operation);
}
} // namespace megafmm::detail

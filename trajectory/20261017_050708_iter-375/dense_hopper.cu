#include "internal.h"
#include "hopper_nine_warp_kernel.cuh"
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include "hopper_inflight_mainloop.cuh"
#include "hopper_batched_mainloop.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace megafmm::detail {
namespace {
using namespace cute;
template<class T,int M,int N,int ClusterM,int Schedule=0,int Stages=0,int K=64,bool DirectStore=false,int ClusterN=1,int Workers=1,bool ZeroBeta=false,bool NineWarp=false,bool StreamK=false,int Splits=1,bool Swapped=false,int MmaInflight=1,bool BatchEight=false>
struct Candidate {
  static_assert(!NineWarp || ZeroBeta,"Nine-warp candidates require beta-zero eligibility");
  static_assert(Splits>=1 && (StreamK || Splits==1),"K splits require the reduction scheduler");
  static_assert(!Swapped || (ZeroBeta && !NineWarp && !StreamK),"Swapped candidates require ordinary beta-zero kernels");
  using Layout=cute::conditional_t<Swapped,cutlass::layout::ColumnMajor,cutlass::layout::RowMajor>;
  using Tile=Shape<Int<M>,Int<N>,Int<K>>;
  using Cluster=Shape<Int<ClusterM>,Int<ClusterN>,_1>;
  using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,float,float,
      cute::conditional_t<ZeroBeta,void,float>,Layout,4,float,Layout,4,
      cute::conditional_t<DirectStore,cutlass::epilogue::NoSmemWarpSpecialized,
        cute::conditional_t<Schedule==0,cutlass::epilogue::TmaWarpSpecializedCooperative,
                            cutlass::epilogue::TmaWarpSpecialized>>>::CollectiveOp;
  using BaseMainloop=typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
      T,Layout,8,T,Layout,8,float,Tile,Cluster,
      cute::conditional_t<Stages==0,
        cutlass::gemm::collective::StageCountAutoCarveout<int(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::collective::StageCount<Stages>>,
      cute::conditional_t<Schedule==0,cutlass::gemm::KernelTmaWarpSpecializedCooperative,
        cute::conditional_t<Schedule==1,cutlass::gemm::KernelTmaWarpSpecialized,
                            cutlass::gemm::KernelTmaWarpSpecializedPingpong>>>::CollectiveOp;
  static_assert(MmaInflight>=1 && (MmaInflight==1 || (ZeroBeta && !NineWarp && !StreamK && Schedule==0)),"Custom MMA windows require ordinary cooperative beta-zero kernels");
  static_assert(!BatchEight || (ZeroBeta && !NineWarp && !StreamK && Schedule==0 && K==32),
      "Batched K loops require ordinary cooperative beta-zero K32 kernels");
  using Mainloop=cute::conditional_t<BatchEight,
      cutlass::gemm::collective::BatchedInflightMainloop<BaseMainloop,MmaInflight>,
      cute::conditional_t<MmaInflight==1,BaseMainloop,
          cutlass::gemm::collective::InflightMainloop<BaseMainloop,MmaInflight>>>;
  using Kernel=cute::conditional_t<NineWarp,
      cutlass::gemm::kernel::NineWarpGemm<Shape<int,int,int,int>,Mainloop,Epilogue>,
      cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue,
          cute::conditional_t<StreamK,cutlass::gemm::StreamKScheduler,void>>>;
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  static typename Gemm::Arguments arguments(const Problem& p,const void* A,const void* B,
      const float* C,float* D,int sm_count,int swizzle=8,int raster=0) {
    typename Kernel::StrideA stride_a{};
    typename Kernel::StrideB stride_b{};
    typename Kernel::StrideC stride_c{};
    typename Kernel::StrideD stride_d{};
    if constexpr(Swapped) {
      // D^T = B^T A^T: column-major views use the original buffers directly.
      stride_a={Int<1>{},p.ldb,int64_t(0)};
      stride_b={p.lda,Int<1>{},int64_t(0)};
      stride_c={Int<1>{},p.beta==0 ? p.ldd:p.ldc,int64_t(0)};
      stride_d={Int<1>{},p.ldd,int64_t(0)};
    } else {
      stride_a={p.lda,Int<1>{},int64_t(0)};
      stride_b={Int<1>{},p.ldb,int64_t(0)};
      stride_c={p.beta==0 ? p.ldd:p.ldc,Int<1>{},int64_t(0)};
      stride_d={p.ldd,Int<1>{},int64_t(0)};
    }
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {int(Swapped ? p.n:p.m),int(Swapped ? p.m:p.n),int(p.k),1},
      {static_cast<const T*>(Swapped ? B:A),stride_a,
       static_cast<const T*>(Swapped ? A:B),stride_b},
      {{},p.beta==0 ? D:C,stride_c,D,stride_d},
      {0,sm_count*Workers}
    };
    args.scheduler.max_swizzle_size=swizzle;
    using Raster=typename Kernel::TileScheduler::RasterOrderOptions;
    if(raster) args.scheduler.raster_order=raster==1 ? Raster::AlongM:Raster::AlongN;
    args.epilogue.thread.alpha=p.alpha;args.epilogue.thread.beta=p.beta;
    if constexpr(StreamK) {
      args.scheduler.reduction_mode=Kernel::TileScheduler::ReductionMode::Deterministic;
      if constexpr(Splits>1) {
        args.scheduler.decomposition_mode=Kernel::TileScheduler::DecompositionMode::SplitK;
        args.scheduler.splits=Splits;
      } else {
        args.scheduler.decomposition_mode=Kernel::TileScheduler::DecompositionMode::StreamK;
      }
    }
    return args;
  }
  static size_t workspace_bytes(const Problem& p,int sm_count,int swizzle=8,int raster=0) {
    return Gemm::get_workspace_size(arguments(p,nullptr,nullptr,nullptr,nullptr,sm_count,swizzle,raster));
  }
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
    if constexpr(ZeroBeta) {if(p.beta!=0)return false;}
    auto args=arguments(p,A,B,C,D,sm_count,swizzle,raster);
    if(Gemm::can_implement(args)!=cutlass::Status::kSuccess) return false;
    // Ordinary persistent candidates need no scratch; Stream-K/Split-K gets caller
    // scratch sized from the same scheduler arguments and captured SM count.
    const size_t required=Gemm::get_workspace_size(args);
    if constexpr(StreamK) {if(required && !workspace)return false;}
    else if(required) return false;
    if(operation==0) return true;
    if constexpr(StreamK) {
      // Every invocation owns its scratch and resets reduction barriers on the
      // caller stream. This memset and the final reduction are timed in run().
      const auto status=Kernel::initialize_workspace(args,workspace,stream);
      if(status!=cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("K-reduction scratch: ")+cutlassGetStatusString(status));
    }
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
    case 39:return Candidate<T,128,256,4>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 40:return Candidate<T,128,256,8>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 41:return Candidate<T,128,256,1,0,0,64,false,2>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 42:return Candidate<T,128,256,2,0,0,64,false,2>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 56:return Candidate<T,64,128,1,1,3>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 57:return Candidate<T,64,256,1,1,2>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 58:return Candidate<T,128,128,1,0,3,64,true,1,2>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 59:return Candidate<T,64,256,1,1,2,64,true,1,2>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 60:return Candidate<T,128,128,1,0,2,64,false,1,2,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 61:return Candidate<T,128,256,1,0,0,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 62:return Candidate<T,128,256,1,0,0,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,1);
    case 63:return Candidate<T,128,256,1,0,0,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,4);
    case 64:return Candidate<T,128,256,1,0,0,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8,1);
    case 65:return Candidate<T,128,256,1,0,0,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8,2);
    case 66:return Candidate<T,128,128,1,0,0,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 67:return Candidate<T,64,256,1,2,0,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 68:return Candidate<T,64,256,1,2,3,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 69:return Candidate<T,64,256,2,2,0,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 70:return Candidate<T,128,256,2,0,0,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 71:return Candidate<T,128,256,1,0,0,64,false,2,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 72:return Candidate<T,128,256,2,0,0,64,false,2,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 73:return Candidate<T,128,256,4,0,0,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 74:return Candidate<T,128,256,2,0,0,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8,1);
    case 75:return Candidate<T,128,256,2,0,0,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8,2);
    case 76:return Candidate<T,128,128,1,0,2,64,false,1,2,true,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 77:return Candidate<T,128,128,1,0,2,64,true,1,2,true,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 78:return Candidate<T,128,128,1,0,3,64,true,1,2,true,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 79:return Candidate<T,128,160,1,0,2,64,false,1,1,true,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 80:return Candidate<T,128,64,1,0,3,64,false,1,2,true,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 81:return Candidate<T,128,256,1,0,0,64,false,1,1,true,false,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 82:return Candidate<T,128,256,2,0,0,64,false,1,1,true,false,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 83:return Candidate<T,128,128,1,0,0,64,false,1,1,true,false,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 84:return Candidate<T,128,128,2,0,0,64,false,1,1,true,false,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 85:return Candidate<T,128,256,1,0,0,64,false,1,1,true,false,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,4,0);
    case 86:return Candidate<T,128,256,1,0,0,64,false,1,1,true,false,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8,1);
    case 87:return Candidate<T,128,256,1,0,0,64,false,1,1,true,false,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8,2);
    case 88:return Candidate<T,128,256,2,0,0,64,false,1,1,true,false,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8,1);
    case 89:return Candidate<T,128,256,2,0,0,64,false,1,1,true,false,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8,2);
    case 90:return Candidate<T,128,256,1,0,4,64,false,1,1,true,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 91:return Candidate<T,128,256,1,0,4,64,true,1,1,true,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 92:return Candidate<T,128,256,2,0,4,64,false,1,1,true,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 93:return Candidate<T,128,256,1,0,4,64,false,2,1,true,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 94:return Candidate<T,128,256,1,0,0,64,false,1,1,true,false,true,2>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 95:return Candidate<T,128,256,1,0,0,64,false,1,1,true,false,true,4>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 96:return Candidate<T,128,256,2,0,0,64,false,1,1,true,false,true,2>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 97:return Candidate<T,128,256,2,0,0,64,false,1,1,true,false,true,4>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 98:return Candidate<T,128,256,1,0,0,128,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 99:return Candidate<T,128,256,2,0,0,128,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 100:return Candidate<T,128,256,1,0,0,32,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 101:return Candidate<T,128,256,2,0,0,32,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 102:return Candidate<T,128,256,1,0,0,64,false,1,1,true,false,false,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 103:return Candidate<T,128,256,2,0,0,64,false,1,1,true,false,false,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 104:return Candidate<T,128,256,1,0,0,32,false,1,1,true,false,false,1,false,3>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 105:return Candidate<T,128,256,2,0,0,32,false,1,1,true,false,false,1,false,3>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 106:return Candidate<T,128,256,1,0,0,32,false,1,1,true,false,false,1,false,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 107:return Candidate<T,128,256,2,0,0,32,false,1,1,true,false,false,1,false,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 108:return Candidate<T,128,256,1,0,0,32,false,1,1,true,false,false,1,false,3,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 109:return Candidate<T,128,256,2,0,0,32,false,1,1,true,false,false,1,false,3,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 110:return Candidate<T,128,128,1,2,0,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    case 111:return Candidate<T,128,128,2,2,0,64,false,1,1,true>::run(p,A,B,C,D,workspace,sm_count,stream,operation,8);
    default:return false;
  }
}
template<class T> size_t stream_workspace(int id,const Problem& p,int sm_count) {
  switch(id) {
    case 81:return Candidate<T,128,256,1,0,0,64,false,1,1,true,false,true>::workspace_bytes(p,sm_count);
    case 82:return Candidate<T,128,256,2,0,0,64,false,1,1,true,false,true>::workspace_bytes(p,sm_count);
    case 83:return Candidate<T,128,128,1,0,0,64,false,1,1,true,false,true>::workspace_bytes(p,sm_count);
    case 84:return Candidate<T,128,128,2,0,0,64,false,1,1,true,false,true>::workspace_bytes(p,sm_count);
    case 85:return Candidate<T,128,256,1,0,0,64,false,1,1,true,false,true>::workspace_bytes(p,sm_count,4,0);
    case 86:return Candidate<T,128,256,1,0,0,64,false,1,1,true,false,true>::workspace_bytes(p,sm_count,8,1);
    case 87:return Candidate<T,128,256,1,0,0,64,false,1,1,true,false,true>::workspace_bytes(p,sm_count,8,2);
    case 88:return Candidate<T,128,256,2,0,0,64,false,1,1,true,false,true>::workspace_bytes(p,sm_count,8,1);
    case 89:return Candidate<T,128,256,2,0,0,64,false,1,1,true,false,true>::workspace_bytes(p,sm_count,8,2);
    case 94:return Candidate<T,128,256,1,0,0,64,false,1,1,true,false,true,2>::workspace_bytes(p,sm_count);
    case 95:return Candidate<T,128,256,1,0,0,64,false,1,1,true,false,true,4>::workspace_bytes(p,sm_count);
    case 96:return Candidate<T,128,256,2,0,0,64,false,1,1,true,false,true,2>::workspace_bytes(p,sm_count);
    case 97:return Candidate<T,128,256,2,0,0,64,false,1,1,true,false,true,4>::workspace_bytes(p,sm_count);
    default:return 0;
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
size_t hopper_streamk_workspace(const Problem& p,int id,int sm_count) {
  if((id<81 || id>89) && (id<94 || id>97))return 0;
  if(sm_count<=0)throw std::invalid_argument("K-reduction workspace requires captured SM count");
  if(p.dtype==DType::BF16)return stream_workspace<cutlass::bfloat16_t>(id,p,sm_count);
  return stream_workspace<cutlass::half_t>(id,p,sm_count);
}
const char* hopper_candidate_name(int id) {
  static const char* names[]={"128x64x64_c1","128x128x64_c1","128x256x64_c1","128x128x64_c2","simt_small","pad_128x64x64_c1","pad_128x128x64_c1","pad_128x256x64_c1","pad_128x128x64_c2",
      "128x128x64_c1_sw4","128x128x64_c1_sw8","128x256x64_c1_sw4",
      "128x256x64_c1_sw8","128x128x64_c2_sw4","128x128x64_c2_sw8",
      "64x32x64_basic_s3","64x64x64_basic_s3","128x128x64_pingpong_s3",
      "256x128x64_c1_sw8","128x256x64_c2_sw8","128x256x64_pingpong_s3_sw8",
      "128x256x64_c1_sw8_rasterN","128x128x64_c2_sw8_rasterN","256x128x64_c1_sw8_rasterM","wmma_32x64x32","wmma_32x64x64","64x32x128_basic_s3","64x64x128_basic_s3","64x64x256_basic_s2","128x64x128_c1","128x128x128_c1_sw8","128x256x128_direct_sw8","128x128x128_direct_sw8","wmma_32x32x32","wmma_32x32x64","mma_32x32x32","mma_32x32x64","even_mma_32x32x32","even_mma_32x32x64","128x256x64_cm4_sw8","128x256x64_cm8_sw8","128x256x64_cn2_sw8","128x256x64_cm2cn2_sw8","even_mma_swizzle_32x32x64","even_mma_swizzle_16x32x64","even_mma_swizzle_32x16x64","register_mma_16x32x16","register_mma_32x16x16","even_mma_32x32x128","even_mma_16x64x128","ampere_32x64x32","ampere_32x64x64","ampere_64x64x32","ampere_64x64x64","ampere_64x32x64","ampere_32x128x32_w32x32","64x128x64_basic_s3_sw8","64x256x64_basic_s2_sw8","128x128x64_direct_s3_2cta_sw8","64x256x64_direct_s2_2cta_sw8","128x128x64_beta0_s2_2cta_sw8","128x256x64_beta0_auto_sw8","128x256x64_beta0_sw1","128x256x64_beta0_sw4","128x256x64_beta0_rasterM","128x256x64_beta0_rasterN","128x128x64_beta0_auto_sw8","64x256x64_beta0_pingpong","64x256x64_beta0_pingpong_s3","64x256x64_beta0_pingpong_c2","128x256x64_beta0_cm2","128x256x64_beta0_cn2","128x256x64_beta0_cm2cn2","128x256x64_beta0_cm4","128x256x64_beta0_cm2_rasterM","128x256x64_beta0_cm2_rasterN","128x128x64_beta0_ninewarp_2workers","128x128x64_ninewarp_direct_s2","128x128x64_ninewarp_direct_s3","128x160x64_ninewarp_s2","128x64x64_ninewarp_s3","128x256x64_beta0_streamk","128x256x64_beta0_streamk_cm2","128x128x64_beta0_streamk","128x128x64_beta0_streamk_cm2","128x256x64_beta0_streamk_sw4","128x256x64_beta0_streamk_rasterM","128x256x64_beta0_streamk_rasterN","128x256x64_beta0_streamk_cm2_rasterM","128x256x64_beta0_streamk_cm2_rasterN","128x256x64_ninewarp_tma_s4","128x256x64_ninewarp_direct_s4","128x256x64_ninewarp_tma_s4_cm2","128x256x64_ninewarp_tma_s4_cn2","128x256x64_beta0_splitk2","128x256x64_beta0_splitk4","128x256x64_beta0_splitk2_cm2","128x256x64_beta0_splitk4_cm2","128x256x128_beta0_auto","128x256x128_beta0_cm2","128x256x32_beta0_auto","128x256x32_beta0_cm2","128x256x64_beta0_swapped","128x256x64_beta0_swapped_cm2","128x256x32_beta0_inflight3","128x256x32_beta0_inflight3_cm2","128x256x32_beta0_batch8","128x256x32_beta0_batch8_cm2","128x256x32_beta0_inflight3_batch8","128x256x32_beta0_inflight3_batch8_cm2","128x128x64_beta0_pingpong","128x128x64_beta0_pingpong_cm2"};
  static_assert(sizeof(names)/sizeof(names[0])==HopperCandidateCount,"Candidate metadata count mismatch");
  return id>=0 && id<HopperCandidateCount ? names[id]:"invalid";
}
bool hopper_dense(int id,const Problem& p,const void* A,const void* B,const float* C,float* D,
                  void* workspace,int sm_count,cudaStream_t stream,int operation) {
  if(id>=56 && id<=60 && (p.m>2048 || p.n>2048 || p.k>2048) && operation!=2)return false;
  if(id>=50 && id<=55)return ampere_dense(id-50,p,A,B,C,D,stream,operation);
  if(id==48 || id==49)return mma_even_dense(id==48?1024:1025,p,A,B,C,D,stream,operation);
  if(id==46 || id==47)return mma_even_dense(id==46?512:513,p,A,B,C,D,stream,operation);
  if(id>=43 && id<=45) {
    constexpr int codes[]={256,128,129};
    return mma_even_dense(codes[id-43],p,A,B,C,D,stream,operation);
  }
  if(id==37 || id==38)return mma_even_dense(id==37?32:64,p,A,B,C,D,stream,operation);
  if(id>=33 && id<=36) {
    constexpr int codes[]={96,128,256,257};
    return wmma_dense(codes[id-33],p,A,B,C,D,stream,operation);
  }
  if(id>=26 && id<=32 && (p.m>2048 || p.n>2048 || p.k>2048) && operation!=2) return false;
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

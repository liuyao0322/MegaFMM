#ifndef PROBE_VARIANT
#define PROBE_VARIANT 0
#endif
#include "internal.h"
#include "../solution/hopper_nine_warp_kernel.cuh"
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include "hopper_split_group369d.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace megafmm::detail {
namespace {
using namespace cute;
template<class T,int M,int N,int ClusterM,int Schedule=0,int Stages=0,int K=64,bool DirectStore=false,int ClusterN=1,int Workers=1,bool ZeroBeta=false,bool NineWarp=false,bool StreamK=false,int Splits=1,int Window=1>
struct Candidate {
  static_assert(!NineWarp || ZeroBeta,"Nine-warp candidates require beta-zero eligibility");
  static_assert(Splits>=1 && (StreamK || Splits==1),"K splits require the reduction scheduler");
  using Tile=Shape<Int<M>,Int<N>,Int<K>>;
  using Cluster=Shape<Int<ClusterM>,Int<ClusterN>,_1>;
  using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,float,float,
      cute::conditional_t<ZeroBeta,void,float>,cutlass::layout::RowMajor,4,float,cutlass::layout::RowMajor,4,
      cute::conditional_t<DirectStore,cutlass::epilogue::NoSmemWarpSpecialized,
        cute::conditional_t<Schedule==0,cutlass::epilogue::TmaWarpSpecializedCooperative,
                            cutlass::epilogue::TmaWarpSpecialized>>>::CollectiveOp;
  using BaseMainloop=typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
      T,cutlass::layout::RowMajor,8,T,cutlass::layout::RowMajor,8,float,Tile,Cluster,
      cute::conditional_t<Stages==0,
        cutlass::gemm::collective::StageCountAutoCarveout<int(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::collective::StageCount<Stages>>,
      cute::conditional_t<Schedule==0,cutlass::gemm::KernelTmaWarpSpecializedCooperative,
        cute::conditional_t<Schedule==1,cutlass::gemm::KernelTmaWarpSpecialized,
                            cutlass::gemm::KernelTmaWarpSpecializedPingpong>>>::CollectiveOp;
  using Mainloop=cutlass::gemm::collective::SplitGroupMainloop<BaseMainloop>;
  using Kernel=cute::conditional_t<NineWarp,
      cutlass::gemm::kernel::NineWarpGemm<Shape<int,int,int,int>,Mainloop,Epilogue>,
      cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue,
          cute::conditional_t<StreamK,cutlass::gemm::StreamKScheduler,void>>>;
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  static typename Gemm::Arguments arguments(const Problem& p,const void* A,const void* B,
      const float* C,float* D,int sm_count,int swizzle=8,int raster=0) {
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {int(p.m),int(p.n),int(p.k),1},
      {static_cast<const T*>(A),typename Kernel::StrideA{p.lda,Int<1>{},int64_t(0)},
       static_cast<const T*>(B),typename Kernel::StrideB{Int<1>{},p.ldb,int64_t(0)}},
      {{},p.beta==0 ? D:C,typename Kernel::StrideC{p.beta==0 ? p.ldd:p.ldc,Int<1>{},int64_t(0)},
       D,typename Kernel::StrideD{p.ldd,Int<1>{},int64_t(0)}},
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

}
}

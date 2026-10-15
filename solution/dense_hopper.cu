#include "internal.h"
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <stdexcept>

namespace megafmm::detail {
namespace {
using namespace cute;
template<class T,int M,int N,int ClusterM>
struct Candidate {
  using Tile=Shape<Int<M>,Int<N>,_64>;
  using Cluster=Shape<Int<ClusterM>,_1,_1>;
  using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,float,float,
      float,cutlass::layout::RowMajor,4,float,cutlass::layout::RowMajor,4,
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
  using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
      T,cutlass::layout::RowMajor,8,T,cutlass::layout::RowMajor,8,float,Tile,Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<int(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue>;
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  static bool run(const Problem& p,const void* A,const void* B,const float* C,float* D,
                  void* workspace,int sm_count,cudaStream_t stream,bool launch) {
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {int(p.m),int(p.n),int(p.k),1},
      {static_cast<const T*>(A),typename Kernel::StrideA{p.lda,Int<1>{},int64_t(0)},
       static_cast<const T*>(B),typename Kernel::StrideB{Int<1>{},p.ldb,int64_t(0)}},
      {{},p.beta==0 ? D:C,typename Kernel::StrideC{p.beta==0 ? p.ldd:p.ldc,Int<1>{},int64_t(0)},
       D,typename Kernel::StrideD{p.ldd,Int<1>{},int64_t(0)}},
      {0,sm_count}
    };
    args.epilogue.thread.alpha=p.alpha;args.epilogue.thread.beta=p.beta;
    if(Gemm::can_implement(args)!=cutlass::Status::kSuccess) return false;
    // The persistent scheduler has zero workspace. No counters or values are
    // retained between invocations. Hardware information is captured at setup.
    if(Gemm::get_workspace_size(args)!=0) return false;
    if(!launch) return true;
    Gemm op;
    auto status=op.initialize(args,workspace,stream);
    if(status==cutlass::Status::kSuccess) status=op.run(stream);
    if(status!=cutlass::Status::kSuccess)
      throw std::runtime_error(std::string("Hopper dense: ")+cutlassGetStatusString(status));
    return true;
  }
};
template<class T>
bool dispatch(int id,const Problem& p,const void* A,const void* B,const float* C,float* D,
              void* workspace,int sm_count,cudaStream_t stream,bool launch) {
  switch(id) {
    case 0:return Candidate<T,128,64,1>::run(p,A,B,C,D,workspace,sm_count,stream,launch);
    case 1:return Candidate<T,128,128,1>::run(p,A,B,C,D,workspace,sm_count,stream,launch);
    case 2:return Candidate<T,128,256,1>::run(p,A,B,C,D,workspace,sm_count,stream,launch);
    case 3:return Candidate<T,128,128,2>::run(p,A,B,C,D,workspace,sm_count,stream,launch);
    default:return false;
  }
}
}
const char* hopper_candidate_name(int id) {
  static const char* names[]={"128x64x64_c1","128x128x64_c1","128x256x64_c1","128x128x64_c2"};
  return id>=0 && id<4 ? names[id]:"invalid";
}
bool hopper_dense(int id,const Problem& p,const void* A,const void* B,const float* C,float* D,
                  void* workspace,int sm_count,cudaStream_t stream,bool launch) {
  if(p.trans_a || p.trans_b || p.lda%8 || p.ldb%8 || p.ldd%4 ||
      (p.beta!=0 && p.ldc%4) || p.k%8 || p.n%8) return false;
  if(reinterpret_cast<uintptr_t>(A)%16 || reinterpret_cast<uintptr_t>(B)%16 ||
      reinterpret_cast<uintptr_t>(D)%16 || (p.beta!=0 && reinterpret_cast<uintptr_t>(C)%16)) return false;
  if(p.dtype==DType::BF16)
    return dispatch<cutlass::bfloat16_t>(id,p,A,B,C,D,workspace,sm_count,stream,launch);
  return dispatch<cutlass::half_t>(id,p,A,B,C,D,workspace,sm_count,stream,launch);
}
} // namespace megafmm::detail

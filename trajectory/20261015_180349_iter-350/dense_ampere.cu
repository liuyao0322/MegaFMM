#include "internal.h"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <stdexcept>

namespace megafmm::detail {
namespace {
template<class T,int M,int N,int K,int WM,int WN>
struct AmpereCandidate {
  using Gemm=cutlass::gemm::device::Gemm<T,cutlass::layout::RowMajor,T,cutlass::layout::RowMajor,
      float,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<M,N,K>,cutlass::gemm::GemmShape<WM,WN,K>,
      cutlass::gemm::GemmShape<16,8,16>,cutlass::epilogue::thread::LinearCombination<float,2,float,float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,2,2,false>;
  using Kernel=typename Gemm::GemmKernel;
  static bool run(const Problem& p,const void* A,const void* B,const float* C,float* D,cudaStream_t stream,int operation) {
    constexpr int shared=int(sizeof(typename Kernel::SharedStorage));
    if(operation==2) {
      cudaError_t e=cudaSuccess;
      if(shared>=(48<<10))e=cudaFuncSetAttribute(cutlass::Kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);
      if(e==cudaSuccess)e=cudaFuncSetAttribute(cutlass::Kernel<Kernel>,cudaFuncAttributePreferredSharedMemoryCarveout,100);
      if(e!=cudaSuccess)throw std::runtime_error(std::string("Ampere setup: ")+cudaGetErrorString(e));
      return true;
    }
    typename Gemm::Arguments args{
      {int(p.m),int(p.n),int(p.k)},
      {static_cast<const T*>(A),int(p.lda)},{static_cast<const T*>(B),int(p.ldb)},
      {p.beta==0?D:C,int(p.beta==0?p.ldd:p.ldc)},{D,int(p.ldd)},
      {p.alpha,p.beta},1};
    if(Gemm::can_implement(args)!=cutlass::Status::kSuccess || Gemm::get_workspace_size(args))return false;
    if(operation==0)return true;
    typename Gemm::ThreadblockSwizzle swizzle;
    auto tiled=swizzle.get_tiled_shape(args.problem_size,{M,N,K},1);
    typename Kernel::Params params{args.problem_size,tiled,args.ref_A.non_const_ref(),args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),args.ref_D,args.epilogue,nullptr};
    // Attributes are prepared once at Plan creation. Construct only current
    // pointer/stride parameters here; no initialize(), allocation, or sync.
    cutlass::Kernel<Kernel><<<swizzle.get_grid_shape(tiled),Kernel::kThreadCount,shared,stream>>>(params);
    auto e=cudaPeekAtLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    return true;
  }
};
template<class T>bool dispatch(int id,const Problem& p,const void* A,const void* B,const float* C,float* D,cudaStream_t stream,int operation) {
  switch(id) {
    case 0:return AmpereCandidate<T,32,64,32,16,32>::run(p,A,B,C,D,stream,operation);
    case 1:return AmpereCandidate<T,32,64,64,16,32>::run(p,A,B,C,D,stream,operation);
    case 2:return AmpereCandidate<T,64,64,32,32,32>::run(p,A,B,C,D,stream,operation);
    case 3:return AmpereCandidate<T,64,64,64,32,32>::run(p,A,B,C,D,stream,operation);
    case 4:return AmpereCandidate<T,64,32,64,32,32>::run(p,A,B,C,D,stream,operation);
    case 5:return AmpereCandidate<T,32,128,32,16,32>::run(p,A,B,C,D,stream,operation);
    default:return false;
  }
}
}
bool ampere_dense(int id,const Problem& p,const void* A,const void* B,const float* C,float* D,cudaStream_t stream,int operation) {
  if(operation!=2 && (p.trans_a || p.trans_b || p.m<=0 || p.n<=0 || p.k<=0 || p.m>2048 || p.n>2048 || p.k>2048 ||
     p.n%2 || p.k%2 || p.lda%2 || p.ldb%2 || p.ldd%2 || p.lda>(1<<20) || p.ldb>(1<<20) ||
     p.ldc>(1<<20) || p.ldd>(1<<20) || reinterpret_cast<uintptr_t>(A)%4 || reinterpret_cast<uintptr_t>(B)%4 ||
     reinterpret_cast<uintptr_t>(D)%8 || (p.beta!=0 && (p.ldc%2 || reinterpret_cast<uintptr_t>(C)%8))))return false;
  if(p.dtype==DType::BF16)return dispatch<cutlass::bfloat16_t>(id,p,A,B,C,D,stream,operation);
  return dispatch<cutlass::half_t>(id,p,A,B,C,D,stream,operation);
}
}

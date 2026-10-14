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
constexpr int BM=64,BN=32,BK=256,Stages=4,ASize=BM*BK,BSize=BK*BN,Elements=BM*BN;
constexpr int Plane=512*512,Threads=768;
constexpr size_t WorkspaceBytes=size_t(8)*Plane*2;
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
template<class T> struct alignas(128) Shared {
  T operand[Stages][ASize+BSize];
  // Row-major P with an XOR permutation inside each 32-float row.
  // Both WGMMA fragment stores and row-major consumer loads use every bank once.
  uint64_t ready[Stages],free[Stages];
  uint64_t pack_ready;
};
static_assert(sizeof(Shared<cute::half_t>)==196736,"shared budget");
__device__ void wait(const uint64_t* b,int phase){Barrier::wait(b,phase);asm volatile("" ::: "memory");}
__device__ void arrive(const uint64_t* b){asm volatile("" ::: "memory");Barrier::arrive(b);}

__device__ __forceinline__ int product_index(int row,int col){
  // Fragment lanes vary column bits 1,2 and row bits 0,1,2.
  // XOR the latter into column bits 0,3,4, the three remaining bank bits.
  return row*BN+(col^((row&1)|((row&6)<<2)));
}

// Zero-based P IDs. Producer and consumers advance barriers by step, not P ID.
__device__ constexpr int product_for_step(int step){
  return step<2 ? step:step==2 ? 5:step==3 ? 6:step==4 ? 4:step==5 ? 2:3;
}

template<class T> __device__ __forceinline__ void mma_product(Shared<T>& s,int step,float (&out)[16]){
 using namespace cute;
 constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
 constexpr auto lb=tile_to_shape(GMMA::Layout_MN_SW64_Atom<T>{},Shape<_32,Int<BK>>{});
 int slot0=(2*step)%Stages,slot1=slot0+1,phase=(2*step/Stages)%2;
 auto a0=make_tensor(make_smem_ptr(s.operand[slot0]),la);
 auto b0=make_tensor(make_smem_ptr(s.operand[slot0]+ASize),lb);
 auto a1=make_tensor(make_smem_ptr(s.operand[slot1]),la);
 auto b1=make_tensor(make_smem_ptr(s.operand[slot1]+ASize),lb);
 auto da0=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(a0,Shape<_64,_16>{}));
 auto db0=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(b0,Shape<_32,_16>{}));
 auto da1=make_tensor<GMMA::smem_desc<GMMA::Major::K>>(tiled_divide(a1,Shape<_64,_16>{}));
 auto db1=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>(tiled_divide(b1,Shape<_32,_16>{}));
 wait(&s.ready[slot0],phase);
 if constexpr(std::is_same_v<T,bfloat16_t>){
  asm volatile(
   "{ .reg .pred p,leader,done; .reg .b64 da,db; .reg .f32 d<16>;\n"
   "setp.ne.u32 p,1,0;\n"
   "setp.eq.u32 leader,%24,0;\n"
   "mov.f32 d0,0f00000000;\n"
   "mov.f32 d1,0f00000000;\n"
   "mov.f32 d2,0f00000000;\n"
   "mov.f32 d3,0f00000000;\n"
   "mov.f32 d4,0f00000000;\n"
   "mov.f32 d5,0f00000000;\n"
   "mov.f32 d6,0f00000000;\n"
   "mov.f32 d7,0f00000000;\n"
   "mov.f32 d8,0f00000000;\n"
   "mov.f32 d9,0f00000000;\n"
   "mov.f32 d10,0f00000000;\n"
   "mov.f32 d11,0f00000000;\n"
   "mov.f32 d12,0f00000000;\n"
   "mov.f32 d13,0f00000000;\n"
   "mov.f32 d14,0f00000000;\n"
   "mov.f32 d15,0f00000000;\n"
   "wgmma.fence.sync.aligned;\n"
   "add.u64 da, %16, %25;\n"
   "add.u64 db, %17, %26;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %27;\n"
   "add.u64 db, %17, %28;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %29;\n"
   "add.u64 db, %17, %30;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %31;\n"
   "add.u64 db, %17, %32;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %33;\n"
   "add.u64 db, %17, %34;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %35;\n"
   "add.u64 db, %17, %36;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %37;\n"
   "add.u64 db, %17, %38;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %39;\n"
   "add.u64 db, %17, %40;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %41;\n"
   "add.u64 db, %17, %42;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %43;\n"
   "add.u64 db, %17, %44;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %45;\n"
   "add.u64 db, %17, %46;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %47;\n"
   "add.u64 db, %17, %48;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %49;\n"
   "add.u64 db, %17, %50;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %51;\n"
   "add.u64 db, %17, %52;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %53;\n"
   "add.u64 db, %17, %54;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %55;\n"
   "add.u64 db, %17, %56;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "wgmma.commit_group.sync.aligned;\n"
   "wgmma.wait_group.sync.aligned 0;\n"
   "bar.warp.sync 0xffffffff;\n"
   "@leader mbarrier.arrive.shared::cta.b64 _,[%21];\n"
   "WAIT_SECOND:\n"
   "mbarrier.try_wait.parity.shared::cta.b64 done,[%20],%23,0x989680;\n"
   "@!done bra WAIT_SECOND;\n"
   "wgmma.fence.sync.aligned;\n"
   "add.u64 da, %18, %25;\n"
   "add.u64 db, %19, %26;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %27;\n"
   "add.u64 db, %19, %28;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %29;\n"
   "add.u64 db, %19, %30;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %31;\n"
   "add.u64 db, %19, %32;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %33;\n"
   "add.u64 db, %19, %34;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %35;\n"
   "add.u64 db, %19, %36;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %37;\n"
   "add.u64 db, %19, %38;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %39;\n"
   "add.u64 db, %19, %40;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %41;\n"
   "add.u64 db, %19, %42;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %43;\n"
   "add.u64 db, %19, %44;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %45;\n"
   "add.u64 db, %19, %46;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %47;\n"
   "add.u64 db, %19, %48;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %49;\n"
   "add.u64 db, %19, %50;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %51;\n"
   "add.u64 db, %19, %52;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %53;\n"
   "add.u64 db, %19, %54;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %55;\n"
   "add.u64 db, %19, %56;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "wgmma.commit_group.sync.aligned;\n"
   "wgmma.wait_group.sync.aligned 0;\n"
   "bar.warp.sync 0xffffffff;\n"
   "@leader mbarrier.arrive.shared::cta.b64 _,[%22];\n"
   "mov.f32 %0,d0;\n"
   "mov.f32 %1,d1;\n"
   "mov.f32 %2,d2;\n"
   "mov.f32 %3,d3;\n"
   "mov.f32 %4,d4;\n"
   "mov.f32 %5,d5;\n"
   "mov.f32 %6,d6;\n"
   "mov.f32 %7,d7;\n"
   "mov.f32 %8,d8;\n"
   "mov.f32 %9,d9;\n"
   "mov.f32 %10,d10;\n"
   "mov.f32 %11,d11;\n"
   "mov.f32 %12,d12;\n"
   "mov.f32 %13,d13;\n"
   "mov.f32 %14,d14;\n"
   "mov.f32 %15,d15;\n"
   "}\n"
   : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]), "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7]), "=f"(out[8]), "=f"(out[9]), "=f"(out[10]), "=f"(out[11]), "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15])
   : "l"(uint64_t(da0(0,0,0))), "l"(uint64_t(db0(0,0,0))), "l"(uint64_t(da1(0,0,0))), "l"(uint64_t(db1(0,0,0))), "r"(uint32_t(__cvta_generic_to_shared(&s.ready[slot1]))), "r"(uint32_t(__cvta_generic_to_shared(&s.free[slot0]))), "r"(uint32_t(__cvta_generic_to_shared(&s.free[slot1]))), "r"(phase), "r"(threadIdx.x&31), "n"(uint32_t(da0.layout()(0,0,Int<0>{}))), "n"(uint32_t(db0.layout()(0,0,Int<0>{}))), "n"(uint32_t(da0.layout()(0,0,Int<1>{}))), "n"(uint32_t(db0.layout()(0,0,Int<1>{}))), "n"(uint32_t(da0.layout()(0,0,Int<2>{}))), "n"(uint32_t(db0.layout()(0,0,Int<2>{}))), "n"(uint32_t(da0.layout()(0,0,Int<3>{}))), "n"(uint32_t(db0.layout()(0,0,Int<3>{}))), "n"(uint32_t(da0.layout()(0,0,Int<4>{}))), "n"(uint32_t(db0.layout()(0,0,Int<4>{}))), "n"(uint32_t(da0.layout()(0,0,Int<5>{}))), "n"(uint32_t(db0.layout()(0,0,Int<5>{}))), "n"(uint32_t(da0.layout()(0,0,Int<6>{}))), "n"(uint32_t(db0.layout()(0,0,Int<6>{}))), "n"(uint32_t(da0.layout()(0,0,Int<7>{}))), "n"(uint32_t(db0.layout()(0,0,Int<7>{}))), "n"(uint32_t(da0.layout()(0,0,Int<8>{}))), "n"(uint32_t(db0.layout()(0,0,Int<8>{}))), "n"(uint32_t(da0.layout()(0,0,Int<9>{}))), "n"(uint32_t(db0.layout()(0,0,Int<9>{}))), "n"(uint32_t(da0.layout()(0,0,Int<10>{}))), "n"(uint32_t(db0.layout()(0,0,Int<10>{}))), "n"(uint32_t(da0.layout()(0,0,Int<11>{}))), "n"(uint32_t(db0.layout()(0,0,Int<11>{}))), "n"(uint32_t(da0.layout()(0,0,Int<12>{}))), "n"(uint32_t(db0.layout()(0,0,Int<12>{}))), "n"(uint32_t(da0.layout()(0,0,Int<13>{}))), "n"(uint32_t(db0.layout()(0,0,Int<13>{}))), "n"(uint32_t(da0.layout()(0,0,Int<14>{}))), "n"(uint32_t(db0.layout()(0,0,Int<14>{}))), "n"(uint32_t(da0.layout()(0,0,Int<15>{}))), "n"(uint32_t(db0.layout()(0,0,Int<15>{}))), "n"(int32_t(GMMA::Major::K)), "n"(int32_t(GMMA::Major::MN))
   : "memory");
 }
 else{
  asm volatile(
   "{ .reg .pred p,leader,done; .reg .b64 da,db; .reg .f32 d<16>;\n"
   "setp.ne.u32 p,1,0;\n"
   "setp.eq.u32 leader,%24,0;\n"
   "mov.f32 d0,0f00000000;\n"
   "mov.f32 d1,0f00000000;\n"
   "mov.f32 d2,0f00000000;\n"
   "mov.f32 d3,0f00000000;\n"
   "mov.f32 d4,0f00000000;\n"
   "mov.f32 d5,0f00000000;\n"
   "mov.f32 d6,0f00000000;\n"
   "mov.f32 d7,0f00000000;\n"
   "mov.f32 d8,0f00000000;\n"
   "mov.f32 d9,0f00000000;\n"
   "mov.f32 d10,0f00000000;\n"
   "mov.f32 d11,0f00000000;\n"
   "mov.f32 d12,0f00000000;\n"
   "mov.f32 d13,0f00000000;\n"
   "mov.f32 d14,0f00000000;\n"
   "mov.f32 d15,0f00000000;\n"
   "wgmma.fence.sync.aligned;\n"
   "add.u64 da, %16, %25;\n"
   "add.u64 db, %17, %26;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %27;\n"
   "add.u64 db, %17, %28;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %29;\n"
   "add.u64 db, %17, %30;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %31;\n"
   "add.u64 db, %17, %32;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %33;\n"
   "add.u64 db, %17, %34;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %35;\n"
   "add.u64 db, %17, %36;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %37;\n"
   "add.u64 db, %17, %38;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %39;\n"
   "add.u64 db, %17, %40;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %41;\n"
   "add.u64 db, %17, %42;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %43;\n"
   "add.u64 db, %17, %44;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %45;\n"
   "add.u64 db, %17, %46;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %47;\n"
   "add.u64 db, %17, %48;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %49;\n"
   "add.u64 db, %17, %50;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %51;\n"
   "add.u64 db, %17, %52;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %53;\n"
   "add.u64 db, %17, %54;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %16, %55;\n"
   "add.u64 db, %17, %56;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "wgmma.commit_group.sync.aligned;\n"
   "wgmma.wait_group.sync.aligned 0;\n"
   "bar.warp.sync 0xffffffff;\n"
   "@leader mbarrier.arrive.shared::cta.b64 _,[%21];\n"
   "WAIT_SECOND:\n"
   "mbarrier.try_wait.parity.shared::cta.b64 done,[%20],%23,0x989680;\n"
   "@!done bra WAIT_SECOND;\n"
   "wgmma.fence.sync.aligned;\n"
   "add.u64 da, %18, %25;\n"
   "add.u64 db, %19, %26;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %27;\n"
   "add.u64 db, %19, %28;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %29;\n"
   "add.u64 db, %19, %30;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %31;\n"
   "add.u64 db, %19, %32;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %33;\n"
   "add.u64 db, %19, %34;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %35;\n"
   "add.u64 db, %19, %36;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %37;\n"
   "add.u64 db, %19, %38;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %39;\n"
   "add.u64 db, %19, %40;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %41;\n"
   "add.u64 db, %19, %42;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %43;\n"
   "add.u64 db, %19, %44;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %45;\n"
   "add.u64 db, %19, %46;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %47;\n"
   "add.u64 db, %19, %48;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %49;\n"
   "add.u64 db, %19, %50;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %51;\n"
   "add.u64 db, %19, %52;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %53;\n"
   "add.u64 db, %19, %54;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "add.u64 da, %18, %55;\n"
   "add.u64 db, %19, %56;\n"
   "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, da, db, p, 1, 1, %57, %58;\n"
   "wgmma.commit_group.sync.aligned;\n"
   "wgmma.wait_group.sync.aligned 0;\n"
   "bar.warp.sync 0xffffffff;\n"
   "@leader mbarrier.arrive.shared::cta.b64 _,[%22];\n"
   "mov.f32 %0,d0;\n"
   "mov.f32 %1,d1;\n"
   "mov.f32 %2,d2;\n"
   "mov.f32 %3,d3;\n"
   "mov.f32 %4,d4;\n"
   "mov.f32 %5,d5;\n"
   "mov.f32 %6,d6;\n"
   "mov.f32 %7,d7;\n"
   "mov.f32 %8,d8;\n"
   "mov.f32 %9,d9;\n"
   "mov.f32 %10,d10;\n"
   "mov.f32 %11,d11;\n"
   "mov.f32 %12,d12;\n"
   "mov.f32 %13,d13;\n"
   "mov.f32 %14,d14;\n"
   "mov.f32 %15,d15;\n"
   "}\n"
   : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]), "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7]), "=f"(out[8]), "=f"(out[9]), "=f"(out[10]), "=f"(out[11]), "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15])
   : "l"(uint64_t(da0(0,0,0))), "l"(uint64_t(db0(0,0,0))), "l"(uint64_t(da1(0,0,0))), "l"(uint64_t(db1(0,0,0))), "r"(uint32_t(__cvta_generic_to_shared(&s.ready[slot1]))), "r"(uint32_t(__cvta_generic_to_shared(&s.free[slot0]))), "r"(uint32_t(__cvta_generic_to_shared(&s.free[slot1]))), "r"(phase), "r"(threadIdx.x&31), "n"(uint32_t(da0.layout()(0,0,Int<0>{}))), "n"(uint32_t(db0.layout()(0,0,Int<0>{}))), "n"(uint32_t(da0.layout()(0,0,Int<1>{}))), "n"(uint32_t(db0.layout()(0,0,Int<1>{}))), "n"(uint32_t(da0.layout()(0,0,Int<2>{}))), "n"(uint32_t(db0.layout()(0,0,Int<2>{}))), "n"(uint32_t(da0.layout()(0,0,Int<3>{}))), "n"(uint32_t(db0.layout()(0,0,Int<3>{}))), "n"(uint32_t(da0.layout()(0,0,Int<4>{}))), "n"(uint32_t(db0.layout()(0,0,Int<4>{}))), "n"(uint32_t(da0.layout()(0,0,Int<5>{}))), "n"(uint32_t(db0.layout()(0,0,Int<5>{}))), "n"(uint32_t(da0.layout()(0,0,Int<6>{}))), "n"(uint32_t(db0.layout()(0,0,Int<6>{}))), "n"(uint32_t(da0.layout()(0,0,Int<7>{}))), "n"(uint32_t(db0.layout()(0,0,Int<7>{}))), "n"(uint32_t(da0.layout()(0,0,Int<8>{}))), "n"(uint32_t(db0.layout()(0,0,Int<8>{}))), "n"(uint32_t(da0.layout()(0,0,Int<9>{}))), "n"(uint32_t(db0.layout()(0,0,Int<9>{}))), "n"(uint32_t(da0.layout()(0,0,Int<10>{}))), "n"(uint32_t(db0.layout()(0,0,Int<10>{}))), "n"(uint32_t(da0.layout()(0,0,Int<11>{}))), "n"(uint32_t(db0.layout()(0,0,Int<11>{}))), "n"(uint32_t(da0.layout()(0,0,Int<12>{}))), "n"(uint32_t(db0.layout()(0,0,Int<12>{}))), "n"(uint32_t(da0.layout()(0,0,Int<13>{}))), "n"(uint32_t(db0.layout()(0,0,Int<13>{}))), "n"(uint32_t(da0.layout()(0,0,Int<14>{}))), "n"(uint32_t(db0.layout()(0,0,Int<14>{}))), "n"(uint32_t(da0.layout()(0,0,Int<15>{}))), "n"(uint32_t(db0.layout()(0,0,Int<15>{}))), "n"(int32_t(GMMA::Major::K)), "n"(int32_t(GMMA::Major::MN))
   : "memory");
 }
}

template<class T> __global__ __launch_bounds__(Threads,1)
void sw_cooperative_hilo(const __grid_constant__ CUtensorMap amap,
                         const __grid_constant__ CUtensorMap bmap,
                         const __grid_constant__ CUtensorMap smap,
                         const __grid_constant__ CUtensorMap tmap,
                         const T* A,const T* B,T* packed,float* D){
  using namespace cute;
  extern __shared__ __align__(128) unsigned char buffer[];
  auto& s=*reinterpret_cast<Shared<T>*>(buffer);
  int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;
  if(tid==0){
    for(int j=0;j<Stages;++j){Barrier::init(&s.ready[j],1);Barrier::init(&s.free[j],4);}
    Barrier::init(&s.pack_ready,608);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  constexpr auto la=tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{});
  constexpr auto lb=tile_to_shape(GMMA::Layout_MN_SW64_Atom<T>{},Shape<_32,Int<BK>>{});
  if(tid==0){
    for(int step=0;step<7;++step)for(int q=0;q<512/BK;++q){
      int p=product_for_step(step);
      int sequence=step*(512/BK)+q,slot=sequence%Stages,k0=q*BK;
      if(sequence==4){
        wait(&s.pack_ready,0);
        // CUDA12.4 cooperative launch gives each invocation its own initialized
        // driver workspace. Only TMA leaders join this publication barrier;
        // the shared pack barrier has already acquired every CUDA writer.
        auto* barrier=&cooperative_groups::details::get_grid_workspace()->barrier;
        uint32_t increment=(blockIdx.x==0 && blockIdx.y==0) ? 0x80000000u-127u:1u;
        uint32_t old,current;
        asm volatile("atom.add.acq_rel.gpu.u32 %0,[%1],%2;" : "=r"(old) : "l"(barrier),"r"(increment) : "memory");
        do{asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(current) : "l"(barrier) : "memory");}while(!((old^current)&0x80000000u));
      }
      if(sequence>=Stages)wait(&s.free[slot],((sequence-Stages)/Stages)%2);
      bool alo=p==2 || p>=4,blo=p==3 || p>=4;
      // Seven-product experiment: only publish high operands; no residual loads.
      TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*(ASize+BSize));
      T* op=s.operand[slot];
      if(alo){
        int plane=p==2 ? 3:p-4;
        SM90_TMA_LOAD_4D::copy(&smap,&s.ready[slot],0,op,0,m0,k0/64,plane);
      }else{
        int col=(p==1 || p==3) ? 512:0,row=p==3 ? 512:0;
        SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],0,op,0,m0+row,(k0+col)/64);
      }
      if(blo){
        int plane=p==3 ? 3:p-4;
        SM90_TMA_LOAD_3D::copy(&tmap,&s.ready[slot],0,op+ASize,n0,k0,plane);
      }else{
        int col=p==2 ? 512:0,row=p==1 || p==2 ? 512:0;
        SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],0,op+ASize,n0+col,k0+row);
      }

    }
  }else if((tid>=32 && tid<128) || tid>=256){
    int worker=tid>=256 ? tid-160:tid-32;
  // All CTAs cooperate in one input transformation per invocation, never across calls.
  // Transform two adjacent elements per thread with aligned 32-bit IO. Experimental seven-product path:
  // retain FP32 transforms and high rounding, omit residual generation.
  for(int local=2*worker;local<Plane/(16*8);local+=2*608){
    int i=(blockIdx.y*gridDim.x+blockIdx.x)*(Plane/(16*8))+local;
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
    arrive(&s.pack_ready);
  }else if(tid>=128 && tid<256){
    float first[16],second[16];
    #pragma unroll 1
    for(int step=0;step<7;++step){
      float acc[16];
      mma_product(s,step,acc);
      #pragma unroll
      for(int j=0;j<8;++j){
        int v=2*j;
        int mn=GMMA::CLayout_64x32{}(tid-128,v);
        int at=(m0+mn%64)*1024+n0+mn/64;
        float2 out;
        #pragma unroll
        for(int lane=0;lane<2;++lane){
          int i=v+lane;
          float value=0;
          if(step==0)first[i]=acc[i];
          else if(step==1)value=__fadd_rn(first[i],acc[i]);
          else if(step==2)first[i]=__fadd_rn(first[i],acc[i]);
          else if(step==3)second[i]=__fadd_rn(first[i],acc[i]);
          else if(step==4){value=__fadd_rn(second[i],acc[i]);first[i]=__fadd_rn(first[i],acc[i]);}
          else if(step==5)value=__fadd_rn(first[i],acc[i]);
          else value=__fsub_rn(second[i],acc[i]);
          if(lane==0)out.x=value;else out.y=value;
        }
        if(step==1)*reinterpret_cast<float2*>(D+at)=out;
        else if(step==4)*reinterpret_cast<float2*>(D+at+512*1024+512)=out;
        else if(step==5)*reinterpret_cast<float2*>(D+at+512)=out;
        else if(step==6)*reinterpret_cast<float2*>(D+at+512*1024)=out;
      }
    }
  }
  // CTA shared storage survives until all role threads return.
}
void check(cudaError_t e,const char* where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
void encode(CUtensorMap& map,const void* ptr,DType dtype,bool is_a,bool transformed){
  if(reinterpret_cast<uintptr_t>(ptr)%16)throw std::invalid_argument("SW TMA input must be16-byte aligned");
  if(is_a){
    // Expose K/64 as an outer tensor dimension. The shared box still packs
    // (64 K elements, BM rows, BK/64 chunks), matching the existing GMMA layout.
    uint64_t dims[4]={64,transformed ? 512ULL:1024ULL,transformed ? 8ULL:16ULL,4};
    uint64_t strides[3]={transformed ? 1024ULL:2048ULL,128,512ULL*512*2};
    uint32_t box[4]={64,BM,BK/64,1},element_strides[4]={1,1,1,1};
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
      is_a ? CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_64B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
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
  if(blocks*prop.multiProcessorCount<128)throw std::invalid_argument("SW cooperative grid requires128 resident CTA capacity");
  return std::make_shared<SWState>(SWState{p.dtype});
}
void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,void* workspace,cudaStream_t stream){
  CUtensorMap a{},b{},s{},t{};encode(a,A,state.dtype,true,false);encode(b,B,state.dtype,false,false);
  encode(s,workspace,state.dtype,true,true);encode(t,static_cast<char*>(workspace)+WorkspaceBytes/2,state.dtype,false,true);
  void* args[]={&a,&b,&s,&t,&A,&B,&workspace,&D};dim3 grid(16,8);
  if(state.dtype==DType::BF16)
    check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(sw_cooperative_hilo<cute::bfloat16_t>),grid,Threads,args,sizeof(Shared<cute::bfloat16_t>),stream),"SW cooperative launch");
  else
    check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(sw_cooperative_hilo<cute::half_t>),grid,Threads,args,sizeof(Shared<cute::half_t>),stream),"SW cooperative launch");
}
} // namespace megafmm::detail

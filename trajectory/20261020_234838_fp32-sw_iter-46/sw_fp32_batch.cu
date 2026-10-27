#include "sw_fp32_batch.h"
#include <cuda.h>
#include <cooperative_groups.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <algorithm>
#include <stdexcept>

namespace megafmm::detail::sw_fp32_batch {
namespace {
constexpr int BK=32;
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ __forceinline__ float read(const float* x,int r,int c,int rows,int cols,int ld) {
  return r<rows && c<cols?x[int64_t(r)*ld+c]:0.0f;
}

// The A output is K-major so the leaf TMA can write vector-friendly shared A
// directly. This tiled transpose keeps both original reads and packed writes
// coalesced; seven transforms are rounded exactly as in the fused pipeline.
template<bool Grid2D=false> __global__ void pack(Problem p,Shape z,const float* A,const float* B,float* ap,float* bp) {
  __shared__ float tile[4][32][33];
  const int x=threadIdx.x,y=threadIdx.y;
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2),kh=int((p.k+1)/2);
  if(blockIdx.z==0) {
    const int tiles_k=z.kp/32;
    if constexpr(Grid2D) {if(blockIdx.x>=tiles_k || blockIdx.y>=z.mp/32)return;}
    else if(blockIdx.x>=(z.mp/32)*tiles_k)return;
    const int r0=Grid2D?int(blockIdx.y)*32:int(blockIdx.x)/tiles_k*32;
    const int q0=Grid2D?int(blockIdx.x)*32:int(blockIdx.x)%tiles_k*32;
    #pragma unroll
    for(int offset=0;offset<32;offset+=8) {
      const int r=r0+y+offset,q=q0+x;
      float a[4]={};
      if(r<mh && q<kh) {
        #pragma unroll
        for(int j=0;j<4;++j)a[j]=read(A,r+(j/2)*mh,q+(j%2)*kh,int(p.m),int(p.k),int(p.lda));
      }
      // Transpose the four original quadrants before forming seven operands.
      // This preserves every RN transform while reducing shared traffic 7 -> 4.
      #pragma unroll
      for(int j=0;j<4;++j)tile[j][y+offset][x]=a[j];
    }
    __syncthreads();
    #pragma unroll
    for(int offset=0;offset<32;offset+=8) {
      const int r=r0+x,q=q0+y+offset;
      float a[4];
      #pragma unroll
      for(int j=0;j<4;++j)a[j]=tile[j][x][y+offset];
      const float s1=__fadd_rn(a[2],a[3]),s2=__fsub_rn(s1,a[0]);
      const float av[7]={a[0],a[1],__fsub_rn(a[1],s2),a[3],s1,s2,__fsub_rn(a[0],a[2])};
      #pragma unroll
      for(int j=0;j<7;++j)ap[size_t(j)*z.mp*z.kp+size_t(q)*z.mp+r]=av[j];
    }
  } else {
    const int tiles_n=z.np/32;
    if constexpr(Grid2D) {if(blockIdx.x>=tiles_n || blockIdx.y>=z.kp/32)return;}
    else if(blockIdx.x>=(z.kp/32)*tiles_n)return;
    const int q0=Grid2D?int(blockIdx.y)*32:int(blockIdx.x)/tiles_n*32;
    const int c0=Grid2D?int(blockIdx.x)*32:int(blockIdx.x)%tiles_n*32;
    #pragma unroll
    for(int offset=0;offset<32;offset+=8) {
      const int q=q0+y+offset,c=c0+x;
      float b[4]={};
      if(q<kh && c<nh) {
        #pragma unroll
        for(int j=0;j<4;++j)b[j]=read(B,q+(j/2)*kh,c+(j%2)*nh,int(p.k),int(p.n),int(p.ldb));
      }
      const float t1=__fsub_rn(b[1],b[0]),t2=__fsub_rn(b[3],t1);
      const float bv[7]={b[0],b[2],b[3],__fsub_rn(t2,b[2]),t1,t2,__fsub_rn(b[3],b[1])};
      #pragma unroll
      for(int j=0;j<7;++j)bp[size_t(j)*z.kp*z.np+size_t(q)*z.np+c]=bv[j];
    }
  }
}

__device__ __forceinline__ float4 add4(float4 a,float4 b) {
  return {__fadd_rn(a.x,b.x),__fadd_rn(a.y,b.y),__fadd_rn(a.z,b.z),__fadd_rn(a.w,b.w)};
}
__device__ __forceinline__ float4 sub4(float4 a,float4 b) {
  return {__fsub_rn(a.x,b.x),__fsub_rn(a.y,b.y),__fsub_rn(a.z,b.z),__fsub_rn(a.w,b.w)};
}
__device__ __forceinline__ float4 read4(const float* x,int r,int c,int rows,int cols,int ld) {
  return r<rows && c+3<cols?*reinterpret_cast<const float4*>(x+int64_t(r)*ld+c):float4{};
}
// Four contiguous elements per thread reduce global load/store instruction count.
template<bool Grid2D=false> __global__ void pack_vector(Problem p,Shape z,const float* A,const float* B,float* ap,float* bp) {
  __shared__ float tile[4][32][33];
  const int x=threadIdx.x,y=threadIdx.y;
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2),kh=int((p.k+1)/2);
  if(blockIdx.z==0) {
    const int tiles_k=z.kp/32;
    if constexpr(Grid2D) {if(blockIdx.x>=tiles_k || blockIdx.y>=z.mp/32)return;}
    else if(blockIdx.x>=(z.mp/32)*tiles_k)return;
    const int r0=Grid2D?int(blockIdx.y)*32:int(blockIdx.x)/tiles_k*32;
    const int q0=Grid2D?int(blockIdx.x)*32:int(blockIdx.x)%tiles_k*32;
    #pragma unroll
    for(int offset=0;offset<32;offset+=32) {
      const int r=r0+y+offset,q=q0+4*x;float4 a[4]={};
      if(r<mh && q<kh) {
        #pragma unroll
        for(int j=0;j<4;++j)a[j]=read4(A,r+(j/2)*mh,q+(j%2)*kh,int(p.m),int(p.k),int(p.lda));
      }
      #pragma unroll
      for(int j=0;j<4;++j) {
        tile[j][y+offset][4*x]=a[j].x;tile[j][y+offset][4*x+1]=a[j].y;
        tile[j][y+offset][4*x+2]=a[j].z;tile[j][y+offset][4*x+3]=a[j].w;
      }
    }
    __syncthreads();
    #pragma unroll
    for(int offset=0;offset<32;offset+=32) {
      const int r=r0+4*x,q=q0+y+offset;
      float4 a[4];
      #pragma unroll
      for(int j=0;j<4;++j)a[j]={tile[j][4*x][y+offset],tile[j][4*x+1][y+offset],
                               tile[j][4*x+2][y+offset],tile[j][4*x+3][y+offset]};
      const float4 s1=add4(a[2],a[3]),s2=sub4(s1,a[0]);
      const float4 av[7]={a[0],a[1],sub4(a[1],s2),a[3],s1,s2,sub4(a[0],a[2])};
      #pragma unroll
      for(int j=0;j<7;++j)*reinterpret_cast<float4*>(ap+size_t(j)*z.mp*z.kp+size_t(q)*z.mp+r)=av[j];
    }
  } else {
    const int tiles_n=z.np/32;
    if constexpr(Grid2D) {if(blockIdx.x>=tiles_n || blockIdx.y>=z.kp/32)return;}
    else if(blockIdx.x>=(z.kp/32)*tiles_n)return;
    const int q0=Grid2D?int(blockIdx.y)*32:int(blockIdx.x)/tiles_n*32;
    const int c0=Grid2D?int(blockIdx.x)*32:int(blockIdx.x)%tiles_n*32;
    #pragma unroll
    for(int offset=0;offset<32;offset+=32) {
      const int q=q0+y+offset,c=c0+4*x;float4 b[4]={};
      if(q<kh && c<nh) {
        #pragma unroll
        for(int j=0;j<4;++j)b[j]=read4(B,q+(j/2)*kh,c+(j%2)*nh,int(p.k),int(p.n),int(p.ldb));
      }
      const float4 t1=sub4(b[1],b[0]),t2=sub4(b[3],t1);
      const float4 bv[7]={b[0],b[2],b[3],sub4(t2,b[2]),t1,t2,sub4(b[3],b[1])};
      #pragma unroll
      for(int j=0;j<7;++j)
        *reinterpret_cast<float4*>(bp+size_t(j)*z.kp*z.np+size_t(q)*z.np+c)=bv[j];
    }
  }
}
__device__ __forceinline__ void cooperative_pack_tile(Problem p,Shape z,const float* A,const float* B,
    float* ap,float* bp,int tile_index,int side,float (&tile)[4][32][33]) {
  constexpr bool Grid2D=false;
  const int x=threadIdx.x%8,y=threadIdx.x/8;
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2),kh=int((p.k+1)/2);
  if(side==0) {
    const int tiles_k=z.kp/32;
    if constexpr(Grid2D) {if(tile_index>=tiles_k || blockIdx.y>=z.mp/32)return;}
    else if(tile_index>=(z.mp/32)*tiles_k)return;
    const int r0=Grid2D?int(blockIdx.y)*32:int(tile_index)/tiles_k*32;
    const int q0=Grid2D?int(tile_index)*32:int(tile_index)%tiles_k*32;
    #pragma unroll
    for(int offset=0;offset<32;offset+=32) {
      const int r=r0+y+offset,q=q0+4*x;float4 a[4]={};
      if(r<mh && q<kh) {
        #pragma unroll
        for(int j=0;j<4;++j)a[j]=read4(A,r+(j/2)*mh,q+(j%2)*kh,int(p.m),int(p.k),int(p.lda));
      }
      #pragma unroll
      for(int j=0;j<4;++j) {
        tile[j][y+offset][4*x]=a[j].x;tile[j][y+offset][4*x+1]=a[j].y;
        tile[j][y+offset][4*x+2]=a[j].z;tile[j][y+offset][4*x+3]=a[j].w;
      }
    }
    __syncthreads();
    #pragma unroll
    for(int offset=0;offset<32;offset+=32) {
      const int r=r0+4*x,q=q0+y+offset;
      float4 a[4];
      #pragma unroll
      for(int j=0;j<4;++j)a[j]={tile[j][4*x][y+offset],tile[j][4*x+1][y+offset],
                               tile[j][4*x+2][y+offset],tile[j][4*x+3][y+offset]};
      const float4 s1=add4(a[2],a[3]),s2=sub4(s1,a[0]);
      const float4 av[7]={a[0],a[1],sub4(a[1],s2),a[3],s1,s2,sub4(a[0],a[2])};
      #pragma unroll
      for(int j=0;j<7;++j)*reinterpret_cast<float4*>(ap+size_t(j)*z.mp*z.kp+size_t(q)*z.mp+r)=av[j];
    }
  } else {
    const int tiles_n=z.np/32;
    if constexpr(Grid2D) {if(tile_index>=tiles_n || blockIdx.y>=z.kp/32)return;}
    else if(tile_index>=(z.kp/32)*tiles_n)return;
    const int q0=Grid2D?int(blockIdx.y)*32:int(tile_index)/tiles_n*32;
    const int c0=Grid2D?int(tile_index)*32:int(tile_index)%tiles_n*32;
    #pragma unroll
    for(int offset=0;offset<32;offset+=32) {
      const int q=q0+y+offset,c=c0+4*x;float4 b[4]={};
      if(q<kh && c<nh) {
        #pragma unroll
        for(int j=0;j<4;++j)b[j]=read4(B,q+(j/2)*kh,c+(j%2)*nh,int(p.k),int(p.n),int(p.ldb));
      }
      const float4 t1=sub4(b[1],b[0]),t2=sub4(b[3],t1);
      const float4 bv[7]={b[0],b[2],b[3],sub4(t2,b[2]),t1,t2,sub4(b[3],b[1])};
      #pragma unroll
      for(int j=0;j<7;++j)
        *reinterpret_cast<float4*>(bp+size_t(j)*z.kp*z.np+size_t(q)*z.np+c)=bv[j];
    }
  }
}

void launch_pack(const Problem& p,Shape z,const float* A,const float* B,float* ap,float* bp,
                  cudaStream_t stream,bool vectorized) {
  const int pack_a=(z.mp/32)*(z.kp/32),pack_b=(z.kp/32)*(z.np/32);
  const int gx=std::max(z.kp,z.np)/32,gy=std::max(z.mp,z.kp)/32;
  const bool grid2d=gx*gy==std::max(pack_a,pack_b);
  const dim3 grid=grid2d?dim3(gx,gy,2):dim3(std::max(pack_a,pack_b),1,2);
  if(vectorized && p.k%8==0 && p.n%8==0 && p.lda%4==0 && p.ldb%4==0 &&
     reinterpret_cast<uintptr_t>(A)%16==0 && reinterpret_cast<uintptr_t>(B)%16==0) {
    if(grid2d)pack_vector<true><<<grid,dim3(8,32),0,stream>>>(p,z,A,B,ap,bp);
    else pack_vector<false><<<grid,dim3(8,32),0,stream>>>(p,z,A,B,ap,bp);
  } else {
    if(grid2d)pack<true><<<grid,dim3(32,8),0,stream>>>(p,z,A,B,ap,bp);
    else pack<false><<<grid,dim3(32,8),0,stream>>>(p,z,A,B,ap,bp);
  }
}
template<int BM,int BN> struct alignas(128) Shared {
  alignas(128) float a[2][BK][BM];
  alignas(128) float b[2][BK][BN];
  uint64_t ready[2];
};
template<int BM,int BN> __device__ __forceinline__
void prefetch(Shared<BM,BN>& s,const CUtensorMap& amap,const CUtensorMap& bmap,
              int slot,int r0,int c0,int q,int product) {
  TxBarrier::arrive_and_expect_tx(&s.ready[slot],(BM+BN)*BK*sizeof(float));
  cute::SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],0,&s.a[slot][0][0],r0,q,product);
  cute::SM90_TMA_LOAD_3D::copy(&bmap,&s.ready[slot],0,&s.b[slot][0][0],c0,q,product);
}

template<int BM,int BN,int RM,int RN>
__global__ __launch_bounds__(BM*BN/(RM*RN))
void leaves(Shape z,const __grid_constant__ CUtensorMap amap,
             const __grid_constant__ CUtensorMap bmap,float* products,int tiles_n) {
  constexpr int WarpCols=BN/(8*RN)>0?BN/(8*RN):1;
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s=*reinterpret_cast<Shared<BM,BN>*>(storage);
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
  const int row=(warp/WarpCols)*(4*RM)+(lane/8)*RM;
  const int col=(warp%WarpCols)*(8*RN)+(lane%8)*RN;
  const int r0=blockIdx.x/tiles_n*BM,c0=blockIdx.x%tiles_n*BN,product=blockIdx.z;
  if(tid==0) {
    Barrier::init(&s.ready[0],1);Barrier::init(&s.ready[1],1);
    cutlass::arch::fence_barrier_init();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  float v[RM][RN]={};int phase0=0,phase1=0;
  if(tid==0)prefetch(s,amap,bmap,0,r0,c0,0,product);
  for(int q=0,seq=0;q<z.kp;q+=BK,++seq) {
    const int slot=seq%2;
    Barrier::wait(&s.ready[slot],slot?phase1:phase0);
    if(slot)phase1^=1;else phase0^=1;
    asm volatile("" ::: "memory");
    if(tid==0 && q+BK<z.kp)prefetch(s,amap,bmap,slot^1,r0,c0,q+BK,product);
    #pragma unroll 8
    for(int k=0;k<BK;++k) {
      float a[RM],b[RN];
      #pragma unroll
      for(int r=0;r<RM;r+=4) {
        const float4 x=*reinterpret_cast<const float4*>(&s.a[slot][k][row+r]);
        a[r]=x.x;a[r+1]=x.y;a[r+2]=x.z;a[r+3]=x.w;
      }
      #pragma unroll
      for(int c=0;c<RN;c+=4) {
        const float4 x=*reinterpret_cast<const float4*>(&s.b[slot][k][col+c]);
        b[c]=x.x;b[c+1]=x.y;b[c+2]=x.z;b[c+3]=x.w;
      }
      #pragma unroll
      for(int r=0;r<RM;++r) {
        #pragma unroll
        for(int c=0;c<RN;++c)v[r][c]=__fmaf_rn(a[r],b[c],v[r][c]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int r=0;r<RM;++r) {
    #pragma unroll
    for(int c=0;c<RN;++c)if(r0+row+r<z.mp && c0+col+c<z.np)
      products[size_t(product)*z.mp*z.np+size_t(r0+row+r)*z.np+c0+col+c]=v[r][c];
  }
}

__global__ void combine(Problem p,Shape z,const float* products,const float* C,float* D) {
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2);
  const size_t pos=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(pos>=size_t(mh)*nh)return;
  const int r=int(pos/nh),c=int(pos%nh);
  float v[7];
  #pragma unroll
  for(int j=0;j<7;++j)v[j]=products[size_t(j)*z.mp*z.np+size_t(r)*z.np+c];
  const float u2=__fadd_rn(v[0],v[5]),u3=__fadd_rn(u2,v[6]),u4=__fadd_rn(u2,v[4]);
  const float out[4]={__fadd_rn(v[0],v[1]),__fadd_rn(u4,v[2]),__fsub_rn(u3,v[3]),__fadd_rn(u3,v[4])};
  #pragma unroll
  for(int j=0;j<4;++j) {
    const int rr=r+(j/2)*mh,cc=c+(j%2)*nh;
    if(rr<p.m && cc<p.n) {
      float value=__fmul_rn(p.alpha,out[j]);
      if(p.beta!=0)value=__fadd_rn(value,__fmul_rn(p.beta,C[int64_t(rr)*p.ldc+cc]));
      D[int64_t(rr)*p.ldd+cc]=value;
    }
  }
}

template<int BM,int BN,int RM,int RN>
SWResources dispatch(const Problem* p,const float* A,const float* B,const float* C,
                     float* D,void* workspace,cudaStream_t stream,const Maps* prepared=nullptr) {
  constexpr size_t Bytes=sizeof(Shared<BM,BN>);constexpr int Threads=BM*BN/(RM*RN);
  if(!p) {
    check(cudaFuncSetAttribute(leaves<BM,BN,RM,RN>,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;
    check(cudaFuncGetAttributes(&attr,leaves<BM,BN,RM,RN>));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,leaves<BM,BN,RM,RN>,Threads,Bytes));
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  const auto z=shape(*p);
  float* ap=static_cast<float*>(workspace),*bp=ap+z.a,*products=bp+z.b;
  const auto& maps=*prepared;
  launch_pack(*p,z,A,B,ap,bp,stream,configurations.at(maps.id).vector_pack);
  const int tiles_n=(z.np+BN-1)/BN,tiles_m=(z.mp+BM-1)/BM;
  leaves<BM,BN,RM,RN><<<dim3(tiles_m*tiles_n,1,7),Threads,Bytes,stream>>>(z,maps.a,maps.b,products,tiles_n);
  const size_t outputs=size_t((p->m+1)/2)*((p->n+1)/2);
  combine<<<unsigned((outputs+255)/256),256,0,stream>>>(*p,z,products,C,D);
  check(cudaPeekAtLastError());return {};
}
// The same seven packed FP32 operand pairs also admit a CUTLASS SIMT leaf.
// This candidate tests its shared/register software pipeline; no Tensor Ops or
// extra products are instantiated and the complete pack/leaf/combine is timed.
template<int BM,int BN,int WM,int WN>
SWResources cutlass_dispatch(const Problem* p,const float* A,const float* B,const float* C,
                            float* D,void* workspace,cudaStream_t stream) {
  using Gemm=cutlass::gemm::device::GemmBatched<float,cutlass::layout::ColumnMajor,
    float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,float,
    cutlass::arch::OpClassSimt,cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<BM,BN,8>,cutlass::gemm::GemmShape<WM,WN,8>,
    cutlass::gemm::GemmShape<1,1,1>,
    cutlass::epilogue::thread::LinearCombination<float,1,float,float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,2>;
  using Kernel=typename Gemm::GemmKernel;
  constexpr int Bytes=sizeof(typename Kernel::SharedStorage);
  if(!p) {
    if(Bytes>=48*1024)check(cudaFuncSetAttribute(cutlass::Kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;
    check(cudaFuncGetAttributes(&attr,cutlass::Kernel<Kernel>));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,cutlass::Kernel<Kernel>,Kernel::kThreadCount,Bytes));
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  const auto z=shape(*p);
  float* ap=static_cast<float*>(workspace),*bp=ap+z.a,*products=bp+z.b;
  const int pack_a=(z.mp/32)*(z.kp/32),pack_b=(z.kp/32)*(z.np/32);
  pack<false><<<dim3(std::max(pack_a,pack_b),1,2),dim3(32,8),0,stream>>>(*p,z,A,B,ap,bp);
  Gemm op;
  typename Gemm::Arguments args({z.mp,z.np,z.kp},{ap,z.mp},int64_t(z.mp)*z.kp,
    {bp,z.np},int64_t(z.kp)*z.np,{products,z.np},int64_t(z.mp)*z.np,
    {products,z.np},int64_t(z.mp)*z.np,{1.0f,0.0f},7);
  if(op.can_implement(args)!=cutlass::Status::kSuccess || op(args,nullptr,stream)!=cutlass::Status::kSuccess)
    throw std::runtime_error("FP32 seven-batch CUTLASS SIMT launch failed");
  const size_t outputs=size_t((p->m+1)/2)*((p->n+1)/2);
  combine<<<unsigned((outputs+255)/256),256,0,stream>>>(*p,z,products,C,D);
  check(cudaPeekAtLastError());return {};
}
template<int BM,int BN,int K> struct alignas(128) FusedShared {
  union {
    struct {
      alignas(128) float a[2][7][K][BM];
      alignas(128) float b[2][7][K][BN];
    } stage;
    float products[7][BM][BN];
  };
  alignas(16) uint64_t ready[2];
  uint64_t empty[2];
};
template<int BM,int BN,int K> __device__ __forceinline__
void fused_prefetch(FusedShared<BM,BN,K>& s,const CUtensorMap& amap,
                    const CUtensorMap& bmap,int slot,int r0,int c0,int q) {
  TxBarrier::arrive_and_expect_tx(&s.ready[slot],7*(BM+BN)*K*4);
  // The third box dimension spans all seven disjoint packed product planes.
  cute::SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],0,s.stage.a[slot],r0,q,0);
  cute::SM90_TMA_LOAD_3D::copy(&bmap,&s.ready[slot],0,s.stage.b[slot],c0,q,0);
}
template<int BM,int BN,int RM,int RN,int K> __device__ __forceinline__
void fused_register_load(FusedShared<BM,BN,K>& s,int slot,int product,int k,int row,int col,
                          float (&a)[RM],float (&b)[RN]) {
  if constexpr(RM==2) {
    const float2 x=*reinterpret_cast<const float2*>(&s.stage.a[slot][product][k][row]);a[0]=x.x;a[1]=x.y;
  } else {
    #pragma unroll
    for(int r=0;r<RM;r+=4) {
      const float4 x=*reinterpret_cast<const float4*>(&s.stage.a[slot][product][k][row+r]);
      a[r]=x.x;a[r+1]=x.y;a[r+2]=x.z;a[r+3]=x.w;
    }
  }
  if constexpr(RN%4==0) {
    #pragma unroll
    for(int c=0;c<RN;c+=4) {
      const float4 x=*reinterpret_cast<const float4*>(&s.stage.b[slot][product][k][col+c]);
      b[c]=x.x;b[c+1]=x.y;b[c+2]=x.z;b[c+3]=x.w;
    }
  } else {
    static_assert(RN%2==0);
    #pragma unroll
    for(int c=0;c<RN;c+=2) {
      const float2 x=*reinterpret_cast<const float2*>(&s.stage.b[slot][product][k][col+c]);b[c]=x.x;b[c+1]=x.y;
    }
  }
}
template<int CoreBN,bool ExtraM,bool FoldN,int APitch,int BPitch,int K,int RM,int RN>
__device__ __forceinline__ void fused_accumulate(FusedShared<APitch,BPitch,K>& s,int slot,int product,
    int k,int lane,bool extra_active,int extra_row,bool fold_last_n,float (&v)[RM][RN],
    float (&extra)[RN],float (&extra_n)[RM],float& corner,float (&a)[RM],float (&b)[RN]) {
  constexpr int LaneCols=CoreBN/RN;
  float ax=0.0f,bx=0.0f;
  if(ExtraM && extra_active)ax=s.stage.a[slot][product][k][extra_row];
  if(FoldN && fold_last_n && lane%LaneCols==0)bx=s.stage.b[slot][product][k][CoreBN];
  #pragma unroll
  for(int r=0;r<RM;++r) {
    #pragma unroll
    for(int c=0;c<RN;++c)v[r][c]=__fmaf_rn(a[r],b[c],v[r][c]);
  }
  if(ExtraM && extra_active) {
    #pragma unroll
    for(int c=0;c<RN;++c)extra[c]=__fmaf_rn(ax,b[c],extra[c]);
  }
  if(FoldN && fold_last_n && lane%LaneCols==0) {
    #pragma unroll
    for(int r=0;r<RM;++r)extra_n[r]=__fmaf_rn(a[r],bx,extra_n[r]);
    if(ExtraM && extra_active)corner=__fmaf_rn(ax,bx,corner);
  }
}
// Spread the seven product-row planes evenly over all eight compute warps.
// Each lane owns seven rows, so both input loads and FMA work are balanced.
// A warp can straddle at most two products; keep both B fragments in registers.
template<int BM,int BN,int K> __device__ __forceinline__
void balanced_load(FusedShared<BM,BN,K>& s,int slot,int k,int warp,int lane,
    float (&a)[7],float (&b)[2][8]) {
  constexpr int LC=BN/8,RG=32/LC;
  const int first=warp*7*RG/BM,last=((warp*7+6)*RG+RG-1)/BM,col=lane%LC*8;
  #pragma unroll
  for(int r=0;r<7;++r) {
    const int linear=(warp*7+r)*RG+lane/LC;
    a[r]=s.stage.a[slot][linear/BM][k][linear%BM];
  }
  #pragma unroll
  for(int j=0;j<2;++j) {
    #pragma unroll
    for(int c=0;c<8;c+=4) {
      const float4 x=*reinterpret_cast<const float4*>(&s.stage.b[slot][j?last:first][k][col+c]);
      b[j][c]=x.x;b[j][c+1]=x.y;b[j][c+2]=x.z;b[j][c+3]=x.w;
    }
  }
}
template<int BM,int BN,int K> __global__ __launch_bounds__(256,1)
void balanced_leaves(Problem p,Shape z,const __grid_constant__ CUtensorMap amap,
    const __grid_constant__ CUtensorMap bmap,const float* C,float* D,int tiles_n) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s=*reinterpret_cast<FusedShared<BM,BN,K>*>(storage);
  constexpr int LC=BN/8,RG=32/LC;
  static_assert(8*7*RG==7*BM);
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
  const int r0=blockIdx.x/tiles_n*BM,c0=blockIdx.x%tiles_n*BN,col=lane%LC*8;
  const int first=warp*7*RG/BM;
  if(tid==0) {
    Barrier::init(&s.ready[0],1);Barrier::init(&s.ready[1],1);
    cutlass::arch::fence_barrier_init();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  const int active_k=int((p.k+1)/2);float v[7][8]={};int phase0=0,phase1=0;
  if(tid==0)fused_prefetch(s,amap,bmap,0,r0,c0,0);
  for(int q=0,seq=0;q<active_k;q+=K,++seq) {
    const int slot=seq%2;
    Barrier::wait(&s.ready[slot],slot?phase1:phase0);
    if(slot)phase1^=1;else phase0^=1;
    asm volatile("" ::: "memory");
    if(tid==0 && q+K<active_k)fused_prefetch(s,amap,bmap,slot^1,r0,c0,q+K);
    #pragma unroll 2
    for(int k=0;k<K;++k) {
      float a[7],b[2][8];balanced_load(s,slot,k,warp,lane,a,b);
      if(q+k<active_k) {
        #pragma unroll
        for(int r=0;r<7;++r) {
          const int linear=(warp*7+r)*RG+lane/LC;
          #pragma unroll
          for(int c=0;c<8;++c)
            v[r][c]=__fmaf_rn(a[r],(linear/BM==first?b[0][c]:b[1][c]),v[r][c]);
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int r=0;r<7;++r) {
    const int linear=(warp*7+r)*RG+lane/LC;
    #pragma unroll
    for(int c=0;c<8;++c) {
      if(active_k%16)v[r][c]=__fmaf_rn(0.0f,0.0f,v[r][c]);
      s.products[linear/BM][linear%BM][col+c]=v[r][c];
    }
  }
  __syncthreads();
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2);
  for(int pos=tid;pos<BM*BN;pos+=256) {
    const int r=pos/BN,c=pos%BN;
    if(r0+r<mh && c0+c<nh) {
      float values[7];
      #pragma unroll
      for(int j=0;j<7;++j)values[j]=s.products[j][r][c];
      const float u2=__fadd_rn(values[0],values[5]),u3=__fadd_rn(u2,values[6]),u4=__fadd_rn(u2,values[4]);
      const float out[4]={__fadd_rn(values[0],values[1]),__fadd_rn(u4,values[2]),__fsub_rn(u3,values[3]),__fadd_rn(u3,values[4])};
      #pragma unroll
      for(int j=0;j<4;++j) {
        const int rr=r0+r+(j/2)*mh,cc=c0+c+(j%2)*nh;
        if(rr<p.m && cc<p.n) {
          float value=__fmul_rn(p.alpha,out[j]);
          if(p.beta!=0)value=__fadd_rn(value,__fmul_rn(p.beta,C[int64_t(rr)*p.ldc+cc]));
          D[int64_t(rr)*p.ldd+cc]=value;
        }
      }
    }
  }
}
template<int BM,int BN,int K>
SWResources balanced_dispatch(const Problem* p,const float* A,const float* B,const float* C,
    float* D,void* workspace,cudaStream_t stream,const Maps* prepared) {
  constexpr int Bytes=sizeof(FusedShared<BM,BN,K>);const auto kernel=balanced_leaves<BM,BN,K>;
  if(!p) {
    check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;check(cudaFuncGetAttributes(&attr,kernel));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,256,Bytes));
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  const auto z=shape(*p);float* ap=static_cast<float*>(workspace),*bp=ap+z.a;
  launch_pack(*p,z,A,B,ap,bp,stream,true);
  const int tiles_n=(int((p->n+1)/2)+BN-1)/BN,tiles_m=(int((p->m+1)/2)+BM-1)/BM;
  kernel<<<tiles_m*tiles_n,256,Bytes,stream>>>(*p,z,prepared->a,prepared->b,C,D,tiles_n);
  check(cudaPeekAtLastError());return {};
}
// Per-warp branches contain no CTA barriers. Every warp rejoins before the
// common stage-reuse barrier; constant product indices remove runtime FSELs.
template<int Warp,int BM,int BN,int K,int Unroll> __device__ __forceinline__
void balanced_static_stage(FusedShared<BM,BN,K>& s,int slot,int q,int active_k,
                          int lane,float (&v)[7][8]) {
  constexpr int LC=BN/8,RG=32/LC;
  constexpr int First=Warp*7*RG/BM,Last=((Warp*7+6)*RG+RG-1)/BM;
  static_assert(First>=0 && Last<7 && Last-First<=1);
  const int row_group=lane/LC,col=lane%LC*8;
  #pragma unroll Unroll
  for(int k=0;k<K;++k) {
    float a[7],b[2][8];
    #pragma unroll
    for(int r=0;r<7;++r) {
      const int product=(Warp*7+r)*RG/BM,row=(Warp*7+r)*RG%BM+row_group;
      a[r]=s.stage.a[slot][product][k][row];
    }
    #pragma unroll
    for(int c=0;c<8;c+=4) {
      const float4 x=*reinterpret_cast<const float4*>(&s.stage.b[slot][First][k][col+c]);
      b[0][c]=x.x;b[0][c+1]=x.y;b[0][c+2]=x.z;b[0][c+3]=x.w;
      if constexpr(First!=Last) {
        const float4 y=*reinterpret_cast<const float4*>(&s.stage.b[slot][Last][k][col+c]);
        b[1][c]=y.x;b[1][c+1]=y.y;b[1][c+2]=y.z;b[1][c+3]=y.w;
      }
    }
    if(q+k<active_k) {
      #pragma unroll
      for(int r=0;r<7;++r) {
        const int product=(Warp*7+r)*RG/BM;
        #pragma unroll
        for(int c=0;c<8;++c)v[r][c]=__fmaf_rn(a[r],b[product==First?0:1][c],v[r][c]);
      }
    }
  }
}
template<int Warp,int BM,int BN,int K> __device__ __forceinline__
void balanced_static_store(FusedShared<BM,BN,K>& s,int lane,int active_k,float (&v)[7][8]) {
  constexpr int LC=BN/8,RG=32/LC;
  #pragma unroll
  for(int r=0;r<7;++r) {
    const int product=(Warp*7+r)*RG/BM,row=(Warp*7+r)*RG%BM+lane/LC;
    #pragma unroll
    for(int c=0;c<8;++c) {
      if(active_k%16)v[r][c]=__fmaf_rn(0.0f,0.0f,v[r][c]);
      s.products[product][row][lane%LC*8+c]=v[r][c];
    }
  }
}
template<int BM,int BN,int K,int Unroll> __global__ __launch_bounds__(256,1)
void balanced_static_leaves(Problem p,Shape z,const __grid_constant__ CUtensorMap amap,
    const __grid_constant__ CUtensorMap bmap,const float* C,float* D,int tiles_n) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s=*reinterpret_cast<FusedShared<BM,BN,K>*>(storage);
  constexpr int LC=BN/8,RG=32/LC;
  static_assert(8*7*RG==7*BM);
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
  const int r0=blockIdx.x/tiles_n*BM,c0=blockIdx.x%tiles_n*BN;
  if(tid==0) {
    Barrier::init(&s.ready[0],1);Barrier::init(&s.ready[1],1);
    cutlass::arch::fence_barrier_init();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  const int active_k=int((p.k+1)/2);float v[7][8]={};int phase0=0,phase1=0;
  if(tid==0)fused_prefetch(s,amap,bmap,0,r0,c0,0);
  for(int q=0,seq=0;q<active_k;q+=K,++seq) {
    const int slot=seq%2;
    Barrier::wait(&s.ready[slot],slot?phase1:phase0);
    if(slot)phase1^=1;else phase0^=1;
    asm volatile("" ::: "memory");
    if(tid==0 && q+K<active_k)fused_prefetch(s,amap,bmap,slot^1,r0,c0,q+K);
    switch(warp) {
      case 0: balanced_static_stage<0,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 1: balanced_static_stage<1,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 2: balanced_static_stage<2,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 3: balanced_static_stage<3,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 4: balanced_static_stage<4,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 5: balanced_static_stage<5,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 6: balanced_static_stage<6,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 7: balanced_static_stage<7,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
    }
    __syncthreads();
  }
  switch(warp) {
    case 0: balanced_static_store<0,BM,BN,K>(s,lane,active_k,v);break;
    case 1: balanced_static_store<1,BM,BN,K>(s,lane,active_k,v);break;
    case 2: balanced_static_store<2,BM,BN,K>(s,lane,active_k,v);break;
    case 3: balanced_static_store<3,BM,BN,K>(s,lane,active_k,v);break;
    case 4: balanced_static_store<4,BM,BN,K>(s,lane,active_k,v);break;
    case 5: balanced_static_store<5,BM,BN,K>(s,lane,active_k,v);break;
    case 6: balanced_static_store<6,BM,BN,K>(s,lane,active_k,v);break;
    case 7: balanced_static_store<7,BM,BN,K>(s,lane,active_k,v);break;
  }
  __syncthreads();
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2);
  for(int pos=tid;pos<BM*BN;pos+=256) {
    const int r=pos/BN,c=pos%BN;
    if(r0+r<mh && c0+c<nh) {
      float values[7];
      #pragma unroll
      for(int j=0;j<7;++j)values[j]=s.products[j][r][c];
      const float u2=__fadd_rn(values[0],values[5]),u3=__fadd_rn(u2,values[6]),u4=__fadd_rn(u2,values[4]);
      const float out[4]={__fadd_rn(values[0],values[1]),__fadd_rn(u4,values[2]),__fsub_rn(u3,values[3]),__fadd_rn(u3,values[4])};
      #pragma unroll
      for(int j=0;j<4;++j) {
        const int rr=r0+r+(j/2)*mh,cc=c0+c+(j%2)*nh;
        if(rr<p.m && cc<p.n) {
          float value=__fmul_rn(p.alpha,out[j]);
          if(p.beta!=0)value=__fadd_rn(value,__fmul_rn(p.beta,C[int64_t(rr)*p.ldc+cc]));
          D[int64_t(rr)*p.ldd+cc]=value;
        }
      }
    }
  }
}
template<int BM,int BN,int K,int Unroll>
SWResources balanced_static_dispatch(const Problem* p,const float* A,const float* B,const float* C,
    float* D,void* workspace,cudaStream_t stream,const Maps* prepared) {
  constexpr int Bytes=sizeof(FusedShared<BM,BN,K>);const auto kernel=balanced_static_leaves<BM,BN,K,Unroll>;
  if(!p) {
    check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;check(cudaFuncGetAttributes(&attr,kernel));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,256,Bytes));
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  const auto z=shape(*p);float* ap=static_cast<float*>(workspace),*bp=ap+z.a;
  launch_pack(*p,z,A,B,ap,bp,stream,true);
  const int tiles_n=(int((p->n+1)/2)+BN-1)/BN,tiles_m=(int((p->m+1)/2)+BM-1)/BM;
  kernel<<<tiles_m*tiles_n,256,Bytes,stream>>>(*p,z,prepared->a,prepared->b,C,D,tiles_n);
  check(cudaPeekAtLastError());return {};
}
// Narrow thread tiles expose more output CTAs and reduce accumulator storage.
// Original seven-product ownership, barriers, and reconstruction are preserved.
template<int Warp,int BM,int BN,int K,int Unroll> __device__ __forceinline__
void balanced_narrow_stage(FusedShared<BM,BN,K>& s,int slot,int q,int active_k,
                          int lane,float (&v)[7][4]) {
  constexpr int LC=BN/4,RG=32/LC;
  constexpr int First=Warp*7*RG/BM,Last=((Warp*7+6)*RG+RG-1)/BM;
  static_assert(First>=0 && Last<7 && Last-First<=1);
  const int row_group=lane/LC,col=lane%LC*4;
  #pragma unroll Unroll
  for(int k=0;k<K;++k) {
    float a[7],b[2][4];
    #pragma unroll
    for(int r=0;r<7;++r) {
      const int product=(Warp*7+r)*RG/BM,row=(Warp*7+r)*RG%BM+row_group;
      a[r]=s.stage.a[slot][product][k][row];
    }
    #pragma unroll
    for(int c=0;c<4;c+=4) {
      const float4 x=*reinterpret_cast<const float4*>(&s.stage.b[slot][First][k][col+c]);
      b[0][c]=x.x;b[0][c+1]=x.y;b[0][c+2]=x.z;b[0][c+3]=x.w;
      if constexpr(First!=Last) {
        const float4 y=*reinterpret_cast<const float4*>(&s.stage.b[slot][Last][k][col+c]);
        b[1][c]=y.x;b[1][c+1]=y.y;b[1][c+2]=y.z;b[1][c+3]=y.w;
      }
    }
    if(q+k<active_k) {
      #pragma unroll
      for(int r=0;r<7;++r) {
        const int product=(Warp*7+r)*RG/BM;
        #pragma unroll
        for(int c=0;c<4;++c)v[r][c]=__fmaf_rn(a[r],b[product==First?0:1][c],v[r][c]);
      }
    }
  }
}
template<int Warp,int BM,int BN,int K> __device__ __forceinline__
void balanced_narrow_store(FusedShared<BM,BN,K>& s,int lane,int active_k,float (&v)[7][4]) {
  constexpr int LC=BN/4,RG=32/LC;
  #pragma unroll
  for(int r=0;r<7;++r) {
    const int product=(Warp*7+r)*RG/BM,row=(Warp*7+r)*RG%BM+lane/LC;
    #pragma unroll
    for(int c=0;c<4;++c) {
      if(active_k%16)v[r][c]=__fmaf_rn(0.0f,0.0f,v[r][c]);
      s.products[product][row][lane%LC*4+c]=v[r][c];
    }
  }
}
template<int BM,int BN,int K,int Unroll> __global__ __launch_bounds__(256,2)
void balanced_narrow_leaves(Problem p,Shape z,const __grid_constant__ CUtensorMap amap,
    const __grid_constant__ CUtensorMap bmap,const float* C,float* D,int tiles_n) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s=*reinterpret_cast<FusedShared<BM,BN,K>*>(storage);
  constexpr int LC=BN/4,RG=32/LC;
  static_assert(8*7*RG==7*BM);
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
  const int r0=blockIdx.x/tiles_n*BM,c0=blockIdx.x%tiles_n*BN;
  if(tid==0) {
    Barrier::init(&s.ready[0],1);Barrier::init(&s.ready[1],1);
    cutlass::arch::fence_barrier_init();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  const int active_k=int((p.k+1)/2);float v[7][4]={};int phase0=0,phase1=0;
  if(tid==0)fused_prefetch(s,amap,bmap,0,r0,c0,0);
  for(int q=0,seq=0;q<active_k;q+=K,++seq) {
    const int slot=seq%2;
    Barrier::wait(&s.ready[slot],slot?phase1:phase0);
    if(slot)phase1^=1;else phase0^=1;
    asm volatile("" ::: "memory");
    if(tid==0 && q+K<active_k)fused_prefetch(s,amap,bmap,slot^1,r0,c0,q+K);
    switch(warp) {
      case 0: balanced_narrow_stage<0,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 1: balanced_narrow_stage<1,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 2: balanced_narrow_stage<2,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 3: balanced_narrow_stage<3,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 4: balanced_narrow_stage<4,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 5: balanced_narrow_stage<5,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 6: balanced_narrow_stage<6,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 7: balanced_narrow_stage<7,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
    }
    __syncthreads();
  }
  switch(warp) {
    case 0: balanced_narrow_store<0,BM,BN,K>(s,lane,active_k,v);break;
    case 1: balanced_narrow_store<1,BM,BN,K>(s,lane,active_k,v);break;
    case 2: balanced_narrow_store<2,BM,BN,K>(s,lane,active_k,v);break;
    case 3: balanced_narrow_store<3,BM,BN,K>(s,lane,active_k,v);break;
    case 4: balanced_narrow_store<4,BM,BN,K>(s,lane,active_k,v);break;
    case 5: balanced_narrow_store<5,BM,BN,K>(s,lane,active_k,v);break;
    case 6: balanced_narrow_store<6,BM,BN,K>(s,lane,active_k,v);break;
    case 7: balanced_narrow_store<7,BM,BN,K>(s,lane,active_k,v);break;
  }
  __syncthreads();
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2);
  for(int pos=tid;pos<BM*BN;pos+=256) {
    const int r=pos/BN,c=pos%BN;
    if(r0+r<mh && c0+c<nh) {
      float values[7];
      #pragma unroll
      for(int j=0;j<7;++j)values[j]=s.products[j][r][c];
      const float u2=__fadd_rn(values[0],values[5]),u3=__fadd_rn(u2,values[6]),u4=__fadd_rn(u2,values[4]);
      const float out[4]={__fadd_rn(values[0],values[1]),__fadd_rn(u4,values[2]),__fsub_rn(u3,values[3]),__fadd_rn(u3,values[4])};
      #pragma unroll
      for(int j=0;j<4;++j) {
        const int rr=r0+r+(j/2)*mh,cc=c0+c+(j%2)*nh;
        if(rr<p.m && cc<p.n) {
          float value=__fmul_rn(p.alpha,out[j]);
          if(p.beta!=0)value=__fadd_rn(value,__fmul_rn(p.beta,C[int64_t(rr)*p.ldc+cc]));
          D[int64_t(rr)*p.ldd+cc]=value;
        }
      }
    }
  }
}
template<int BM,int BN,int K,int Unroll>
SWResources balanced_narrow_dispatch(const Problem* p,const float* A,const float* B,const float* C,
    float* D,void* workspace,cudaStream_t stream,const Maps* prepared) {
  constexpr int Bytes=sizeof(FusedShared<BM,BN,K>);const auto kernel=balanced_narrow_leaves<BM,BN,K,Unroll>;
  if(!p) {
    check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;check(cudaFuncGetAttributes(&attr,kernel));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,256,Bytes));
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  const auto z=shape(*p);float* ap=static_cast<float*>(workspace),*bp=ap+z.a;
  launch_pack(*p,z,A,B,ap,bp,stream,true);
  const int tiles_n=(int((p->n+1)/2)+BN-1)/BN,tiles_m=(int((p->m+1)/2)+BM-1)/BM;
  kernel<<<tiles_m*tiles_n,256,Bytes,stream>>>(*p,z,prepared->a,prepared->b,C,D,tiles_n);
  check(cudaPeekAtLastError());return {};
}
// Alternative A-register distribution; original static-warp entries are
// retained verbatim so their compiled instructions can be audited separately.
template<int Warp,int BM,int BN,int K,int Unroll> __device__ __forceinline__
void balanced_shuffle_stage(FusedShared<BM,BN,K>& s,int slot,int q,int active_k,
                          int lane,float (&v)[7][8]) {
  constexpr int LC=BN/8,RG=32/LC;
  constexpr int First=Warp*7*RG/BM,Last=((Warp*7+6)*RG+RG-1)/BM;
  static_assert(First>=0 && Last<7 && Last-First<=1);
  const int row_group=lane/LC,col=lane%LC*8;
  #pragma unroll Unroll
  for(int k=0;k<K;++k) {
    float a[7],b[2][8];
    // Each warp loads only its distinct A rows. All lanes execute all seven
    // shuffles, including lanes whose unused tail loads are suppressed.
    constexpr int Rows=7*RG;
    const int first=Warp*Rows+lane,second=first+32;
    float cached0=0.0f,cached1=0.0f;
    if(lane<Rows)cached0=s.stage.a[slot][first/BM][k][first%BM];
    if constexpr(Rows>32) {
      if(lane<Rows-32)cached1=s.stage.a[slot][second/BM][k][second%BM];
    }
    #pragma unroll
    for(int r=0;r<7;++r) {
      const int source=(r*RG)%32+row_group;
      a[r]=__shfl_sync(0xffffffffu,r*RG<32?cached0:cached1,source);
    }
    #pragma unroll
    for(int c=0;c<8;c+=4) {
      const float4 x=*reinterpret_cast<const float4*>(&s.stage.b[slot][First][k][col+c]);
      b[0][c]=x.x;b[0][c+1]=x.y;b[0][c+2]=x.z;b[0][c+3]=x.w;
      if constexpr(First!=Last) {
        const float4 y=*reinterpret_cast<const float4*>(&s.stage.b[slot][Last][k][col+c]);
        b[1][c]=y.x;b[1][c+1]=y.y;b[1][c+2]=y.z;b[1][c+3]=y.w;
      }
    }
    if(q+k<active_k) {
      #pragma unroll
      for(int r=0;r<7;++r) {
        const int product=(Warp*7+r)*RG/BM;
        #pragma unroll
        for(int c=0;c<8;++c)v[r][c]=__fmaf_rn(a[r],b[product==First?0:1][c],v[r][c]);
      }
    }
  }
}
template<int BM,int BN,int K,int Unroll> __global__ __launch_bounds__(256,1)
void balanced_shuffle_leaves(Problem p,Shape z,const __grid_constant__ CUtensorMap amap,
    const __grid_constant__ CUtensorMap bmap,const float* C,float* D,int tiles_n) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s=*reinterpret_cast<FusedShared<BM,BN,K>*>(storage);
  constexpr int LC=BN/8,RG=32/LC;
  static_assert(8*7*RG==7*BM);
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
  const int r0=blockIdx.x/tiles_n*BM,c0=blockIdx.x%tiles_n*BN;
  if(tid==0) {
    Barrier::init(&s.ready[0],1);Barrier::init(&s.ready[1],1);
    cutlass::arch::fence_barrier_init();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  const int active_k=int((p.k+1)/2);float v[7][8]={};int phase0=0,phase1=0;
  if(tid==0)fused_prefetch(s,amap,bmap,0,r0,c0,0);
  for(int q=0,seq=0;q<active_k;q+=K,++seq) {
    const int slot=seq%2;
    Barrier::wait(&s.ready[slot],slot?phase1:phase0);
    if(slot)phase1^=1;else phase0^=1;
    asm volatile("" ::: "memory");
    if(tid==0 && q+K<active_k)fused_prefetch(s,amap,bmap,slot^1,r0,c0,q+K);
    switch(warp) {
      case 0: balanced_shuffle_stage<0,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 1: balanced_shuffle_stage<1,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 2: balanced_shuffle_stage<2,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 3: balanced_shuffle_stage<3,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 4: balanced_shuffle_stage<4,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 5: balanced_shuffle_stage<5,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 6: balanced_shuffle_stage<6,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
      case 7: balanced_shuffle_stage<7,BM,BN,K,Unroll>(s,slot,q,active_k,lane,v);break;
    }
    __syncthreads();
  }
  switch(warp) {
    case 0: balanced_static_store<0,BM,BN,K>(s,lane,active_k,v);break;
    case 1: balanced_static_store<1,BM,BN,K>(s,lane,active_k,v);break;
    case 2: balanced_static_store<2,BM,BN,K>(s,lane,active_k,v);break;
    case 3: balanced_static_store<3,BM,BN,K>(s,lane,active_k,v);break;
    case 4: balanced_static_store<4,BM,BN,K>(s,lane,active_k,v);break;
    case 5: balanced_static_store<5,BM,BN,K>(s,lane,active_k,v);break;
    case 6: balanced_static_store<6,BM,BN,K>(s,lane,active_k,v);break;
    case 7: balanced_static_store<7,BM,BN,K>(s,lane,active_k,v);break;
  }
  __syncthreads();
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2);
  for(int pos=tid;pos<BM*BN;pos+=256) {
    const int r=pos/BN,c=pos%BN;
    if(r0+r<mh && c0+c<nh) {
      float values[7];
      #pragma unroll
      for(int j=0;j<7;++j)values[j]=s.products[j][r][c];
      const float u2=__fadd_rn(values[0],values[5]),u3=__fadd_rn(u2,values[6]),u4=__fadd_rn(u2,values[4]);
      const float out[4]={__fadd_rn(values[0],values[1]),__fadd_rn(u4,values[2]),__fsub_rn(u3,values[3]),__fadd_rn(u3,values[4])};
      #pragma unroll
      for(int j=0;j<4;++j) {
        const int rr=r0+r+(j/2)*mh,cc=c0+c+(j%2)*nh;
        if(rr<p.m && cc<p.n) {
          float value=__fmul_rn(p.alpha,out[j]);
          if(p.beta!=0)value=__fadd_rn(value,__fmul_rn(p.beta,C[int64_t(rr)*p.ldc+cc]));
          D[int64_t(rr)*p.ldd+cc]=value;
        }
      }
    }
  }
}
template<int BM,int BN,int K,int Unroll>
SWResources balanced_shuffle_dispatch(const Problem* p,const float* A,const float* B,const float* C,
    float* D,void* workspace,cudaStream_t stream,const Maps* prepared) {
  constexpr int Bytes=sizeof(FusedShared<BM,BN,K>);const auto kernel=balanced_shuffle_leaves<BM,BN,K,Unroll>;
  if(!p) {
    check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;check(cudaFuncGetAttributes(&attr,kernel));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,256,Bytes));
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  const auto z=shape(*p);float* ap=static_cast<float*>(workspace),*bp=ap+z.a;
  launch_pack(*p,z,A,B,ap,bp,stream,true);
  const int tiles_n=(int((p->n+1)/2)+BN-1)/BN,tiles_m=(int((p->m+1)/2)+BM-1)/BM;
  kernel<<<tiles_m*tiles_n,256,Bytes,stream>>>(*p,z,prepared->a,prepared->b,C,D,tiles_n);
  check(cudaPeekAtLastError());return {};
}
// One raw shared tile. Preserve its last Tail K operands in registers before
// letting TMA overwrite it; all arithmetic still visits real K in order.
template<int BM,int BN,int K> struct alignas(128) SingleShared {
  union {
    struct {alignas(128) float a[7][K][BM];alignas(128) float b[7][K][BN];} raw;
    float products[7][BM][BN];
  };
  alignas(16) uint64_t ready;
};
template<int BM,int BN,int K> __device__ __forceinline__
void single_prefetch(SingleShared<BM,BN,K>& s,const CUtensorMap& amap,
                     const CUtensorMap& bmap,int r0,int c0,int q) {
  TxBarrier::arrive_and_expect_tx(&s.ready,7*(BM+BN)*K*4);
  cute::SM90_TMA_LOAD_3D::copy(&amap,&s.ready,0,s.raw.a,r0,q,0);
  cute::SM90_TMA_LOAD_3D::copy(&bmap,&s.ready,0,s.raw.b,c0,q,0);
}
template<int BM,int BN,int K> __device__ __forceinline__
void single_load(SingleShared<BM,BN,K>& s,int product,int k,int row,int col,
                 float (&a)[8],float (&b)[8]) {
  #pragma unroll
  for(int i=0;i<8;i+=4) {
    const float4 x=*reinterpret_cast<const float4*>(&s.raw.a[product][k][row+i]);
    const float4 y=*reinterpret_cast<const float4*>(&s.raw.b[product][k][col+i]);
    a[i]=x.x;a[i+1]=x.y;a[i+2]=x.z;a[i+3]=x.w;
    b[i]=y.x;b[i+1]=y.y;b[i+2]=y.z;b[i+3]=y.w;
  }
}
__device__ __forceinline__ void single_multiply(const float (&a)[8],const float (&b)[8],float (&v)[8][8]) {
  #pragma unroll
  for(int r=0;r<8;++r) {
    #pragma unroll
    for(int c=0;c<8;++c)v[r][c]=__fmaf_rn(a[r],b[c],v[r][c]);
  }
}
template<int BM,int BN,int K,int Tail,bool Pipeline=false> __global__ __launch_bounds__(256,1)
void register_tail_leaves(Problem p,Shape z,const __grid_constant__ CUtensorMap amap,
    const __grid_constant__ CUtensorMap bmap,const float* C,float* D,int tiles_n) {
  static_assert(BM==64 && BN==32 && Tail>0 && Tail<K);
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s=*reinterpret_cast<SingleShared<BM,BN,K>*>(storage);
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
  const int row=lane/4*8,col=lane%4*8;
  const int r0=blockIdx.x/tiles_n*BM,c0=blockIdx.x%tiles_n*BN;
  if(tid==224) {
    Barrier::init(&s.ready,1);cutlass::arch::fence_barrier_init();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  const int active_k=int((p.k+1)/2);int phase=0;float v[8][8]={};
  if(tid==224)single_prefetch(s,amap,bmap,r0,c0,0);
  for(int q=0;q<active_k;q+=K) {
    Barrier::wait(&s.ready,phase);phase^=1;
    asm volatile("" ::: "memory");
    float last_a[Tail][8],last_b[Tail][8];
    if(warp<7) {
      if(q+K<=active_k) {
        if constexpr(Pipeline) {
          float a[2][8],b[2][8];single_load(s,warp,0,row,col,a[0],b[0]);
          #pragma unroll (K-Tail)
          for(int k=0;k<K-Tail;++k) {
            if(k+1<K-Tail)single_load(s,warp,k+1,row,col,a[(k+1)%2],b[(k+1)%2]);
            single_multiply(a[k%2],b[k%2],v);
          }
        } else {
          #pragma unroll 4
          for(int k=0;k<K-Tail;++k) {
            float a[8],b[8];single_load(s,warp,k,row,col,a,b);single_multiply(a,b,v);
          }
        }
      } else {
        #pragma unroll 1
        for(int k=0;k<K-Tail && q+k<active_k;++k) {
          float a[8],b[8];single_load(s,warp,k,row,col,a,b);single_multiply(a,b,v);
        }
      }
      #pragma unroll
      for(int k=0;k<Tail;++k)single_load(s,warp,K-Tail+k,row,col,last_a[k],last_b[k]);
    }
    // All raw reads must precede the async-proxy overwrite. The memory clobber
    // also prevents sinking any cached-tail LDS past this handoff.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if(tid==224 && q+K<active_k)single_prefetch(s,amap,bmap,r0,c0,q+K);
    if(warp<7) {
      #pragma unroll
      for(int k=0;k<Tail;++k)if(q+K-Tail+k<active_k)single_multiply(last_a[k],last_b[k],v);
    }
    __syncthreads();
  }
  if(warp<7) {
    #pragma unroll
    for(int r=0;r<8;++r) {
      #pragma unroll
      for(int c=0;c<8;++c) {
        if(active_k%16)v[r][c]=__fmaf_rn(0.0f,0.0f,v[r][c]);
        s.products[warp][row+r][col+c]=v[r][c];
      }
    }
  }
  __syncthreads();
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2);
  for(int pos=tid;pos<BM*BN;pos+=256) {
    const int r=pos/BN,c=pos%BN;
    if(r0+r<mh && c0+c<nh) {
      float values[7];
      #pragma unroll
      for(int j=0;j<7;++j)values[j]=s.products[j][r][c];
      const float u2=__fadd_rn(values[0],values[5]),u3=__fadd_rn(u2,values[6]),u4=__fadd_rn(u2,values[4]);
      const float out[4]={__fadd_rn(values[0],values[1]),__fadd_rn(u4,values[2]),__fsub_rn(u3,values[3]),__fadd_rn(u3,values[4])};
      #pragma unroll
      for(int j=0;j<4;++j) {
        const int rr=r0+r+(j/2)*mh,cc=c0+c+(j%2)*nh;
        if(rr<p.m && cc<p.n) {
          float value=__fmul_rn(p.alpha,out[j]);
          if(p.beta!=0)value=__fadd_rn(value,__fmul_rn(p.beta,C[int64_t(rr)*p.ldc+cc]));
          D[int64_t(rr)*p.ldd+cc]=value;
        }
      }
    }
  }
}
template<int BM,int BN,int K,int Tail,bool Pipeline=false>
SWResources register_tail_dispatch(const Problem* p,const float* A,const float* B,const float* C,
    float* D,void* workspace,cudaStream_t stream,const Maps* prepared) {
  constexpr int Bytes=sizeof(SingleShared<BM,BN,K>);const auto kernel=register_tail_leaves<BM,BN,K,Tail,Pipeline>;
  if(!p) {
    check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;check(cudaFuncGetAttributes(&attr,kernel));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,256,Bytes));
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  const auto z=shape(*p);float* ap=static_cast<float*>(workspace),*bp=ap+z.a;
  launch_pack(*p,z,A,B,ap,bp,stream,true);
  const int tiles_n=(int((p->n+1)/2)+BN-1)/BN,tiles_m=(int((p->m+1)/2)+BM-1)/BM;
  kernel<<<tiles_m*tiles_n,256,Bytes,stream>>>(*p,z,prepared->a,prepared->b,C,D,tiles_n);
  check(cudaPeekAtLastError());return {};
}
template<int BM,int BN,int RM,int RN,int K,bool Pipeline=false,int MinBlocks=2,bool ActiveK=false,bool FoldM=false,int Threads=256,int Unroll=0,bool FoldN=false,int ProductWarps=1,bool Specialized=false,bool VectorEpilogue=false>
__device__ __forceinline__ void fused_body(Problem p,Shape z,const CUtensorMap& amap,
                  const CUtensorMap& bmap,const float* C,float* D,int tiles_n,int tile_index=-1) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s=*reinterpret_cast<FusedShared<BM+(FoldM?4:0),BN+(FoldN?4:0),K>*>(storage);
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
  constexpr int LaneCols=BN/RN;
  constexpr int RowGroups=32/LaneCols,MainRows=RM*RowGroups*ProductWarps,ExtraRows=BM-MainRows;
  static_assert(32%LaneCols==0 && ExtraRows>=0 && ExtraRows<=RowGroups && BM%4==0);
  static_assert(!(ExtraRows>0 && FoldM));
  static_assert(!Specialized || (ProductWarps==1 && Threads==256));
  constexpr int ComputeWarps=7*ProductWarps;
  constexpr int Issuer=Threads==32*ComputeWarps?0:32*ComputeWarps;
  const int product=warp/ProductWarps;
  const int row=(warp%ProductWarps)*RM*RowGroups+lane/LaneCols*RM,col=lane%LaneCols*RN;
  const unsigned output_index=tile_index<0?blockIdx.x:unsigned(tile_index);
  const int r0=output_index/tiles_n*BM,c0=output_index%tiles_n*BN;
  if(tid==Issuer) {
    Barrier::init(&s.ready[0],1);Barrier::init(&s.ready[1],1);
    if constexpr(Specialized) {Barrier::init(&s.empty[0],7);Barrier::init(&s.empty[1],7);}
    cutlass::arch::fence_barrier_init();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2);
  const bool fold_last=FoldM && mh>BM && mh%BM==1 && r0+BM==mh-1;
  const bool fold_last_n=FoldN && nh>BN && nh%BN==1 && c0+BN==nh-1;
  const bool extra_active=(ExtraRows>0 && lane/LaneCols<ExtraRows) || (FoldM && fold_last && lane<LaneCols);
  const int extra_row=ExtraRows>0?MainRows+lane/LaneCols:BM;
  const int active_k=ActiveK?int((p.k+1)/2):z.kp;
  float extra[RN]={},extra_n[RM]={},corner=0.0f;
  float v[RM][RN]={};int phase0=0,phase1=0;
  if constexpr(Specialized) {
    if(tid==Issuer) {
      int empty0=0,empty1=0;
      for(int q=0,seq=0;q<active_k;q+=K,++seq) {
        const int slot=seq%2;
        if(seq>=2) {
          Barrier::wait(&s.empty[slot],slot?empty1:empty0);
          if(slot)empty1^=1;else empty0^=1;
        }
        fused_prefetch(s,amap,bmap,slot,r0,c0,q);
      }
    }
  } else if(tid==Issuer)fused_prefetch(s,amap,bmap,0,r0,c0,0);
  if(!Specialized || warp<ComputeWarps) {
  for(int q=0,seq=0;q<active_k;q+=K,++seq) {
    const int slot=seq%2;
    Barrier::wait(&s.ready[slot],slot?phase1:phase0);
    if(slot)phase1^=1;else phase0^=1;
    asm volatile("" ::: "memory");
    if(!Specialized && tid==Issuer && q+K<active_k)fused_prefetch(s,amap,bmap,slot^1,r0,c0,q+K);
    if(warp<ComputeWarps) {
      if(ActiveK && q+K>active_k) {
        #pragma unroll 1
        for(int k=0;k<active_k-q;++k) {
          float a[RM],b[RN];fused_register_load(s,slot,product,k,row,col,a,b);
          fused_accumulate<BN,(ExtraRows>0)||FoldM,FoldN>(s,slot,product,k,lane,extra_active,extra_row,fold_last_n,v,extra,extra_n,corner,a,b);
        }
      } else if constexpr(Pipeline) {
        float a[2][RM],b[2][RN];
        fused_register_load(s,slot,product,0,row,col,a[0],b[0]);
        #pragma unroll (Unroll==0?K:Unroll)
        for(int k=0;k<K;++k) {
          if(k+1<K)fused_register_load(s,slot,product,k+1,row,col,a[(k+1)%2],b[(k+1)%2]);
          fused_accumulate<BN,(ExtraRows>0)||FoldM,FoldN>(s,slot,product,k,lane,extra_active,extra_row,fold_last_n,v,extra,extra_n,corner,a[k%2],b[k%2]);
        }
      } else {
        #pragma unroll (Unroll==0?16:Unroll)
        for(int k=0;k<K;++k) {
          float a[RM],b[RN];fused_register_load(s,slot,product,k,row,col,a,b);
          fused_accumulate<BN,(ExtraRows>0)||FoldM,FoldN>(s,slot,product,k,lane,extra_active,extra_row,fold_last_n,v,extra,extra_n,corner,a,b);
        }
      }
    }
    if constexpr(Specialized) {
      // One release per product warp after every reader has finished the slot.
      __syncwarp();
      if(lane==0)Barrier::arrive(&s.empty[slot]);
    } else __syncthreads();
  }
  }
  if constexpr(Specialized)__syncthreads();
  if(ActiveK && warp<ComputeWarps && active_k%16!=0) {
    // All omitted padding operands are +0. Repeated fma(+0,+0,x) is
    // idempotent, including signed zero; one preserves the BK16 reference.
    #pragma unroll
    for(int r=0;r<RM;++r) {
      #pragma unroll
      for(int c=0;c<RN;++c)v[r][c]=__fmaf_rn(0.0f,0.0f,v[r][c]);
    }
  }
  if(((ExtraRows>0)||FoldM) && ActiveK && warp<ComputeWarps && extra_active && active_k%16!=0) {
    #pragma unroll
    for(int c=0;c<RN;++c)extra[c]=__fmaf_rn(0.0f,0.0f,extra[c]);
  }
  if(FoldN && ActiveK && warp<ComputeWarps && fold_last_n && lane%LaneCols==0 && active_k%16!=0) {
    #pragma unroll
    for(int r=0;r<RM;++r)extra_n[r]=__fmaf_rn(0.0f,0.0f,extra_n[r]);
    if(((ExtraRows>0)||FoldM) && extra_active)corner=__fmaf_rn(0.0f,0.0f,corner);
  }
  // All stage readers have finished; reuse the same storage for reconstruction.
  if(warp<ComputeWarps) {
    #pragma unroll
    for(int r=0;r<RM;++r) {
      #pragma unroll
      for(int c=0;c<RN;++c)s.products[product][row+r][col+c]=v[r][c];
    }
  }
  if(((ExtraRows>0)||FoldM) && warp<ComputeWarps && extra_active) {
    #pragma unroll
    for(int c=0;c<RN;++c)s.products[product][extra_row][col+c]=extra[c];
  }
  if(FoldN && warp<ComputeWarps && fold_last_n && lane%LaneCols==0) {
    #pragma unroll
    for(int r=0;r<RM;++r)s.products[product][row+r][BN]=extra_n[r];
    if(((ExtraRows>0)||FoldM) && extra_active)s.products[product][extra_row][BN]=corner;
  }
  __syncthreads();
  if constexpr(VectorEpilogue) {
    static_assert(BN%4==0 && !FoldM && !FoldN);
    // Shared planes are padded and16-byte aligned. Four adjacent columns
    // belong to one thread; global vector accesses additionally check bounds
    // and alignment independently for each quadrant and C/D binding.
    for(int pos=tid*4;pos<BM*BN;pos+=Threads*4) {
      const int r=pos/BN,c=pos%BN;
      if(r0+r<mh && c0+c<nh) {
        float4 terms[7],out[4];
        #pragma unroll
        for(int j=0;j<7;++j)terms[j]=*reinterpret_cast<const float4*>(&s.products[j][r][c]);
        #pragma unroll
        for(int t=0;t<4;++t) {
          float v[7];
          #pragma unroll
          for(int j=0;j<7;++j)v[j]=reinterpret_cast<const float*>(&terms[j])[t];
          const float u2=__fadd_rn(v[0],v[5]),u3=__fadd_rn(u2,v[6]),u4=__fadd_rn(u2,v[4]);
          reinterpret_cast<float*>(&out[0])[t]=__fadd_rn(v[0],v[1]);
          reinterpret_cast<float*>(&out[1])[t]=__fadd_rn(u4,v[2]);
          reinterpret_cast<float*>(&out[2])[t]=__fsub_rn(u3,v[3]);
          reinterpret_cast<float*>(&out[3])[t]=__fadd_rn(u3,v[4]);
        }
        #pragma unroll
        for(int j=0;j<4;++j) {
          const int rr=r0+r+(j/2)*mh,cc=c0+c+(j%2)*nh;
          if(rr<p.m && cc<p.n) {
            float* dest=D+int64_t(rr)*p.ldd+cc;
            const bool full=c0+c+3<nh && cc+3<p.n;
            if(full && (reinterpret_cast<uintptr_t>(dest)&15)==0) {
              float4 value=out[j];
              #pragma unroll
              for(int t=0;t<4;++t)reinterpret_cast<float*>(&value)[t]=__fmul_rn(p.alpha,reinterpret_cast<const float*>(&out[j])[t]);
              if(p.beta!=0) {
                const float* src=C+int64_t(rr)*p.ldc+cc;
                float4 prior;
                if((reinterpret_cast<uintptr_t>(src)&15)==0)prior=*reinterpret_cast<const float4*>(src);
                else {
                  #pragma unroll
                  for(int t=0;t<4;++t)reinterpret_cast<float*>(&prior)[t]=src[t];
                }
                #pragma unroll
                for(int t=0;t<4;++t)reinterpret_cast<float*>(&value)[t]=__fadd_rn(reinterpret_cast<const float*>(&value)[t],__fmul_rn(p.beta,reinterpret_cast<const float*>(&prior)[t]));
              }
              *reinterpret_cast<float4*>(dest)=value;
            } else {
              #pragma unroll
              for(int t=0;t<4;++t)if(c0+c+t<nh && cc+t<p.n) {
                float value=__fmul_rn(p.alpha,reinterpret_cast<const float*>(&out[j])[t]);
                if(p.beta!=0)value=__fadd_rn(value,__fmul_rn(p.beta,C[int64_t(rr)*p.ldc+cc+t]));
                dest[t]=value;
              }
            }
          }
        }
      }
    }
  } else {
  const int output_cols=BN+int(fold_last_n);
  for(int pos=tid;pos<(BM+int(fold_last))*output_cols;pos+=Threads) {
    const int r=pos/output_cols,c=pos%output_cols;
    if(r0+r<mh && c0+c<nh) {
      float v[7];
      #pragma unroll
      for(int j=0;j<7;++j)v[j]=s.products[j][r][c];
      const float u2=__fadd_rn(v[0],v[5]),u3=__fadd_rn(u2,v[6]),u4=__fadd_rn(u2,v[4]);
      const float out[4]={__fadd_rn(v[0],v[1]),__fadd_rn(u4,v[2]),__fsub_rn(u3,v[3]),__fadd_rn(u3,v[4])};
      #pragma unroll
      for(int j=0;j<4;++j) {
        const int rr=r0+r+(j/2)*mh,cc=c0+c+(j%2)*nh;
        if(rr<p.m && cc<p.n) {
          float value=__fmul_rn(p.alpha,out[j]);
          if(p.beta!=0)value=__fadd_rn(value,__fmul_rn(p.beta,C[int64_t(rr)*p.ldc+cc]));
          D[int64_t(rr)*p.ldd+cc]=value;
        }
      }
    }
  }
  }
}
template<int BM,int BN,int RM,int RN,int K,bool Pipeline=false,int MinBlocks=2,bool ActiveK=false,bool FoldM=false,int Threads=256,int Unroll=0,bool FoldN=false,int ProductWarps=1,bool Specialized=false>
__global__ __launch_bounds__(Threads,MinBlocks)
void fused_leaves(Problem p,Shape z,const __grid_constant__ CUtensorMap amap,
    const __grid_constant__ CUtensorMap bmap,const float* C,float* D,int tiles_n) {
  fused_body<BM,BN,RM,RN,K,Pipeline,MinBlocks,ActiveK,FoldM,Threads,Unroll,FoldN,ProductWarps,Specialized>(p,z,amap,bmap,C,D,tiles_n);
}
template<int BM,int BN,int RM,int RN,int K,bool Pipeline,int MinBlocks,bool ActiveK,bool Specialized=false,bool Helpers=false,bool ElectPerSM=false>
__global__ __launch_bounds__(256,MinBlocks)
void cooperative_fused_leaves(Problem p,Shape z,const __grid_constant__ CUtensorMap amap,
    const __grid_constant__ CUtensorMap bmap,const float* A,const float* B,const float* C,float* D,
    float* ap,float* bp,int tiles_n) {
  const auto grid=cooperative_groups::this_grid();
  extern __shared__ __align__(128) unsigned char storage[];
  auto& tile=*reinterpret_cast<float(*)[4][32][33]>(storage);
  unsigned nsmid=0;
  auto* scheduling=reinterpret_cast<unsigned*>(bp+z.b);
  if constexpr(ElectPerSM) {
    // SM identifiers may have holes. Size flags using %nsmid, never SM count.
    // The unused product region belongs to this invocation's workspace.
    asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsmid));
    if(size_t(nsmid)<z.c)for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<=nsmid;i+=size_t(gridDim.x)*256)
      scheduling[i]=0;
  }
  const int pack_a=(z.mp/32)*(z.kp/32),pack_b=(z.kp/32)*(z.np/32);
  for(int task=int(blockIdx.x);task<pack_a+pack_b;task+=int(gridDim.x)) {
    const int side=task>=pack_a;
    cooperative_pack_tile(p,z,A,B,ap,bp,side?task-pack_a:task,side,tile);
    __syncthreads();
  }
  // Generic global writes become available to TMA's async proxy before readers
  // pass the cooperative grid barrier. All writing threads execute the fence.
  asm volatile("fence.proxy.async.global;" ::: "memory");
  grid.sync();
  if constexpr(ElectPerSM) {
    if(size_t(nsmid)<z.c) {
      __shared__ int next_tile;
      if(threadIdx.x==0) {
        unsigned smid=0;asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        next_tile=atomicCAS(scheduling+smid,0u,1u)==0?0:-1;
      }
      __syncthreads();
      const int elected=next_tile;
      __syncthreads();
      if(elected<0)return;
      // The counter supplies every tile even if placement/preemption changes
      // the number or physical location of elected CTAs. SM IDs affect only
      // scheduling, never correctness or an assumed contiguous SM mapping.
      const int output_blocks=((int((p.m+1)/2)+BM-1)/BM)*tiles_n;
      while(true) {
        if(threadIdx.x==0)next_tile=int(atomicAdd(scheduling+nsmid,1u));
        __syncthreads();
        const int index=next_tile;
        if(index>=output_blocks)return;
        fused_body<BM,BN,RM,RN,K,Pipeline,MinBlocks,ActiveK,false,256,0,false,1,Specialized>(p,z,amap,bmap,C,D,tiles_n,index);
        __syncthreads();
      }
    }
  }
  if constexpr(Helpers) {
    // Helper CTAs retire only after their packed writes are visible to TMA.
    // No grid-wide synchronization occurs after this uniform per-CTA return.
    const int output_blocks=((int((p.m+1)/2)+BM-1)/BM)*tiles_n;
    if(int(blockIdx.x)>=output_blocks)return;
  }
  fused_body<BM,BN,RM,RN,K,Pipeline,MinBlocks,ActiveK,false,256,0,false,1,Specialized>(p,z,amap,bmap,C,D,tiles_n);
}

template<int BM,int K,bool Specialized>
__global__ __launch_bounds__(256,1)
void vector_epilogue_leaves(Problem p,Shape z,const __grid_constant__ CUtensorMap amap,
    const __grid_constant__ CUtensorMap bmap,const float* C,float* D,int tiles_n) {
  fused_body<BM,32,(BM==104?12:8),8,K,(BM!=104),1,true,false,256,0,false,1,Specialized,true>(p,z,amap,bmap,C,D,tiles_n);
}
template<int BM,int K,bool Specialized>
SWResources vector_epilogue_dispatch(const Problem* p,const float* A,const float* B,const float* C,
    float* D,void* workspace,cudaStream_t stream,const Maps* prepared) {
  constexpr int Bytes=sizeof(FusedShared<BM,32,K>);
  const auto kernel=vector_epilogue_leaves<BM,K,Specialized>;
  if(!p) {
    check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;
    check(cudaFuncGetAttributes(&attr,kernel));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,256,Bytes));
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  const auto z=shape(*p);float* ap=static_cast<float*>(workspace),*bp=ap+z.a;
  launch_pack(*p,z,A,B,ap,bp,stream,true);
  const int mh=int((p->m+1)/2),nh=int((p->n+1)/2),tiles_n=(nh+31)/32;
  kernel<<<((mh+BM-1)/BM)*tiles_n,256,Bytes,stream>>>(*p,z,prepared->a,prepared->b,C,D,tiles_n);
  check(cudaPeekAtLastError());return {};
}
template<int BM,int BN,int RM,int RN,int K,bool Pipeline=false,int MinBlocks=2,bool ActiveK=false,bool FoldM=false,int Threads=256,int Unroll=0,bool FoldN=false,int ProductWarps=1,bool Specialized=false>
SWResources fused_dispatch(const Problem* p,const float* A,const float* B,const float* C,
                           float* D,void* workspace,cudaStream_t stream,const Maps* prepared) {
  constexpr int Bytes=sizeof(FusedShared<BM+(FoldM?4:0),BN+(FoldN?4:0),K>);
  if(!p) {
    check(cudaFuncSetAttribute(fused_leaves<BM,BN,RM,RN,K,Pipeline,MinBlocks,ActiveK,FoldM,Threads,Unroll,FoldN,ProductWarps,Specialized>,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;
    check(cudaFuncGetAttributes(&attr,fused_leaves<BM,BN,RM,RN,K,Pipeline,MinBlocks,ActiveK,FoldM,Threads,Unroll,FoldN,ProductWarps,Specialized>));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fused_leaves<BM,BN,RM,RN,K,Pipeline,MinBlocks,ActiveK,FoldM,Threads,Unroll,FoldN,ProductWarps,Specialized>,Threads,Bytes));
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  const auto z=shape(*p);float* ap=static_cast<float*>(workspace),*bp=ap+z.a;
  const auto& maps=*prepared;
  launch_pack(*p,z,A,B,ap,bp,stream,configurations.at(maps.id).vector_pack);
  const int mh=int((p->m+1)/2),nh=int((p->n+1)/2);
  const int tiles_n=FoldN && nh>BN && nh%BN==1?(nh-1)/BN:(nh+BN-1)/BN;
  const int tiles_m=FoldM && mh>BM && mh%BM==1?(mh-1)/BM:(mh+BM-1)/BM;
  fused_leaves<BM,BN,RM,RN,K,Pipeline,MinBlocks,ActiveK,FoldM,Threads,Unroll,FoldN,ProductWarps,Specialized><<<tiles_m*tiles_n,Threads,Bytes,stream>>>(*p,z,maps.a,maps.b,C,D,tiles_n);
  check(cudaPeekAtLastError());return {};
}
template<int BM,int BN,int RM,int RN,int K,bool Pipeline,int MinBlocks,bool ActiveK,bool Specialized=false,bool Helpers=false,bool ElectPerSM=false>
SWResources cooperative_dispatch(const Problem* p,const float* A,const float* B,const float* C,
    float* D,void* workspace,cudaStream_t stream,const Maps* prepared) {
  constexpr int Bytes=sizeof(FusedShared<BM,BN,K>);
  const auto kernel=cooperative_fused_leaves<BM,BN,RM,RN,K,Pipeline,MinBlocks,ActiveK,Specialized,Helpers,ElectPerSM>;
  if(!p) {
    // The fallback is a distinct entry point and needs its own shared limit.
    fused_dispatch<BM,BN,RM,RN,K,Pipeline,MinBlocks,ActiveK,false,256,0,false,1,Specialized>(
      nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr);
    check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;
    check(cudaFuncGetAttributes(&attr,kernel));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,256,Bytes));
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  auto z=shape(*p);const auto& maps=*prepared;
  int tiles_n=(int((p->n+1)/2)+BN-1)/BN;
  int blocks=((int((p->m+1)/2)+BM-1)/BM)*tiles_n;
  const bool aligned=p->k%8==0 && p->n%8==0 && p->lda%4==0 && p->ldb%4==0 &&
    reinterpret_cast<uintptr_t>(A)%16==0 && reinterpret_cast<uintptr_t>(B)%16==0;
  if(!aligned || blocks>maps.cooperative_max_blocks)
    return fused_dispatch<BM,BN,RM,RN,K,Pipeline,MinBlocks,ActiveK,false,256,0,false,1,Specialized>(p,A,B,C,D,workspace,stream,prepared);
  if constexpr(Helpers) {
    const int pack_tasks=(z.mp/32)*(z.kp/32)+(z.kp/32)*(z.np/32);
    blocks=std::min(maps.cooperative_max_blocks,std::max(blocks,pack_tasks));
  }
  if constexpr(ElectPerSM)blocks=maps.cooperative_max_blocks;
  float* ap=static_cast<float*>(workspace),*bp=ap+z.a;
  void* args[]={const_cast<Problem*>(p),&z,const_cast<CUtensorMap*>(&maps.a),const_cast<CUtensorMap*>(&maps.b),
    &A,&B,&C,&D,&ap,&bp,&tiles_n};
  check(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel),dim3(blocks),dim3(256),args,Bytes,stream));
  check(cudaPeekAtLastError());return {};
}

int cooperative_resident(int id) {
  int resident=0;
  if(id==113)check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
    cooperative_fused_leaves<64,32,8,8,32,true,1,true>,256,sizeof(FusedShared<64,32,32>)));
  else if(id==114)check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
    cooperative_fused_leaves<16,32,4,4,32,true,2,false>,256,sizeof(FusedShared<16,32,32>)));
  if(id==116)check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
    cooperative_fused_leaves<64,32,8,8,32,true,1,true,true>,256,sizeof(FusedShared<64,32,32>)));
  else if(id==117)check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
    cooperative_fused_leaves<16,32,4,4,32,true,2,false,true>,256,sizeof(FusedShared<16,32,32>)));
  if(id==123)check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
    cooperative_fused_leaves<64,32,8,8,16,true,2,true,false,true>,256,sizeof(FusedShared<64,32,16>)));
  else if(id==124)check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
    cooperative_fused_leaves<64,32,8,8,16,true,2,true,true,true>,256,sizeof(FusedShared<64,32,16>)));
  if(id==128)check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
    cooperative_fused_leaves<64,32,8,8,16,true,2,true,false,true,true>,256,sizeof(FusedShared<64,32,16>)));
  else if(id==129)check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
    cooperative_fused_leaves<64,32,8,8,16,true,2,true,true,true,true>,256,sizeof(FusedShared<64,32,16>)));
  return resident;
}

SWResources select(int id,const Problem* p,const float* A,const float* B,const float* C,
                   float* D,void* workspace,cudaStream_t stream,const Maps* prepared=nullptr) {
  if(p && !configurations.at(id).cutlass && !prepared) {
    const auto encoded=prepare(*p,id,workspace);return select(id,p,A,B,C,D,workspace,stream,&encoded);
  }
  switch(id) {
#define CONFIG(ID,BM,BN,RM,RN) case ID:return dispatch<BM,BN,RM,RN>(p,A,B,C,D,workspace,stream,prepared)
    CONFIG(0,64,64,4,4);CONFIG(1,32,64,4,4);CONFIG(2,64,32,4,4);
    CONFIG(3,32,32,4,4);CONFIG(4,64,64,8,8);CONFIG(5,64,32,8,4);
#define CUTLASS_CONFIG(ID,BM,BN,WM,WN) case ID:return cutlass_dispatch<BM,BN,WM,WN>(p,A,B,C,D,workspace,stream)
    CUTLASS_CONFIG(6,64,64,32,32);CUTLASS_CONFIG(7,64,128,32,64);
    CUTLASS_CONFIG(8,128,64,64,32);CUTLASS_CONFIG(9,128,128,32,64);
    CUTLASS_CONFIG(10,32,64,16,32);CUTLASS_CONFIG(11,64,32,32,16);
#undef CUTLASS_CONFIG
#define FUSED_CONFIG(ID,BM,BN,RM,RN,K) case ID:return fused_dispatch<BM,BN,RM,RN,K>(p,A,B,C,D,workspace,stream,prepared)
    FUSED_CONFIG(12,32,64,4,16,16);FUSED_CONFIG(13,64,32,8,8,16);
    FUSED_CONFIG(14,32,32,4,8,16);FUSED_CONFIG(15,32,16,4,4,16);
    FUSED_CONFIG(16,16,32,2,8,16);FUSED_CONFIG(17,16,16,2,4,16);
    FUSED_CONFIG(18,32,32,4,8,32);FUSED_CONFIG(19,32,16,4,4,32);
    FUSED_CONFIG(20,16,16,2,4,32);FUSED_CONFIG(21,16,32,2,8,32);
    FUSED_CONFIG(22,64,32,16,4,16);FUSED_CONFIG(23,64,32,16,4,32);
    FUSED_CONFIG(24,32,64,8,8,16);FUSED_CONFIG(25,32,64,16,4,16);
    FUSED_CONFIG(26,32,64,16,4,32);FUSED_CONFIG(27,32,32,8,4,16);
    FUSED_CONFIG(28,32,32,8,4,32);FUSED_CONFIG(29,32,16,4,4,64);
    FUSED_CONFIG(30,16,16,2,4,64);FUSED_CONFIG(31,16,32,4,4,32);
    FUSED_CONFIG(32,16,32,4,4,16);
#undef FUSED_CONFIG
#define PIPE_CONFIG(ID,BM,BN,RM,RN,K,MIN) case ID:return fused_dispatch<BM,BN,RM,RN,K,true,MIN>(p,A,B,C,D,workspace,stream,prepared)
    PIPE_CONFIG(33,64,32,8,8,16,1);PIPE_CONFIG(34,64,32,8,8,16,2);
    PIPE_CONFIG(35,64,32,8,8,32,1);PIPE_CONFIG(36,32,64,8,8,16,1);
    PIPE_CONFIG(37,32,32,8,4,16,2);PIPE_CONFIG(38,32,32,8,4,32,2);
    PIPE_CONFIG(39,32,16,4,4,32,2);PIPE_CONFIG(40,16,32,4,4,32,2);PIPE_CONFIG(105,16,32,4,4,32,2);
    PIPE_CONFIG(41,32,64,16,4,16,1);PIPE_CONFIG(42,64,32,16,4,16,1);
    PIPE_CONFIG(46,64,32,8,8,16,2);PIPE_CONFIG(47,32,64,8,8,16,1);
    PIPE_CONFIG(48,32,16,4,4,32,2);
#undef PIPE_CONFIG
#define EXTRA_CONFIG(ID,BM,BN,RM,RN,K,PIPE,MIN,TAIL) case ID:return fused_dispatch<BM,BN,RM,RN,K,PIPE,MIN,TAIL>(p,A,B,C,D,workspace,stream,prepared)
    EXTRA_CONFIG(43,32,16,4,4,64,false,2,false);EXTRA_CONFIG(44,16,32,4,4,32,false,2,false);
    EXTRA_CONFIG(45,64,32,8,8,16,false,2,false);
    EXTRA_CONFIG(49,64,32,8,8,16,false,2,true);EXTRA_CONFIG(50,64,32,8,8,16,true,2,true);
    EXTRA_CONFIG(51,64,32,8,8,32,true,1,true);EXTRA_CONFIG(104,64,32,8,8,32,true,1,true);EXTRA_CONFIG(52,32,64,8,8,16,false,2,true);
    EXTRA_CONFIG(53,32,32,8,4,16,false,2,true);EXTRA_CONFIG(54,32,32,8,4,32,true,2,true);
    EXTRA_CONFIG(64,64,64,16,8,16,false,1,true);EXTRA_CONFIG(65,64,64,16,8,16,true,1,true);
    EXTRA_CONFIG(72,64,64,8,16,16,false,1,true);
#undef EXTRA_CONFIG
#define FOLD_CONFIG(ID,BM,BN,RM,RN,K,PIPE,MIN) case ID:return fused_dispatch<BM,BN,RM,RN,K,PIPE,MIN,true,true>(p,A,B,C,D,workspace,stream,prepared)
    FOLD_CONFIG(55,64,32,8,8,16,false,2);FOLD_CONFIG(56,64,32,8,8,16,true,2);
    FOLD_CONFIG(57,64,32,8,8,16,true,1);FOLD_CONFIG(58,64,32,8,8,32,true,1);
    FOLD_CONFIG(59,32,64,8,8,16,false,2);FOLD_CONFIG(60,32,32,8,4,16,false,2);
    FOLD_CONFIG(61,32,32,8,4,32,true,2);FOLD_CONFIG(62,32,16,4,4,32,true,2);
    FOLD_CONFIG(63,16,32,4,4,32,true,2);
    FOLD_CONFIG(66,64,64,16,8,16,false,1);FOLD_CONFIG(67,64,64,16,8,16,true,1);
    FOLD_CONFIG(68,128,32,16,8,16,false,1);FOLD_CONFIG(69,128,32,16,8,16,true,1);
    FOLD_CONFIG(70,32,128,16,8,16,false,1);FOLD_CONFIG(71,32,128,16,8,16,true,1);
    FOLD_CONFIG(73,64,64,8,16,16,true,1);
#undef FOLD_CONFIG
#define SCHEDULE_CONFIG(ID,BM,BN,RM,RN,K,PIPE,MIN,TAIL,FOLD,THREADS,UNROLL) case ID:return fused_dispatch<BM,BN,RM,RN,K,PIPE,MIN,TAIL,FOLD,THREADS,UNROLL>(p,A,B,C,D,workspace,stream,prepared)
    SCHEDULE_CONFIG(74,64,64,16,8,16,false,1,true,false,256,4);
    SCHEDULE_CONFIG(75,64,64,16,8,16,false,1,true,false,256,8);
    SCHEDULE_CONFIG(76,64,64,16,8,16,false,1,true,false,256,1);
    SCHEDULE_CONFIG(77,64,64,16,8,16,true,1,true,false,256,4);
    SCHEDULE_CONFIG(78,64,64,16,8,16,false,1,true,false,224,4);
    SCHEDULE_CONFIG(79,64,64,16,8,16,true,1,true,false,224,4);
    SCHEDULE_CONFIG(80,128,32,16,8,16,false,1,true,true,224,4);
    SCHEDULE_CONFIG(81,32,128,16,8,16,false,1,true,true,224,4);
    SCHEDULE_CONFIG(82,64,32,8,8,32,true,1,true,false,224,8);
    SCHEDULE_CONFIG(83,64,32,8,8,16,true,2,true,false,224,8);
    SCHEDULE_CONFIG(84,32,16,4,4,32,true,2,false,false,224,8);
    SCHEDULE_CONFIG(85,32,16,4,4,64,false,2,false,false,224,8);
    SCHEDULE_CONFIG(86,16,32,4,4,32,true,2,false,false,224,8);
    SCHEDULE_CONFIG(87,64,64,16,8,16,true,1,true,true,224,4);
    SCHEDULE_CONFIG(88,64,64,8,16,16,false,1,true,false,224,4);
    SCHEDULE_CONFIG(89,64,32,8,8,16,true,2,true,true,224,8);
    SCHEDULE_CONFIG(125,64,32,8,8,32,true,1,true,false,256,4);
    SCHEDULE_CONFIG(126,64,32,8,8,32,true,1,true,false,256,8);
    SCHEDULE_CONFIG(127,64,32,8,8,32,true,1,true,false,256,16);
#undef SCHEDULE_CONFIG
#define BALANCED_CONFIG(ID,BM,BN,RM,RN,K,PIPE,MIN,FOLDM,THREADS,UNROLL,FOLDN) case ID:return fused_dispatch<BM,BN,RM,RN,K,PIPE,MIN,true,FOLDM,THREADS,UNROLL,FOLDN>(p,A,B,C,D,workspace,stream,prepared)
    BALANCED_CONFIG(90,104,32,12,8,16,false,1,false,256,0,false);
    BALANCED_CONFIG(91,104,32,12,8,16,true,1,false,256,0,false);
    BALANCED_CONFIG(92,104,32,12,8,16,false,1,false,224,8,false);
    BALANCED_CONFIG(93,104,32,12,8,16,false,1,false,256,8,false);
    BALANCED_CONFIG(94,48,64,12,8,16,false,1,false,256,0,true);
    BALANCED_CONFIG(95,48,64,12,8,16,true,1,false,256,0,true);
    BALANCED_CONFIG(96,48,64,12,8,32,true,1,false,256,16,true);
    BALANCED_CONFIG(97,64,48,8,12,16,false,1,true,256,0,true);
    BALANCED_CONFIG(98,64,48,8,12,16,true,1,true,256,0,true);
    BALANCED_CONFIG(99,64,48,8,12,32,true,1,true,256,16,true);
    BALANCED_CONFIG(100,32,96,8,12,16,true,1,true,256,0,true);
    BALANCED_CONFIG(101,72,48,8,12,16,true,1,false,256,0,true);
    BALANCED_CONFIG(102,64,24,8,6,16,true,2,true,256,0,false);
    BALANCED_CONFIG(103,64,24,8,6,16,true,2,true,256,0,true);
#undef BALANCED_CONFIG
#undef CONFIG
#define MULTI_CONFIG(ID,BM,BN,RM,RN,K,PIPE,TAIL,THREADS,PP) case ID:return fused_dispatch<BM,BN,RM,RN,K,PIPE,1,TAIL,false,THREADS,0,false,PP>(p,A,B,C,D,workspace,stream,prepared)
    MULTI_CONFIG(106,64,32,4,8,32,true,true,480,2);
    MULTI_CONFIG(107,64,32,4,8,16,true,true,480,2);
    MULTI_CONFIG(108,64,32,2,8,32,true,true,928,4);
    MULTI_CONFIG(109,64,32,2,8,16,true,true,928,4);
    MULTI_CONFIG(110,16,32,2,4,32,true,false,480,2);
    MULTI_CONFIG(111,32,16,2,4,64,false,false,480,2);
    MULTI_CONFIG(112,64,32,4,8,32,false,true,480,2);
#undef MULTI_CONFIG
    case 113:return cooperative_dispatch<64,32,8,8,32,true,1,true>(p,A,B,C,D,workspace,stream,prepared);
    case 114:return cooperative_dispatch<16,32,4,4,32,true,2,false>(p,A,B,C,D,workspace,stream,prepared);
    case 115:return fused_dispatch<64,32,8,8,32,true,1,true,false,256,0,false,1,true>(p,A,B,C,D,workspace,stream,prepared);
    case 116:return cooperative_dispatch<64,32,8,8,32,true,1,true,true>(p,A,B,C,D,workspace,stream,prepared);
    case 117:return cooperative_dispatch<16,32,4,4,32,true,2,false,true>(p,A,B,C,D,workspace,stream,prepared);
    case 118:return fused_dispatch<104,32,12,8,16,false,1,true,false,256,0,false,1,true>(p,A,B,C,D,workspace,stream,prepared);
    case 123:return cooperative_dispatch<64,32,8,8,16,true,2,true,false,true>(p,A,B,C,D,workspace,stream,prepared);
    case 124:return cooperative_dispatch<64,32,8,8,16,true,2,true,true,true>(p,A,B,C,D,workspace,stream,prepared);
    case 128:return cooperative_dispatch<64,32,8,8,16,true,2,true,false,true,true>(p,A,B,C,D,workspace,stream,prepared);
    case 129:return cooperative_dispatch<64,32,8,8,16,true,2,true,true,true,true>(p,A,B,C,D,workspace,stream,prepared);
    case 130:return balanced_static_dispatch<64,32,32,2>(p,A,B,C,D,workspace,stream,prepared);
    case 131:return balanced_static_dispatch<32,64,32,2>(p,A,B,C,D,workspace,stream,prepared);
    case 132:return balanced_static_dispatch<64,32,16,2>(p,A,B,C,D,workspace,stream,prepared);
    case 133:return balanced_static_dispatch<64,32,32,8>(p,A,B,C,D,workspace,stream,prepared);
    case 134:return balanced_shuffle_dispatch<64,32,32,2>(p,A,B,C,D,workspace,stream,prepared);
    case 135:return balanced_shuffle_dispatch<32,64,32,2>(p,A,B,C,D,workspace,stream,prepared);
    case 136:return balanced_narrow_dispatch<32,32,32,2>(p,A,B,C,D,workspace,stream,prepared);
    case 137:return balanced_narrow_dispatch<32,32,16,2>(p,A,B,C,D,workspace,stream,prepared);
    case 138:return balanced_narrow_dispatch<64,16,16,2>(p,A,B,C,D,workspace,stream,prepared);
    case 139:return balanced_narrow_dispatch<16,64,16,2>(p,A,B,C,D,workspace,stream,prepared);
    case 140:return balanced_narrow_dispatch<32,32,32,8>(p,A,B,C,D,workspace,stream,prepared);
    case 141:return balanced_narrow_dispatch<32,32,16,8>(p,A,B,C,D,workspace,stream,prepared);
    case 142:return register_tail_dispatch<64,32,64,4>(p,A,B,C,D,workspace,stream,prepared);
    case 143:return register_tail_dispatch<64,32,64,8>(p,A,B,C,D,workspace,stream,prepared);
    case 144:return register_tail_dispatch<64,32,48,4>(p,A,B,C,D,workspace,stream,prepared);
    case 145:return register_tail_dispatch<64,32,48,8>(p,A,B,C,D,workspace,stream,prepared);
    case 146:return register_tail_dispatch<64,32,64,4,true>(p,A,B,C,D,workspace,stream,prepared);
    case 147:return register_tail_dispatch<64,32,64,8,true>(p,A,B,C,D,workspace,stream,prepared);
    case 148:return register_tail_dispatch<64,32,48,4,true>(p,A,B,C,D,workspace,stream,prepared);
    case 149:return register_tail_dispatch<64,32,48,8,true>(p,A,B,C,D,workspace,stream,prepared);
    case 150:return vector_epilogue_dispatch<64,32,false>(p,A,B,C,D,workspace,stream,prepared);
    case 151:return vector_epilogue_dispatch<64,16,false>(p,A,B,C,D,workspace,stream,prepared);
    case 152:return vector_epilogue_dispatch<64,32,true>(p,A,B,C,D,workspace,stream,prepared);
    case 153:return vector_epilogue_dispatch<104,16,true>(p,A,B,C,D,workspace,stream,prepared);
    case 119:return balanced_dispatch<64,32,32>(p,A,B,C,D,workspace,stream,prepared);
    case 120:return balanced_dispatch<32,64,32>(p,A,B,C,D,workspace,stream,prepared);
    case 121:return balanced_dispatch<64,32,16>(p,A,B,C,D,workspace,stream,prepared);
    case 122:return balanced_dispatch<32,64,16>(p,A,B,C,D,workspace,stream,prepared);
    default:throw std::invalid_argument("invalid FP32 packed configuration");
  }
}
}
bool problem(const Problem& p) {
  return p.dtype==DType::FP32 && !p.trans_a && !p.trans_b && p.m>0 && p.n>0 && p.k>0 &&
    std::max({p.m,p.n,p.k,p.lda,p.ldb})<=32768;
}
Shape shape(const Problem& p) {
  const auto align=[](int64_t x){return int((x+31)/32*32);};
  const int mp=align((p.m+1)/2),np=align((p.n+1)/2),kp=align((p.k+1)/2);
  const size_t a=size_t(7)*mp*kp,b=size_t(7)*kp*np,c=size_t(7)*mp*np;
  return {mp,np,kp,a,b,c,(a+b+c)*4};
}
Maps prepare(const Problem& p,int id,void* workspace) {
  const auto& c=configurations.at(id);const auto z=shape(p);
  if(!problem(p) || !workspace || reinterpret_cast<uintptr_t>(workspace)%256 || c.cutlass)
    throw std::invalid_argument("incompatible FP32 packed map preparation");
  Maps maps;maps.workspace=workspace;maps.id=id;maps.mp=z.mp;maps.np=z.np;maps.kp=z.kp;
  if(c.cooperative) {
    int device=0,sms=0,supported=0;
    check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
    check(cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,device));
    if(supported)maps.cooperative_max_blocks=cooperative_resident(id)*sms;
  }
  float* ap=static_cast<float*>(workspace),*bp=ap+z.a;
  for(int side=0;side<2;++side) {
    const uint64_t dims[3]={uint64_t(side?z.np:z.mp),uint64_t(z.kp),7};
    const uint64_t strides[2]={dims[0]*4,dims[0]*z.kp*4};
    const uint32_t box[3]={uint32_t(side?c.bn+(c.fold_n?4:0):c.bm+(c.fold_m?4:0)),uint32_t(c.fused?c.bk:32),uint32_t(c.fused?7:1)},es[3]={1,1,1};
    const auto result=cuTensorMapEncodeTiled(side?&maps.b:&maps.a,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      3,side?bp:ap,dims,strides,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(result!=CUDA_SUCCESS)throw std::runtime_error("FP32 packed tensor map: "+std::to_string(int(result)));
  }
  return maps;
}
void run_prepared(const Problem& p,int id,const float* A,const float* B,const float* C,
                  float* D,void* workspace,cudaStream_t stream,const Maps& maps) {
  const auto z=shape(p);
  if(!problem(p) || maps.id!=id || maps.workspace!=workspace || maps.mp!=z.mp || maps.np!=z.np || maps.kp!=z.kp)
    throw std::invalid_argument("prepared FP32 packed maps do not match binding");
  select(id,&p,A,B,C,D,workspace,stream,&maps);
}
SWResources setup(int id) {return select(id,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr);}
std::string name(int id) {
  const auto& c=configurations.at(id);
  if(c.register_tail)return "sw_fp32_batch_register_tail;leaf_terms=7;tile=64x32;thread_tile=8x8;bk="+std::to_string(c.bk)+
    ";threads=256;product_warps=7;issuer=thread224;transforms=fp32;leaf_math=fp32_simt;accum=fp32;load=tma;copies_per_stage=2;raw_stages=1;active_k=1;k_unroll="+std::to_string(c.unroll)+";prefix_pipeline="+std::to_string(c.tail_pipeline)+";register_tail="+std::to_string(c.register_tail)+";pack_vector_if_aligned=1;calls=pack+fused_leaves_combine;resource_scope=leaf;residual_correction=off";
  if(c.balanced)return "sw_fp32_batch_balanced;leaf_terms=7;tile="+std::to_string(c.bm)+"x"+std::to_string(c.bn)+
    ";thread_tile=7x"+std::to_string(c.rn)+";bk="+std::to_string(c.bk)+";threads=256;product_warps=8;issuer=thread0;transforms=fp32;leaf_math=fp32_simt;accum=fp32;load=tma;copies_per_stage=2;raw_stages=2;active_k=1;register_pipeline=0;static_warps="+std::to_string(c.static_warps)+";k_unroll="+std::to_string(c.unroll?c.unroll:2)+";a_operand_shuffle="+std::to_string(id>=134 && id<=135)+";pack_vector_if_aligned=1;calls=pack+fused_leaves_combine;resource_scope=leaf;residual_correction=off";
  if(c.fused)return (c.vector_epilogue?std::string("epilogue_vector=4;"):std::string())+"sw_fp32_batch_fused;leaf_terms=7;tile="+std::to_string(c.bm)+"x"+std::to_string(c.bn)+
    ";thread_tile="+std::to_string(c.rm)+"x"+std::to_string(c.rn)+";bk="+std::to_string(c.bk)+
    ";threads="+std::to_string(c.threads)+";k_unroll="+std::to_string(c.unroll?c.unroll:(c.pipeline?c.bk:16))+
    ";extra_m_rows="+std::to_string(c.bm-c.rm*(32/(c.bn/c.rn))*c.warps_per_product)+";fold_one_n="+std::to_string(c.fold_n)+";b_box_cols="+std::to_string(c.bn+(c.fold_n?4:0))+
    ";fold_one_m="+std::to_string(c.fold_m)+";a_box_rows="+std::to_string(c.bm+(c.fold_m?4:0))+
    ";elect_per_sm="+std::to_string(c.elect_per_sm)+";helper_ctas="+std::to_string(c.helper_ctas)+";warp_specialized="+std::to_string(c.warp_specialized)+";cooperative_pack="+std::to_string(c.cooperative)+";pack_vector_shared="+std::to_string(c.vector_pack?16896:0)+";pack_vector_if_aligned="+std::to_string(c.vector_pack)+";active_k="+std::to_string(c.active_k)+
    ";register_pipeline="+std::to_string(c.pipeline)+";min_blocks="+std::to_string(c.min_blocks)+
    ";transforms=fp32;leaf_math=fp32_simt;accum=fp32;load=tma;box_products=7;copies_per_stage=2;"
    "raw_stages=2;issuer=thread"+std::to_string(c.threads==32*7*c.warps_per_product?0:32*7*c.warps_per_product)+";warps_per_product="+std::to_string(c.warps_per_product)+";product_warps="+std::to_string(7*c.warps_per_product)+";calls="+(c.cooperative?std::string("cooperative_pack_grid_sync_fused_if_eligible"):std::string("pack+fused_leaves_combine"))+";resource_scope="+(c.cooperative?std::string("cooperative_kernel"):std::string("leaf"))+";pack_shared=16896;pack_grid_2d_if_no_extra_ctas="+std::to_string(!c.cooperative)+";residual_correction=off";
  if(c.cutlass)return "sw_fp32_batch;leaf_terms=7;batch_count=7;tile="+std::to_string(c.bm)+"x"+std::to_string(c.bn)+
    ";warp_tile="+std::to_string(c.wm)+"x"+std::to_string(c.wn)+
    ";bk=8;transforms=fp32;leaf_math=fp32_simt;accum=fp32;load=cutlass_simt;stages=2;"
    "calls=pack+seven_leaves+combine;resource_scope=leaf;pack_shared=16896;residual_correction=off";
  return "sw_fp32_batch;leaf_terms=7;batch_count=7;tile="+std::to_string(c.bm)+"x"+std::to_string(c.bn)+
    ";thread_tile="+std::to_string(c.rm)+"x"+std::to_string(c.rn)+
    ";bk=32;transforms=fp32;leaf_math=fp32_simt;accum=fp32;load=tma;raw_stages=2;issuer=thread0;"
    "calls=pack+seven_leaves+combine;resource_scope=leaf;pack_shared=16896;pack_grid_2d_if_no_extra_ctas=1;residual_correction=off";
}
void run(const Problem& p,int id,const float* A,const float* B,const float* C,
         float* D,void* workspace,cudaStream_t stream) {
  if(!problem(p) || !workspace || reinterpret_cast<uintptr_t>(workspace)%256)
    throw std::invalid_argument("FP32 packed binding/workspace is incompatible");
  select(id,&p,A,B,C,D,workspace,stream);
}
}

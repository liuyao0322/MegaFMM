#include "sw_fp32_batch.h"
#include <cuda.h>
#include <cooperative_groups.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <algorithm>
#include <stdexcept>

namespace megafmm::detail::sw_fp32_batch {
namespace {
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class Kernel,class... Args> void launch_dependent(dim3 grid,int threads,int bytes,
    cudaStream_t stream,Kernel kernel,Args&&... args) {
  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed=1;
  cudaLaunchConfig_t cfg{};cfg.gridDim=grid;cfg.blockDim=dim3(threads);
  cfg.dynamicSmemBytes=bytes;cfg.stream=stream;cfg.attrs=&attr;cfg.numAttrs=1;
  check(cudaLaunchKernelEx(&cfg,kernel,std::forward<Args>(args)...));
}
__device__ __forceinline__ float read(const float* x,int r,int c,int rows,int cols,int ld) {
  return r<rows && c<cols?x[int64_t(r)*ld+c]:0.0f;
}

// The A output is K-major so the leaf TMA can write vector-friendly shared A
// directly. This tiled transpose keeps both original reads and packed writes
// coalesced; seven transforms are rounded exactly as in the fused pipeline.
template<bool Grid2D=false,bool Programmatic=false> __global__ void pack(Problem p,Shape z,const float* A,const float* B,float* ap,float* bp) {
  if constexpr(Programmatic)cudaTriggerProgrammaticLaunchCompletion();
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
// Keep contiguous scalar input loads; regroup only packed output stores.
template<bool Grid2D=false,bool Programmatic=false,bool Compact=false,bool VectorB=false,int APlanes=7,int BPlanes=7> __global__ void pack_coalesced_vector(Problem p,Shape z,const float* A,const float* B,float* ap,float* bp) {
  if constexpr(Programmatic)cudaTriggerProgrammaticLaunchCompletion();
  __shared__ float tile[4][32][33];
  const int x=threadIdx.x,y=threadIdx.y;
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2),kh=int((p.k+1)/2);
  const int pack_a=(z.mp/32)*(z.kp/32);
  const bool side=Compact?int(blockIdx.x)>=pack_a:blockIdx.z!=0;
  const int task=Compact?int(blockIdx.x)-(side?pack_a:0):int(blockIdx.x);
  const int tid=y*32+x,ox=tid%8,oy=tid/8;
  if(!side) {
    const int tiles_k=z.kp/32;
    if constexpr(Grid2D) {if(blockIdx.x>=tiles_k || blockIdx.y>=z.mp/32)return;}
    else if(task>=(z.mp/32)*tiles_k)return;
    const int r0=Grid2D?int(blockIdx.y)*32:task/tiles_k*32;
    const int q0=Grid2D?int(blockIdx.x)*32:task%tiles_k*32;
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
    const int r=r0+4*ox,q=q0+oy;
    float4 a[4];
    #pragma unroll
    for(int j=0;j<4;++j)a[j]={tile[j][4*ox][oy],tile[j][4*ox+1][oy],tile[j][4*ox+2][oy],tile[j][4*ox+3][oy]};
    const float4 s1=add4(a[2],a[3]),s2=sub4(s1,a[0]);
    const float4 av[7]={a[0],a[1],sub4(a[1],s2),a[3],s1,s2,sub4(a[0],a[2])};
    #pragma unroll
    for(int j=0;j<APlanes;++j)*reinterpret_cast<float4*>(ap+size_t(j)*z.mp*z.kp+size_t(q)*z.mp+r)=APlanes==4?a[j]:av[j];
  } else {
    const int tiles_n=z.np/32;
    if constexpr(Grid2D) {if(blockIdx.x>=tiles_n || blockIdx.y>=z.kp/32)return;}
    else if(task>=(z.kp/32)*tiles_n)return;
    const int q0=Grid2D?int(blockIdx.y)*32:task/tiles_n*32;
    const int c0=Grid2D?int(blockIdx.x)*32:task%tiles_n*32;
    #pragma unroll
    for(int offset=0;offset<32;offset+=8) {
      const int q=q0+y+offset,c=c0+x;
      float b[4]={};
      if(q<kh && c<nh) {
        #pragma unroll
        for(int j=0;j<4;++j)b[j]=read(B,q+(j/2)*kh,c+(j%2)*nh,int(p.k),int(p.n),int(p.ldb));
      }
      if constexpr(VectorB) {
        #pragma unroll
        for(int j=0;j<4;++j)tile[j][y+offset][x]=b[j];
      } else {
      const float t1=__fsub_rn(b[1],b[0]),t2=__fsub_rn(b[3],t1);
      const float bv[7]={b[0],b[2],b[3],__fsub_rn(t2,b[2]),t1,t2,__fsub_rn(b[3],b[1])};
      #pragma unroll
      for(int j=0;j<BPlanes;++j)bp[size_t(j)*z.kp*z.np+size_t(q)*z.np+c]=BPlanes==4?b[j]:bv[j];
      }
    }
    if constexpr(VectorB) {
      __syncthreads();
      const int q=q0+oy,c=c0+4*ox;
      float4 b[4];
      #pragma unroll
      for(int j=0;j<4;++j)b[j]={tile[j][oy][4*ox],tile[j][oy][4*ox+1],tile[j][oy][4*ox+2],tile[j][oy][4*ox+3]};
      const float4 t1=sub4(b[1],b[0]),t2=sub4(b[3],t1);
      const float4 bv[7]={b[0],b[2],b[3],sub4(t2,b[2]),t1,t2,sub4(b[3],b[1])};
      #pragma unroll
      for(int j=0;j<BPlanes;++j)*reinterpret_cast<float4*>(bp+size_t(j)*z.kp*z.np+size_t(q)*z.np+c)=BPlanes==4?b[j]:bv[j];
    }
  }
}

// Logical product to physical float4 slot; B rotation repeats each32cols.
template<bool A> __host__ __device__ constexpr size_t interleaved_offset(int k,int width,int coordinate,int product) {
  const int rotation=A?7*((coordinate/8)&1):7*((coordinate/8)%4);
  return (size_t(k)*(width/4)+coordinate/4)*32+((product+rotation)&7)*4+coordinate%4;
}
// Select exactly one of the original rounded transforms; no dynamic register
// array indexing and no eighth product is evaluated by the leaf computation.
template<bool A> __device__ __forceinline__ float4 interleaved_selected_operand(
    int product,float4 x0,float4 x1,float4 x2,float4 x3) {
  if constexpr(A) {
    switch(product) {
      case 0:return x0; case 1:return x1;
      case 2:return sub4(x1,sub4(add4(x2,x3),x0));
      case 3:return x3; case 4:return add4(x2,x3);
      case 5:return sub4(add4(x2,x3),x0); case 6:return sub4(x0,x2);
      default:return float4{};
    }
  } else {
    switch(product) {
      case 0:return x0; case 1:return x2; case 2:return x3;
      case 3:return sub4(sub4(x3,sub4(x1,x0)),x2);
      case 4:return sub4(x1,x0); case 5:return sub4(x3,sub4(x1,x0));
      case 6:return sub4(x3,x1); default:return float4{};
    }
  }
}
template<bool Grid2D=false,bool Programmatic=false,bool Compact=false,bool VectorB=false,int APlanes=7,int BPlanes=7,bool PhysicalSlots=false> __global__ void pack_interleaved(Problem p,Shape z,const float* A,const float* B,float* ap,float* bp) {
  if constexpr(Programmatic)cudaTriggerProgrammaticLaunchCompletion();
  __shared__ float tile[4][32][33];
  const int x=threadIdx.x,y=threadIdx.y;
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2),kh=int((p.k+1)/2);
  const int pack_a=(z.mp/32)*(z.kp/32);
  const bool side=Compact?int(blockIdx.x)>=pack_a:blockIdx.z!=0;
  const int task=Compact?int(blockIdx.x)-(side?pack_a:0):int(blockIdx.x);
  const int tid=y*32+x,ox=tid%8,oy=tid/8;
  if(!side) {
    const int tiles_k=z.kp/32;
    if constexpr(Grid2D) {if(blockIdx.x>=tiles_k || blockIdx.y>=z.mp/32)return;}
    else if(task>=(z.mp/32)*tiles_k)return;
    const int r0=Grid2D?int(blockIdx.y)*32:task/tiles_k*32;
    const int q0=Grid2D?int(blockIdx.x)*32:task%tiles_k*32;
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
    if constexpr(PhysicalSlots) {
      // Eight adjacent lanes fill all physical product slots of one float4.
      // Each warp writes 512 consecutive bytes after the branch reconverges.
      #pragma unroll 1
      for(int step=0;step<8;++step) {
        const int vi=tid/8+32*step,coord4=vi%8,localq=vi/8,slot=tid%8;
        const int q=q0+localq,coordinate=r0+4*coord4;
        float4 x[4];
        #pragma unroll
        for(int j=0;j<4;++j)x[j]={tile[j][4*coord4][localq],tile[j][4*coord4+1][localq],tile[j][4*coord4+2][localq],tile[j][4*coord4+3][localq]};
        const int rotation=7*((coordinate/8)&1);
        const int product=(slot-rotation)&7;
        const float4 output=interleaved_selected_operand<true>(product,x[0],x[1],x[2],x[3]);
    *reinterpret_cast<float4*>(ap+interleaved_offset<true>(q,z.mp,coordinate,product))=output;
      }
    } else {
    const int r=r0+4*ox,q=q0+oy;
    float4 a[4];
    #pragma unroll
    for(int j=0;j<4;++j)a[j]={tile[j][4*ox][oy],tile[j][4*ox+1][oy],tile[j][4*ox+2][oy],tile[j][4*ox+3][oy]};
    const float4 s1=add4(a[2],a[3]),s2=sub4(s1,a[0]);
    const float4 av[7]={a[0],a[1],sub4(a[1],s2),a[3],s1,s2,sub4(a[0],a[2])};
    #pragma unroll
    for(int j=0;j<8;++j)*reinterpret_cast<float4*>(ap+interleaved_offset<true>(q,z.mp,r,j))=j==7?float4{}:av[j];
    }
  } else {
    const int tiles_n=z.np/32;
    if constexpr(Grid2D) {if(blockIdx.x>=tiles_n || blockIdx.y>=z.kp/32)return;}
    else if(task>=(z.kp/32)*tiles_n)return;
    const int q0=Grid2D?int(blockIdx.y)*32:task/tiles_n*32;
    const int c0=Grid2D?int(blockIdx.x)*32:task%tiles_n*32;
    #pragma unroll
    for(int offset=0;offset<32;offset+=8) {
      const int q=q0+y+offset,c=c0+x;
      float b[4]={};
      if(q<kh && c<nh) {
        #pragma unroll
        for(int j=0;j<4;++j)b[j]=read(B,q+(j/2)*kh,c+(j%2)*nh,int(p.k),int(p.n),int(p.ldb));
      }
      if constexpr(VectorB) {
        #pragma unroll
        for(int j=0;j<4;++j)tile[j][y+offset][x]=b[j];
      } else {
      const float t1=__fsub_rn(b[1],b[0]),t2=__fsub_rn(b[3],t1);
      const float bv[7]={b[0],b[2],b[3],__fsub_rn(t2,b[2]),t1,t2,__fsub_rn(b[3],b[1])};
      #pragma unroll
      for(int j=0;j<BPlanes;++j)bp[size_t(j)*z.kp*z.np+size_t(q)*z.np+c]=BPlanes==4?b[j]:bv[j];
      }
    }
    if constexpr(VectorB) {
      __syncthreads();
      if constexpr(PhysicalSlots) {
        // Eight adjacent lanes fill all physical product slots of one float4.
        // Each warp writes 512 consecutive bytes after the branch reconverges.
        #pragma unroll 1
        for(int step=0;step<8;++step) {
          const int vi=tid/8+32*step,coord4=vi%8,localq=vi/8,slot=tid%8;
          const int q=q0+localq,coordinate=c0+4*coord4;
          float4 x[4];
          #pragma unroll
          for(int j=0;j<4;++j)x[j]={tile[j][localq][4*coord4],tile[j][localq][4*coord4+1],tile[j][localq][4*coord4+2],tile[j][localq][4*coord4+3]};
          const int rotation=7*((coordinate/8)%4);
          const int product=(slot-rotation)&7;
          const float4 output=interleaved_selected_operand<false>(product,x[0],x[1],x[2],x[3]);
      *reinterpret_cast<float4*>(bp+interleaved_offset<false>(q,z.np,coordinate,product))=output;
        }
      } else {
      const int q=q0+oy,c=c0+4*ox;
      float4 b[4];
      #pragma unroll
      for(int j=0;j<4;++j)b[j]={tile[j][oy][4*ox],tile[j][oy][4*ox+1],tile[j][oy][4*ox+2],tile[j][oy][4*ox+3]};
      const float4 t1=sub4(b[1],b[0]),t2=sub4(b[3],t1);
      const float4 bv[7]={b[0],b[2],b[3],sub4(t2,b[2]),t1,t2,sub4(b[3],b[1])};
      #pragma unroll
      for(int j=0;j<8;++j)*reinterpret_cast<float4*>(bp+interleaved_offset<false>(q,z.np,c,j))=j==7?float4{}:bv[j];
      }
    }
  }
}

// Four contiguous elements per thread reduce global load/store instruction count.
template<bool Grid2D=false,bool DirectB=false,bool Programmatic=false,bool Compact=false,int APlanes=7,int BPlanes=7> __global__ void pack_vector(Problem p,Shape z,const float* A,const float* B,float* ap,float* bp) {
  if constexpr(Programmatic)cudaTriggerProgrammaticLaunchCompletion();
  __shared__ float tile[4][32][33];
  const int x=threadIdx.x,y=threadIdx.y;
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2),kh=int((p.k+1)/2);
  // A and B have independent task counts. Compact launches retain the same
  // coalesced reads, shared transpose, RN transforms and vector writes.
  const int pack_a=APlanes?(z.mp/32)*(z.kp/32):0;
  const bool side=Compact?int(blockIdx.x)>=pack_a:blockIdx.z!=0;
  const int task=Compact?int(blockIdx.x)-(side?pack_a:0):int(blockIdx.x);
  if(!side) {
    const int tiles_k=z.kp/32;
    if constexpr(Grid2D) {if(blockIdx.x>=tiles_k || blockIdx.y>=z.mp/32)return;}
    else if(task>=(z.mp/32)*tiles_k)return;
    const int r0=Grid2D?int(blockIdx.y)*32:task/tiles_k*32;
    const int q0=Grid2D?int(blockIdx.x)*32:task%tiles_k*32;
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
      for(int j=0;j<APlanes;++j)
        *reinterpret_cast<float4*>(ap+size_t(j)*z.mp*z.kp+size_t(q)*z.mp+r)=APlanes==4?a[j]:av[j];
    }
  } else {
    const int tiles_n=z.np/32;
    if constexpr(Grid2D) {if(blockIdx.x>=tiles_n || blockIdx.y>=z.kp/32)return;}
    else if(task>=(z.kp/32)*tiles_n)return;
    const int q0=Grid2D?int(blockIdx.y)*32:task/tiles_n*32;
    const int c0=Grid2D?int(blockIdx.x)*32:task%tiles_n*32;
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
      for(int j=(DirectB && BPlanes==7)?3:0;j<BPlanes;++j)
        *reinterpret_cast<float4*>(bp+size_t(j)*z.kp*z.np+size_t(q)*z.np+c)=BPlanes==4?b[j]:bv[j];
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
    for(int offset=0;offset<32 && y<32;offset+=32) {
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
    for(int offset=0;offset<32 && y<32;offset+=32) {
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
    for(int offset=0;offset<32 && y<32;offset+=32) {
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

template<bool Programmatic=false> void launch_pack(const Problem& p,Shape z,const float* A,const float* B,float* ap,float* bp,
                  cudaStream_t stream,bool vectorized,bool direct_b=false) {
  const int pack_a=(z.mp/32)*(z.kp/32),pack_b=(z.kp/32)*(z.np/32);
  const int gx=std::max(z.kp,z.np)/32,gy=std::max(z.mp,z.kp)/32;
  const bool grid2d=gx*gy==std::max(pack_a,pack_b);
  const dim3 grid=grid2d?dim3(gx,gy,2):dim3(std::max(pack_a,pack_b),1,2);
  if(vectorized && p.k%8==0 && p.n%8==0 && p.lda%4==0 && p.ldb%4==0 &&
     reinterpret_cast<uintptr_t>(A)%16==0 && reinterpret_cast<uintptr_t>(B)%16==0) {
    if(direct_b) {
      if(grid2d)pack_vector<true,true,Programmatic><<<grid,dim3(8,32),0,stream>>>(p,z,A,B,ap,bp);
      else pack_vector<false,true,Programmatic><<<grid,dim3(8,32),0,stream>>>(p,z,A,B,ap,bp);
    } else {
      if(grid2d)pack_vector<true,false,Programmatic><<<grid,dim3(8,32),0,stream>>>(p,z,A,B,ap,bp);
      else pack_vector<false,false,Programmatic><<<grid,dim3(8,32),0,stream>>>(p,z,A,B,ap,bp);
    }
  } else {
    if(grid2d)pack<true,Programmatic><<<grid,dim3(32,8),0,stream>>>(p,z,A,B,ap,bp);
    else pack<false,Programmatic><<<grid,dim3(32,8),0,stream>>>(p,z,A,B,ap,bp);
  }
}
template<bool A> __device__ __forceinline__ void form_operands(const float (&x)[4],float (&v)[7]) {
  if constexpr(A) {
    const float s1=__fadd_rn(x[2],x[3]),s2=__fsub_rn(s1,x[0]);
    v[0]=x[0];v[1]=x[1];v[2]=__fsub_rn(x[1],s2);v[3]=x[3];
    v[4]=s1;v[5]=s2;v[6]=__fsub_rn(x[0],x[2]);
  } else {
    const float t1=__fsub_rn(x[1],x[0]),t2=__fsub_rn(x[3],t1);
    v[0]=x[0];v[1]=x[2];v[2]=x[3];v[3]=__fsub_rn(t2,x[2]);
    v[4]=t1;v[5]=t2;v[6]=__fsub_rn(x[3],x[1]);
  }
}
// A compact task range contains only the operands actually stored by this
// policy. Scalar guarded reads also support odd quadrants and arbitrary strides.
template<class C> __device__ void policy_pack_tile(Problem p,Shape z,const float* A,const float* B,
    float* ap,float* bp,int task,bool direct_b=false) {
  const int pack_a=C::a_planes?(z.mp/32)*(z.kp/32):0;
  const bool side=task>=pack_a;
  const int index=side?task-pack_a:task;
  const int width=side?z.np:z.mp,tiles=side?z.np/32:z.kp/32;
  const int outer=index/tiles*32,inner=index%tiles*32;
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2),kh=int((p.k+1)/2);
  for(int i=threadIdx.x;i<32*32;i+=blockDim.x) {
    const int q=side?outer+i/32:inner+i%32;
    const int rc=side?inner+i%32:outer+i/32;
    float x[4]={},v[7];
    if(q<kh && rc<(side?nh:mh)) {
      #pragma unroll
      for(int j=0;j<4;++j)x[j]=side
        ?read(B,q+(j/2)*kh,rc+(j%2)*nh,int(p.k),int(p.n),int(p.ldb))
        :read(A,rc+(j/2)*mh,q+(j%2)*kh,int(p.m),int(p.k),int(p.lda));
    }
    if(side) {
      if constexpr(C::b_source==OperandSource::PackedQuadrants) {
        #pragma unroll
        for(int j=0;j<4;++j)bp[size_t(j)*width*z.kp+size_t(q)*width+rc]=x[j];
      } else if constexpr(C::b_planes>0) {
        form_operands<false>(x,v);
        #pragma unroll
        for(int j=(C::b_source==OperandSource::HybridRawB && direct_b)?3:0;j<7;++j)
          bp[size_t(j)*width*z.kp+size_t(q)*width+rc]=v[j];
      }
    } else {
      if constexpr(C::a_source==OperandSource::PackedQuadrants) {
        #pragma unroll
        for(int j=0;j<4;++j)ap[size_t(j)*width*z.kp+size_t(q)*width+rc]=x[j];
      } else if constexpr(C::a_planes>0) {
        form_operands<true>(x,v);
        #pragma unroll
        for(int j=0;j<7;++j)ap[size_t(j)*width*z.kp+size_t(q)*width+rc]=v[j];
      }
    }
  }
}
template<class C> __global__ void policy_pack(Problem p,Shape z,const float* A,const float* B,float* ap,float* bp,bool direct_b) {
  if constexpr(C::pdl_pack)cudaTriggerProgrammaticLaunchCompletion();
  policy_pack_tile<C>(p,z,A,B,ap,bp,int(blockIdx.x),direct_b);
}
template<class C> void launch_policy_pack(Problem p,Shape z,const float* A,const float* B,float* ap,float* bp,cudaStream_t stream,bool direct_b=false) {
  const int tasks=(C::a_planes?(z.mp/32)*(z.kp/32):0)+(C::b_planes?(z.np/32)*(z.kp/32):0);
  if(!tasks)return;
  if(p.k%8==0 && p.n%8==0 && p.lda%4==0 && p.ldb%4==0 &&
     reinterpret_cast<uintptr_t>(A)%16==0 && reinterpret_cast<uintptr_t>(B)%16==0) {
    if(direct_b)pack_vector<false,true,C::pdl_pack,true,C::a_planes,C::b_planes><<<tasks,dim3(8,32),0,stream>>>(p,z,A,B,ap,bp);
    else pack_vector<false,false,C::pdl_pack,true,C::a_planes,C::b_planes><<<tasks,dim3(8,32),0,stream>>>(p,z,A,B,ap,bp);
  } else policy_pack<C><<<tasks,256,0,stream>>>(p,z,A,B,ap,bp,direct_b);
}

template<class C> void launch_config_pack(Problem p,Shape z,const float* A,const float* B,
    float* ap,float* bp,cudaStream_t stream,bool direct_b=false) {
  if constexpr(!C::vector_packing) {
    if constexpr(C::policy_pack)launch_policy_pack<C>(p,z,A,B,ap,bp,stream,direct_b);
    else launch_pack<C::pdl_pack>(p,z,A,B,ap,bp,stream,true,direct_b);
  } else {
    if(p.k%8==0 && p.n%8==0 && p.lda%4==0 && p.ldb%4==0 &&
       reinterpret_cast<uintptr_t>(A)%16==0 && reinterpret_cast<uintptr_t>(B)%16==0) {
      if constexpr(C::policy_pack)launch_policy_pack<C>(p,z,A,B,ap,bp,stream,direct_b);
      else launch_pack<C::pdl_pack>(p,z,A,B,ap,bp,stream,true,direct_b);
      return;
    }
    const int a=(z.mp/32)*(z.kp/32),b=(z.kp/32)*(z.np/32);
    const int gx=std::max(z.kp,z.np)/32,gy=std::max(z.mp,z.kp)/32;
    if constexpr(C::compact_pack)
      pack_coalesced_vector<false,C::pdl_pack,true,C::vector_pack_b,C::a_planes,C::b_planes><<<a+b,dim3(32,8),0,stream>>>(p,z,A,B,ap,bp);
    else {
      if(gx*gy==std::max(a,b))
        pack_coalesced_vector<true,C::pdl_pack,false,C::vector_pack_b,C::a_planes,C::b_planes><<<dim3(gx,gy,2),dim3(32,8),0,stream>>>(p,z,A,B,ap,bp);
      else pack_coalesced_vector<false,C::pdl_pack,false,C::vector_pack_b,C::a_planes,C::b_planes><<<dim3(std::max(a,b),1,2),dim3(32,8),0,stream>>>(p,z,A,B,ap,bp);
    }
  }
}

template<class C> struct alignas(128) Shared {
  alignas(128) float a[2][C::bk][C::bm];
  alignas(128) float b[2][C::bk][C::bn];
  alignas(16) uint64_t ready[2];
};
template<class C> __device__ __forceinline__ void prefetch(Shared<C>& s,const CUtensorMap& amap,
    const CUtensorMap& bmap,int slot,int r0,int c0,int q,int product) {
  TxBarrier::arrive_and_expect_tx(&s.ready[slot],(C::bm+C::bn)*C::bk*4);
  cute::SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],0,s.a[slot],r0,q,product);
  cute::SM90_TMA_LOAD_3D::copy(&bmap,&s.ready[slot],0,s.b[slot],c0,q,product);
}
template<class C> __device__ __forceinline__ void load(Shared<C>& s,int slot,int k,int row,int col,
                                    float (&a)[C::tm],float (&b)[C::tn]) {
  #pragma unroll
  for(int j=0;j<C::tm;j+=4) {
    const float4 av=*reinterpret_cast<const float4*>(&s.a[slot][k][row+j]);
    a[j]=av.x;a[j+1]=av.y;a[j+2]=av.z;a[j+3]=av.w;
  }
  #pragma unroll
  for(int j=0;j<C::tn;j+=4) {
    const float4 bv=*reinterpret_cast<const float4*>(&s.b[slot][k][col+j]);
    b[j]=bv.x;b[j+1]=bv.y;b[j+2]=bv.z;b[j+3]=bv.w;
  }
}
template<class C> __device__ __forceinline__ void accumulate(const float (&a)[C::tm],const float (&b)[C::tn],
                                                            float (&v)[C::tm][C::tn]) {
  if constexpr(C::zigzag_fma) {
    // Visit independent outputs in adjacent 2x2 groups. This changes only
    // intra-K scheduling: each accumulator retains its original K/RN order.
    #pragma unroll
    for(int c=0;c<C::tn;c+=2) {
      #pragma unroll
      for(int pair=0;pair<C::tm;pair+=2) {
        const int r=(c%4)?C::tm-2-pair:pair;
        v[r][c]=__fmaf_rn(a[r],b[c],v[r][c]);
        v[r+1][c]=__fmaf_rn(a[r+1],b[c],v[r+1][c]);
        v[r+1][c+1]=__fmaf_rn(a[r+1],b[c+1],v[r+1][c+1]);
        v[r][c+1]=__fmaf_rn(a[r],b[c+1],v[r][c+1]);
      }
    }
  } else {
    #pragma unroll
    for(int r=0;r<C::tm;++r) {
      #pragma unroll
      for(int c=0;c<C::tn;++c)v[r][c]=__fmaf_rn(a[r],b[c],v[r][c]);
    }
  }
}
// Transfer granularity and the expanded FMA body can be tuned independently.
template<class C> __device__ __forceinline__ void compute_chunk(Shared<C>& s,int slot,int base,int row,int col,
                                                               float (&v)[C::tm][C::tn]) {
  float a[2][C::tm],b[2][C::tn];load<C>(s,slot,base,row,col,a[0],b[0]);
  #pragma unroll
  for(int k=0;k<C::compute_k;++k) {
    if(k+1<C::compute_k)load<C>(s,slot,base+k+1,row,col,a[(k+1)%2],b[(k+1)%2]);
    accumulate<C>(a[k%2],b[k%2],v);
  }
}
template<class C> __device__ __forceinline__ void store_product(float* dest,float4 value) {
  if constexpr(!C::retain_products)*reinterpret_cast<float4*>(dest)=value;
  else {
    uint64_t policy;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;"
      :: "l"(dest),"f"(value.x),"f"(value.y),"f"(value.z),"f"(value.w),"l"(policy) : "memory");
  }
}
// Every configuration executes the same original seven-product algorithm.
template<class C> __global__ __launch_bounds__(C::threads,1)
void leaves(Problem p,const __grid_constant__ CUtensorMap amap,
            const __grid_constant__ CUtensorMap bmap,float* products,int mp,int np,int tiles_n) {
  if constexpr(C::pdl_combine)cudaTriggerProgrammaticLaunchCompletion();
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s=*reinterpret_cast<Shared<C>*>(storage);
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32,product=blockIdx.z;
  const int row=(warp/C::warps_n)*(8*C::tm)+lane/4*C::tm;
  const int col=(warp%C::warps_n)*(4*C::tn)+lane%4*C::tn;
  const int tiles_m=(int((p.m+1)/2)+C::bm-1)/C::bm;
  const int first_m=int(blockIdx.x)/(C::group_m*tiles_n)*C::group_m;
  const int group_m=min(tiles_m-first_m,C::group_m);
  const int within=int(blockIdx.x)%(C::group_m*tiles_n);
  const int r0=(first_m+within%group_m)*C::bm,c0=(within/group_m)*C::bn;
  if(tid==0) {
    Barrier::init(&s.ready[0],1);Barrier::init(&s.ready[1],1);
    cutlass::arch::fence_barrier_init();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  const int active_k=int((p.k+1)/2);
  float v[C::tm][C::tn]={};
  // Independent initialization above may overlap; never read packed operands
  // until the preceding grid has completed and flushed its results.
  if constexpr(C::pdl_pack)cudaGridDependencySynchronize();
  if(tid==0)prefetch<C>(s,amap,bmap,0,r0,c0,0,product);
  for(int q=0,seq=0;q<active_k;q+=C::bk,++seq) {
    const int slot=seq%2;
    Barrier::wait(&s.ready[slot],(seq/2)&1);
    asm volatile("" ::: "memory");
    if(tid==0 && q+C::bk<active_k)prefetch<C>(s,amap,bmap,slot^1,r0,c0,q+C::bk,product);
    if(q+C::bk>active_k) {
      #pragma unroll 1
      for(int k=0;k<active_k-q;++k) {
        float a[C::tm],b[C::tn];load<C>(s,slot,k,row,col,a,b);accumulate<C>(a,b,v);
      }
    } else {
      #pragma unroll 1
      for(int base=0;base<C::bk;base+=C::compute_k)compute_chunk<C>(s,slot,base,row,col,v);
    }
    // Order every reader before a later TMA write to the reused slot.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
  }
  #pragma unroll
  for(int r=0;r<C::tm;++r) {
    #pragma unroll
    for(int c=0;c<C::tn;c+=4) {
      const int rr=r0+row+r,cc=c0+col+c;
      if(rr<mp && cc+3<np) {
        float4 value;
        #pragma unroll
        for(int t=0;t<4;++t) {
          // Preserve the canonical rounded zero-FMA tail across BK choices.
          if(active_k%16!=0)v[r][c+t]=__fmaf_rn(0.0f,0.0f,v[r][c+t]);
          reinterpret_cast<float*>(&value)[t]=v[r][c+t];
        }
        store_product<C>(products+size_t(product)*mp*np+size_t(rr)*np+cc,value);
      }
    }
  }
}
// Products0/1/2 use unmodified B quadrants; load them directly from original B.
// Remaining four products retain exactly the original packed RN transforms.
template<class C> __device__ __forceinline__ void direct_b_prefetch(Shared<C>& s,const CUtensorMap& amap,
    const CUtensorMap& bmap,const CUtensorMap& raw_bmap,Problem p,int slot,int r0,int c0,int q,int product) {
  TxBarrier::arrive_and_expect_tx(&s.ready[slot],(C::bm+C::bn)*C::bk*4);
  cute::SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],0,s.a[slot],r0,q,product);
  if(product<3) {
    const int nh=int((p.n+1)/2),kh=int((p.k+1)/2);
    cute::SM90_TMA_LOAD_3D::copy(&raw_bmap,&s.ready[slot],0,s.b[slot],c0+(product==2?nh:0),q+(product>=1?kh:0),0);
  } else cute::SM90_TMA_LOAD_3D::copy(&bmap,&s.ready[slot],0,s.b[slot],c0,q,product);
}
template<class C> __global__ __launch_bounds__(C::threads,1)
void direct_b_leaves(Problem p,const __grid_constant__ CUtensorMap amap,
            const __grid_constant__ CUtensorMap bmap,const __grid_constant__ CUtensorMap raw_bmap,float* products,int mp,int np,int tiles_n) {
  if constexpr(C::pdl_combine)cudaTriggerProgrammaticLaunchCompletion();
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s=*reinterpret_cast<Shared<C>*>(storage);
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32,product=blockIdx.z;
  const int row=(warp/C::warps_n)*(8*C::tm)+lane/4*C::tm;
  const int col=(warp%C::warps_n)*(4*C::tn)+lane%4*C::tn;
  const int tiles_m=(int((p.m+1)/2)+C::bm-1)/C::bm;
  const int first_m=int(blockIdx.x)/(C::group_m*tiles_n)*C::group_m;
  const int group_m=min(tiles_m-first_m,C::group_m);
  const int within=int(blockIdx.x)%(C::group_m*tiles_n);
  const int r0=(first_m+within%group_m)*C::bm,c0=(within/group_m)*C::bn;
  if(tid==0) {
    Barrier::init(&s.ready[0],1);Barrier::init(&s.ready[1],1);
    cutlass::arch::fence_barrier_init();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  const int active_k=int((p.k+1)/2);
  float v[C::tm][C::tn]={};
  // Independent initialization above may overlap; never read packed operands
  // until the preceding grid has completed and flushed its results.
  if constexpr(C::pdl_pack)cudaGridDependencySynchronize();
  if(tid==0)direct_b_prefetch<C>(s,amap,bmap,raw_bmap,p,0,r0,c0,0,product);
  for(int q=0,seq=0;q<active_k;q+=C::bk,++seq) {
    const int slot=seq%2;
    Barrier::wait(&s.ready[slot],(seq/2)&1);
    asm volatile("" ::: "memory");
    if(tid==0 && q+C::bk<active_k)direct_b_prefetch<C>(s,amap,bmap,raw_bmap,p,slot^1,r0,c0,q+C::bk,product);
    if(q+C::bk>active_k) {
      #pragma unroll 1
      for(int k=0;k<active_k-q;++k) {
        float a[C::tm],b[C::tn];load<C>(s,slot,k,row,col,a,b);accumulate<C>(a,b,v);
      }
    } else {
      #pragma unroll 1
      for(int base=0;base<C::bk;base+=C::compute_k)compute_chunk<C>(s,slot,base,row,col,v);
    }
    // Order every reader before a later TMA write to the reused slot.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
  }
  #pragma unroll
  for(int r=0;r<C::tm;++r) {
    #pragma unroll
    for(int c=0;c<C::tn;c+=4) {
      const int rr=r0+row+r,cc=c0+col+c;
      if(rr<mp && cc+3<np) {
        float4 value;
        #pragma unroll
        for(int t=0;t<4;++t) {
          // Preserve the canonical rounded zero-FMA tail across BK choices.
          if(active_k%16!=0)v[r][c+t]=__fmaf_rn(0.0f,0.0f,v[r][c+t]);
          reinterpret_cast<float*>(&value)[t]=v[r][c+t];
        }
        store_product<C>(products+size_t(product)*mp*np+size_t(rr)*np+cc,value);
      }
    }
  }
}
template<bool Programmatic=false> __global__ void combine(Problem p,Shape z,const float* products,const float* C,float* D) {
  if constexpr(Programmatic)cudaGridDependencySynchronize();
  const int tid=threadIdx.x,r0=0,c0=0,mh=int((p.m+1)/2),nh=int((p.n+1)/2);
  for(int pos=(blockIdx.x*blockDim.x+tid)*4;pos<z.mp*z.np;pos+=gridDim.x*blockDim.x*4) {
      const int r=pos/z.np,c=pos%z.np;
      if(r0+r<mh && c0+c<nh) {
        float4 terms[7],out[4];
        #pragma unroll
        for(int j=0;j<7;++j)terms[j]=*reinterpret_cast<const float4*>(products+size_t(j)*z.mp*z.np+size_t(r)*z.np+c);
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
}
template<class C> struct alignas(128) FusedShared {
  union {
    struct {
      alignas(128) float a[C::fused_slots][7][C::bk][C::bm];
      alignas(128) float b[C::fused_slots][7][C::bk][C::bn];
    } stage;
    float products[7][C::bm][C::bn];
  };
  alignas(16) uint64_t ready[C::fused_slots];
  uint64_t empty[C::fused_slots];
};
template<class C> struct alignas(128) WarpProducerShared : FusedShared<C> {
  alignas(16) uint64_t transformed[C::fused_slots];
};
template<class C> using FusedStorage=std::conditional_t<C::warp_transform,WarpProducerShared<C>,FusedShared<C>>;
template<class C> __device__ __forceinline__ void fused_prefetch(FusedStorage<C>& s,
    const CUtensorMap& amap,const CUtensorMap& bmap,int slot,int r0,int c0,int q) {
  TxBarrier::arrive_and_expect_tx(&s.ready[slot],7*(C::bm+C::bn)*C::bk*4);
  cute::SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],0,s.stage.a[slot],r0,q,0);
  cute::SM90_TMA_LOAD_3D::copy(&bmap,&s.ready[slot],0,s.stage.b[slot],c0,q,0);
}
// A keeps all seven packed transforms. B0/B1/B2 are original quadrants;
// B3..B6 share one four-product packed TMA transfer into matching shared planes.
template<class C> __device__ __forceinline__ void fused_raw_prefetch(FusedStorage<C>& s,
    const CUtensorMap& amap,const CUtensorMap& rest_b,const CUtensorMap& raw_b,
    Problem p,int slot,int r0,int c0,int q) {
  TxBarrier::arrive_and_expect_tx(&s.ready[slot],7*(C::bm+C::bn)*C::bk*4);
  cute::SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],0,s.stage.a[slot],r0,q,0);
  cute::SM90_TMA_LOAD_3D::copy(&rest_b,&s.ready[slot],0,s.stage.b[slot][3],c0,q,3);
  const int nh=int((p.n+1)/2),kh=int((p.k+1)/2);
  cute::SM90_TMA_LOAD_3D::copy(&raw_b,&s.ready[slot],0,s.stage.b[slot][0],c0,q,0);
  cute::SM90_TMA_LOAD_3D::copy(&raw_b,&s.ready[slot],0,s.stage.b[slot][1],c0,q+kh,0);
  cute::SM90_TMA_LOAD_3D::copy(&raw_b,&s.ready[slot],0,s.stage.b[slot][2],c0+nh,q+kh,0);
}
template<class C,bool RawB> __device__ void policy_prefetch(FusedStorage<C>& s,
    const CUtensorMap& amap,const CUtensorMap& bmap,const CUtensorMap* raw_bmap,
    Problem p,int slot,int r0,int c0,int q) {
  constexpr int BPlanes=RawB && C::raw_b_tma?4:C::b_planes;
  TxBarrier::arrive_and_expect_tx(&s.ready[slot],(C::a_planes*C::bm+BPlanes*C::bn)*C::bk*4);
  if constexpr(C::a_planes>0)
    cute::SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],0,s.stage.a[slot],r0,q,0);
  if constexpr(RawB) {
    if constexpr(C::raw_b_tma) {
      const int nh=int(p.n/2),kh=int(p.k/2);
      cute::SM90_TMA_LOAD_3D::copy(raw_bmap,&s.ready[slot],0,s.stage.b[slot][0],c0,q,0);
      cute::SM90_TMA_LOAD_3D::copy(raw_bmap,&s.ready[slot],0,s.stage.b[slot][1],c0+nh,q,0);
      cute::SM90_TMA_LOAD_3D::copy(raw_bmap,&s.ready[slot],0,s.stage.b[slot][2],c0,q+kh,0);
      cute::SM90_TMA_LOAD_3D::copy(raw_bmap,&s.ready[slot],0,s.stage.b[slot][3],c0+nh,q+kh,0);
    } else {
      const int nh=int((p.n+1)/2),kh=int((p.k+1)/2);
      cute::SM90_TMA_LOAD_3D::copy(&bmap,&s.ready[slot],0,s.stage.b[slot][3],c0,q,3);
      cute::SM90_TMA_LOAD_3D::copy(raw_bmap,&s.ready[slot],0,s.stage.b[slot][0],c0,q,0);
      cute::SM90_TMA_LOAD_3D::copy(raw_bmap,&s.ready[slot],0,s.stage.b[slot][1],c0,q+kh,0);
      cute::SM90_TMA_LOAD_3D::copy(raw_bmap,&s.ready[slot],0,s.stage.b[slot][2],c0+nh,q+kh,0);
    }
  } else if constexpr(C::b_planes>0)
    cute::SM90_TMA_LOAD_3D::copy(&bmap,&s.ready[slot],0,s.stage.b[slot],c0,q,0);
}
template<class C,bool A> __device__ void expand_operand(FusedStorage<C>& s,Problem p,
    const float* original,int slot,int r0,int c0,int q0,int lane) {
  constexpr auto Source=A?C::a_source:C::b_source;
  if constexpr(Source==OperandSource::PackedQuadrants || Source==OperandSource::RawQuadrants) {
    constexpr int Width=A?C::bm:C::bn;
    const int mh=int((p.m+1)/2),nh=int((p.n+1)/2),kh=int((p.k+1)/2);
    float* dest=A?&s.stage.a[slot][0][0][0]:&s.stage.b[slot][0][0][0];
    if constexpr(Source==OperandSource::PackedQuadrants) {
      // Each producer lane owns four adjacent elements. Read all four source
      // planes before publishing seven planes in place, retaining RN order.
      for(int i=4*lane;i<C::bk*Width;i+=4*32) {
        float4 x[4],v[7];
        #pragma unroll
        for(int j=0;j<4;++j)x[j]=*reinterpret_cast<const float4*>(dest+j*C::bk*Width+i);
        if constexpr(A) {
          const float4 s1=add4(x[2],x[3]),s2=sub4(s1,x[0]);
          v[0]=x[0];v[1]=x[1];v[2]=sub4(x[1],s2);v[3]=x[3];
          v[4]=s1;v[5]=s2;v[6]=sub4(x[0],x[2]);
        } else {
          const float4 t1=sub4(x[1],x[0]),t2=sub4(x[3],t1);
          v[0]=x[0];v[1]=x[2];v[2]=x[3];v[3]=sub4(t2,x[2]);
          v[4]=t1;v[5]=t2;v[6]=sub4(x[3],x[1]);
        }
        #pragma unroll
        for(int j=0;j<7;++j)*reinterpret_cast<float4*>(dest+j*C::bk*Width+i)=v[j];
      }
    } else {
      for(int i=lane;i<C::bk*Width;i+=32) {
        float x[4]={},v[7];
        const int q=q0+i/Width,rc=(A?r0:c0)+i%Width;
        if(q<kh && rc<(A?mh:nh)) {
          #pragma unroll
          for(int j=0;j<4;++j)x[j]=A
            ?read(original,rc+(j/2)*mh,q+(j%2)*kh,int(p.m),int(p.k),int(p.lda))
            :read(original,q+(j/2)*kh,rc+(j%2)*nh,int(p.k),int(p.n),int(p.ldb));
        }
        form_operands<A>(x,v);
        #pragma unroll
        for(int j=0;j<7;++j)dest[j*C::bk*Width+i]=v[j];
      }
    }
  }
}
// TMA has completed all four original B quadrants before this producer work.
// Each lane owns disjoint float4 positions and reads all sources before writes.
template<class C> __device__ __forceinline__ void expand_tma_b(FusedStorage<C>& s,int slot,int lane) {
  float* dest=&s.stage.b[slot][0][0][0];
  for(int i=4*lane;i<C::bk*C::bn;i+=4*32) {
    float4 x[4],v[7];
    #pragma unroll
    for(int j=0;j<4;++j)x[j]=*reinterpret_cast<const float4*>(dest+j*C::bk*C::bn+i);
    const float4 t1=sub4(x[1],x[0]),t2=sub4(x[3],t1);
    v[0]=x[0];v[1]=x[2];v[2]=x[3];v[3]=sub4(t2,x[2]);
    v[4]=t1;v[5]=t2;v[6]=sub4(x[3],x[1]);
    #pragma unroll
    for(int j=0;j<7;++j)*reinterpret_cast<float4*>(dest+j*C::bk*C::bn+i)=v[j];
  }
}
template<class C,bool RawB> __device__ void warp_produce(FusedStorage<C>& s,Problem p,
    const CUtensorMap& amap,const CUtensorMap& bmap,const CUtensorMap* raw_bmap,
    const float* A,const float* B,int r0,int c0,int limit,int lane) {
  const int stages=(limit+C::bk-1)/C::bk;
  if(lane==0)for(int seq=0;seq<min(stages,C::fused_slots);++seq)
    policy_prefetch<C,RawB>(s,amap,bmap,raw_bmap,p,seq,r0,c0,seq*C::bk);
  __syncwarp();
  for(int seq=0;seq<stages;++seq) {
    const int slot=seq%C::fused_slots;
    if(seq>=C::fused_slots) {
      Barrier::wait(&s.empty[slot],((seq-C::fused_slots)/C::fused_slots)&1);
      if(lane==0)policy_prefetch<C,RawB>(s,amap,bmap,raw_bmap,p,slot,r0,c0,seq*C::bk);
    }
    Barrier::wait(&s.ready[slot],(seq/C::fused_slots)&1);
    asm volatile("" ::: "memory");
    expand_operand<C,true>(s,p,A,slot,r0,c0,seq*C::bk,lane);
    if constexpr(C::raw_b_tma && RawB)expand_tma_b<C>(s,slot,lane);
    else expand_operand<C,false>(s,p,B,slot,r0,c0,seq*C::bk,lane);
    // Expansion stores must precede future asynchronous slot overwrites.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
    // Every writing lane publishes its own stores; every reader releases its
    // own loads below. No cross-warp dependency relies on a representative lane.
    Barrier::arrive(&s.transformed[slot]);
  }
}
template<class C> __device__ __forceinline__ void fused_load(FusedStorage<C>& s,
    int slot,int product,int k,int row,int col,float (&a)[C::tm],float (&b)[C::tn]) {
  if constexpr(C::tm==2) {
    const float2 x=*reinterpret_cast<const float2*>(&s.stage.a[slot][product][k][row]);
    a[0]=x.x;a[1]=x.y;
  } else {
  #pragma unroll
  for(int r=0;r<C::tm;r+=4) {
    const float4 x=*reinterpret_cast<const float4*>(&s.stage.a[slot][product][k][row+r]);
    a[r]=x.x;a[r+1]=x.y;a[r+2]=x.z;a[r+3]=x.w;
  }
  }
  #pragma unroll
  for(int c=0;c<C::tn;c+=4) {
    const float4 x=*reinterpret_cast<const float4*>(&s.stage.b[slot][product][k][col+c]);
    b[c]=x.x;b[c+1]=x.y;b[c+2]=x.z;b[c+3]=x.w;
  }
}
// Full tiles retain the original double-buffered register pipeline and zigzag
// traversal. A shortened row group omits only padded output accumulators.
template<class C,int Rows> __device__ __forceinline__ void fused_load_rows(
    FusedStorage<C>& s,int slot,int product,int k,int row,int col,
    float (&a)[C::tm],float (&b)[C::tn]) {
  if constexpr(Rows==C::tm)fused_load<C>(s,slot,product,k,row,col,a,b);
  else {
    #pragma unroll
    for(int r=0;r<Rows;++r)a[r]=s.stage.a[slot][product][k][row+r];
    #pragma unroll
    for(int c=0;c<C::tn;c+=4) {
      const float4 x=*reinterpret_cast<const float4*>(&s.stage.b[slot][product][k][col+c]);
      b[c]=x.x;b[c+1]=x.y;b[c+2]=x.z;b[c+3]=x.w;
    }
  }
}
template<class C,int Rows> __device__ __forceinline__ void accumulate_rows(
    const float (&a)[C::tm],const float (&b)[C::tn],float (&v)[C::tm][C::tn]) {
  if constexpr(Rows==C::tm)accumulate<C>(a,b,v);
  else {
    #pragma unroll
    for(int r=0;r<Rows;++r) {
      #pragma unroll
      for(int c=0;c<C::tn;++c)v[r][c]=__fmaf_rn(a[r],b[c],v[r][c]);
    }
  }
}
// Explicit compile-time steps keep both register buffers scalar even when the
// two output-row branches would make the compiler retain a runtime K loop.
template<class C,int Rows,int K> __device__ __forceinline__ void fused_register_steps(
    FusedStorage<C>& s,int slot,int product,int base,int row,int col,
    float (&a)[C::tm],float (&b)[C::tn],float (&next_a)[C::tm],float (&next_b)[C::tn],
    float (&v)[C::tm][C::tn]) {
  if constexpr(K+1<C::compute_k)
    fused_load_rows<C,Rows>(s,slot,product,base+K+1,row,col,next_a,next_b);
  accumulate_rows<C,Rows>(a,b,v);
  if constexpr(K+1<C::compute_k)
    fused_register_steps<C,Rows,K+1>(s,slot,product,base,row,col,next_a,next_b,a,b,v);
}
template<class C,int Rows> __device__ __forceinline__ void fused_step_rows(
    FusedStorage<C>& s,int slot,int product,int q,int limit,int row,int col,
    float (&v)[C::tm][C::tn]) {
  if(C::tail_policy==TailPolicy::GuardedFinalTile && q+C::bk>limit) {
    #pragma unroll 1
    for(int k=0;k<limit-q;++k) {
      float a[C::tm],b[C::tn];
      fused_load_rows<C,Rows>(s,slot,product,k,row,col,a,b);
      accumulate_rows<C,Rows>(a,b,v);
    }
  } else {
    #pragma unroll 1
    for(int base=0;base<C::bk;base+=C::compute_k) {
      float a[C::tm],b[C::tn],next_a[C::tm],next_b[C::tn];
      fused_load_rows<C,Rows>(s,slot,product,base,row,col,a,b);
      fused_register_steps<C,Rows,0>(s,slot,product,base,row,col,a,b,next_a,next_b,v);
    }
  }
}
template<class C> __device__ __forceinline__ int product_column(int row,int col) {
  if constexpr(C::product_layout==ProductLayout::RowXor4)
    return col ^ (((unsigned(row)/C::tm)&1u)*4u);
  return col;
}
template<class Cfg,bool RawB=false> __device__ __forceinline__ void fused_body(Problem p,Shape z,
    const CUtensorMap& amap,const CUtensorMap& bmap,const float* C,float* D,int tiles_n,int tile_index,
    const CUtensorMap* raw_bmap=nullptr,const float* A=nullptr,const float* B=nullptr) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s=*reinterpret_cast<FusedStorage<Cfg>*>(storage);
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
  constexpr int LaneCols=Cfg::bn/(Cfg::tn*Cfg::warps_n),Issuer=Cfg::compute_warps*32;
  const int product=warp/Cfg::product_warps,within=warp%Cfg::product_warps;
  const int row=(within/Cfg::warps_n)*(32/LaneCols)*Cfg::tm+lane/LaneCols*Cfg::tm;
  const int col=(within%Cfg::warps_n)*LaneCols*Cfg::tn+lane%LaneCols*Cfg::tn;
  const int r0=tile_index/tiles_n*Cfg::bm,c0=tile_index%tiles_n*Cfg::bn;
  if(tid==Issuer) {
    if constexpr(Cfg::persistent_tasks) {
      // Previous task's final CTA join completed every reader and TMA access.
      // Invalidate live mbarriers before resetting their phase for another task.
      if(tile_index>=int(gridDim.x)) {
        if constexpr(Cfg::warp_transform) {
          #pragma unroll
          for(int i=0;i<Cfg::fused_slots;++i)Barrier::invalidate(&s.transformed[i]);
        }
        if constexpr(Cfg::fused_slots==2) {
        Barrier::invalidate(&s.ready[0]);Barrier::invalidate(&s.ready[1]);
        Barrier::invalidate(&s.empty[0]);Barrier::invalidate(&s.empty[1]);
        } else {
          #pragma unroll
          for(int i=0;i<Cfg::fused_slots;++i) {
            Barrier::invalidate(&s.ready[i]);Barrier::invalidate(&s.empty[i]);
          }
        }
      }
    }
    if constexpr(Cfg::fused_slots==2) {
    Barrier::init(&s.ready[0],1);Barrier::init(&s.ready[1],1);
    Barrier::init(&s.empty[0],Cfg::compute_warps*(Cfg::warp_transform?32:1));
    Barrier::init(&s.empty[1],Cfg::compute_warps*(Cfg::warp_transform?32:1));
    } else {
      #pragma unroll
      for(int i=0;i<Cfg::fused_slots;++i) {
        Barrier::init(&s.ready[i],1);Barrier::init(&s.empty[i],Cfg::compute_warps*(Cfg::warp_transform?32:1));
      }
    }
    if constexpr(Cfg::warp_transform) {
      #pragma unroll
      for(int i=0;i<Cfg::fused_slots;++i)Barrier::init(&s.transformed[i],32);
    }
    cutlass::arch::fence_barrier_init();
  }
  // Every thread orders its prior cooperative-pack or previous-task accesses.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  float v[Cfg::tm][Cfg::tn]={};
  const int limit=Cfg::tail_policy==TailPolicy::GuardedFinalTile?int((p.k+1)/2):z.kp;
  // Barrier and register initialization may overlap the preceding pack grid.
  // All threads wait for its completion and global visibility before any TMA
  // or generic load can consume packed operands.
  if constexpr(Cfg::pdl_pack)cudaGridDependencySynchronize();
  // A dedicated producer can fill the configured slots. Each product warp releases
  // a slot only after every lane has consumed its operands.
  if constexpr(Cfg::warp_transform) {
    if(warp==Cfg::compute_warps)warp_produce<Cfg,RawB>(s,p,amap,bmap,raw_bmap,A,B,r0,c0,limit,lane);
  } else if(tid==Issuer) {
    for(int q=0,seq=0;q<limit;q+=Cfg::bk,++seq) {
      const int slot=seq%Cfg::fused_slots;
      if(seq>=Cfg::fused_slots)Barrier::wait(&s.empty[slot],((seq-Cfg::fused_slots)/Cfg::fused_slots)&1);
      if constexpr(RawB)fused_raw_prefetch<Cfg>(s,amap,bmap,*raw_bmap,p,slot,r0,c0,q);
      else fused_prefetch<Cfg>(s,amap,bmap,slot,r0,c0,q);
    }
  }
  if(warp<Cfg::compute_warps) {
    for(int q=0,seq=0;q<limit;q+=Cfg::bk,++seq) {
      const int slot=seq%Cfg::fused_slots;
      if constexpr(Cfg::warp_transform)Barrier::wait(&s.transformed[slot],(seq/Cfg::fused_slots)&1);
      else Barrier::wait(&s.ready[slot],(seq/Cfg::fused_slots)&1);
      asm volatile("" ::: "memory");
      if constexpr(Cfg::output_tail_rows>0) {
        // Every valid lane satisfies row+r<remaining. When remaining<=Rows,
        // no valid output can use an accumulator with r>=Rows.
        if(int((p.m+1)/2)-r0<=Cfg::output_tail_rows)
          fused_step_rows<Cfg,Cfg::output_tail_rows>(s,slot,product,q,limit,row,col,v);
        else fused_step_rows<Cfg,Cfg::tm>(s,slot,product,q,limit,row,col,v);
      } else {
      if(Cfg::tail_policy==TailPolicy::GuardedFinalTile && q+Cfg::bk>limit) {
        #pragma unroll 1
        for(int k=0;k<limit-q;++k) {
          float a[Cfg::tm],b[Cfg::tn];
          fused_load<Cfg>(s,slot,product,k,row,col,a,b);accumulate<Cfg>(a,b,v);
        }
      } else {
        #pragma unroll 1
        for(int base=0;base<Cfg::bk;base+=Cfg::compute_k) {
          float a[2][Cfg::tm],b[2][Cfg::tn];
          fused_load<Cfg>(s,slot,product,base,row,col,a[0],b[0]);
          #pragma unroll
          for(int k=0;k<Cfg::compute_k;++k) {
            if(k+1<Cfg::compute_k)fused_load<Cfg>(s,slot,product,base+k+1,row,col,a[(k+1)%2],b[(k+1)%2]);
            accumulate<Cfg>(a[k%2],b[k%2],v);
          }
        }
      }
      }
      // Fence each lane before its release (or the warp leader release).
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      if constexpr(Cfg::warp_transform)Barrier::arrive(&s.empty[slot]);
      else if(lane==0)Barrier::arrive(&s.empty[slot]);
    }
    if constexpr(Cfg::tail_policy==TailPolicy::GuardedFinalTile) {
      // One zero FMA is equivalent to the original padded tail, including -0.
      if(limit<z.kp) {
        #pragma unroll
        for(int r=0;r<Cfg::tm;++r) {
          #pragma unroll
          for(int c=0;c<Cfg::tn;++c)v[r][c]=__fmaf_rn(0.0f,0.0f,v[r][c]);
        }
      }
    }
  }
  // The raw operand/result union is reused only after producer and readers exit.
  __syncthreads();
  if(warp<Cfg::compute_warps) {
    #pragma unroll
    for(int r=0;r<Cfg::tm;++r) {
      if constexpr(Cfg::product_layout==ProductLayout::RowXor4) {
        #pragma unroll
        for(int c=0;c<Cfg::tn;c+=4) {
          const float4 value={v[r][c],v[r][c+1],v[r][c+2],v[r][c+3]};
          *reinterpret_cast<float4*>(&s.products[product][row+r][product_column<Cfg>(row+r,col+c)])=value;
        }
      } else {
        #pragma unroll
        for(int c=0;c<Cfg::tn;++c)s.products[product][row+r][col+c]=v[r][c];
      }
    }
  }
  __syncthreads();
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2);
  for(int pos=tid*4;pos<Cfg::bm*Cfg::bn;pos+=Cfg::threads*4) {
      const int r=pos/Cfg::bn,c=pos%Cfg::bn;
      if(r0+r<mh && c0+c<nh) {
        float4 terms[7],out[4];
        #pragma unroll
        for(int j=0;j<7;++j)terms[j]=*reinterpret_cast<const float4*>(&s.products[j][r][product_column<Cfg>(r,c)]);
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
  // All readers must finish before a persistent CTA reinitializes the union.
  __syncthreads();
}
template<class C> __global__ __launch_bounds__(C::threads,C::min_blocks)
void fused_leaves(Problem p,Shape z,const __grid_constant__ CUtensorMap amap,
    const __grid_constant__ CUtensorMap bmap,const float* prior,float* D,int tiles_n,
    const float* A,const float* B) {
  fused_body<C>(p,z,amap,bmap,prior,D,tiles_n,int(blockIdx.x),nullptr,A,B);
}
template<class C> __global__ __launch_bounds__(C::threads,C::min_blocks)
void raw_b_fused_leaves(Problem p,Shape z,const __grid_constant__ CUtensorMap amap,
    const __grid_constant__ CUtensorMap rest_b,const __grid_constant__ CUtensorMap raw_b,
    const float* prior,float* D,int tiles_n,const float* A,const float* B) {
  fused_body<C,true>(p,z,amap,rest_b,prior,D,tiles_n,int(blockIdx.x),&raw_b,A,B);
}
template<class C> __global__ __launch_bounds__(C::threads,C::min_blocks)
void cooperative_fused_leaves(Problem p,Shape z,const __grid_constant__ CUtensorMap amap,
    const __grid_constant__ CUtensorMap bmap,const float* A,const float* B,const float* prior,
    float* D,float* ap,float* bp,int tiles_n) {
  const auto grid=cooperative_groups::this_grid();
  extern __shared__ __align__(128) unsigned char storage[];
  auto& tile=*reinterpret_cast<float(*)[4][32][33]>(storage);
  const int pack_a=(C::a_planes?(z.mp/32)*(z.kp/32):0),pack_b=(C::b_planes?(z.kp/32)*(z.np/32):0);
  for(int task=int(blockIdx.x);task<pack_a+pack_b;task+=int(gridDim.x)) {
    if constexpr(C::policy_pack)policy_pack_tile<C>(p,z,A,B,ap,bp,task);
    else {
      const int side=task>=pack_a;
      cooperative_pack_tile(p,z,A,B,ap,bp,side?task-pack_a:task,side,tile);
    }
    __syncthreads();
  }
  // Every writing thread publishes its generic global writes to the TMA proxy.
  asm volatile("fence.proxy.async.global;" ::: "memory");
  grid.sync();
  const int tiles_m=(int((p.m+1)/2)+C::bm-1)/C::bm;
  for(int task=int(blockIdx.x);task<tiles_m*tiles_n;task+=int(gridDim.x))
    fused_body<C>(p,z,amap,bmap,prior,D,tiles_n,task,nullptr,A,B);
}
template<class Cfg> SWResources fused_dispatch(const Problem* p,const float* A,const float* B,const float* C,
    float* D,void* workspace,cudaStream_t stream,const Maps* maps) {
  constexpr int Bytes=sizeof(FusedStorage<Cfg>),Threads=Cfg::threads;
  static_assert(Bytes>=sizeof(float[4][32][33]),"cooperative pack must fit in shared storage");
  const auto kernel=cooperative_fused_leaves<Cfg>;
  if(!p) {
    check(cudaFuncSetAttribute(fused_leaves<Cfg>,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    if constexpr(Cfg::fused_raw_b || Cfg::raw_b_tma)check(cudaFuncSetAttribute(raw_b_fused_leaves<Cfg>,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;
    if constexpr(Cfg::pack_execution==PackExecution::SeparateKernel) {
      check(cudaFuncGetAttributes(&attr,fused_leaves<Cfg>));
      check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fused_leaves<Cfg>,Threads,Bytes));
    } else {
      check(cudaFuncGetAttributes(&attr,kernel));
      check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,Threads,Bytes));
    }
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  auto z=shape(*p,maps->id);float* ap=static_cast<float*>(workspace),*bp=z.a?ap+z.a:ap;
  int tiles_n=(int((p->n+1)/2)+Cfg::bn-1)/Cfg::bn;
  const int blocks=((int((p->m+1)/2)+Cfg::bm-1)/Cfg::bm)*tiles_n;
  if(uses_cooperative(*p,maps->id,A,B,*maps)) {
    void* args[]={const_cast<Problem*>(p),&z,const_cast<CUtensorMap*>(&maps->a),const_cast<CUtensorMap*>(&maps->b),
      &A,&B,&C,&D,&ap,&bp,&tiles_n};
    const int resident_blocks=std::min(blocks,maps->cooperative_max_blocks);
    // The launched grid, rather than the number of output tasks, must fit.
    if(resident_blocks<=0 || resident_blocks>maps->cooperative_max_blocks)
      throw std::runtime_error("invalid cooperative resident grid");
    check(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel),dim3(resident_blocks),dim3(Threads),args,Bytes,stream));
  } else {
    if constexpr(Cfg::fused_raw_b || Cfg::raw_b_tma) {
      const bool raw=uses_direct_b(*p,maps->id,A,B,*maps);
      launch_config_pack<Cfg>(*p,z,A,B,ap,bp,stream,raw);
      if(raw) {
        alignas(64) CUtensorMap raw_b=maps->raw_b;
        const auto result=cuTensorMapReplaceAddress(&raw_b,const_cast<float*>(B));
        if(result!=CUDA_SUCCESS)throw std::runtime_error("fused raw B binding: "+std::to_string(int(result)));
        if constexpr(Cfg::pdl_pack)launch_dependent(dim3(blocks),Threads,Bytes,stream,raw_b_fused_leaves<Cfg>,*p,z,maps->a,maps->rest_b,raw_b,C,D,tiles_n,A,B);
        else raw_b_fused_leaves<Cfg><<<blocks,Threads,Bytes,stream>>>(*p,z,maps->a,maps->rest_b,raw_b,C,D,tiles_n,A,B);
      } else {
        if constexpr(Cfg::pdl_pack)launch_dependent(dim3(blocks),Threads,Bytes,stream,fused_leaves<Cfg>,*p,z,maps->a,maps->b,C,D,tiles_n,A,B);
        else fused_leaves<Cfg><<<blocks,Threads,Bytes,stream>>>(*p,z,maps->a,maps->b,C,D,tiles_n,A,B);
      }
    } else {
    launch_config_pack<Cfg>(*p,z,A,B,ap,bp,stream);
    if constexpr(Cfg::pdl_pack)launch_dependent(dim3(blocks),Threads,Bytes,stream,fused_leaves<Cfg>,*p,z,maps->a,maps->b,C,D,tiles_n,A,B);
    else fused_leaves<Cfg><<<blocks,Threads,Bytes,stream>>>(*p,z,maps->a,maps->b,C,D,tiles_n,A,B);
    }
  }
  check(cudaPeekAtLastError());return {};
}

template<class Cfg> SWResources materialized_dispatch(const Problem* p,const float* A,const float* B,const float* C,
    float* D,void* workspace,cudaStream_t stream,const Maps* prepared) {
  constexpr int BM=Cfg::bm,BN=Cfg::bn,Threads=Cfg::threads,Bytes=sizeof(Shared<Cfg>);
  if(!p) {
    check(cudaFuncSetAttribute(leaves<Cfg>,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;
    if constexpr(Cfg::direct_b) {
      check(cudaFuncSetAttribute(direct_b_leaves<Cfg>,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
      check(cudaFuncGetAttributes(&attr,direct_b_leaves<Cfg>));
      check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,direct_b_leaves<Cfg>,Threads,Bytes));
    } else {
      check(cudaFuncGetAttributes(&attr,leaves<Cfg>));
      check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,leaves<Cfg>,Threads,Bytes));
    }
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  const auto z=shape(*p);float* ap=static_cast<float*>(workspace),*bp=ap+z.a;
  float* products=bp+z.b;
  const bool direct_b=Cfg::direct_b && uses_direct_b(*p,prepared->id,A,B,*prepared);
  launch_config_pack<Cfg>(*p,z,A,B,ap,bp,stream,direct_b);
  const int mh=int((p->m+1)/2),nh=int((p->n+1)/2),tiles_n=(nh+BN-1)/BN;
  if constexpr(Cfg::direct_b) {
    if(direct_b) {
      alignas(64) CUtensorMap raw_b=prepared->raw_b;
      const auto result=cuTensorMapReplaceAddress(&raw_b,const_cast<float*>(B));
      if(result!=CUDA_SUCCESS)throw std::runtime_error("direct B tensor map binding: "+std::to_string(int(result)));
      if constexpr(Cfg::pdl_pack)launch_dependent(dim3(((mh+BM-1)/BM)*tiles_n,1,7),Threads,Bytes,stream,direct_b_leaves<Cfg>,*p,prepared->a,prepared->b,raw_b,products,z.mp,z.np,tiles_n);
      else direct_b_leaves<Cfg><<<dim3(((mh+BM-1)/BM)*tiles_n,1,7),Threads,Bytes,stream>>>(*p,prepared->a,prepared->b,raw_b,products,z.mp,z.np,tiles_n);
    } else {
      if constexpr(Cfg::pdl_pack)launch_dependent(dim3(((mh+BM-1)/BM)*tiles_n,1,7),Threads,Bytes,stream,leaves<Cfg>,*p,prepared->a,prepared->b,products,z.mp,z.np,tiles_n);
      else leaves<Cfg><<<dim3(((mh+BM-1)/BM)*tiles_n,1,7),Threads,Bytes,stream>>>(*p,prepared->a,prepared->b,products,z.mp,z.np,tiles_n);
    }
  } else {
      if constexpr(Cfg::pdl_pack)launch_dependent(dim3(((mh+BM-1)/BM)*tiles_n,1,7),Threads,Bytes,stream,leaves<Cfg>,*p,prepared->a,prepared->b,products,z.mp,z.np,tiles_n);
      else leaves<Cfg><<<dim3(((mh+BM-1)/BM)*tiles_n,1,7),Threads,Bytes,stream>>>(*p,prepared->a,prepared->b,products,z.mp,z.np,tiles_n);
  }
  constexpr int EpilogueThreads=256;
  if constexpr(Cfg::pdl_combine)launch_dependent(dim3(std::min((z.mp*z.np+EpilogueThreads*4-1)/(EpilogueThreads*4),65535)),EpilogueThreads,0,stream,combine<true>,*p,z,products,C,D);
  else combine<false><<<std::min((z.mp*z.np+EpilogueThreads*4-1)/(EpilogueThreads*4),65535),EpilogueThreads,0,stream>>>(*p,z,products,C,D);
  check(cudaPeekAtLastError());return {};
}
// Separate prototype storage: never changes the retained product-major kernels.
template<class C> struct alignas(128) InterleavedBase {
  union {
    struct alignas(128) {
      float a[C::fused_slots][C::bk*C::bm*8];
      float b[C::fused_slots][C::bk*C::bn*8];
    } stage;
    float products[C::lane_reconstruct?6*7*C::tm*C::tn:7*C::bm*C::bn];
  };
  alignas(16) uint64_t ready[C::fused_slots];
  uint64_t empty[C::fused_slots];
};
template<class C> struct alignas(128) InterleavedShared:InterleavedBase<C> {
  alignas(16) uint64_t transformed[C::fused_slots];
};
template<class C> __device__ __forceinline__ void interleaved_load(
    InterleavedShared<C>& s,int slot,int product,int k,int row,int col,
    float (&a)[C::tm],float (&b)[C::tn]) {
  #pragma unroll
  for(int r=0;r<C::tm;r+=4) {
    const float4 x=*reinterpret_cast<const float4*>(s.stage.a[slot]+interleaved_offset<true>(k,C::bm,row+r,product));
    a[r]=x.x;a[r+1]=x.y;a[r+2]=x.z;a[r+3]=x.w;
  }
  #pragma unroll
  for(int c=0;c<C::tn;c+=4) {
    const float4 x=*reinterpret_cast<const float4*>(s.stage.b[slot]+interleaved_offset<false>(k,C::bn,col+c,product));
    b[c]=x.x;b[c+1]=x.y;b[c+2]=x.z;b[c+3]=x.w;
  }
}
template<class C,int K> __device__ __forceinline__ void interleaved_register_steps(
    InterleavedShared<C>& s,int slot,int product,int base,int row,int col,
    float (&a)[C::tm],float (&b)[C::tn],float (&next_a)[C::tm],float (&next_b)[C::tn],
    float (&v)[C::tm][C::tn]) {
  if constexpr(K+1<C::compute_k)
    interleaved_load<C>(s,slot,product,base+K+1,row,col,next_a,next_b);
  accumulate<C>(a,b,v);
  if constexpr(K+1<C::compute_k)
    interleaved_register_steps<C,K+1>(s,slot,product,base,row,col,next_a,next_b,a,b,v);
}
template<class C> __device__ __forceinline__ void interleaved_step(
    InterleavedShared<C>& s,int slot,int product,int row,int col,float (&v)[C::tm][C::tn]) {
  #pragma unroll 1
  for(int base=0;base<C::bk;base+=C::compute_k) {
    float a[C::tm],b[C::tn],next_a[C::tm],next_b[C::tn];
    interleaved_load<C>(s,slot,product,base,row,col,a,b);
    interleaved_register_steps<C,0>(s,slot,product,base,row,col,a,b,next_a,next_b,v);
  }
}
template<class C> __device__ __forceinline__ void interleaved_prefetch(
    InterleavedShared<C>& s,const CUtensorMap& amap,const CUtensorMap& bmap,
    int slot,int r0,int c0,int q) {
  TxBarrier::arrive_and_expect_tx(&s.ready[slot],8*(C::bm+C::bn)*C::bk*4);
  cute::SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],0,s.stage.a[slot],0,r0/4,q);
  cute::SM90_TMA_LOAD_3D::copy(&bmap,&s.ready[slot],0,s.stage.b[slot],0,c0/4,q);
}
template<class C> __device__ __forceinline__ void interleaved_produce(
    InterleavedShared<C>& s,const CUtensorMap& amap,const CUtensorMap& bmap,
    int r0,int c0,int limit,int lane) {
  const int stages=limit/C::bk;
  if(lane==0)for(int seq=0;seq<min(stages,C::fused_slots);++seq)
    interleaved_prefetch<C>(s,amap,bmap,seq,r0,c0,seq*C::bk);
  __syncwarp();
  for(int seq=0;seq<stages;++seq) {
    const int slot=seq%C::fused_slots;
    if(seq>=C::fused_slots) {
      Barrier::wait(&s.empty[slot],((seq-C::fused_slots)/C::fused_slots)&1);
      if(lane==0)interleaved_prefetch<C>(s,amap,bmap,slot,r0,c0,seq*C::bk);
    }
    Barrier::wait(&s.ready[slot],(seq/C::fused_slots)&1);
    asm volatile("" ::: "memory");
    // Keep the per-lane proxy ordering and all32 publications of accepted242.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
    Barrier::arrive(&s.transformed[slot]);
  }
}
__host__ __device__ constexpr int interleaved_seam(int position) {
  return position==4?0:position==9?1:position==13?2:position==18?3:position==22?4:position==27?5:-1;
}
__host__ __device__ constexpr int interleaved_seam_position(int seam) {
  return seam==0?4:seam==1?9:seam==2?13:seam==3?18:seam==4?22:27;
}
__device__ __forceinline__ void interleaved_store(Problem p,const float* C,float* D,
    int r0,int c0,int r,int c,int quadrant,float4 out) {
  const int mh=int((p.m+1)/2),nh=int((p.n+1)/2);
  const int rr=r0+r+(quadrant/2)*mh,cc=c0+c+(quadrant%2)*nh;
  if(r0+r>=mh || c0+c>=nh || rr>=p.m || cc>=p.n)return;
  float* dest=D+int64_t(rr)*p.ldd+cc;
  if(c0+c+3<nh && cc+3<p.n && (reinterpret_cast<uintptr_t>(dest)&15)==0) {
    float4 value=out;
    #pragma unroll
    for(int t=0;t<4;++t)reinterpret_cast<float*>(&value)[t]=__fmul_rn(p.alpha,reinterpret_cast<const float*>(&out)[t]);
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
      float value=__fmul_rn(p.alpha,reinterpret_cast<const float*>(&out)[t]);
      if(p.beta!=0)value=__fadd_rn(value,__fmul_rn(p.beta,C[int64_t(rr)*p.ldc+cc+t]));
      dest[t]=value;
    }
  }
}
__device__ __forceinline__ void interleaved_combine(Problem p,const float* C,float* D,
    int r0,int c0,int r,int c,const float4 (&v)[7]) {
  const float4 u2=add4(v[0],v[5]),u3=add4(u2,v[6]),u4=add4(u2,v[4]);
  interleaved_store(p,C,D,r0,c0,r,c,0,add4(v[0],v[1]));
  interleaved_store(p,C,D,r0,c0,r,c,1,add4(u4,v[2]));
  interleaved_store(p,C,D,r0,c0,r,c,2,sub4(u3,v[3]));
  interleaved_store(p,C,D,r0,c0,r,c,3,add4(u3,v[4]));
}
template<int R,int Col,class C> __device__ __forceinline__ float4 interleaved_gather(
    const float (&v)[C::tm][C::tn],int source) {
  return {__shfl_sync(0xffffffffu,v[R][Col],source),__shfl_sync(0xffffffffu,v[R][Col+1],source),
          __shfl_sync(0xffffffffu,v[R][Col+2],source),__shfl_sync(0xffffffffu,v[R][Col+3],source)};
}
template<int R,int Col,class C> __device__ __forceinline__ void interleaved_shuffle_write(
    Problem p,const float* prior,float* D,int r0,int c0,int warp,int lane,const float (&v)[C::tm][C::tn]) {
  const int first=(32*warp+6)/7,last=(32*(warp+1)-7)/7;
  const int position=first+lane/4,quad=lane%4;
  const bool valid=position<=last;
  const int source=7*(valid?position:first)-32*warp;
  const int r=(position/4)*8+R,c=(position%4)*8+Col;
  if constexpr(C::register_gather==RegisterGather::SelectedFour) {
  const float4 p0=interleaved_gather<R,Col,C>(v,source);
  const float4 second=interleaved_gather<R,Col,C>(v,source+(quad==0?1:5));
  const float4 u2=add4(p0,second);
  const float4 third=interleaved_gather<R,Col,C>(v,source+(quad==1?4:6));
  const float4 u34=add4(u2,third);
  const float4 fourth=interleaved_gather<R,Col,C>(v,source+(quad==1?2:quad==2?3:4));
  const float4 tail=quad==2?sub4(u34,fourth):add4(u34,fourth);
  const float4 out=quad==0?u2:tail;
  if(valid)interleaved_store(p,prior,D,r0,c0,r,c,quad,out);
  } else if constexpr(C::register_gather==RegisterGather::SelectedSeven) {
  const float4 p0=interleaved_gather<R,Col,C>(v,source);
  const float4 p1=interleaved_gather<R,Col,C>(v,source+1);
  float4 out=add4(p0,p1);
  const float4 p5=interleaved_gather<R,Col,C>(v,source+5);
  const float4 u2=add4(p0,p5);
  const float4 p6=interleaved_gather<R,Col,C>(v,source+6);
  const float4 u3=add4(u2,p6);
  const float4 p4=interleaved_gather<R,Col,C>(v,source+4);
  const float4 u4=add4(u2,p4);
  if(quad==3)out=add4(u3,p4);
  const float4 p2=interleaved_gather<R,Col,C>(v,source+2);
  if(quad==1)out=add4(u4,p2);
  const float4 p3=interleaved_gather<R,Col,C>(v,source+3);
  if(quad==2)out=sub4(u3,p3);
  if(valid)interleaved_store(p,prior,D,r0,c0,r,c,quad,out);
  } else {
  const float4 p0=interleaved_gather<R,Col,C>(v,source);
  const float4 p1=interleaved_gather<R,Col,C>(v,source+1);
  if(valid && quad==0)interleaved_store(p,prior,D,r0,c0,r,c,0,add4(p0,p1));
  const float4 p5=interleaved_gather<R,Col,C>(v,source+5);
  const float4 u2=add4(p0,p5);
  const float4 p6=interleaved_gather<R,Col,C>(v,source+6);
  const float4 u3=add4(u2,p6);
  const float4 p4=interleaved_gather<R,Col,C>(v,source+4);
  const float4 u4=add4(u2,p4);
  if(valid && quad==3)interleaved_store(p,prior,D,r0,c0,r,c,3,add4(u3,p4));
  const float4 p2=interleaved_gather<R,Col,C>(v,source+2);
  if(valid && quad==1)interleaved_store(p,prior,D,r0,c0,r,c,1,add4(u4,p2));
  const float4 p3=interleaved_gather<R,Col,C>(v,source+3);
  if(valid && quad==2)interleaved_store(p,prior,D,r0,c0,r,c,2,sub4(u3,p3));
  }
}
template<class C,int R=0> __device__ __forceinline__ void interleaved_shuffle_rows(
    Problem p,const float* prior,float* D,int r0,int c0,int warp,int lane,const float (&v)[C::tm][C::tn]) {
  interleaved_shuffle_write<R,0,C>(p,prior,D,r0,c0,warp,lane,v);
  interleaved_shuffle_write<R,4,C>(p,prior,D,r0,c0,warp,lane,v);
  if constexpr(R+1<C::tm)interleaved_shuffle_rows<C,R+1>(p,prior,D,r0,c0,warp,lane,v);
}
template<class C> __global__ __launch_bounds__(C::threads,C::min_blocks)
void interleaved_fused_leaves(Problem p,Shape z,const __grid_constant__ CUtensorMap amap,
    const __grid_constant__ CUtensorMap bmap,const float* prior,float* D,int tiles_n) {
  static_assert(C::tail_policy==TailPolicy::Padded && C::compute_warps==7 && C::threads==256);
  static_assert(C::bk%C::compute_k==0);
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s=*reinterpret_cast<InterleavedShared<C>*>(storage);
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
  const int r0=int(blockIdx.x)/tiles_n*C::bm,c0=int(blockIdx.x)%tiles_n*C::bn;
  const int product=C::lane_reconstruct?tid%7:warp;
  const int position=C::lane_reconstruct?tid/7:lane;
  const int row=(position/4)*8,col=(position%4)*8;
  if(tid==224) {
    #pragma unroll
    for(int i=0;i<C::fused_slots;++i) {
      Barrier::init(&s.ready[i],1);Barrier::init(&s.empty[i],224);Barrier::init(&s.transformed[i],32);
    }
    cutlass::arch::fence_barrier_init();
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  float v[C::tm][C::tn]={};
  if constexpr(C::pdl_pack)cudaGridDependencySynchronize();
  if(warp==7)interleaved_produce<C>(s,amap,bmap,r0,c0,z.kp,lane);
  else {
    for(int q=0,seq=0;q<z.kp;q+=C::bk,++seq) {
      const int slot=seq%C::fused_slots;
      Barrier::wait(&s.transformed[slot],(seq/C::fused_slots)&1);
      asm volatile("" ::: "memory");
      interleaved_step<C>(s,slot,product,row,col,v);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      Barrier::arrive(&s.empty[slot]);
    }
  }
  __syncthreads();
  if(warp<7) {
    if constexpr(C::lane_reconstruct) {
      const int seam=interleaved_seam(position);
      if(seam>=0) {
        #pragma unroll
        for(int r=0;r<C::tm;++r) {
          #pragma unroll
          for(int c=0;c<C::tn;++c)s.products[(seam*7+product)*C::tm*C::tn+r*C::tn+c]=v[r][c];
        }
      }
      // Every math lane participates, including seam sources and nonwriters.
      interleaved_shuffle_rows<C>(p,prior,D,r0,c0,warp,lane,v);
    } else {
      #pragma unroll
      for(int r=0;r<C::tm;++r) {
        #pragma unroll
        for(int c=0;c<C::tn;++c)s.products[(product*C::bm+row+r)*C::bn+col+c]=v[r][c];
      }
    }
  }
  __syncthreads();
  if constexpr(C::lane_reconstruct) {
    for(int i=4*tid;i<6*C::tm*C::tn;i+=4*C::threads) {
      const int seam=i/(C::tm*C::tn),offset=i%(C::tm*C::tn),position=interleaved_seam_position(seam);
      float4 products[7];
      #pragma unroll
      for(int j=0;j<7;++j)products[j]=*reinterpret_cast<const float4*>(s.products+(seam*7+j)*C::tm*C::tn+offset);
      interleaved_combine(p,prior,D,r0,c0,(position/4)*8+offset/C::tn,(position%4)*8+offset%C::tn,products);
    }
  } else {
    for(int i=4*tid;i<C::bm*C::bn;i+=4*C::threads) {
      float4 products[7];
      #pragma unroll
      for(int j=0;j<7;++j)products[j]=*reinterpret_cast<const float4*>(s.products+j*C::bm*C::bn+i);
      interleaved_combine(p,prior,D,r0,c0,i/C::bn,i%C::bn,products);
    }
  }
  __syncthreads();
}
template<class Cfg> SWResources interleaved_dispatch(const Problem* p,const float* A,const float* B,const float* C,
    float* D,void* workspace,cudaStream_t stream,const Maps* maps) {
  constexpr int Bytes=sizeof(InterleavedShared<Cfg>),Threads=Cfg::threads;
  if(!p) {
    check(cudaFuncSetAttribute(interleaved_fused_leaves<Cfg>,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
    cudaFuncAttributes attr{};int resident=0;
    check(cudaFuncGetAttributes(&attr,interleaved_fused_leaves<Cfg>));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,interleaved_fused_leaves<Cfg>,Threads,Bytes));
    return {Bytes+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  const auto z=shape(*p,maps->id);float* ap=static_cast<float*>(workspace),*bp=ap+z.a;
  const int pa=(z.mp/32)*(z.kp/32),pb=(z.kp/32)*(z.np/32);
  pack_interleaved<false,Cfg::pdl_pack,true,true,7,7,Cfg::coalesced_pack><<<pa+pb,dim3(32,8),0,stream>>>(*p,z,A,B,ap,bp);
  const int tiles_n=(int((p->n+1)/2)+Cfg::bn-1)/Cfg::bn;
  const int blocks=((int((p->m+1)/2)+Cfg::bm-1)/Cfg::bm)*tiles_n;
  if constexpr(Cfg::pdl_pack)launch_dependent(dim3(blocks),Threads,Bytes,stream,interleaved_fused_leaves<Cfg>,*p,z,maps->a,maps->b,C,D,tiles_n);
  else interleaved_fused_leaves<Cfg><<<blocks,Threads,Bytes,stream>>>(*p,z,maps->a,maps->b,C,D,tiles_n);
  check(cudaPeekAtLastError());return {};
}

template<class Cfg> SWResources dispatch(const Problem* p,const float* A,const float* B,const float* C,
    float* D,void* workspace,cudaStream_t stream,const Maps* maps) {
  if constexpr(Cfg::interleaved_operands)return interleaved_dispatch<Cfg>(p,A,B,C,D,workspace,stream,maps);
  else if constexpr(Cfg::fused)return fused_dispatch<Cfg>(p,A,B,C,D,workspace,stream,maps);
  else return materialized_dispatch<Cfg>(p,A,B,C,D,workspace,stream,maps);
}
// Dispatch is generated from the same tuple as the metadata and tuner registry.
template<class C> struct ConfigTag {using type=C;};
template<size_t I=0,class F> auto visit(int id,F&& f)
    -> decltype(f(ConfigTag<std::tuple_element_t<0,KernelConfigurations>>{})) {
  if constexpr(I<std::tuple_size_v<KernelConfigurations>) {
    if(id==int(I))return f(ConfigTag<std::tuple_element_t<I,KernelConfigurations>>{});
    return visit<I+1>(id,std::forward<F>(f));
  } else throw std::invalid_argument("invalid FP32 packed configuration");
}
const Config& config(int id) {
  if(id<0 || id>=int(configurations.size()))throw std::invalid_argument("invalid FP32 packed configuration");
  return configurations[size_t(id)];
}
} // namespace
bool problem(const Problem& p) {
  return p.dtype==DType::FP32 && !p.trans_a && !p.trans_b && p.m>0 && p.n>0 && p.k>0 &&
    std::max({p.m,p.n,p.k,p.lda,p.ldb})<=32768;
}
Shape shape(const Problem& p) {
  const auto align=[](int64_t x){return int((x+31)/32*32);};
  const int mp=align((p.m+1)/2),np=align((p.n+1)/2),kp=align((p.k+1)/2);
  const size_t a=size_t(7)*mp*kp,b=size_t(7)*kp*np;
  const size_t products=size_t(7)*mp*np;
  return {mp,np,kp,a,b,products,(a+b+products)*4};
}
Shape shape(const Problem& p,int id) {
  auto z=shape(p);const auto& c=config(id);
  z.a=size_t(c.a_planes)*z.mp*z.kp;z.b=size_t(c.b_planes)*z.kp*z.np;
  z.bytes=(z.a+z.b+z.products)*4;
  return z;
}
size_t workspace_size(const Problem& p,int id) {
  const auto z=shape(p,id);
  return config(id).strategy==ExecutionStrategy::CooperativeFused?(z.a+z.b)*4:z.bytes;
}
bool uses_cooperative(const Problem& p,int id,const float* A,const float* B,const Maps& maps) {
  const auto& c=config(id);
  if(c.strategy!=ExecutionStrategy::CooperativeFused || c.pack_execution==PackExecution::SeparateKernel || maps.id!=id || maps.cooperative_max_blocks<=0)return false;
  const int tiles_m=(int((p.m+1)/2)+c.bm-1)/c.bm,tiles_n=(int((p.n+1)/2)+c.bn-1)/c.bn;
  if(!c.persistent_tasks && int64_t(tiles_m)*tiles_n>maps.cooperative_max_blocks)return false;
  return p.k%8==0 && p.n%8==0 &&
    p.lda%4==0 && p.ldb%4==0 && reinterpret_cast<uintptr_t>(A)%16==0 && reinterpret_cast<uintptr_t>(B)%16==0;
}
bool uses_direct_b(const Problem& p,int id,const float* A,const float* B,const Maps& maps) {
  const auto& c=config(id);
  const bool raw_tma=c.producer_policy==ProducerPolicy::WarpTransformTmaB;
  // Cooperative packing remains its validated fully-packed path. Raw B is an
  // alternative only for ordinary pack+fused execution outside that capacity.
  if((c.fused_raw_b || raw_tma) && uses_cooperative(p,id,A,B,maps))return false;
  if((c.fused_raw_b || raw_tma) && p.k%64!=0)return false;
  return (c.strategy==ExecutionStrategy::DirectBMaterialized || c.fused_raw_b || raw_tma) && maps.id==id && maps.raw_b_available &&
    p.n==maps.raw_n && p.k==maps.raw_k && p.ldb==maps.raw_ldb &&
    p.n%8==0 && p.k%8==0 && p.lda%4==0 && p.ldb%4==0 &&
    reinterpret_cast<uintptr_t>(A)%16==0 && reinterpret_cast<uintptr_t>(B)%16==0;
}
Maps prepare(const Problem& p,int id,void* workspace) {
  const auto& cfg=config(id);
  if(!problem(p) || (!workspace && workspace_size(p,id)>0) || reinterpret_cast<uintptr_t>(workspace)%256)
    throw std::invalid_argument("incompatible FP32 packed map preparation");
  const auto z=shape(p,id);
  Maps maps;maps.workspace=workspace;maps.id=id;maps.mp=z.mp;maps.np=z.np;maps.kp=z.kp;
  if(cfg.strategy==ExecutionStrategy::CooperativeFused && cfg.pack_execution!=PackExecution::SeparateKernel) {
    int device=0,sms=0,supported=0;
    check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
    check(cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,device));
    if(supported)maps.cooperative_max_blocks=visit(id,[&](auto tag) {
      using Cfg=typename decltype(tag)::type;
      int resident=0;
      if constexpr(Cfg::fused && !Cfg::interleaved_operands)check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident,cooperative_fused_leaves<Cfg>,Cfg::threads,sizeof(FusedStorage<Cfg>)));
      return resident*sms;
    });
  }
  float* ap=static_cast<float*>(workspace),*bp=z.a?ap+z.a:ap;
  for(int side=0;side<2;++side) {
    const int planes=side?cfg.b_planes:cfg.a_planes;
    if(!planes)continue;
    const bool interleaved=planes==8;
    const uint64_t width=uint64_t(side?z.np:z.mp);
    const uint64_t dims[3]={interleaved?32:width,interleaved?width/4:uint64_t(z.kp),interleaved?uint64_t(z.kp):uint64_t(planes)};
    const uint64_t strides[2]={interleaved?128:dims[0]*4,interleaved?width*32:dims[0]*z.kp*4};
    const uint32_t box[3]={interleaved?32u:uint32_t(side?cfg.bn:cfg.bm),interleaved?uint32_t((side?cfg.bn:cfg.bm)/4):uint32_t(cfg.bk),interleaved?uint32_t(cfg.bk):uint32_t(cfg.strategy==ExecutionStrategy::CooperativeFused?planes:1)},es[3]={1,1,1};
    const auto result=cuTensorMapEncodeTiled(side?&maps.b:&maps.a,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      3,side?bp:ap,dims,strides,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(result!=CUDA_SUCCESS)throw std::runtime_error("FP32 packed tensor map: "+std::to_string(int(result)));
  }
  if((cfg.strategy==ExecutionStrategy::DirectBMaterialized || cfg.fused_raw_b || cfg.producer_policy==ProducerPolicy::WarpTransformTmaB) && p.n%8==0 && p.k%8==0 && p.ldb%4==0) {
    const uint64_t dims[3]={uint64_t(p.n),uint64_t(p.k),1};
    const uint64_t strides[2]={uint64_t(p.ldb)*4,uint64_t(p.ldb)*p.k*4};
    const uint32_t box[3]={uint32_t(cfg.bn),uint32_t(cfg.bk),1},es[3]={1,1,1};
    const auto result=cuTensorMapEncodeTiled(&maps.raw_b,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,3,workspace,dims,strides,box,es,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(result!=CUDA_SUCCESS)throw std::runtime_error("direct B tensor map preparation: "+std::to_string(int(result)));
    maps.raw_b_available=true;maps.raw_n=int(p.n);maps.raw_k=int(p.k);maps.raw_ldb=int(p.ldb);
  }
  if(cfg.fused_raw_b) {
    const uint64_t dims[3]={uint64_t(z.np),uint64_t(z.kp),7};
    const uint64_t strides[2]={dims[0]*4,dims[0]*z.kp*4};
    const uint32_t box[3]={uint32_t(cfg.bn),uint32_t(cfg.bk),4},es[3]={1,1,1};
    const auto result=cuTensorMapEncodeTiled(&maps.rest_b,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,3,bp,dims,strides,box,es,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(result!=CUDA_SUCCESS)throw std::runtime_error("fused remaining B map: "+std::to_string(int(result)));
  }
  return maps;
}
void run_prepared(const Problem& p,int id,const float* A,const float* B,const float* C,
                  float* D,void* workspace,cudaStream_t stream,const Maps& maps) {
  const auto z=shape(p,id);
  config(id);
  if(!problem(p) || maps.id!=id || maps.workspace!=workspace || maps.mp!=z.mp || maps.np!=z.np || maps.kp!=z.kp)
    throw std::invalid_argument("prepared FP32 packed maps do not match binding");
  visit(id,[&](auto tag) {
    using Cfg=typename decltype(tag)::type;
    dispatch<Cfg>(&p,A,B,C,D,workspace,stream,&maps);
  });
}
SWResources setup(int id) {
  return visit(id,[](auto tag) {
    using Cfg=typename decltype(tag)::type;
    return dispatch<Cfg>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr);
  });
}
std::string name(int id) {
  const auto& c=config(id);
  const auto source=[](OperandSource s) {
    switch(s) {
      case OperandSource::PackedTransforms:return "packed_transforms";
      case OperandSource::PackedQuadrants:return "packed_quadrants";
      case OperandSource::RawQuadrants:return "raw_quadrants";
      case OperandSource::HybridRawB:return "hybrid_raw_b";
    }
    return "invalid";
  };
  return std::string("sw_fp32_batch;strategy=")+strategy_name(c.strategy)+";leaf_terms=7;tile="+std::to_string(c.bm)+"x"+std::to_string(c.bn)+
    ";bk="+std::to_string(c.bk)+";compute_k="+std::to_string(c.compute_k)+
    ";group_m="+std::to_string(c.group_m)+";threads="+std::to_string(c.threads)+
    ";thread_tile="+std::to_string(c.tm)+"x"+std::to_string(c.tn)+
    ";warp_grid="+std::to_string(c.warps_m)+"x"+std::to_string(c.warps_n)+
    ";programmatic_mask="+std::to_string(c.pdl_mask)+";zigzag_fma="+std::to_string(c.zigzag_fma)+";retain_products_l2="+std::to_string(c.retain_products)+";fused_raw_b="+std::to_string(c.fused_raw_b)+";fused_slots="+std::to_string(c.fused_slots)+";persistent_tasks="+std::to_string(c.persistent_tasks)+
    ";a_source="+source(c.a_source)+";b_source="+source(c.b_source)+
    ";pack_execution="+(c.pack_execution==PackExecution::SeparateKernel?"separate":c.pack_execution==PackExecution::CooperativeResident?"cooperative_resident":"automatic")+
    ";tail_policy="+(c.tail_policy==TailPolicy::GuardedFinalTile?"guarded_final_tile":"padded")+
    ";producer_policy="+(c.producer_policy==ProducerPolicy::WarpTransformTmaB?"warp_transform_tma_b":c.producer_policy==ProducerPolicy::WarpTransform?"warp_transform":"tma_issuer")+
    ";product_layout="+(c.product_layout==ProductLayout::RowXor4?"row_xor4":"linear")+
    ";a_planes="+std::to_string(c.a_planes)+";b_planes="+std::to_string(c.b_planes)+
    (c.vector_packing?";vector_packing=1;vector_pack_b="+std::to_string(c.vector_pack_b)+";compact_pack="+std::to_string(c.compact_pack):std::string())+
    (c.output_tail_rows?";output_tail_rows="+std::to_string(c.output_tail_rows):std::string())+
    ";tuning_task_waves="+std::to_string(c.max_task_waves)+
    ";epilogue_vector="+std::to_string(4)+";transforms=fp32;leaf_math=fp32_simt;accum=fp32;load="+
    (c.producer_policy==ProducerPolicy::WarpTransformTmaB?"tma":c.a_source==OperandSource::RawQuadrants && c.b_source==OperandSource::RawQuadrants?"producer_global":
      c.a_source==OperandSource::RawQuadrants || c.b_source==OperandSource::RawQuadrants?"tma+producer_global":"tma")+
    (c.producer_policy==ProducerPolicy::WarpTransformTmaB?";raw_b_tma=1;raw_b_source_planes=4;raw_b_fallback=producer_global":"")+
    ";raw_stages="+std::to_string(c.strategy==ExecutionStrategy::CooperativeFused?c.fused_slots:2)+";compute_warps="+
    std::to_string(c.strategy==ExecutionStrategy::CooperativeFused?7*c.warps_m*c.warps_n:c.threads/32)+";unroll="+std::to_string(c.compute_k)+
    (c.strategy==ExecutionStrategy::DirectBMaterialized
      ?";direct_b_products=0,1,2;packed_b_products=3,4,5,6;calls=pack+seven_leaves+combine;fallback=fully_packed;residual_correction=off"
      :c.strategy==ExecutionStrategy::Materialized
      ?";calls=pack+seven_leaves+combine;residual_correction=off"
      :";issuer=thread"+std::to_string(224*c.warps_m*c.warps_n)+";empty_arrivals="+std::to_string(7*c.warps_m*c.warps_n*(c.producer_policy!=ProducerPolicy::TmaIssuer?32:1))+
        ";box_products_a="+std::to_string(c.a_planes)+";box_products_b="+std::to_string(c.b_planes)+
        (c.pack_execution==PackExecution::SeparateKernel?";calls=pack+fused_leaves_combine":";calls=resident_cooperative_pack_grid_sync_fused_tasks_if_eligible;fallback=pack+fused_leaves_combine")+";residual_correction=off");
}
void run(const Problem& p,int id,const float* A,const float* B,const float* C,
         float* D,void* workspace,cudaStream_t stream) {
  const auto maps=prepare(p,id,workspace);
  run_prepared(p,id,A,B,C,D,workspace,stream,maps);
}
} // namespace megafmm::detail::sw_fp32_batch

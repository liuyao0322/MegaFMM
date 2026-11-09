#include "sw_hopper.h"
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace megafmm::detail::sw_hopper {
namespace {
using Barrier=cutlass::arch::ClusterBarrier;
using TxBarrier=cutlass::arch::ClusterTransactionBarrier;
using namespace cute;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Shape {int mh,nh,kh;size_t sa,sb,bytes;};
Shape shape(const Problem& p,const Configuration& c) {
  auto align=[](int64_t n,int a){return int((n+a-1)/a*a);};
  int mh=c.padded?align((p.m+1)/2,c.n):int(p.m/2);
  int nh=c.padded?align((p.n+1)/2,64*c.groups):int(p.n/2);
  int kh=c.padded?align((p.k+1)/2,c.bk):int(p.k/2);
  size_t sa=size_t(mh)*kh,sb=size_t(nh)*kh;
  return {mh,nh,kh,sa,sb,(c.padded?16:8)*(sa+sb)};
}
template<class T,class C> struct alignas(128) Shared {
  T operand[C::stages][(64*C::groups+C::n)*C::bk];
  uint64_t ready[C::stages],free[C::stages];
  alignas(128) float u3[C::shared_u3?C::groups*C::u3_components*128:1];
};
__device__ constexpr int product_for_step(int step) {
  return step<2?step:step==2?5:step==3?6:step==4?4:step==5?2:3;
}
template<class T> __device__ uint32_t pair(float a,float b) {
  uint32_t out;
  if constexpr(std::is_same_v<T,bfloat16_t>)asm("cvt.rn.bf16x2.f32 %0,%2,%1;":"=r"(out):"f"(a),"f"(b));
  else asm("cvt.rn.f16x2.f32 %0,%2,%1;":"=r"(out):"f"(a),"f"(b));
  return out;
}
// Native aligned inputs read vector pairs. The padded policy also materializes
// raw quadrants, so boundary and odd dimensions use the identical MMA pipeline.
template<class T,class C> __global__ __launch_bounds__(C::pack_threads)
void transform(const T* A,const T* B,T* workspace,int M,int N,int K,Shape z) {
  cudaTriggerProgrammaticLaunchCompletion();
  if constexpr(C::k_tiles)z.kh=C::k_tiles*C::bk;
  if constexpr(C::k_tiles && !C::padded)K=2*C::k_tiles*C::bk;
  const int lm=(M+1)/2,ln=(N+1)/2,lk=(K+1)/2;
  for(int side=C::split_pack?int(blockIdx.y):0;side<(C::split_pack?int(blockIdx.y)+1:2);++side) {
    int cols=side?z.nh:z.kh,halfrows=side?z.kh:z.mh;
    size_t plane=side?z.sb:z.sa;
    T* out=workspace+(side?4*z.sa:0);
    T* raw=workspace+4*(z.sa+z.sb)+(side?4*z.sa:0);
    constexpr int Vector=C::pack_vector;
    for(size_t i=Vector*(size_t(blockIdx.x)*C::pack_threads+threadIdx.x);i<plane;i+=size_t(Vector)*C::pack_threads*gridDim.x) {
      // Each plane fits in signed 32 bits for every eligible problem.
      // Native addressing only needs the row base, not a 64-bit quotient.
      int r=0,col=0;
      if constexpr(C::padded){r=int(i)/cols;col=int(i)%cols;}
      uint4 packed[4];
      if constexpr(!C::padded) {
        const T* input=side?B:A;
        size_t at=i+((cols&(cols-1))==0?(i&~size_t(cols-1)):size_t(int(i)/cols)*cols);
        #pragma unroll
        for(int j=0;j<4;++j) {
          auto src=input+at+(j&1)*cols+(j/2)*size_t(halfrows)*2*cols;
          if constexpr(Vector==8)packed[j]=*reinterpret_cast<const uint4*>(src);
          else packed[j].x=*reinterpret_cast<const uint32_t*>(src);
        }
      }
      uint4 result[4];
      #pragma unroll
      for(int part=0;part<Vector/2;++part) {
        float x[4][2]={},v[4][2];
        #pragma unroll
        for(int l=0;l<2;++l) {
          int cc=col+2*part+l;
          if constexpr(C::padded) {
            if(side==0) {
              if(r<lm && cc<lk) {
                x[0][l]=float(A[size_t(r)*K+cc]);
                if(cc+lk<K)x[1][l]=float(A[size_t(r)*K+cc+lk]);
                if(r+lm<M)x[2][l]=float(A[size_t(r+lm)*K+cc]);
                if(r+lm<M && cc+lk<K)x[3][l]=float(A[size_t(r+lm)*K+cc+lk]);
              }
            } else if(r<lk && cc<ln) {
              x[0][l]=float(B[size_t(r)*N+cc]);
              if(cc+ln<N)x[1][l]=float(B[size_t(r)*N+cc+ln]);
              if(r+lk<K)x[2][l]=float(B[size_t(r+lk)*N+cc]);
              if(r+lk<K && cc+ln<N)x[3][l]=float(B[size_t(r+lk)*N+cc+ln]);
            }
          } else {
            #pragma unroll
            for(int j=0;j<4;++j)x[j][l]=float(T::bitcast(uint16_t(reinterpret_cast<uint32_t*>(&packed[j])[part]>>(16*l))));
          }
          if(side==0) {
            v[0][l]=__fadd_rn(x[2][l],x[3][l]);v[1][l]=__fsub_rn(v[0][l],x[0][l]);
            v[2][l]=__fsub_rn(x[0][l],x[2][l]);v[3][l]=__fsub_rn(x[1][l],v[1][l]);
          } else {
            v[0][l]=__fsub_rn(x[1][l],x[0][l]);v[1][l]=__fsub_rn(x[3][l],v[0][l]);
            v[2][l]=__fsub_rn(x[3][l],x[1][l]);v[3][l]=__fsub_rn(v[1][l],x[2][l]);
          }
        }
        #pragma unroll
        for(int j=0;j<4;++j) {
          if constexpr(C::padded) {
            size_t at=size_t(r)*2*cols+col+(j&1)*cols+(j/2)*size_t(halfrows)*2*cols;
            *reinterpret_cast<uint32_t*>(raw+at)=pair<T>(x[j][0],x[j][1]);
            *reinterpret_cast<uint32_t*>(out+j*plane+i)=pair<T>(v[j][0],v[j][1]);
          } else reinterpret_cast<uint32_t*>(&result[j])[part]=pair<T>(v[j][0],v[j][1]);
        }
      }
      if constexpr(!C::padded) {
        #pragma unroll
        for(int j=0;j<4;++j) {
          if constexpr(Vector==8)*reinterpret_cast<uint4*>(out+j*plane+i)=result[j];
          else *reinterpret_cast<uint32_t*>(out+j*plane+i)=result[j].x;
        }
      }
    }
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
}
#include "sw_tensorcore_ptx.cuh"
// One implementation for every registered tile/schedule and both native dtypes.
template<class T,class C> __global__ __launch_bounds__(C::threads,1)
void compute(const __grid_constant__ CUtensorMap amap,const __grid_constant__ CUtensorMap bmap,
             const __grid_constant__ CUtensorMap smap,const __grid_constant__ CUtensorMap tmap,
             Shape z,float* D,int M,int N) {
  if constexpr(C::k_tiles)z.kh=C::k_tiles*C::bk;
  extern __shared__ __align__(128) unsigned char bytes[];
  auto& s=*reinterpret_cast<Shared<T,C>*>(bytes);
  int tid=threadIdx.x,n0=blockIdx.x*C::n,m0=blockIdx.y*64*C::groups;
  if constexpr(C::raster>0) {
    int pid=blockIdx.x+blockIdx.y*gridDim.x,g=pid/(C::raster*gridDim.y),first=g*C::raster;
    int count=min(C::raster,int(gridDim.x)-first),within=pid%(C::raster*gridDim.y);
    n0=(first+within%count)*C::n;m0=(within/count)*64*C::groups;
  }
  if(tid==32)asm volatile("prefetch.tensormap [%0]; prefetch.tensormap [%1]; prefetch.tensormap [%2]; prefetch.tensormap [%3];"::"l"(&amap),"l"(&bmap),"l"(&smap),"l"(&tmap):"memory");
  if(tid==0) {
    for(int i=0;i<C::stages;++i){Barrier::init(&s.ready[i],1);Barrier::init(&s.free[i],4*C::groups);}
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  if(tid<128) {
    if constexpr(C::producer_regs)cutlass::arch::warpgroup_reg_dealloc<C::producer_regs>();
    if(tid==0) {
      int kt=C::k_tiles?C::k_tiles:z.kh/C::bk;
      const uint64_t ah=z.kh>=8192?uint64_t(TMA::CacheHintSm90::EVICT_LAST):0;
      const uint64_t bh=z.kh>=8192?uint64_t(TMA::CacheHintSm90::EVICT_FIRST):0;
      #pragma unroll (C::k_tiles==1?7:1)
      for(int step=0;step<7;++step) {
        #pragma unroll (C::k_tiles?C::k_tiles:1)
        for(int q=0;q<kt;++q) {
          int p=product_for_step(step),seq=step*kt+q,slot=seq%C::stages,k0=q*C::bk;
          if(seq==(C::padded?0:2*kt))cudaGridDependencySynchronize();
          if(seq>=C::stages)Barrier::wait(&s.free[slot],((seq-C::stages)/C::stages)%2);
          TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*(64*C::groups+C::n)*C::bk);
          T* dst=s.operand[slot];
          if(p==3 || p>=4) {
            if constexpr(C::groups==1)SM90_TMA_LOAD_3D::copy(&tmap,&s.ready[slot],bh,dst,m0,k0,p==3?3:p-4);
            else SM90_TMA_LOAD_4D::copy(&tmap,&s.ready[slot],bh,dst,0,k0,m0/64,p==3?3:p-4);
          } else {
            int col=p==2?z.nh:0,row=p==1 || p==2?z.kh:0;
            if constexpr(C::groups==1)SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],bh,dst,m0+col,k0+row);
            else SM90_TMA_LOAD_3D::copy(&bmap,&s.ready[slot],bh,dst,0,k0+row,(m0+col)/64);
          }
          if(p==2 || p>=4)SM90_TMA_LOAD_4D::copy(&smap,&s.ready[slot],ah,dst+64*C::groups*C::bk,0,n0,k0/64,p==2?3:p-4);
          else {int col=p==1 || p==3?z.kh:0,row=p==3?z.mh:0;SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],ah,dst+64*C::groups*C::bk,0,n0+row,(k0+col)/64);}
        }
      }
    }
  } else {
    if constexpr(C::consumer_regs)cutlass::arch::warpgroup_reg_alloc<C::consumer_regs>();
    consume<T,C>(s,C::groups==1?0:(tid-128)/128,m0,n0,z,D,M,N);
  }
}
void encode(CUtensorMap& map,const void* ptr,DType type,bool is_a,bool transformed,Shape z,const Configuration& c) {
  auto dtype=type==DType::BF16?CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  uint64_t dims[4],strides[3];uint32_t box[4],es[4]={1,1,1,1};int rank=transformed?4:3;
  if(is_a) {
    dims[0]=64;dims[1]=transformed?z.mh:2*z.mh;dims[2]=(transformed?z.kh:2*z.kh)/64;dims[3]=4;
    strides[0]=uint64_t(transformed?z.kh:2*z.kh)*2;strides[1]=128;strides[2]=uint64_t(z.mh)*z.kh*2;
    box[0]=64;box[1]=c.n;box[2]=c.bk/64;box[3]=1;
  } else if(c.groups==1) {
    dims[0]=transformed?z.nh:2*z.nh;dims[1]=transformed?z.kh:2*z.kh;dims[2]=4;
    strides[0]=dims[0]*2;strides[1]=dims[0]*dims[1]*2;
    box[0]=64;box[1]=c.bk;box[2]=1;rank=transformed?3:2;
  } else {
    dims[0]=64;dims[1]=transformed?z.kh:2*z.kh;dims[2]=(transformed?z.nh:2*z.nh)/64;dims[3]=4;
    strides[0]=uint64_t(transformed?z.nh:2*z.nh)*2;strides[1]=128;strides[2]=uint64_t(z.nh)*z.kh*2;
    box[0]=64;box[1]=c.bk;box[2]=c.groups;box[3]=1;
  }
  auto e=cuTensorMapEncodeTiled(&map,dtype,rank,const_cast<void*>(ptr),dims,strides,box,es,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(e!=CUDA_SUCCESS)throw std::runtime_error("Tensor Core SW tensor map: "+std::to_string(int(e)));
}
template<class T,class C> SWResources action(const Problem& p,int id,const void* A,const void* B,float* D,void* ws,cudaStream_t stream,bool init) {
  constexpr size_t smem=sizeof(Shared<T,C>);
  if(init) {
    check(cudaFuncSetAttribute(transform<T,C>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
    check(cudaFuncSetAttribute(compute<T,C>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    check(cudaFuncSetAttribute(compute<T,C>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
    cudaFuncAttributes attr{};int resident=0;
    check(cudaFuncGetAttributes(&attr,compute<T,C>));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,compute<T,C>,C::threads,smem));
    return {smem+attr.sharedSizeBytes,attr.numRegs,resident};
  }
  auto c=configurations[id];auto z=shape(p,c);
  const void* rawA=C::padded?static_cast<char*>(ws)+8*(z.sa+z.sb):A;
  const void* rawB=C::padded?static_cast<const char*>(rawA)+8*z.sa:B;
  CUtensorMap amap{},bmap{},smap{},tmap{};
  encode(amap,rawA,p.dtype,true,false,z,c);encode(bmap,rawB,p.dtype,false,false,z,c);
  encode(smap,ws,p.dtype,true,true,z,c);encode(tmap,static_cast<char*>(ws)+8*z.sa,p.dtype,false,true,z,c);
  constexpr int vector=C::pack_vector;
  constexpr int sides=C::split_pack?2:1;
  int blocks=int(std::min<size_t>(C::pack_cap/sides,(std::max(z.sa,z.sb)+vector*C::pack_threads-1)/(vector*C::pack_threads)));
  transform<T,C><<<dim3(blocks,sides),C::pack_threads,0,stream>>>(static_cast<const T*>(A),static_cast<const T*>(B),static_cast<T*>(ws),p.m,p.n,p.k,z);
  check(cudaPeekAtLastError());
  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3((z.mh+C::n-1)/C::n,(z.nh+64*C::groups-1)/(64*C::groups));
  cfg.blockDim=dim3(C::threads);cfg.dynamicSmemBytes=smem;cfg.stream=stream;cfg.attrs=&attr;cfg.numAttrs=1;
  check(cudaLaunchKernelEx(&cfg,compute<T,C>,amap,bmap,smap,tmap,z,D,int(p.m),int(p.n)));
  return {};
}
template<size_t I=0> SWResources dispatch(const Problem& p,int id,const void* A,const void* B,float* D,void* ws,cudaStream_t stream,bool init) {
  if constexpr(I<configuration_count) {
    if(id==int(I)) {
      using C=std::tuple_element_t<I,KernelConfigurations>;
      if(p.dtype==DType::BF16)return action<bfloat16_t,C>(p,id,A,B,D,ws,stream,init);
      return action<half_t,C>(p,id,A,B,D,ws,stream,init);
    }
    return dispatch<I+1>(p,id,A,B,D,ws,stream,init);
  } else throw std::invalid_argument("unknown Tensor Core SW configuration");
}
}
bool eligible(const Problem& p,int id) {
  if(id<0 || id>=configuration_count || p.dtype==DType::FP32 || p.m<=0 || p.n<=0 || p.k<=0 ||
     std::max({p.m,p.n,p.k})>32768 || p.trans_a || p.trans_b || p.lda!=p.k || p.ldb!=p.n || p.ldd!=p.n || p.alpha!=1 || p.beta!=0)return false;
  auto c=configurations[id];
  if(c.padded)return !c.k_tiles || shape(p,c).kh==c.k_tiles*c.bk;
  if(p.m%16 || (!(c.n&(c.n-1)) && p.m%(2*c.n)) || p.n%(128*c.groups) || p.k%(2*c.bk))return false;
  return !c.k_tiles || p.k/2==c.k_tiles*c.bk;
}
size_t workspace_size(const Problem& p,int id) {return eligible(p,id)?shape(p,configurations[id]).bytes:0;}
SWResources setup(int id) {
  Problem p;p.dtype=DType::BF16;auto b=dispatch(p,id,nullptr,nullptr,nullptr,nullptr,nullptr,true);
  p.dtype=DType::FP16;auto h=dispatch(p,id,nullptr,nullptr,nullptr,nullptr,nullptr,true);
  return {std::max(b.shared,h.shared),std::max(b.registers,h.registers),std::min(b.resident,h.resident)};
}
void run(const Problem& p,int id,const void* A,const void* B,float* D,void* ws,cudaStream_t stream) {
  if(!eligible(p,id))throw std::invalid_argument("ineligible Tensor Core SW configuration");
  dispatch(p,id,A,B,D,ws,stream,false);
}
std::string name(int id) {
  const auto& c=configurations.at(id);
  return "sw_hopper;template=unified;leaf_terms=7;tile="+std::to_string(c.n)+"x"+std::to_string(64*c.groups)+
    ";bk="+std::to_string(c.bk)+";stages="+std::to_string(c.stages)+";groups="+std::to_string(c.groups)+
    ";shared_u3_components="+std::to_string(c.u3_components)+";pending="+std::to_string(c.pending)+";k_tiles="+std::to_string(c.k_tiles)+
    ";padded="+std::to_string(c.padded)+";pack_threads="+std::to_string(c.pack_threads)+
    ";pack_vector="+std::to_string(c.pack_vector)+";pack_cap="+std::to_string(c.pack_cap)+
    ";split_pack="+std::to_string(c.split_pack)+";raster="+std::to_string(c.raster);
}
}
namespace megafmm::detail {
size_t sw_hopper_workspace(const Problem& p){return sw_hopper::workspace_size(p,0);}
void setup_sw_hopper(const Problem&){sw_hopper::setup(0);}
void run_sw_hopper(const Problem& p,const void* A,const void* B,float* D,void* ws,cudaStream_t stream){sw_hopper::run(p,0,A,B,D,ws,stream);}
const char* sw_hopper_config(const Problem&){return "sw_hopper;template=unified;tile=64x64;bk=128;stages=4;leaf_terms=7;padded=1";}
std::vector<SWConfig> sw_configs(const Problem& p) {
  std::vector<SWConfig> out;
  for(const auto& c:sw_hopper::configurations)if(sw_hopper::eligible(p,c.id))out.push_back({c.id,c.n,c.bk,c.stages,c.pack_cap,c.raster,c.padded,c.groups});
  return out;
}
size_t sw_config_workspace(const Problem& p,const SWConfig& c){return sw_hopper::workspace_size(p,c.id);}
SWResources setup_sw_config(const Problem&,const SWConfig& c){return sw_hopper::setup(c.id);}
void run_sw_config(const Problem& p,const SWConfig& c,const void* A,const void* B,float* D,void* ws,cudaStream_t stream){sw_hopper::run(p,c.id,A,B,D,ws,stream);}
std::string sw_config_name(const SWConfig& c){return sw_hopper::name(c.id);}
}

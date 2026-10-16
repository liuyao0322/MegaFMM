"""Generate P55 in isolation from frozen P39/P51; never touch working sources."""
from pathlib import Path
import argparse
import hashlib
import json

h = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
prefix = (h/'p39/pack.cuh').read_text().split('template<class T,bool SideA,bool Balance,bool FastBalance=false>')[0]
prefix = prefix.replace('(9*32+9*64+1024)', '(9*64+9*64+1024)').replace('p+=9*32*sizeof(int)', 'p+=9*64*sizeof(int)')
exponent = (h/'quant_exponent_proposal.cuh').read_text().replace('#pragma once\n', '')
pack = prefix+exponent+r'''
constexpr int PackThreads=768,PackBlocks=128;
struct alignas(128) DirectPackShared {
 union {
  struct {float ma[64],mb[8];} inside;
  struct {float maxima[5][24];int exponent[5];int8_t transpose[9][512*12];} pack;
 };
};
static_assert(sizeof(DirectPackShared)==55808);
static_assert(WorkBytes==12329472);
template<bool SideA> using PairPackShared=DirectPackShared;
static __device__ __noinline__ float pair_general_pow2(float x,int e){return ldexpf(x,e);}
__device__ __forceinline__ float pair_pow2(float x,int e){
 if(e>=-126&&e<=127)return __fmul_rn(x,__int_as_float((e+127)<<23));
 return pair_general_pow2(x,e);
}
struct Tuple {float a,b,c,d,u,v,w,z;};
template<class T,bool SideA,bool Balance> __device__ __forceinline__ Tuple tuple(
 const T* X,const int* ex,int r,int c){
 int at=r*1024+c;
 Tuple x{float(X[at]),float(X[at+512]),float(X[at+512*1024]),float(X[at+512*1024+512]),0,0,0,0};
 if constexpr(Balance){
  int e0=SideA?-ex[c]:ex[r],e1=SideA?-ex[c+512]:ex[r+512];
  x.a=pair_pow2(x.a,e0);x.d=pair_pow2(x.d,e1);
  if constexpr(SideA){x.b=pair_pow2(x.b,e1);x.c=pair_pow2(x.c,e0);}
  else{x.b=pair_pow2(x.b,e0);x.c=pair_pow2(x.c,e1);}
 }
 if constexpr(SideA){x.u=__fadd_rn(x.c,x.d);x.v=__fsub_rn(x.u,x.a);x.w=__fsub_rn(x.a,x.c);x.z=__fsub_rn(x.b,x.v);}
 else{x.u=__fsub_rn(x.b,x.a);x.v=__fsub_rn(x.d,x.u);x.w=__fsub_rn(x.d,x.b);x.z=__fsub_rn(x.v,x.c);}
 return x;
}
template<class T,bool SideA,int QP> __device__ __forceinline__ void quant_store(
 float v,int i,int r,int c,T* high,int8_t* q,DirectPackShared& s){
 constexpr int Bits=std::is_same_v<T,cute::half_t> ? 11:8;
 float hv=float(T(v));int eh=s.pack.exponent[QP<4?QP:4];
 int qh=max(-127,min(127,__float2int_rn(pair_pow2(hv,-eh))));
 int target=SideA?r*512+c:c*512+r;
 if constexpr(SideA)q[QP*Plane+target]=int8_t(qh);
 else s.pack.transpose[QP][(i/8)*12+i%8]=int8_t(qh);
 if constexpr(QP<4){
  int ql=max(-127,min(127,__float2int_rn(pair_pow2(__fsub_rn(v,hv),Bits-eh))));
  if constexpr(SideA)q[(QP+4)*Plane+target]=int8_t(ql);
  else s.pack.transpose[QP+4][(i/8)*12+i%8]=int8_t(ql);
 }
}
template<class T,bool SideA,bool Balance> __device__ __forceinline__ void direct_pack(
 const T* X,T* high,int8_t* q,int* exponent,const int* inside,DirectPackShared& s,int task){
 int tid=threadIdx.x;float peak[5]={};
 for(int i=tid;i<4096;i+=PackThreads){
  int r=SideA?task*8+i/512:i/8,c=SideA?i%512:task*8+i%8,at=r*512+c;
  Tuple x=tuple<T,SideA,Balance>(X,inside,r,c);
  high[0*Plane+at]=T(x.a);high[1*Plane+at]=T(SideA?x.b:x.c);high[(SideA?3:2)*Plane+at]=T(x.d);
  high[4*Plane+at]=T(x.u);high[5*Plane+at]=T(x.v);high[6*Plane+at]=T(x.w);high[(SideA?2:3)*Plane+at]=T(x.z);
  peak[0]=fmaxf(peak[0],fabsf(float(T(x.u))));peak[1]=fmaxf(peak[1],fabsf(float(T(x.v))));
  peak[2]=fmaxf(peak[2],fabsf(float(T(x.w))));peak[3]=fmaxf(peak[3],fabsf(float(T(x.z))));peak[4]=fmaxf(peak[4],fabsf(float(T(x.d))));
 }
 #pragma unroll
 for(int p=0;p<5;++p){
  for(int delta=16;delta;delta/=2)peak[p]=fmaxf(peak[p],__shfl_xor_sync(0xffffffff,peak[p],delta));
  if((tid&31)==0)s.pack.maxima[p][tid/32]=peak[p];
 }
 __syncthreads();
 if(tid<5){
  float p=0;for(int w=0;w<24;++w)p=fmaxf(p,s.pack.maxima[tid][w]);
  int e=native_high_exponent(p);s.pack.exponent[tid]=e;
  constexpr int Bits=std::is_same_v<T,cute::half_t> ? 11:8;
  if(tid<4){exponent[tid*64+task]=e;exponent[(tid+4)*64+task]=e-Bits;}
  else exponent[8*64+task]=e;
 }
 __syncthreads();
 for(int i=tid;i<4096;i+=PackThreads){
  int r=SideA?task*8+i/512:i/8,c=SideA?i%512:task*8+i%8;
  Tuple x=tuple<T,SideA,Balance>(X,inside,r,c);
  quant_store<T,SideA,0>(x.u,i,r,c,high,q,s);quant_store<T,SideA,1>(x.v,i,r,c,high,q,s);
  quant_store<T,SideA,2>(x.w,i,r,c,high,q,s);quant_store<T,SideA,3>(x.z,i,r,c,high,q,s);
  quant_store<T,SideA,8>(x.d,i,r,c,high,q,s);
 }
 if constexpr(!SideA){
  __syncthreads();
  for(int i=tid;i<4096;i+=PackThreads){
   int c=i/512,r=i%512;
   #pragma unroll
   for(int p=0;p<9;++p)q[p*Plane+(task*8+c)*512+r]=s.pack.transpose[p][r*12+c];
  }
 }
 asm volatile("fence.proxy.async.global;" ::: "memory");
}
template<class T,bool SideA,bool Balance> __global__ __launch_bounds__(768,1)
void pack(const T* X,T* high,int8_t* q,int* exponent,const int* inside){
 extern __shared__ __align__(128) unsigned char buffer[];
 direct_pack<T,SideA,Balance>(X,high,q,exponent,inside,*reinterpret_cast<DirectPackShared*>(buffer),blockIdx.x);
}
template<class T> __global__ __launch_bounds__(768,1)
void preprocess_cooperative_columns(const T* A,const T* B,T* ha,T* hb,int8_t* qa,int8_t* qb,
 int* ea,int* eb,int* inside,float* partial){
 extern __shared__ __align__(128) unsigned char buffer[];
 auto& s=*reinterpret_cast<DirectPackShared*>(buffer);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,k0=blockIdx.x*8;
 if(tid<256){
  float am=0;
  #pragma unroll 4
  for(int row=tid/8;row<1024;row+=32)am=fmaxf(am,fabsf(float(A[row*1024+k0+tid%8])));
  am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,8));am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,16));
  if(lane<8)s.inside.ma[warp*8+lane]=am;
  float bm=0;
  #pragma unroll 4
  for(int col=lane;col<1024;col+=32)bm=fmaxf(bm,fabsf(float(B[(k0+warp)*1024+col])));
  for(int delta=16;delta;delta/=2)bm=fmaxf(bm,__shfl_xor_sync(0xffffffff,bm,delta));
  if(lane==0)s.inside.mb[warp]=bm;
 }
 __syncthreads();
 if(tid<8){
  float am=0,bm=s.inside.mb[tid];for(int w=0;w<8;++w)am=fmaxf(am,s.inside.ma[w*8+tid]);
  inside[k0+tid]=(am>0&&bm>0)?__float2int_rn(.5f*(log2f(am)-log2f(bm))):0;
 }
 cooperative_groups::this_grid().sync();
 if(blockIdx.x<64)direct_pack<T,false,true>(B,hb,qb,eb,inside,s,blockIdx.x);
 else direct_pack<T,true,true>(A,ha,qa,ea,inside,s,blockIdx.x-64);
}
template<class T> void init_cooperative_preprocess(){
 check(cudaFuncSetAttribute(preprocess_cooperative_columns<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(DirectPackShared)),"direct fine prep shared");
 int dev=0,supported=0,sms=0,active=0;check(cudaGetDevice(&dev),"direct device");
 check(cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,dev),"cooperative support");
 check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev),"SM count");
 check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,preprocess_cooperative_columns<T>,768,sizeof(DirectPackShared)),"direct fine prep capacity");
 printf("COOPERATIVE_INIT dtype=%s supported=%d sm_count=%d active_blocks_per_sm=%d grid=128 threads=768 shared_bytes=%zu\n",std::is_same_v<T,cute::half_t> ? "fp16":"bf16",supported,sms,active,sizeof(DirectPackShared));
 if(!supported||active*sms<128)throw std::runtime_error("Insufficient direct fine cooperative capacity");
}
template<class T,bool Balance> void preprocess(const T* A,const T* B,Work w,cudaStream_t stream){
 if constexpr(Balance){
  T* ha=static_cast<T*>(w.ha);T* hb=static_cast<T*>(w.hb);
  void* args[]={&A,&B,&ha,&hb,&w.qa,&w.qb,&w.ea,&w.eb,&w.inside,&w.partial};
  check(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(preprocess_cooperative_columns<T>),dim3(128),dim3(768),args,sizeof(DirectPackShared),stream),"direct fine cooperative prep");
 }else{
  pack<T,true,false><<<64,768,sizeof(DirectPackShared),stream>>>(A,static_cast<T*>(w.ha),w.qa,w.ea,w.inside);
  pack<T,false,false><<<64,768,sizeof(DirectPackShared),stream>>>(B,static_cast<T*>(w.hb),w.qb,w.eb,w.inside);
 }
 check(cudaPeekAtLastError(),"direct fine prep launch");
}
} // namespace int8_research
'''
consumer = (h/'p39/corrected.cuh').read_text()
consumer = consumer.replace('uint64_t ready[Stages],free[Stages];', 'uint64_t ready[Stages],free[Stages];int ae[9][4],be[9][8];')
consumer = consumer.replace('sizeof(Shared)==196736', 'sizeof(Shared)==197248')
start = consumer.index('void diagnose_uniform_scales(')
end = consumer.index('template<class T> __device__ __forceinline__ void float_mma', start)
consumer = consumer[:start]+r'''void diagnose_uniform_scales(const char* dtype,int pattern,int trial,Work w,cudaStream_t stream,int stream_id=0){
 std::vector<int> a(9*64),b(9*64);
 check(cudaMemcpyAsync(a.data(),w.ea,a.size()*4,cudaMemcpyDeviceToHost,stream),"fine A metadata diagnostic");
 check(cudaMemcpyAsync(b.data(),w.eb,b.size()*4,cudaMemcpyDeviceToHost,stream),"fine B metadata diagnostic");
 check(cudaStreamSynchronize(stream),"fine metadata diagnostic ready");
 int errors=0;int bits=std::string(dtype)=="fp16"?11:8;
 for(int p=0;p<4;++p)for(int g=0;g<64;++g){errors+=a[(p+4)*64+g]!=a[p*64+g]-bits;errors+=b[(p+4)*64+g]!=b[p*64+g]-bits;}
 printf("FINE_SCALE_METADATA dtype=%s pattern=%d trial=%d stream=%d exponents=1152 relative_errors=%d diagnostic_only=1\n",dtype,pattern,trial,stream_id,errors);
 if(errors)throw std::runtime_error("Invalid paired fine metadata");
}

'''+consumer[end:]
start = consumer.index(' // One correction/K metadata entry per lane')
end = consumer.index(' float c[4][16]={};', start)
consumer = consumer[:start]+consumer[end:]
start = consumer.index('  }else{\n   int owner=(step-7)*4;')
end = consumer.index('  // Product/term is uniform', start)
consumer = consumer[:start]+r'''  }else{
   uint32_t integer[16]={};
   #pragma unroll
   for(int j=0;j<16;++j)warpgroup_fence_operand(integer[j]);
   warpgroup_arrive();
   #pragma unroll 1
   for(int k=0;k<4;++k){
    int seq=step*4+k,slot=seq%Stages;
    Barrier::wait(&s.ready[slot],seq/Stages%2);asm volatile("" ::: "memory");
    uint64_t offset=slot*24576/16;
    #pragma unroll
    for(int inner=0;inner<4;++inner)
     integer_mma(uint64_t(daq(0,0,inner))+offset,uint64_t(dbq(0,0,inner))+offset,integer,(k==0&&inner==0)?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One);
    if(step>=9){
     #pragma unroll
     for(int inner=0;inner<4;++inner)
      integer_mma(uint64_t(daq(0,0,inner))+offset+8192/16,uint64_t(dbq(0,0,inner))+offset+4096/16,integer,GMMA::ScaleOut::One);
    }
    warpgroup_commit_batch();
    if(k>0){
     warpgroup_wait<1>();asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
     if((tid&31)==0)Barrier::arrive(&s.free[(seq-1)%Stages]);
    }
   }
   warpgroup_wait<0>();
   #pragma unroll
   for(int j=0;j<16;++j)warpgroup_fence_operand(integer[j]);
   asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
   if((tid&31)==0)Barrier::arrive(&s.free[(step*4+3)%Stages]);
   int term=first_term(step-7),ap=qa_plane(term),bp=qb_plane(term);
   #pragma unroll
   for(int j=0;j<16;++j){
    int mn=GMMA::CLayout_64x32{}(tid-128,j);
    int e=s.ae[ap][(mn/64)/8]+s.be[bp][(mn%64)/8];
    d[j]=scale_pow2(float(static_cast<int32_t>(integer[j])),e);
   }
  }
'''+consumer[end:]
needle=' int tid=threadIdx.x,m0=blockIdx.y*64,n0=blockIdx.x*32;'
assert consumer.count(needle) == 1
consumer = consumer.replace(needle, needle+r'''
 for(int i=tid;i<36;i+=256)s.ae[i/4][i%4]=ea[(i/4)*64+n0/8+i%4];
 for(int i=tid;i<72;i+=256)s.be[i/8][i%8]=eb[(i/8)*64+m0/8+i%8];''')
start = consumer.index('template<class T,bool Balance> void init_shared_packer(){')
end = consumer.index('void init(){', start)
consumer = consumer[:start]+r'''template<class T,bool Balance> void init_shared_packer(){
 check(cudaFuncSetAttribute(pack<T,true,Balance>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(DirectPackShared)),"direct fine standalone A");
 check(cudaFuncSetAttribute(pack<T,false,Balance>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(DirectPackShared)),"direct fine standalone B");
}
template<class T> void diagnose_consumer_capacity(){
 int dev=0,sms=0,active=0;check(cudaGetDevice(&dev),"consumer device");
 check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev),"consumer SM count");
 check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,corrected<T>,256,sizeof(Shared)),"fine consumer capacity");
 printf("FINE_CONSUMER_INIT dtype=%s sm_count=%d active_blocks_per_sm=%d grid=128 shared_bytes=%zu\n",std::is_same_v<T,cute::half_t> ? "fp16":"bf16",sms,active,sizeof(Shared));
 if(active<1)throw std::runtime_error("Fine consumer cannot reside");
}
'''+consumer[end:]
needle='void run(megafmm::DType dt,'
pos = consumer.index(needle)
head = consumer[:pos]
assert head.endswith('}\n')
head = head[:-2]+' diagnose_consumer_capacity<cute::half_t>();diagnose_consumer_capacity<cute::bfloat16_t>();\n}\n'
consumer = head+consumer[pos:]
compare = (h/'p39/compare.cu').read_text()
compare = compare.replace('<<<3*32,256,', '<<<64,768,').replace('<<<3*64,256,', '<<<64,768,')
for name, source in [('pack.cuh', pack), ('corrected.cuh', consumer), ('compare.cu', compare), ('pow2.cuh', (h/'p51/pow2.cuh').read_text())]:
    (args.output/name).write_text(source)
print(json.dumps({name: hashlib.sha256((args.output/name).read_bytes()).hexdigest()
                  for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']}, indent=2))

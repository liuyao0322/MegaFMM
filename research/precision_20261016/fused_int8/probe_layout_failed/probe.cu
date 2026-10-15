#define FUSED_PROBE
#include "prototype.cuh"
#include <vector>
#include <random>
#include <cstdio>
#include <cmath>
using namespace fused_int8;
struct Dump {uint8_t a[4][8192],b[4][16384],slot[49152];int e[4],inside[2];float result[3][2048];};
template<class T,int Prod> __global__ void probe(const __grid_constant__ CUtensorMap A,const __grid_constant__ CUtensorMap B,Dump* out){
 extern __shared__ __align__(1024) unsigned char buf[];auto& s=*reinterpret_cast<Shared*>(buf);int tid=threadIdx.x;
 if(tid==0){Barrier::init(&s.raw_ready,1);Barrier::init(&s.ready[0],1);cutlass::arch::fence_barrier_init();}__syncthreads();
 if(tid<128){
  if(tid==0){TxBarrier::arrive_and_expect_tx(&s.raw_ready,98304);for(int j=0;j<4;++j){
   cute::SM90_TMA_LOAD_3D::copy(&A,&s.raw_ready,0,s.a[j],0,(j/2)*512,(j%2)*8);
   cute::SM90_TMA_LOAD_2D::copy(&B,&s.raw_ready,0,s.b[j],(j%2)*512,(j/2)*512);
  }}
  Barrier::wait(&s.raw_ready,0);asm volatile("" ::: "memory");choose_inside<T>(s,tid);pack_leaf<T,Prod>(s,s.slot[0],tid);
  if(tid==0)Barrier::arrive(&s.ready[0]);
  for(int j=0;j<4;++j){for(int i=tid;i<8192;i+=128)out->a[j][i]=s.a[j][i];for(int i=tid;i<16384;i+=128)out->b[j][i]=s.b[j][i];}
  for(int i=tid;i<49152;i+=128)out->slot[i]=s.slot[0].bytes[i];
  if(tid==0){out->e[0]=s.slot[0].eah;out->e[1]=s.slot[0].eal;out->e[2]=s.slot[0].ebh;out->e[3]=s.slot[0].ebl;out->inside[0]=s.inside[0];out->inside[1]=s.inside[1];}
 }else{
  Barrier::wait(&s.ready[0],0);asm volatile("" ::: "memory");float d[16]={};
  native_product<T>(s.slot[0].bytes+16384,s.slot[0].bytes,d);
  #pragma unroll
  for(int j=0;j<16;++j){int mn=cute::GMMA::CLayout_64x32{}(tid-128,j);out->result[0][(mn/64)*64+mn%64]=d[j];}
  correction<T,Prod,0>(s.slot[0],d);
  #pragma unroll
  for(int j=0;j<16;++j){int mn=cute::GMMA::CLayout_64x32{}(tid-128,j);out->result[1][(mn/64)*64+mn%64]=d[j];}
  if constexpr(Prod!=2&&Prod!=3){correction<T,Prod,1>(s.slot[0],d);
   #pragma unroll
   for(int j=0;j<16;++j){int mn=cute::GMMA::CLayout_64x32{}(tid-128,j);out->result[2][(mn/64)*64+mn%64]=d[j];}
  }
 }
}
float transform_host(float* x,int p,bool a){
 if(a){float s1=x[2]+x[3],s2=s1-x[0];return p==3?x[3]:p==4?s1:p==5?s2:p==6?x[0]-x[2]:x[1]-s2;}
 float t1=x[1]-x[0],t2=x[3]-t1;return p==2?x[3]:p==4?t1:p==5?t2:p==6?x[3]-x[1]:t2-x[2];
}
int quant(float x,int e){return std::max(-127,std::min(127,int(std::nearbyint(std::ldexp(x,-e)))));}
template<class T,int Prod> int run_probe(){
 std::mt19937 g(7823);std::uniform_real_distribution<float> u(-1,1);std::vector<T>a(1024*1024),b(a.size());for(size_t i=0;i<a.size();++i){a[i]=T(u(g));b[i]=T(u(g));}
 T *da,*db;Dump* dd;check(cudaMalloc(&da,a.size()*2),"Aalloc");check(cudaMalloc(&db,b.size()*2),"Balloc");check(cudaMalloc(&dd,sizeof(Dump)),"dumpalloc");
 check(cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice),"Acopy");check(cudaMemcpy(db,b.data(),b.size()*2,cudaMemcpyHostToDevice),"Bcopy");
 CUtensorMap ma,mb;encode(ma,da,(std::is_same_v<T, cute::half_t>) ? megafmm::DType::FP16:megafmm::DType::BF16,true);encode(mb,db,(std::is_same_v<T, cute::half_t>) ? megafmm::DType::FP16:megafmm::DType::BF16,false);
 check(cudaFuncSetAttribute(probe<T,Prod>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)),"probe shared");
 probe<T,Prod><<<1,256,sizeof(Shared)>>>(ma,mb,dd);check(cudaDeviceSynchronize(),"probe run");Dump d;check(cudaMemcpy(&d,dd,sizeof(d),cudaMemcpyDeviceToHost),"dumpcopy");
 for(int pp=0;pp<4;++pp)for(int kk : {0,1,16,32,64}){
  printf("RAW p=%d k=%d Aoff=%d Boff=%d A=%g Awant=%g B=%g Bwant=%g\n",pp,kk,int(Traits<T>::ah()(0,kk)),int(Traits<T>::bh()(0,kk)),float(reinterpret_cast<T*>(d.a[pp])[Traits<T>::ah()(0,kk)]),float(a[((pp/2)*512)*1024+(pp%2)*512+kk]),float(reinterpret_cast<T*>(d.b[pp])[Traits<T>::bh()(0,kk)]),float(b[((pp/2)*512+kk)*1024+(pp%2)*512]));
 }
 int raw=0,high=0,qerror=0,exp=0,integer=0;std::vector<float> ah(4096),al(4096),bh(8192),bl(8192);float peaks[4]={};
 for(int j=0;j<4;++j){
  for(int r=0;r<32;++r)for(int k=0;k<128;++k)raw+=float(reinterpret_cast<T*>(d.a[j])[Traits<T>::ah()(r,k)])!=float(a[((j/2)*512+r)*1024+(j%2)*512+k]);
  for(int n=0;n<64;++n)for(int k=0;k<128;++k)raw+=float(reinterpret_cast<T*>(d.b[j])[Traits<T>::bh()(n,k)])!=float(b[((j/2)*512+k)*1024+(j%2)*512+n]);
 }
 for(int r=0;r<32;++r)for(int k=0;k<128;++k){float x[4];for(int j=0;j<4;++j)x[j]=std::ldexp(float(a[((j/2)*512+r)*1024+(j%2)*512+k]),-d.inside[j%2]);
  float v=transform_host(x,Prod,true),h=float(T(v)),l=v-h;ah[r*128+k]=h;al[r*128+k]=l;peaks[0]=fmaxf(peaks[0],fabsf(h));peaks[1]=fmaxf(peaks[1],fabsf(l));high+=h!=float(reinterpret_cast<T*>(d.slot+16384)[Traits<T>::ah()(r,k)]);
 }
 for(int n=0;n<64;++n)for(int k=0;k<128;++k){float x[4];for(int j=0;j<4;++j)x[j]=std::ldexp(float(b[((j/2)*512+k)*1024+(j%2)*512+n]),d.inside[j/2]);
  float v=transform_host(x,Prod,false),h=float(T(v)),l=v-h;bh[k*64+n]=h;bl[k*64+n]=l;peaks[2]=fmaxf(peaks[2],fabsf(h));peaks[3]=fmaxf(peaks[3],fabsf(l));high+=h!=float(reinterpret_cast<T*>(d.slot)[Traits<T>::bh()(n,k)]);
 }
 for(int j=0;j<4;++j)exp+=d.e[j]!=(peaks[j]>0?int(ceilf(log2f(peaks[j]/127))):0);
 for(int r=0;r<32;++r)for(int k=0;k<128;++k){int off=Traits<T>::aq()(r,k);qerror+=quant(ah[r*128+k],d.e[0])!=reinterpret_cast<int8_t*>(d.slot+45056)[off];qerror+=quant(al[r*128+k],d.e[1])!=reinterpret_cast<int8_t*>(d.slot+32768)[off];}
 for(int n=0;n<64;++n)for(int k=0;k<128;++k){int off=Traits<T>::bq()(n,k);qerror+=quant(bh[k*64+n],d.e[2])!=reinterpret_cast<int8_t*>(d.slot+24576)[off];qerror+=quant(bl[k*64+n],d.e[3])!=reinterpret_cast<int8_t*>(d.slot+36864)[off];}
 double e2=0,den=0;
 for(int r=0;r<32;++r)for(int n=0;n<64;++n){double v=0;for(int k=0;k<128;++k)v+=double(ah[r*128+k])*bh[k*64+n];double err=d.result[0][r*64+n]-v;e2+=err*err;den+=v*v;
  for(int pair=0;pair<(Prod==2||Prod==3?1:2);++pair){bool la=Prod==2||(Prod!=3&&pair==0);int iv=0;for(int k=0;k<128;++k)iv+=quant((la?al:ah)[r*128+k],d.e[la?1:0])*quant((la?bh:bl)[k*64+n],d.e[la?2:3]);float want=std::ldexp(float(iv),d.e[la?1:0]+d.e[la?2:3]);integer+=want!=d.result[1+pair][r*64+n];}
 }
 printf("PROBE dtype=%s product=%d raw_errors=%d high_errors=%d exponent_errors=%d quant_errors=%d integer_errors=%d native_ef=%.12g\n",(std::is_same_v<T, cute::half_t>) ? "fp16":"bf16",Prod,raw,high,exp,qerror,integer,sqrt(e2/den));fflush(stdout);
 cudaFree(da);cudaFree(db);cudaFree(dd);return raw+high+exp+qerror+integer;
}
int main(){try{return run_probe<cute::half_t,5>()+run_probe<cute::bfloat16_t,5>()?2:0;}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}

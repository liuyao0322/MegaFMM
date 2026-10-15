// Fresh alternating K-column imbalance; original rounded-input FP64 reference.
#define main original_sw_test_main
#include "../../../tests/test_sw1024.cu"
#undef main
#include "alternating_p9/corrected.cuh"
#include "alternating_p10/corrected.cuh"

template<class T> __global__ void alternating_fill(T* a,T* b,double* ad,double* bd,int seed){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Count;i+=gridDim.x*blockDim.x){
  int r=i/N,c=i%N;
  float x=float(hash(uint64_t(i)^seed)>>40)*(2.f/16777216.f)-1;
  float y=float(hash(uint64_t(i)^seed^0x632be59bd9b4e019ULL)>>40)*(2.f/16777216.f)-1;
  bool af=((c%2)==1)^(c>=512),bf=((r%2)==1)^(r>=512);
  x=ldexpf(x,af?-12:0);y=ldexpf(y,bf?0:-12);
  T xx=rounded<T>(x),yy=rounded<T>(y);a[i]=xx;b[i]=yy;ad[i]=float(xx);bd[i]=float(yy);
 }
}

template<class CT> void run_p9(megafmm::DType dtype,void* A,void* B,float* D,void* workspace,cudaStream_t stream){
 namespace ns=alternating_p9;ns::Work w(workspace);CUtensorMap ha,hb,qa,qb;
 ns::encode(ha,w.ha,dtype,true,false);ns::encode(hb,w.hb,dtype,false,false);ns::encode(qa,w.qa,dtype,true,true);ns::encode(qb,w.qb,dtype,false,true);
 ns::preprocess<CT,true>(static_cast<const CT*>(A),static_cast<const CT*>(B),w,stream);
 ns::corrected<CT><<<dim3(16,8),256,sizeof(ns::Shared),stream>>>(ha,hb,qa,qb,w.ea,w.eb,D);
 check(cudaPeekAtLastError());
}
template<class CT> void run_p10(megafmm::DType dtype,void* A,void* B,float* D,void* workspace,cudaStream_t stream){
 namespace ns=alternating_p10;ns::Work w(workspace);CUtensorMap ha,hb,qa,qb;
 ns::encode(ha,w.ha,dtype,true,false);ns::encode(hb,w.hb,dtype,false,false);ns::encode(qa,w.qa,dtype,true,true);ns::encode(qb,w.qb,dtype,false,true);
 ns::preprocess<CT,true>(static_cast<const CT*>(A),static_cast<const CT*>(B),w,stream);
 ns::corrected<CT><<<dim3(16,8),256,sizeof(ns::Shared),stream>>>(ha,hb,qa,qb,w.ea,w.eb,D);
 check(cudaPeekAtLastError());
}

template<class T,class CT> void compare(megafmm::DType dtype){
 Stream stream;Blas blas(stream.s);Buffer a(Count*2),b(Count*2),ad(Count*8),bd(Count*8),ref(Count*8),denseout(Count*4),out9(Count*4),out10(Count*4),scratch(128*sizeof(Metric));
 Buffer w9(alternating_p9::WorkBytes),w10(alternating_p10::WorkBytes);
 megafmm::Problem p;p.m=p.n=p.k=N;p.dtype=dtype;megafmm::Policy policy;policy.algorithm=megafmm::Algorithm::Dense;
 auto dense=megafmm::create_plan(p,policy);Buffer dw(dense->workspace_bytes());
 const char* dt=dtype==megafmm::DType::BF16?"bf16":"fp16";
 for(int trial=0;trial<3;++trial){
  int seed=2801+trial*271;alternating_fill<<<512,256,0,stream.s>>>(a.as<T>(),b.as<T>(),ad.as<double>(),bd.as<double>(),seed);
  if(trial==0)megafmm::tune(*dense,a.p,b.p,nullptr,denseout.as<float>(),dw.p,stream.s);
  double one=1,zero=0;check(cublasDgemm(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&one,bd.as<double>(),N,ad.as<double>(),N,&zero,ref.as<double>(),N));
  megafmm::run(*dense,a.p,b.p,nullptr,denseout.as<float>(),dw.p,stream.s);
  run_p9<CT>(dtype,a.p,b.p,out9.as<float>(),w9.p,stream.s);run_p10<CT>(dtype,a.p,b.p,out10.as<float>(),w10.p,stream.s);
  auto dm=metric(denseout.as<float>(),ref.as<double>(),scratch,stream.s);double gate=3*ef(dm)+(dtype==megafmm::DType::BF16?1e-3:1e-4);
  for(int variant=0;variant<2;++variant){auto cm=metric((variant?out10:out9).as<float>(),ref.as<double>(),scratch,stream.s);
   printf("ALTERNATING candidate=%s dtype=%s trial=%d seed=%d dense_ef=%.12g corrected_ef=%.12g gate=%.12g nonfinite=%llu pass=%d\n",variant?"p10":"p9",dt,trial,seed,ef(dm),ef(cm),gate,cm.nonfinite,int(!cm.nonfinite&&!dm.nonfinite&&ef(cm)<=gate));fflush(stdout);
  }
 }
}
int main(){try{
 alternating_p9::init();alternating_p10::init();
 compare<__nv_bfloat16,cute::bfloat16_t>(megafmm::DType::BF16);compare<__half,cute::half_t>(megafmm::DType::FP16);return 0;
}catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}

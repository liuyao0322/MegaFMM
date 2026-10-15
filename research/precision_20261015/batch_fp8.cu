// Prequantized eight-leaf correction GEMM timing only, NOT full-call latency.
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#define CU(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define BL(x) do{auto e=(x);if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"BLAS %d line %d\n",int(e),__LINE__);exit(1);}}while(0)
constexpr int N=512,Batch=8,P=N*N;
__global__ void init(__nv_fp8_e4m3* a,__nv_fp8_e4m3* b){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<Batch*P){a[i]=__nv_fp8_e4m3(float((i*17+3)%127-63)/64);b[i]=__nv_fp8_e4m3(float((i*31+7)%127-63)/64);}
}
struct Plan{
 cublasLtMatmulDesc_t desc;
 cublasLtMatrixLayout_t a,b,c;
 cublasLtMatmulAlgo_t algo;
 int count;
};
int main(){
 cublasLtHandle_t handle;BL(cublasLtCreate(&handle));
 __nv_fp8_e4m3 *a,*b;float *c,*scale;void*workspace;
 CU(cudaMalloc(&a,Batch*P));CU(cudaMalloc(&b,Batch*P));CU(cudaMalloc(&c,Batch*P*sizeof(float)));
 CU(cudaMalloc(&scale,sizeof(float)));CU(cudaMalloc(&workspace,32<<20));
 float one=1,zero=0;CU(cudaMemcpy(scale,&one,4,cudaMemcpyHostToDevice));
 init<<<(Batch*P+255)/256,256>>>(a,b);CU(cudaDeviceSynchronize());
 cudaEvent_t start,end;CU(cudaEventCreate(&start));CU(cudaEventCreate(&end));
 cudaStream_t stream;CU(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 for(int count:{1,8}){
  Plan p{};p.count=count;BL(cublasLtMatmulDescCreate(&p.desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t trans=CUBLAS_OP_T;BL(cublasLtMatmulDescSetAttribute(p.desc,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
  BL(cublasLtMatmulDescSetAttribute(p.desc,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&scale,sizeof(scale)));
  BL(cublasLtMatmulDescSetAttribute(p.desc,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&scale,sizeof(scale)));
  BL(cublasLtMatrixLayoutCreate(&p.a,CUDA_R_8F_E4M3,N,N,N));BL(cublasLtMatrixLayoutCreate(&p.b,CUDA_R_8F_E4M3,N,N,N));
  BL(cublasLtMatrixLayoutCreate(&p.c,CUDA_R_32F,N,N,N));
  int64_t stride=P;
  for(auto layout:{p.a,p.b,p.c}){
   BL(cublasLtMatrixLayoutSetAttribute(layout,CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&count,sizeof(count)));
   BL(cublasLtMatrixLayoutSetAttribute(layout,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&stride,sizeof(stride)));
  }
  cublasLtMatmulPreference_t pref;BL(cublasLtMatmulPreferenceCreate(&pref));size_t bytes=32<<20;
  BL(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
  cublasLtMatmulHeuristicResult_t results[32];int returned=0;
  BL(cublasLtMatmulAlgoGetHeuristic(handle,p.desc,p.a,p.b,p.c,p.c,pref,32,results,&returned));
  printf("batch_count=%d candidates=%d\n",count,returned);fflush(stdout);
  if(!returned)return 2;
  auto run=[&](const cublasLtMatmulAlgo_t&algo){
   for(int i=0;i<Batch;i+=count)BL(cublasLtMatmul(handle,p.desc,&one,a+i*P,p.a,b+i*P,p.b,&zero,c+i*P,p.c,c+i*P,p.c,&algo,workspace,bytes,stream));
  };
  float best=1e9;
  for(int j=0;j<returned;++j){
   if(results[j].state!=CUBLAS_STATUS_SUCCESS)continue;
   run(results[j].algo);CU(cudaEventRecord(start,stream));for(int i=0;i<10;++i)run(results[j].algo);
   CU(cudaEventRecord(end,stream));CU(cudaEventSynchronize(end));float ms;CU(cudaEventElapsedTime(&ms,start,end));
   if(ms<best){best=ms;p.algo=results[j].algo;}
  }
  cudaGraph_t graph;cudaGraphExec_t exec;CU(cudaStreamSynchronize(stream));CU(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));run(p.algo);
  CU(cudaStreamEndCapture(stream,&graph));CU(cudaGraphInstantiate(&exec,graph,0));
  for(int i=0;i<20;++i)CU(cudaGraphLaunch(exec,stream));
  std::vector<float> times;
  for(int group=0;group<5;++group){
   CU(cudaEventRecord(start,stream));for(int i=0;i<100;++i)CU(cudaGraphLaunch(exec,stream));CU(cudaEventRecord(end,stream));
   CU(cudaEventSynchronize(end));float ms;CU(cudaEventElapsedTime(&ms,start,end));times.push_back(ms*10);
  }
  std::sort(times.begin(),times.end());printf("batch_count=%d eight_products_graph_median_us=%.6f\n",count,times[2]);
  // Sample all batches against exact FP64 dots of the same quantized FP8 values.
  std::vector<__nv_fp8_e4m3> ha(Batch*P),hb(Batch*P);std::vector<float>hc(Batch*P);
  CU(cudaMemcpy(ha.data(),a,Batch*P,cudaMemcpyDeviceToHost));CU(cudaMemcpy(hb.data(),b,Batch*P,cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(hc.data(),c,Batch*P*sizeof(float),cudaMemcpyDeviceToHost));double maxerr=0;
  for(int batch=0;batch<Batch;++batch)for(int sample=0;sample<16;++sample){
   int row=(sample*31)%N,col=(sample*17)%N;double ref=0;
   for(int k=0;k<N;++k)ref+=double(float(ha[batch*P+k+row*N]))*double(float(hb[batch*P+k+col*N]));
   maxerr=std::max(maxerr,std::abs(ref-hc[batch*P+row+col*N]));
  }
  printf("batch_count=%d sampled_maxabs=%.9g\n",count,maxerr);fflush(stdout);if(maxerr>1e-3)return 3;
  CU(cudaGraphExecDestroy(exec));CU(cudaGraphDestroy(graph));BL(cublasLtMatmulPreferenceDestroy(pref));
  BL(cublasLtMatrixLayoutDestroy(p.a));BL(cublasLtMatrixLayoutDestroy(p.b));BL(cublasLtMatrixLayoutDestroy(p.c));BL(cublasLtMatmulDescDestroy(p.desc));
 }
 return 0;
}

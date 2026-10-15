"""P43: independent native main/eight BF16 correction dots from frozen P39.

Unscaled residuals isolate the representation change; no performance/precision
claim until the generated hardware is measured. Four K128 slots, 49,152B each.
"""
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p39'
s=(base/'pack.cuh').read_text()
s=s.replace('Stages=8','Stages=4').replace('QuantBytes=9*Plane;','QuantBytes=9*Plane*2;')
s=s.replace('// Seven main planes per side, nine INT8 planes per side, integer scale metadata.',
 '// Seven native main planes and nine BF16 correction planes per side.\n// Historical integer metadata remains allocated but is neither read nor written.')
s=s.replace('int8_t *qa,*qb','cute::bfloat16_t *qa,*qb').replace('reinterpret_cast<int8_t*>(p)','reinterpret_cast<cute::bfloat16_t*>(p)')
# Remove the old independent scalar packer, not instantiated by this candidate.
start=s.index('template<class T,bool SideA,bool Balance,int PlaneId>')
end=s.index('template<class T,bool SideA,bool Balance,int Group>')
new='''// Paired FP32 transforms remain shared; B correction planes are physically
// transposed, giving K-major BF16 operands on both sides of correction MMA.
// A 34-halfword padded row is 17 words, coprime with the 32 shared banks.
template<bool SideA> struct PairPackShared {
 float values[2][SideA?8192:4096];
 cute::bfloat16_t transpose[SideA?1:2][SideA?1:128*34];
};
static __device__ __noinline__ float pair_general_pow2(float x,int e){return ldexpf(x,e);}
__device__ __forceinline__ float pair_pow2(float x,int e){
 if(e>=-126&&e<=127)return __fmul_rn(x,__int_as_float((e+127)<<23));
 return pair_general_pow2(x,e);
}

template<class T,bool SideA,int NativePlane,int QP> __device__ __forceinline__
void pair_store(int which,T* high,cute::bfloat16_t* q,int* exp,
 PairPackShared<SideA>& s,int tile,int r0,int c0){
 constexpr int Rows=SideA?64:128,Cols=SideA?128:32,Size=Rows*Cols,PT=Size/256;
 int tid=threadIdx.x;
 #pragma unroll 4
 for(int j=0;j<PT;++j){
  int i=tid+256*j,at=(r0+i/Cols)*512+c0+i%Cols;
  float v=s.values[which][i];T native=T(v);float high_value=float(native);
  high[NativePlane*Plane+at]=native;
  cute::bfloat16_t qh=cute::bfloat16_t(high_value);
  cute::bfloat16_t ql=cute::bfloat16_t(__fsub_rn(v,high_value));
  if constexpr(SideA){
   q[QP*Plane+at]=qh;if constexpr(QP<4)q[(QP+4)*Plane+at]=ql;
  }else{
   int shared_at=(i/Cols)*34+i%Cols;
   s.transpose[0][shared_at]=qh;if constexpr(QP<4)s.transpose[1][shared_at]=ql;
  }
 }
 if constexpr(!SideA){
  // Publish every warp's values before other warps read transposed rows.
  __syncthreads();
  for(int i=tid;i<Size;i+=256){
   int c=i/Rows,r=i%Rows;
   q[QP*Plane+(c0+c)*512+r0+r]=s.transpose[0][r*34+c];
   if constexpr(QP<4)q[(QP+4)*Plane+(c0+c)*512+r0+r]=s.transpose[1][r*34+c];
  }
  // The next transformed plane reuses the two transpose buffers.
  __syncthreads();
 }
}
template<class T,bool SideA,int NP0,int QP0,int NP1,int QP1>
__device__ __forceinline__ void pair_store_both(T* high,cute::bfloat16_t* q,int* exp,
 PairPackShared<SideA>& s,int tile,int r0,int c0){
 pair_store<T,SideA,NP0,QP0>(0,high,q,exp,s,tile,r0,c0);
 pair_store<T,SideA,NP1,QP1>(1,high,q,exp,s,tile,r0,c0);
}

'''
s=s[:start]+new+s[end:]
s=s.replace('int8_t*','cute::bfloat16_t*')
(h/'pack.cuh').write_text(s)
s=(base/'corrected.cuh').read_text()
s=s.replace('operand[Stages][24576]','operand[Stages][49152]').replace('using type=int8_t','using type=cute::bfloat16_t')
start=s.index('// Five correction groups')
end=s.index('template<class T> __device__ __forceinline__ void float_mma')
s=s[:start]+'''// Five groups retain all eight original cross terms, now in BF16.
__host__ __device__ constexpr int first_term(int group){return group<2?group:2+2*(group-2);}
// No adaptive quantization exponents exist in this representation. This host-only
// diagnostic stays outside every timing/capture arm and never reads old metadata.
void diagnose_uniform_scales(const char* dtype,int pattern,int trial,Work w,cudaStream_t stream,int stream_id=0){
 printf("BF16_CORRECTION dtype=%s pattern=%d trial=%d stream=%d low_scale_power=0 diagnostic_only=1\\n",dtype,pattern,trial,stream_id);
}

'''+s[end:]
start=s.index('__device__ __forceinline__ void integer_mma')
end=s.index('template<class T> __device__ __forceinline__ void consume')
s=s[:start]+'''__device__ __forceinline__ void correction_mma(uint64_t a,uint64_t b,float* d,cute::GMMA::ScaleOut scale){
 using namespace cute;
 SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>::fma(a,b,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],scale);
}

'''+s[end:]
s=s.replace('s.operand[0]+16384)),lbq','s.operand[0]+32768)),lbq')
s=s.replace('tiled_divide(aq,Shape<_64,_32>{})','tiled_divide(aq,Shape<_64,_16>{})').replace('tiled_divide(bq,Shape<_32,_32>{})','tiled_divide(bq,Shape<_32,_16>{})')
start=s.index(' // One correction/K metadata entry')
end=s.index('  // Product/term is uniform')
new=''' float c[4][16]={};
 #pragma unroll 1
 for(int step=0;step<12;++step){
  float d[16]={};
  #pragma unroll
  for(int j=0;j<16;++j)warpgroup_fence_operand(d[j]);
  warpgroup_arrive();
  #pragma unroll 1
  for(int k=0;k<4;++k){
   int seq=step*4+k,slot=seq%Stages;
   Barrier::wait(&s.ready[slot],seq/Stages%2);asm volatile("" ::: "memory");
   uint64_t offset=slot*49152/16;
   if(step<7){
    #pragma unroll
    for(int inner=0;inner<8;++inner)
     float_mma<T>(uint64_t(dah(0,0,inner))+offset,uint64_t(dbh(0,0,inner))+offset,d,(k==0&&inner==0)?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One);
   }else{
    #pragma unroll
    for(int inner=0;inner<8;++inner)
     correction_mma(uint64_t(daq(0,0,inner))+offset,uint64_t(dbq(0,0,inner))+offset,d,(k==0&&inner==0)?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One);
    if(step>=9){
     #pragma unroll
     for(int inner=0;inner<8;++inner)
      correction_mma(uint64_t(daq(0,0,inner))+offset+16384/16,uint64_t(dbq(0,0,inner))+offset+8192/16,d,GMMA::ScaleOut::One);
    }
   }
   warpgroup_commit_batch();
   if(k>0){
    // After wait<1>, the preceding batch no longer references its operand slot.
    warpgroup_wait<1>();asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
    if((tid&31)==0)Barrier::arrive(&s.free[(seq-1)%Stages]);
   }
  }
  warpgroup_wait<0>();
  #pragma unroll
  for(int j=0;j<16;++j)warpgroup_fence_operand(d[j]);
  asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
  if((tid&31)==0)Barrier::arrive(&s.free[(step*4+3)%Stages]);
'''
s=s[:start]+new+s[end:]
s=s.replace('TxBarrier::arrive_and_expect_tx(&s.ready[slot],(step<7||step>=9)?24576:12288);','TxBarrier::arrive_and_expect_tx(&s.ready[slot],step>=9?49152:24576);')
s=s.replace('s.operand[slot]+16384,k*128,n0,qa_plane(term)', 's.operand[slot]+32768,k*128,n0,qa_plane(term)')
s=s.replace('s.operand[slot]+8192,k*128,m0,qb_plane(term+1)', 's.operand[slot]+16384,k*128,m0,qb_plane(term+1)')
s=s.replace('s.operand[slot]+20480,k*128,n0,qa_plane(term+1)', 's.operand[slot]+40960,k*128,n0,qa_plane(term+1)')
s=s.replace('512ULL*(integer?1:2),uint64_t(Plane)*(integer?1:2)', '512ULL*2,uint64_t(Plane)*2')
s=s.replace('integer?CU_TENSOR_MAP_DATA_TYPE_UINT8:', 'integer?CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:')
# 16bit correction K128 requires a rank4 split into two contiguous K64 boxes.
# With 128B swizzle, boxDim[0]*sizeof(element) must not exceed 128.
s=s.replace('SM90_TMA_LOAD_3D::copy(&quant_b,&s.ready[slot],0,s.operand[slot],k*128,m0,qb_plane(term));',
 'SM90_TMA_LOAD_4D::copy(&quant_b,&s.ready[slot],0,s.operand[slot],0,m0,k*2,qb_plane(term));')
s=s.replace('SM90_TMA_LOAD_3D::copy(&quant_a,&s.ready[slot],0,s.operand[slot]+32768,k*128,n0,qa_plane(term));',
 'SM90_TMA_LOAD_4D::copy(&quant_a,&s.ready[slot],0,s.operand[slot]+32768,0,n0,k*2,qa_plane(term));')
s=s.replace('SM90_TMA_LOAD_3D::copy(&quant_b,&s.ready[slot],0,s.operand[slot]+16384,k*128,m0,qb_plane(term+1));',
 'SM90_TMA_LOAD_4D::copy(&quant_b,&s.ready[slot],0,s.operand[slot]+16384,0,m0,k*2,qb_plane(term+1));')
s=s.replace('SM90_TMA_LOAD_3D::copy(&quant_a,&s.ready[slot],0,s.operand[slot]+40960,k*128,n0,qa_plane(term+1));',
 'SM90_TMA_LOAD_4D::copy(&quant_a,&s.ready[slot],0,s.operand[slot]+40960,0,n0,k*2,qa_plane(term+1));')
anchor=' if(side_a&&!integer){rank=4;dims[0]=64;dims[1]=512;dims[2]=8;dims[3]=7;strides[0]=1024;strides[1]=128;strides[2]=Plane*2;box[2]=2;}'
assert s.count(anchor)==1
s=s.replace(anchor,anchor+'\n if(integer){rank=4;dims[0]=64;dims[1]=512;dims[2]=8;dims[3]=9;strides[0]=1024;strides[1]=128;strides[2]=Plane*2;box[0]=64;box[2]=2;}')
(h/'corrected.cuh').write_text(s)
for f in ('compare.cu','pow2.cuh'):(h/f).write_text((base/f).read_text())
print('generated P43 native main + eight unscaled BF16 correction dots, four49,152B slots; no adaptive quantizer/metadata reads')

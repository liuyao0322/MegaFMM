"""P45: alias native correction highs; four scaled native lows, eleven planes."""
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p44'
s=(base/'pack.cuh').read_text()
s=s.replace('HighBytes=7*Plane*2, QuantBytes=9*Plane*2','HighBytes=11*Plane*2, QuantBytes=0')
s=s.replace('// Seven native main planes and nine BF16 correction planes per side.','// Seven native main planes followed by four scaled native residual planes.')
s=s.replace('void *ha,*hb;cute::bfloat16_t *qa,*qb;', 'void *ha,*hb,*qa,*qb;')
s=s.replace('qa=reinterpret_cast<cute::bfloat16_t*>(p);p+=QuantBytes;qb=reinterpret_cast<cute::bfloat16_t*>(p);p+=QuantBytes;',
 'qa=static_cast<char*>(ha)+7*Plane*2;qb=static_cast<char*>(hb)+7*Plane*2;')
s=s.replace('cute::bfloat16_t*','T*')
s=s.replace('''  cute::bfloat16_t qh=cute::bfloat16_t(high_value);
  cute::bfloat16_t ql=cute::bfloat16_t(__fsub_rn(v,high_value));
  // BF16 supports MN-major B, so retain original row-major coordinates.
  q[QP*Plane+at]=qh;if constexpr(QP<4)q[(QP+4)*Plane+at]=ql;''',
 '''  // Correction highs already reside in their seven native main planes.
  // RN residual * 2^significand_bits stays finite whenever v/native are finite.
  if constexpr(QP<4){
   constexpr float scale=std::is_same_v<T,cute::half_t>?2048.f:256.f;
   q[QP*Plane+at]=T(__fmul_rn(__fsub_rn(v,high_value),scale));
  }''')
needle='T* ha=static_cast<T*>(w.ha);T* hb=static_cast<T*>(w.hb);'
assert s.count(needle)==1;s=s.replace(needle,needle+'T* qa=static_cast<T*>(w.qa);T* qb=static_cast<T*>(w.qb);')
s=s.replace('&ha,&hb,&w.qa,&w.qb,&w.ea', '&ha,&hb,&qa,&qb,&w.ea')
s=s.replace('w.qa,w.qb,w.ea,w.eb,w.inside);','static_cast<T*>(w.qa),static_cast<T*>(w.qb),w.ea,w.eb,w.inside);')
s=s.replace('// Paired FP32 transforms stay shared. BF16 correction B is MN-major and', '// Paired FP32 transforms stay shared. Native correction B is MN-major and')
(h/'pack.cuh').write_text(s)
s=(base/'corrected.cuh').read_text().replace('using type=cute::bfloat16_t','using type=T')
s=s.replace('return term<2?7+term:(term-2)/2+((term&1)?0:4);', 'return term<2?(term==0?10:3):(term-2)/2+((term&1)?4:7);')
s=s.replace('return term<2?8-term:(term-2)/2+((term&1)?4:0);', 'return term<2?(term==0?2:10):(term-2)/2+((term&1)?7:4);')
s=s.replace('// Five groups retain all eight original cross terms, now in BF16.', '// Five groups retain all eight original cross terms, in native T with scaled low.')
s=s.replace('printf("BF16_CORRECTION dtype=%s pattern=%d trial=%d stream=%d low_scale_power=0 diagnostic_only=1\\n",dtype,pattern,trial,stream_id);',
 'printf("NATIVE_CORRECTION dtype=%s pattern=%d trial=%d stream=%d low_scale_power=%d diagnostic_only=1\\n",dtype,pattern,trial,stream_id,std::string(dtype)=="fp16"?11:8);')
start=s.index('__device__ __forceinline__ void correction_mma(');end=s.index('template<class T> __device__ __forceinline__ void consume',start)
s=s[:start]+s[end:]
s=s.replace('correction_mma(', 'float_mma<T>(')
needle='  // Product/term is uniform across all16outputs; select before the vector loop.'
assert s.count(needle)==1;s=s.replace(needle,'''  if(step>=7){
   constexpr float inverse=std::is_same_v<T,cute::half_t>?(1.f/2048.f):(1.f/256.f);
   #pragma unroll
   for(int j=0;j<16;++j)d[j]=__fmul_rn(d[j],inverse);
  }
'''+needle)
s=s.replace('const __grid_constant__ CUtensorMap quant_a,const __grid_constant__ CUtensorMap quant_b,\n ', '')
s=s.replace('&quant_a,','&high_a,').replace('&quant_b,','&high_b,')
start=s.index('void encode(');end=s.index('template<class T,bool Balance> void init_shared_packer()',start)
s=s[:start]+'''void encode(CUtensorMap& map,const void* ptr,megafmm::DType dt,bool side_a){
 uint64_t dims[4]={512,512,11,1},strides[3]={1024,uint64_t(Plane)*2,0};
 uint32_t box[4]={64,128,1,1},steps[4]={1,1,1,1};
 int rank=3;
 if(side_a){rank=4;dims[0]=64;dims[1]=512;dims[2]=8;dims[3]=11;strides[0]=1024;strides[1]=128;strides[2]=Plane*2;box[1]=32;box[2]=2;}
 CUtensorMapDataType type=dt==megafmm::DType::BF16?CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
 auto e=cuTensorMapEncodeTiled(&map,type,rank,const_cast<void*>(ptr),dims,strides,box,steps,
  CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 if(e!=CUDA_SUCCESS)throw std::runtime_error("native correction tensor map "+std::to_string(int(e)));
}
'''+s[end:]
s=s.replace('Work w(workspace);CUtensorMap ha,hb,qa,qb;\n encode(ha,w.ha,dt,true,false);encode(hb,w.hb,dt,false,false);encode(qa,w.qa,dt,true,true);encode(qb,w.qb,dt,false,true);',
 'Work w(workspace);CUtensorMap ha,hb;\n encode(ha,w.ha,dt,true);encode(hb,w.hb,dt,false);')
s=s.replace('(ha,hb,qa,qb,w.ea,w.eb,D);','(ha,hb,w.ea,w.eb,D);')
(h/'corrected.cuh').write_text(s)
s=(base/'compare.cu').read_text().replace('w.qa,w.ea','static_cast<CT*>(w.qa),w.ea').replace('w.qb,w.eb','static_cast<CT*>(w.qb),w.eb')
(h/'compare.cu').write_text(s)
(h/'pow2.cuh').write_text((base/'pow2.cuh').read_text())
print('generated P45: eleven native planes/side, p11/8 scaled lows, two maps, all eight corrections retained')

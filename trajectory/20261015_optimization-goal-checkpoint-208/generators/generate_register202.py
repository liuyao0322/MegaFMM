from pathlib import Path
import subprocess
s=Path('solution/sw_pipeline.cu').read_text()
s=s.replace('  float product[2][Elements];\n','').replace('uint64_t ready[Stages],free[Stages],product_ready[2],product_free[2];','uint64_t ready[Stages],free[Stages];').replace('==213120','==196736')
s=s.replace('    for(int j=0;j<2;++j){Barrier::init(&s.product_ready[j],128);Barrier::init(&s.product_free[j],128);}\n','')
a=s.index('template<class T,class TA,class TB>');b=s.index('template<class T> __global__',a)
fn='''template<class T> __device__ __forceinline__ void mma_product(Shared<T>& s,int step,float (&out)[16]){
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
'''
for dtype,cond in [('bf16','if constexpr(std::is_same_v<T,bfloat16_t>)'),('f16','else')]:
 lines=['{ .reg .pred p,leader,done; .reg .b64 da,db; .reg .f32 d<16>;', 'setp.ne.u32 p,1,0;', 'setp.eq.u32 leader,%24,0;']
 lines += ['mov.f32 d%d,0f00000000;'%j for j in range(16)]
 for q in range(2):
  if q==1: lines += ['WAIT_SECOND:','mbarrier.try_wait.parity.shared::cta.b64 done,[%20],%23,0x989680;','@!done bra WAIT_SECOND;']
  lines += ['wgmma.fence.sync.aligned;']
  for k in range(16):
   lines += ['add.u64 da, %%%d, %%%d;'%(16+2*q,25+2*k), 'add.u64 db, %%%d, %%%d;'%(17+2*q,26+2*k), 'wgmma.mma_async.sync.aligned.m64n32k16.f32.%s.%s {%s}, da, db, p, 1, 1, %%57, %%58;'%(dtype,dtype,', '.join('d%d'%j for j in range(16)))]
  lines += ['wgmma.commit_group.sync.aligned;','wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;','@leader mbarrier.arrive.shared::cta.b64 _,[%%%d];'%(21+q)]
 lines += ['mov.f32 %%%d,d%d;'%(j,j) for j in range(16)]+['}']
 fn+=' '+cond+'{\n  asm volatile(\n'+''.join('   "'+v+'\\n"\n' for v in lines)
 fn+='   : '+', '.join('"=f"(out[%d])'%j for j in range(16))+'\n'
 ins=['"l"(uint64_t(da0(0,0,0)))','"l"(uint64_t(db0(0,0,0)))','"l"(uint64_t(da1(0,0,0)))','"l"(uint64_t(db1(0,0,0)))','"r"(uint32_t(__cvta_generic_to_shared(&s.ready[slot1])))','"r"(uint32_t(__cvta_generic_to_shared(&s.free[slot0])))','"r"(uint32_t(__cvta_generic_to_shared(&s.free[slot1])))','"r"(phase)','"r"(threadIdx.x&31)']
 for k in range(16):ins+=['"n"(uint32_t(da0.layout()(0,0,Int<%d>{})))'%k,'"n"(uint32_t(db0.layout()(0,0,Int<%d>{})))'%k]
 ins+=['"n"(int32_t(GMMA::Major::K))','"n"(int32_t(GMMA::Major::MN))']
 fn+='   : '+', '.join(ins)+'\n   : "memory");\n }\n'
fn+='}\n\n'
s=s[:a]+fn+s[b:]
a=s.index('    if(worker<128){');b=s.index('  }else if(tid>=128 && tid<256)',a);s=s[:a]+s[b:]
a=s.index('  }else if(tid>=128 && tid<256)');b=s.index('  // CTA shared storage',a)
old=subprocess.check_output('git show $(git log --format=%H --grep="iter 119" -1):solution/sw_pipeline.cu',shell=True,text=True)
ep=old[old.index('      #pragma unroll\n      for(int j=0;j<8;++j)'):old.index('  // CTA',old.index('      #pragma unroll\n      for(int j=0;j<8;++j)'))] if '  // CTA' in old[old.index('      #pragma unroll\n      for(int j=0;j<8;++j)'):] else ''
# extract just pair recombination loop by matching braces
beg=old.index('      #pragma unroll\n      for(int j=0;j<8;++j)');brace=old.index('{',beg);depth=1;end=brace+1
while depth:
 if old[end]=='{':depth+=1
 if old[end]=='}':depth-=1
 end+=1
pair=old[beg:end]
s=s[:a]+'''  }else if(tid>=128 && tid<256){
    float first[16],second[16];
    #pragma unroll 1
    for(int step=0;step<7;++step){
      float acc[16];
      mma_product(s,step,acc);
'''+pair+'\n    }\n  }\n'+s[b:]
Path('solution/sw_pipeline.cu').write_text(s)
p=Path('solution/plan.cpp');s=p.read_text().replace('smem=213120','smem=196736').replace('4simt+4wgmma+15pack_only','4wgmma+19pack_only').replace('recombine=streaming_fixed_chunk_offsets','recombine=tc_register_private_ptx_accumulators').replace('product_slots=2','product_slots=0');p.write_text(s)

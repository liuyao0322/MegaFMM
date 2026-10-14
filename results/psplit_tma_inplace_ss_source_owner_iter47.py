from pathlib import Path
import hashlib,json
base=Path('solution/megakernel_wgmma_psplit_ss_bn16.cu').read_text(); s=base; steps=[]
h=lambda s:hashlib.sha256(s.encode()).hexdigest()
def rep(old,new,count=1):
 global s
 assert s.count(old)==count,(old[:70],s.count(old),count)
 before=h(s); s=s.replace(old,new); steps.append({'old':old,'new':new,'count':count,'before_sha256':before,'after_sha256':h(s)})
rep(s[:s.index('#include')],'''// Native51 BN32 / Native52 BN16 inplace SS TMA, BM64/BK32, two banks.
// Eight raw quadrant TMA boxes alias first four final planes. Each consumer pair
// loads all four words before overwriting; final RN ST cast once per bank stage.
// Both current A/B maps encoded per ordinary call; fresh four-instance CAN required.
''')
rep('#include <cute/tensor.hpp>','#include <cute/tensor.hpp>\n#include <cute/arch/copy_sm90_tma.hpp>\n#include <cute/arch/copy_sm90_desc.hpp>\n#include <cuda.h>')
rep('#include <cstdint>','#include <cstdint>\n#include <cstddef>')
rep('psplit_ss_bn16','psplit_tma_inplace_ss',s.count('psplit_ss_bn16'))
rep('PSplitSSBN16Traits','PSplitTmaInplaceSSTraits',s.count('PSplitSSBN16Traits'))
rep('static_assert(Cols == 16,"this uncompiled SS companion uses typed BN16 only");','static_assert(Cols == 16 || Cols == 32,"typed BN16/BN32 only");')
rep('GMMA::Layout_K_SW32_Atom<Element>','GMMA::Layout_K_SW64_Atom<Element>')
rep('''  using LayoutB = decltype(tile_to_shape(GMMA::Layout_MN_SW32_Atom<Element>{},
      make_shape(Int<BN>{},Int<BK>{},Int<7>{})));
  using Op = std::conditional_t<std::is_same_v<Element,cutlass::half_t>,
      SM90_64x16x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::MN>,
      SM90_64x16x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>>;''','''  using LayoutB = std::conditional_t<Cols == 32,
      decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Element>{},make_shape(Int<BN>{},Int<BK>{},Int<7>{}))),
      decltype(tile_to_shape(GMMA::Layout_MN_SW32_Atom<Element>{},make_shape(Int<BN>{},Int<BK>{},Int<7>{})))>;
  using Op = decltype(GMMA::ss_op_selector<Element,Element,float,Tile,
      GMMA::Major::K,GMMA::Major::MN>());''')
rep('struct alignas(128) SharedStorage','struct alignas(1024) SharedStorage')
rep('    typename Pipeline::SharedStorage barriers;','    typename Pipeline::SharedStorage barriers;\n    alignas(8) uint64_t raw_done[Stages];')
rep('  static constexpr size_t SharedBytes = sizeof(SharedStorage);','''  static constexpr size_t SharedBytes = sizeof(SharedStorage);
  static constexpr uint32_t RawBytes = 4*(BM+BN)*BK*sizeof(Element);
  static_assert(RawBytes == (Cols==32 ? 24576 : 20480),"all eight raw boxes contribute TX bytes");
  static_assert(BM*BK*sizeof(Element)%512==0 && BN*BK*sizeof(Element)%512==0 &&
      ElementsA*sizeof(Element)%512==0 && BankElements*sizeof(Element)%512==0,
      "all absolute SW64/SW32 boxes have zero swizzle pattern offsets");
  static_assert(offsetof(SharedStorage,raw_done)>=offsetof(SharedStorage,barriers)+sizeof(typename Pipeline::SharedStorage),
      "raw barriers never overlap bank operands or pipeline barriers");''')
a=s.index('struct PSplitBN16SSFuture38PackedEight'); b=s.index('// Tail FP32 rendezvous',a)
helpers=r'''// Raw and final share the same flagged CuTe layout. Each pair task owns
// its coordinates in all seven planes. Four explicit loads are one asm statement;
// every mutating store is a later memory-clobbered statement.
CUTE_DEVICE uint4 psplit_tma_inplace_ss_load_four(const void* p0,const void* p1,
    const void* p2,const void* p3) {
  const uint32_t a0=uint32_t(__cvta_generic_to_shared(p0));
  const uint32_t a1=uint32_t(__cvta_generic_to_shared(p1));
  const uint32_t a2=uint32_t(__cvta_generic_to_shared(p2));
  const uint32_t a3=uint32_t(__cvta_generic_to_shared(p3));
  uint4 r;
  asm volatile("ld.shared.b32 %0,[%4]; ld.shared.b32 %1,[%5]; ld.shared.b32 %2,[%6]; ld.shared.b32 %3,[%7];"
      : "=r"(r.x),"=r"(r.y),"=r"(r.z),"=r"(r.w)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3) : "memory");
  return r;
}
CUTE_DEVICE void psplit_tma_inplace_ss_store_pair(void* p,uint32_t v) {
  const uint32_t a=uint32_t(__cvta_generic_to_shared(p));
  asm volatile("st.shared.b32 [%0],%1;" :: "r"(a),"r"(v) : "memory");
}
template<class Element>
CUTE_DEVICE float psplit_tma_inplace_ss_unpack_word(uint32_t x,int j) {
  return float(Element::bitcast(uint16_t(x>>(16*j))));
}
template<class Element>
CUTE_DEVICE uint32_t psplit_tma_inplace_ss_round_pair(float x,float y) {
  cutlass::Array<float,2> f; f[0]=x; f[1]=y;
  using C=cutlass::NumericArrayConverter<Element,float,2,cutlass::FloatRoundStyle::round_to_nearest>;
  const auto r=C{}(f);
  return reinterpret_cast<const uint32_t&>(r);
}
template<class Element,class SmemA>
CUTE_DEVICE void psplit_tma_inplace_ss_transform_a(int task,SmemA a) {
  const int row=task/16,k=(task%16)*2;
  const auto raw=psplit_tma_inplace_ss_load_four(&a(row,k,0),&a(row,k,1),&a(row,k,2),&a(row,k,3));
  float s1[2],s2[2],s3[2],s4[2];
  CUTE_UNROLL
  for(int j=0;j<2;++j) {
    const float a11=psplit_tma_inplace_ss_unpack_word<Element>(raw.x,j);
    const float a12=psplit_tma_inplace_ss_unpack_word<Element>(raw.y,j);
    const float a21=psplit_tma_inplace_ss_unpack_word<Element>(raw.z,j);
    const float a22=psplit_tma_inplace_ss_unpack_word<Element>(raw.w,j);
    s1[j]=__fadd_rn(a21,a22);
    s2[j]=__fsub_rn(s1[j],a11);
    s3[j]=__fsub_rn(a11,a21);
    s4[j]=__fsub_rn(a12,s2[j]);
  }
  // Raw A11/A12/A22 already reside in their unchanged final slots0/1/3.
  psplit_tma_inplace_ss_store_pair(&a(row,k,4),psplit_tma_inplace_ss_round_pair<Element>(s1[0],s1[1]));
  psplit_tma_inplace_ss_store_pair(&a(row,k,5),psplit_tma_inplace_ss_round_pair<Element>(s2[0],s2[1]));
  psplit_tma_inplace_ss_store_pair(&a(row,k,6),psplit_tma_inplace_ss_round_pair<Element>(s3[0],s3[1]));
  psplit_tma_inplace_ss_store_pair(&a(row,k,2),psplit_tma_inplace_ss_round_pair<Element>(s4[0],s4[1]));
}
template<class Element,int Cols,class SmemB>
CUTE_DEVICE void psplit_tma_inplace_ss_transform_b(int task,SmemB b) {
  const int k=task/(Cols/2),col=(task%(Cols/2))*2;
  const auto raw=psplit_tma_inplace_ss_load_four(&b(col,k,0),&b(col,k,1),&b(col,k,2),&b(col,k,3));
  // All raw loads precede the two bitcopy moves that overwrite raw B12/B21.
  psplit_tma_inplace_ss_store_pair(&b(col,k,1),raw.z);
  psplit_tma_inplace_ss_store_pair(&b(col,k,2),raw.w);
  float t1[2],t2[2],t3[2],t4[2];
  CUTE_UNROLL
  for(int j=0;j<2;++j) {
    const float b11=psplit_tma_inplace_ss_unpack_word<Element>(raw.x,j);
    const float b12=psplit_tma_inplace_ss_unpack_word<Element>(raw.y,j);
    const float b21=psplit_tma_inplace_ss_unpack_word<Element>(raw.z,j);
    const float b22=psplit_tma_inplace_ss_unpack_word<Element>(raw.w,j);
    t1[j]=__fsub_rn(b12,b11);
    t2[j]=__fsub_rn(b22,t1[j]);
    t3[j]=__fsub_rn(b22,b12);
    t4[j]=__fsub_rn(t2[j],b21);
  }
  psplit_tma_inplace_ss_store_pair(&b(col,k,4),psplit_tma_inplace_ss_round_pair<Element>(t1[0],t1[1]));
  psplit_tma_inplace_ss_store_pair(&b(col,k,5),psplit_tma_inplace_ss_round_pair<Element>(t2[0],t2[1]));
  psplit_tma_inplace_ss_store_pair(&b(col,k,6),psplit_tma_inplace_ss_round_pair<Element>(t3[0],t3[1]));
  psplit_tma_inplace_ss_store_pair(&b(col,k,3),psplit_tma_inplace_ss_round_pair<Element>(t4[0],t4[1]));
}
template<class Element,int Cols,class SmemA,class SmemB>
CUTE_DEVICE void psplit_tma_inplace_ss_transform_stage(SmemA a,SmemB b) {
  unsigned fresh_tid;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(fresh_tid) :: "memory");
  const int consumer=int(fresh_tid)-128;
  CUTE_UNROLL
  for(int task=consumer;task<64*32/2;task+=256)
    psplit_tma_inplace_ss_transform_a<Element>(task,a);
  CUTE_UNROLL
  for(int task=consumer;task<Cols*32/2;task+=256)
    psplit_tma_inplace_ss_transform_b<Element,Cols>(task,b);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  // All256 own writers must retire generic stores before either SS WG reads.
  asm volatile("barrier.sync 1,256;" ::: "memory");
}
struct alignas(64) TmaInplaceSSDescriptors { CUtensorMap a; CUtensorMap b; };

'''
rep(s[a:b],helpers)
rep('void winograd_wgmma_psplit_tma_inplace_ss(Problem p,const Element* __restrict__ A,\n    const Element* __restrict__ B,const float* C0,float* D)', 'void winograd_wgmma_psplit_tma_inplace_ss(Problem p,\n    const __grid_constant__ TmaInplaceSSDescriptors maps,const float* C0,float* D)')
rep('extern __shared__ __align__(128) unsigned char raw_storage[];','extern __shared__ __align__(1024) unsigned char raw_storage[];')
rep('  // Initial PipelineAsync construction and full/empty init are collective384.','''  if(tid==0) {
    CUTE_UNROLL
    for(int stage=0;stage<T::Stages;++stage) cutlass::arch::initialize_barrier(storage.raw_done[stage],1);
    cutlass::arch::fence_barrier_init();
  }
  // Initial PipelineAsync construction and full/empty init are collective384.''')
rep('cutlass::arch::warpgroup_reg_dealloc<64>();','cutlass::arch::warpgroup_reg_dealloc<48>();')
a=s.index('      CUTE_UNROLL\n      for (int task = tid;'); b=s.index('      pipeline.producer_commit(state);',a)
producer=r'''      auto& raw_done=storage.raw_done[state.index()];
      if(cute::elect_one_sync() && tid<32) {
        cutlass::arch::set_barrier_transaction_bytes(raw_done,T::RawBytes);
        constexpr uint64_t hint=uint64_t(TMA::CacheHintSm90::EVICT_NORMAL);
        const int k=int(k_base),r=int(row_base),c=int(col_base);
        const int mh=int(p.m/2),nh=int(p.n/2),kh=int(p.k/2);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,0),k,r);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,1),k+kh,r);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,2),k,r+mh);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,3),k+kh,r+mh);
        SM90_TMA_LOAD_2D::copy(&maps.b,raw_done,hint,&sB(0,0,0),c,k);
        SM90_TMA_LOAD_2D::copy(&maps.b,raw_done,hint,&sB(0,0,1),c+nh,k);
        SM90_TMA_LOAD_2D::copy(&maps.b,raw_done,hint,&sB(0,0,2),c,k+kh);
        SM90_TMA_LOAD_2D::copy(&maps.b,raw_done,hint,&sB(0,0,3),c+nh,k+kh);
      }
      cutlass::arch::wait_barrier(raw_done,state.phase()^1);
      asm volatile("" ::: "memory");
      // Full publishes RAW operands. Consumerbar1 publishes final ST later.
'''
rep(s[a:b],producer)
rep('''    // Conservative source hypothesis, not actual compiler resource evidence.
    // Source pool need=128*(64+80+80)=28672; hypothetical I80 owns30720,
    // plus actual role-register bounds and no spills must be proved first.
    cutlass::arch::warpgroup_reg_alloc<80>();''','''    // Source quotas are hypotheses until all four actual machine instances pass.
    constexpr int WG0Quota=Cols==32 ? 88 : 64;
    constexpr int WG1Quota=Cols==32 ? 104 : 72;''')
rep('    if (tid < 256) {\n      clear(q0);','    if (tid < 256) {\n      cutlass::arch::warpgroup_reg_alloc<WG0Quota>();\n      clear(q0);')
rep('    } else {\n      clear(q0);','    } else {\n      cutlass::arch::warpgroup_reg_alloc<WG1Quota>();\n      clear(q0);')
rep('''        auto sB = make_tensor(make_smem_ptr(bank+T::ElementsA),typename T::LayoutB{});
        auto tCrA''','''        auto sB = make_tensor(make_smem_ptr(bank+T::ElementsA),typename T::LayoutB{});
        psplit_tma_inplace_ss_transform_stage<Element,Cols>(sA,sB);
        auto tCrA''',2)
a=s.index('template <class Element>\nvoid prepare_wgmma_');
host=r'''template<class Element,int Cols>
void prepare_wgmma_psplit_tma_inplace_ss_typed() {
  constexpr size_t bytes=PSplitTmaInplaceSSTraits<Element,Cols>::SharedBytes;
  static_assert(bytes>48*1024 && bytes<227*1024,"H100 two-bank shared budget");
  const auto e=cudaFuncSetAttribute(winograd_wgmma_psplit_tma_inplace_ss<Element,Cols>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes));
  if(e!=cudaSuccess) throw std::runtime_error(std::string("Psplit shared opt-in: ")+cudaGetErrorString(e));
}
template<int Cols>
TmaInplaceSSDescriptors make_tma_inplace_ss_descriptors(const Problem& p,const void* A,const void* B) {
  TmaInplaceSSDescriptors maps{};
  const auto type=p.dtype==DType::BF16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  const cuuint64_t adims[2]={cuuint64_t(p.k),cuuint64_t(p.m)},bdims[2]={cuuint64_t(p.n),cuuint64_t(p.k)};
  const cuuint64_t astride[1]={cuuint64_t(p.lda)*2},bstride[1]={cuuint64_t(p.ldb)*2};
  const cuuint32_t abox[2]={32,64},bbox[2]={Cols,32},step[2]={1,1};
  auto e=cuTensorMapEncodeTiled(&maps.a,type,2,const_cast<void*>(A),adims,astride,abox,step,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_64B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(e!=CUDA_SUCCESS) throw std::runtime_error("Inplace SS A tensor-map encode error="+std::to_string(int(e)));
  e=cuTensorMapEncodeTiled(&maps.b,type,2,const_cast<void*>(B),bdims,bstride,bbox,step,
      CU_TENSOR_MAP_INTERLEAVE_NONE,Cols==32 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_32B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(e!=CUDA_SUCCESS) throw std::runtime_error("Inplace SS B tensor-map encode error="+std::to_string(int(e)));
  return maps;
}
template<class Element,int Cols>
void launch_wgmma_psplit_tma_inplace_ss_typed(const Problem& p,const void* A,const void* B,
    const float* C0,float* D,cudaStream_t stream) {
  using T=PSplitTmaInplaceSSTraits<Element,Cols>;
  // Both current-input encodes are inside every ordinary call/capture.
  const auto maps=make_tma_inplace_ss_descriptors<Cols>(p,A,B);
  const dim3 grid(unsigned(p.m/(2*T::BM)),unsigned(p.n/(2*T::BN)));
  winograd_wgmma_psplit_tma_inplace_ss<Element,Cols><<<grid,T::Threads,T::SharedBytes,stream>>>(p,maps,C0,D);
}
} // namespace
bool wgmma_psplit_tma_inplace_ss_can_implement(const Problem& p,int columns,int configuration) {
  return configuration==0 && (columns==32 || columns==16) && megakernel_can_implement(p,0) &&
      p.m%128==0 && p.n%(2*columns)==0 && p.k%64==0 && p.lda%8==0 && p.ldb%8==0 &&
      uint64_t(p.lda)<(uint64_t(1)<<39) && uint64_t(p.ldb)<(uint64_t(1)<<39);
}
bool wgmma_psplit_tma_inplace_ss_full_tiles(const Problem& p,const void* A,const void* B,int columns,int configuration) {
  return wgmma_psplit_tma_inplace_ss_can_implement(p,columns,configuration) && A && B &&
      (reinterpret_cast<uintptr_t>(A)&15)==0 && (reinterpret_cast<uintptr_t>(B)&15)==0;
}
void prepare_wgmma_psplit_tma_inplace_ss(const Problem& p,int columns,int configuration) {
  if(!wgmma_psplit_tma_inplace_ss_can_implement(p,columns,configuration)) throw std::invalid_argument("unsupported inplace SS metadata/configuration");
  if(columns==32) {
    if(p.dtype==DType::BF16) prepare_wgmma_psplit_tma_inplace_ss_typed<cutlass::bfloat16_t,32>();
    else prepare_wgmma_psplit_tma_inplace_ss_typed<cutlass::half_t,32>();
  } else {
    if(p.dtype==DType::BF16) prepare_wgmma_psplit_tma_inplace_ss_typed<cutlass::bfloat16_t,16>();
    else prepare_wgmma_psplit_tma_inplace_ss_typed<cutlass::half_t,16>();
  }
}
void launch_wgmma_psplit_tma_inplace_ss(const Problem& p,const void* A,const void* B,const float* C0,float* D,
    cudaStream_t stream,int columns,int configuration) {
  if(!wgmma_psplit_tma_inplace_ss_full_tiles(p,A,B,columns,configuration)) throw std::invalid_argument("inplace SS requires native full tiles/currentAB16B");
  if(!D || (p.beta!=0.0f && !C0)) throw std::invalid_argument("inplace SS active output/source null");
  if(columns==32) {
    if(p.dtype==DType::BF16) launch_wgmma_psplit_tma_inplace_ss_typed<cutlass::bfloat16_t,32>(p,A,B,C0,D,stream);
    else launch_wgmma_psplit_tma_inplace_ss_typed<cutlass::half_t,32>(p,A,B,C0,D,stream);
  } else {
    if(p.dtype==DType::BF16) launch_wgmma_psplit_tma_inplace_ss_typed<cutlass::bfloat16_t,16>(p,A,B,C0,D,stream);
    else launch_wgmma_psplit_tma_inplace_ss_typed<cutlass::half_t,16>(p,A,B,C0,D,stream);
  }
}
} // namespace megafmm
'''
rep(s[a:],host)
out=Path('solution/megakernel_wgmma_psplit_tma_inplace_ss.cu'); assert not out.exists();out.write_text(s)
Path('results/psplit_tma_inplace_ss_parent43_source_iter47.cu').write_text(base)
Path('results/psplit_tma_inplace_ss_source_delta_iter47.json').write_text(json.dumps({'schema':'megafmm.iter47.inplace-ss-source-delta.v1','parent_path':'results/psplit_tma_inplace_ss_parent43_source_iter47.cu','parent_sha256':h(base),'new_path':str(out),'new_sha256':h(s),'steps':steps,'scope':'new inplace TMA raw-to-final shared SS template; original perP math/wholeK/tailU from parent43, no compiled or GPU proof'},indent=2)+'\n')
print(json.dumps({'source':str(out),'sha256':h(s),'delta_steps':len(steps)}))

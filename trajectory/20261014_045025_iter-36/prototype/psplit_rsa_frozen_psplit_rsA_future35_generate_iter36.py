from pathlib import Path
import hashlib,difflib
R=Path('/share/home/liuyao/workspace/MegaFMM')
p=R/'results/psplit_revision6_megakernel_wgmma_psplit_iter34.cu'
x=p.read_text();assert hashlib.sha256(p.read_bytes()).hexdigest()=='fdaae75a80893ee161a9db202d4746e86daf3d38135332759098bd9f7faa7f91'
orig=x
x=x[x.index('#include'):].replace('PSplit','PSplitRSA').replace('psplit','psplit_rsA')
x='''// FUTURE CPU-ONLY UNCOMPILED RS-A experiment, not iteration35 formal source.
// BK64 raw A4 shared + final B7 shared; whole-stage distinct RS A registers.
// Source64/192/192 and initialI152/160/168 are CPU algebra, not fit/launch proof.
// Original onecast ST, whole seven FP32 P, fixedU/alphaRN/betaFMA retained.
'''+x
x=x.replace('static constexpr int BK = 32;','static constexpr int BK = 64;\n  static constexpr int AtomsK = BK/16;')
x=x.replace('make_shape(Int<BM>{},Int<BK>{},Int<7>{}))','make_shape(Int<BM>{},Int<BK>{},Int<4>{}))',1)
x=x.replace('GMMA::ss_op_selector','GMMA::rs_op_selector').replace('ElementsA == 7*BM*BK','ElementsA == 4*BM*BK')
start=x.index('// Each task owns eight K elements');end=x.index('// BN16:',start)
x=x[:start]+'''// Producer writes four raw original A quadrants, with no A transform/cast.
template <class Element,int BK,class SmemA>
CUTE_DEVICE void psplit_rsA_fill_a(const Problem& p,const Element* A,
    int64_t row_base,int64_t k_base,int task,SmemA sA) {
  const int64_t mh=p.m/2,kh=p.k/2;
  const int row=task/(BK/8),k=(task%(BK/8))*8;
  const int64_t gr=row_base+row,gk=k_base+k;
  const auto a11=psplit_rsA_read_eight(A+gr*p.lda+gk);
  const auto a12=psplit_rsA_read_eight(A+gr*p.lda+gk+kh);
  const auto a21=psplit_rsA_read_eight(A+(gr+mh)*p.lda+gk);
  const auto a22=psplit_rsA_read_eight(A+(gr+mh)*p.lda+gk+kh);
  psplit_rsA_store_eight(&sA(row,k,0),a11);
  psplit_rsA_store_eight(&sA(row,k,1),a12);
  psplit_rsA_store_eight(&sA(row,k,2),a21);
  psplit_rsA_store_eight(&sA(row,k,3),a22);
}

CUTE_DEVICE uint32_t psplit_rsA_load_pair(const void* ptr) {
  uint32_t word;const uint32_t address=uint32_t(__cvta_generic_to_shared(ptr));
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(word) : "r"(address) : "memory");
  return word;
}
template <class Element>
CUTE_DEVICE float psplit_rsA_pair_half(uint32_t word,int lane) {
  return float(Element::bitcast(uint16_t(word>>(16*lane))));
}

// First0: [A11,A12,S4]. First3: [A22,S1,S2,S3], no rawA12 load.
// Each group performs three FP32 RN ops/lane; total six versus SS four.
// Four derived finals still receive one cast each, with no intermediate cast.
template <class Element,int First,class ThrMma,class SmemRawA,
          class RA0,class RA1,class RA2,class RA3>
CUTE_DEVICE void psplit_rsA_build_stage(const ThrMma& thr_mma,SmemRawA raw,
    RA0& a0,RA1& a1,RA2& a2,RA3& a3) {
  static_assert(First==0 || First==3,"wholeP3+4 only");
  auto coords=thr_mma.partition_A(make_identity_tensor(make_shape(Int<64>{},Int<64>{})));
  CUTE_STATIC_ASSERT_V(size<0>(coords)==Int<8>{});
  CUTE_STATIC_ASSERT_V(size<1>(coords)==Int<1>{});
  CUTE_STATIC_ASSERT_V(size<2>(coords)==Int<4>{});
  auto u0=recast<uint32_t>(a0);auto u1=recast<uint32_t>(a1);
  auto u2=recast<uint32_t>(a2);auto u3=recast<uint32_t>(a3);
  CUTE_UNROLL
  for(int atom=0;atom<4;++atom) {
    CUTE_UNROLL
    for(int pair=0;pair<4;++pair) {
      const auto c=coords(2*pair,0,atom);
      const int row=get<0>(c),k=get<1>(c);
      const uint32_t x11=psplit_rsA_load_pair(&raw(row,k,0));
      const uint32_t x21=psplit_rsA_load_pair(&raw(row,k,2));
      const uint32_t x22=psplit_rsA_load_pair(&raw(row,k,3));
      if constexpr(First==0) {
        const uint32_t x12=psplit_rsA_load_pair(&raw(row,k,1));
        float s4[2];
        CUTE_UNROLL
        for(int j=0;j<2;++j) {
          const float f11=psplit_rsA_pair_half<Element>(x11,j);
          const float f12=psplit_rsA_pair_half<Element>(x12,j);
          const float f21=psplit_rsA_pair_half<Element>(x21,j);
          const float f22=psplit_rsA_pair_half<Element>(x22,j);
          const float s1=__fadd_rn(f21,f22);
          const float s2=__fsub_rn(s1,f11);
          s4[j]=__fsub_rn(f12,s2);
        }
        u0(pair,0,atom)=x11;u1(pair,0,atom)=x12;
        u2(pair,0,atom)=psplit_rsA_round_pair<Element>(s4[0],s4[1]);
      } else {
        float s1[2],s2[2],s3[2];
        CUTE_UNROLL
        for(int j=0;j<2;++j) {
          const float f11=psplit_rsA_pair_half<Element>(x11,j);
          const float f21=psplit_rsA_pair_half<Element>(x21,j);
          const float f22=psplit_rsA_pair_half<Element>(x22,j);
          s1[j]=__fadd_rn(f21,f22);s2[j]=__fsub_rn(s1[j],f11);
          s3[j]=__fsub_rn(f11,f21);
        }
        u0(pair,0,atom)=x22;
        u1(pair,0,atom)=psplit_rsA_round_pair<Element>(s1[0],s1[1]);
        u2(pair,0,atom)=psplit_rsA_round_pair<Element>(s2[0],s2[1]);
        u3(pair,0,atom)=psplit_rsA_round_pair<Element>(s3[0],s3[1]);
      }
    }
  }
}

'''+x[end:]
x=x.replace('psplit_rsA_fill_a(p,A,row_base,k_base,task,sA);','psplit_rsA_fill_a<Element,T::BK>(p,A,row_base,k_base,task,sA);')
start=x.index('    // Conservative source hypothesis');end=x.index('    if (tid < 256)',start)
x=x[:start]+'''    // Element arrays have no initializer; pinned default ctors are defaulted.
    // Decltype obtains layout without constructing an owning ArrayEngine.
    using ARLayout=typename decltype(thr_mma.make_fragment_A(thr_mma.partition_A(
        make_identity_tensor(make_shape(Int<T::BM>{},Int<T::BK>{})))) )::layout_type;
    alignas(16) Element a0_data[cosize_v<ARLayout>],a1_data[cosize_v<ARLayout>];
    alignas(16) Element a2_data[cosize_v<ARLayout>],a3_data[cosize_v<ARLayout>];
    auto a0=make_tensor(make_rmem_ptr(a0_data),ARLayout{});
    auto a1=make_tensor(make_rmem_ptr(a1_data),ARLayout{});
    auto a2=make_tensor(make_rmem_ptr(a2_data),ARLayout{});
    auto a3=make_tensor(make_rmem_ptr(a3_data),ARLayout{});
    static_assert(size(a0)==8*T::AtomsK,"all distinct stage RS input values");
    // Source quota hypothesis only. Must bind future actual initial allocation,
    // role widths, scalarization, zero spills and asynchronous A liveness first.
    cutlass::arch::warpgroup_reg_alloc<192>();
'''+x[end:]
line='        auto tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA));'
assert x.count(line)==2
x=x.replace(line,'        psplit_rsA_build_stage<Element,0>(thr_mma,sA,a0,a1,a2,a3);',1)
x=x.replace(line,'        psplit_rsA_build_stage<Element,3>(thr_mma,sA,a0,a1,a2,a3);',1)
for pid,aid in [(0,0),(1,1),(2,2),(3,0),(4,1),(5,2),(6,3)]:
 x=x.replace(f'tCrA(_,_,_,Int<{pid}>{{}})',f'a{aid}')
first=x.index('    if (tid < 256)');second=x.index('    } else {',first);tail=x.index('    // All384 threads',second)
g0=x[first:second];g1=x[second:tail]
for name,buf,ids in [('g0',g0,[0,1,2]),('g1',g1,[0,1,2,3])]:
 fences='\n'.join('        warpgroup_fence_operand(a%d);'%i for i in ids)
 assert buf.count('        warpgroup_arrive();')==buf.count('        warpgroup_wait<0>();')==1
 buf=buf.replace('        warpgroup_arrive();',fences+'\n        warpgroup_arrive();')
 buf=buf.replace('        warpgroup_wait<0>();','        warpgroup_wait<0>();\n'+fences)
 if name=='g0':g0=buf
 else:g1=buf
x=x[:first]+g0+g1+x[tail:]
x=x.replace('p.k%64 == 0','p.k%128 == 0')
x=x.replace('canonical GMMA descriptors for all seven P','rawA4 and canonical finalB7 descriptors')
x=x.replace('// BN16:64 tasks; BN32:128 tasks. All use vector8 original reads/stores.\n// A temporaries are dead before this helper starts; there is no raw shared bank.', '// BK64/BN32 has256 B vector tasks; the finalB7 onecast DAG is unchanged.\n// Producer rawA copy temporaries are dead before constructing finalB slots.')
p=R/'.scratch/psplit_rsA_future35_candidate.cu';assert not p.exists();p.write_text(x)
d=R/'.scratch/psplit_rsA_future35_candidate.diff';assert not d.exists();d.write_text(''.join(difflib.unified_diff(orig.splitlines(True),x.splitlines(True),fromfile='frozen_revision6_fdaae75a',tofile='CPU_future_RS_A_BK64')))
print(str(p),hashlib.sha256(p.read_bytes()).hexdigest())

"""CPU-only selective rawA TMA hypothesis: no compiler, GPU, Git or network."""
from pathlib import Path
from collections import Counter
import hashlib
import itertools
import json
import random
import sys

ROOT = Path("/share/home/liuyao/workspace/MegaFMM")
PREFIX = ROOT/".scratch/psplit_selective_tmaA_root_future40_"
PARENT = ROOT/"solution/megakernel_wgmma_psplit_rsa.cu"
PARENT_SHA = "1f964339034c3e800455704b6d33482d072ea99a2b6eb9f08773df57f705f054"
HEADERS = [
    ROOT/"third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp",
    ROOT/"third_party/cutlass/include/cute/pointer_flagged.hpp",
    ROOT/"third_party/cutlass/include/cute/swizzle.hpp",
    ROOT/"third_party/cutlass/include/cute/arch/copy_sm90_tma.hpp",
    ROOT/"third_party/cutlass/include/cute/arch/copy_sm90_desc.hpp",
    ROOT/"third_party/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp",
    ROOT/"third_party/cutlass/include/cute/atom/copy_traits_sm90_tma_swizzle.hpp",
    ROOT/"third_party/cutlass/include/cutlass/arch/barrier.h",
    ROOT/"third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp",
    ROOT/"third_party/cutlass/include/cute/algorithm/gemm.hpp",
    Path("/usr/local/cuda/include/cuda.h"),
]
DOMAINS = [1,2,3,4,6,7,8,9,10,16,17,18,33,34]
OWNED = [(0,1,2),(3,4,5,6)]
OLD = "psplit_rsa"
NEW = "psplit_selective_tmaA_root_future40"
NEW_TRAITS = "PSplitSelectiveTmaARootFuture40Traits"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def guard(event, args):
    if event == "open" and args and isinstance(args[0], (str,bytes)):
        mode,flags = args[1] or "", args[2] if len(args)>2 else 0
        writes = isinstance(mode,str) and any(c in mode for c in "wax+")
        writes |= isinstance(flags,int) and bool(flags & (1|2|64|512|1024))
        if writes and not str(Path(args[0]).absolute()).startswith(str(PREFIX)):
            raise RuntimeError("write outside selective TMA-A prefix")


sys.addaudithook(guard)
assert sha(PARENT)==PARENT_SHA
protected = {str(p):sha(p) for p in [PARENT,ROOT/"solution/megakernel_wgmma_tma_alias.cu",
              ROOT/"solution/megakernel_hopper.cu",*HEADERS]}
old_manifest = ROOT/".scratch/psplit_bn16_microbatch_root_future39_manifest.json"
assert sha(old_manifest)=="bf10bfb85350f18254e677dedd9f4828217c4dd6597f160c036d9d5c0838a522"
protected[str(old_manifest)] = sha(old_manifest)
for p,h in json.loads(old_manifest.read_text())["files"].items():
    assert sha(p)==h
    protected[p]=h
texts = {str(p):p.read_text() for p in HEADERS}
traits = texts[str(HEADERS[0])]
cuda = texts[str(HEADERS[-1])]
barrier = texts[str(HEADERS[7])]
pipeline = texts[str(HEADERS[8])]
assert "Layout<Shape<_8,_1024>,Stride<_1024,_1>>>" in traits
assert "using ALayout_64x16 = CLayout_64x16;" in traits
assert "case 3:   return TMA::SmemSwizzleBits::B128;" in texts[str(HEADERS[6])]
assert "return offset ^ shiftr(offset & yyy_msk{}, msk_sft{});" in texts[str(HEADERS[2])]
assert "make_tensor(make_smem_ptr(ptr.get(), layout.layout_a())," in texts[str(HEADERS[1])]
assert "must be less than or equal to the swizzle size." in cuda
assert "CU_TENSOR_MAP_SWIZZLE_128B* require the bounding box inner dimension to be <= 128." in cuda
assert "mbarrier.arrive.expect_tx.shared::cta.b64" in barrier
assert "constexpr uint32_t InitialProducerPhase = 1;" in pipeline
assert "PipelineState(): index_{}, phase_{}, count_{} {}" in pipeline
assert "empty_barrier_ptr_[stage].arrive(params_.dst_blockid);" in pipeline

original = PARENT.read_text()
source = original
changes = []


def replace(old,new,count=1):
    global source
    assert source.count(old)==count, (old[:80],source.count(old),count)
    changes.append({"old":old,"new":new,"count":count})
    source=source.replace(old,new)


replace(OLD,NEW,source.count(OLD))
replace("PSplitRSATraits",NEW_TRAITS,source.count("PSplitRSATraits"))
replace("// Iteration36 RS-A candidate: raw A4 shared, final B7 shared, whole-stage RS A.\n",
        "// CPU-only selective rawA TMA future40; uncompiled and unregistered.\n")
replace("#include <cute/tensor.hpp>\n", "#include <cute/tensor.hpp>\n#include <cute/arch/copy_sm90_tma.hpp>\n#include <cute/arch/copy_sm90_desc.hpp>\n#include <cuda.h>\n")
replace("#include <cstdint>\n","#include <cstdint>\n#include <cstddef>\n")
marker="template <class Element, int Cols>\nstruct "+NEW_TRAITS
dense_helper=r"""// Flagged CuTe swizzles are byte functions, not element-address functions.
// A full BK64 row is128B, so four one-box raw copies require SW128.
template <class Layout>
constexpr bool selective_tmaA_dense_box_sentinels() {
  for(int q=0;q<4;++q) {
    for(int ri=0;ri<4;++ri) {
      const int r=ri==0 ? 0 : ri==1 ? 7 : ri==2 ? 8 : 63;
      for(int ki=0;ki<6;++ki) {
        const int k=ki==0 ? 0 : ki==1 ? 7 : ki==2 ? 8 : ki==3 ? 15 : ki==4 ? 16 : 63;
        const int plain=q*4096+r*64+k;
        const int actual_raw=Layout{}.layout_b()(make_coord(r,k,q));
        if(actual_raw!=plain) return false;
        const int actual=Layout{}.layout_a()(2*actual_raw)/2;
        const int hardware=plain^(((plain>>6)&7)<<3);
        if(actual!=hardware) return false;
      }
    }
  }
  return true;
}

"""
replace(marker,dense_helper+marker)
replace("GMMA::Layout_K_SW32_Atom<Element>{}","GMMA::Layout_K_SW128_Atom<Element>{}")
replace("struct alignas(128) SharedStorage {","struct alignas(1024) SharedStorage {")
replace("    typename Pipeline::SharedStorage barriers;\n",
        "    typename Pipeline::SharedStorage barriers;\n    alignas(8) uint64_t raw_done[Stages];\n")
replace("  static constexpr size_t SharedBytes = sizeof(SharedStorage);\n",
        "  static constexpr size_t SharedBytes = sizeof(SharedStorage);\n"
        "  static constexpr uint32_t RawABytes = 4*BM*BK*sizeof(Element);\n"
        "  static_assert(RawABytes==32768,\"only four rawA boxes count as transaction bytes\");\n"
        "  static_assert(BankElements*sizeof(Element)%1024==0 && ElementsA*sizeof(Element)%1024==0 &&\n"
        "      BM*BK*sizeof(Element)%1024==0,\"all absolute SW128 pattern bases are zero-offset\");\n"
        "  static_assert(offsetof(SharedStorage,raw_done)>=offsetof(SharedStorage,barriers)+sizeof(typename Pipeline::SharedStorage),\n"
        "      \"raw transaction barriers never alias operands or pipeline barriers\");\n")
replace(NEW+"_contiguous_eight_sentinels<LayoutA,1>()","selective_tmaA_dense_box_sentinels<LayoutA>()")
replace('"A edge vectors must be contiguous and 16B aligned in every product slot"',
        '"A dense SW128 TMA box sentinels match flagged CuTe byte coordinates"')
start=source.index("// Producer writes four raw original A quadrants")
end=source.index("CUTE_DEVICE uint32_t "+NEW+"_load_pair",start)
replace(source[start:end],"// Four raw A quadrants are copied by TMA, without transform or cast.\n")
descriptor=r"""// One current-A descriptor, no B descriptor and no pointer-bound cache.
struct alignas(64) SelectiveTmaADescriptor {
  CUtensorMap a;
};
static_assert(sizeof(SelectiveTmaADescriptor)==128 && alignof(SelectiveTmaADescriptor)==64 &&
    std::is_trivially_copyable<SelectiveTmaADescriptor>::value,
    "immutable A descriptor is passed by value in kernel parameter space");

"""
replace("template <class Element,int Cols>\n__global__",descriptor+"template <class Element,int Cols>\n__global__")
replace("void winograd_wgmma_"+NEW+"(Problem p,const Element* __restrict__ A,\n",
        "void winograd_wgmma_"+NEW+"(Problem p,CUTE_GRID_CONSTANT const SelectiveTmaADescriptor maps,\n")
replace("extern __shared__ __align__(128) unsigned char raw_storage[];",
        "extern __shared__ __align__(1024) unsigned char raw_storage[];")
replace("  Pipeline pipeline(storage.barriers,params);\n",
        "  Pipeline pipeline(storage.barriers,params);\n"
        "  if(tid==0) {\n"
        "    CUTE_UNROLL\n"
        "    for(int b=0;b<T::Stages;++b)\n"
        "      cutlass::arch::ClusterTransactionBarrier::init(&storage.raw_done[b],1);\n"
        "    cutlass::arch::fence_barrier_init();\n"
        "  }\n")
start=source.index("      CUTE_UNROLL\n      for (int task = tid; task < T::BM*T::BK/8;")
end=source.index("      CUTE_UNROLL\n      for (int task = tid; task < T::BN*T::BK/8;",start)
replace(source[start:end],r"""      // A raw transaction is distinct from final-ready PipelineAsync.
      if(tid==0) {
        auto* raw_done=&storage.raw_done[state.index()];
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(raw_done,T::RawABytes);
        constexpr uint64_t hint=uint64_t(TMA::CacheHintSm90::EVICT_NORMAL);
        const int32_t r=int32_t(row_base),k=int32_t(k_base);
        const int32_t mh=int32_t(p.m/2),kh=int32_t(p.k/2);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,0),k,r);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,1),k+kh,r);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,2),k,r+mh);
        SM90_TMA_LOAD_2D::copy(&maps.a,raw_done,hint,&sA(0,0,3),k+kh,r+mh);
      }
      // The unchanged SIMT B producer can overlap rawA TMA completion.
""")
replace('      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\n      pipeline.producer_commit(state);',
        '      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\n'
        '      // raw_done starts phase0, opposite the producer start phase1.\n'
        '      // Every writer waits for all32768 A bytes before final publication.\n'
        '      cutlass::arch::ClusterTransactionBarrier::wait(\n'
        '          &storage.raw_done[state.index()],state.phase()^1u);\n'
        '      asm volatile("" ::: "memory"); // pinned wait lacks compiler memory clobber\n'
        '      pipeline.producer_commit(state);')
host=r"""// This encode remains inside each ordinary timed host invocation/capture.
// Graph replay retains fixed captured addresses and reads fresh A content.
SelectiveTmaADescriptor make_selective_tmaA_descriptor(const Problem& p,const void* A) {
  SelectiveTmaADescriptor maps{};
  const CUtensorMapDataType type=p.dtype==DType::BF16 ?
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  const cuuint64_t dims[2]={cuuint64_t(p.k),cuuint64_t(p.m)};
  const cuuint64_t stride[1]={cuuint64_t(p.lda)*2};
  const cuuint32_t box[2]={64,64},step[2]={1,1};
  const auto result=cuTensorMapEncodeTiled(&maps.a,type,2,const_cast<void*>(A),
      dims,stride,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(result!=CUDA_SUCCESS)
    throw std::runtime_error("Selective rawA TMA tensor-map encode error="+std::to_string(int(result)));
  return maps;
}

"""
replace("template <class Element>\nvoid launch_wgmma_"+NEW+"_typed",host+"template <class Element>\nvoid launch_wgmma_"+NEW+"_typed")
replace("  using T = "+NEW_TRAITS+"<Element,32>;\n  const dim3 grid",
        "  using T = "+NEW_TRAITS+"<Element,32>;\n  const auto maps=make_selective_tmaA_descriptor(p,A);\n  const dim3 grid")
replace("      p,static_cast<const Element*>(A),static_cast<const Element*>(B),C0,D);",
        "      p,maps,static_cast<const Element*>(B),C0,D);")
replace("p.k%128 == 0 && p.lda%8 == 0 && p.ldb%8 == 0;",
        "p.k%128 == 0 && p.lda%8 == 0 && p.ldb%8 == 0 &&\n"
        "      uint64_t(p.lda)<(uint64_t(1)<<39); // TMA byte stride must be <2^40")
restored=source
for c in reversed(changes):
    assert restored.count(c["new"])==c["count"], c["new"][:80]
    restored=restored.replace(c["new"],c["old"])
assert restored==original
norm=source.replace(NEW,OLD).replace(NEW_TRAITS,"PSplitRSATraits")
sections={}
for label,begin,end in [
    ("A_math_and_register_builder","CUTE_DEVICE uint32_t psplit_rsa_load_pair","// BK64/BN32"),
    ("producer_B_math_and_layout","// BK64/BN32","// Tail FP32"),
    ("consumer_wholeK_issue_release","  } else {\n    // Do not construct","    // All384 threads rendezvous"),
    ("tail_fixedU_alphaRN_betaFMA","    // All384 threads rendezvous","template <class Element>\nvoid prepare_"),
]:
    def section(s):
        i=s.index(begin); return s[i:s.index(end,i)]
    assert section(norm)==section(original),label
    sections[label]=hashlib.sha256(section(original).encode()).hexdigest()
assert source.count("SM90_TMA_LOAD_2D::copy(")==4
assert "maps.b" not in source and "_fill_a<" not in source
assert source.count("pipeline.consumer_release(state);")==2

# Literal independent byte coordinates; no execution/import of an older model.
BM,BN,BK=64,32,64
AEL,BEL=4*BM*BK,7*BN*BK
BANK=2*(AEL+BEL)
TAIL=7*BM*BN*4


def sw(addr,bits):
    return addr ^ ((addr & (((1<<bits)-1)<<7))>>3)


def old_A(r,k,q):
    return (q*8*4+(k//16)*8+r//8)*128+(r%8)*16+k%16


def new_A(r,k,q):
    return q*4096+r*64+k


def raw_B(n,k,q):
    return (q*2*8+(k//8)*2+n//16)*128+n%16+(k%8)*16


def rs(t,v,atom):
    linear=128*(t%4)+(t//4%8)+16*(t//32%4)+64*(v%2)+8*(v//2%2)+512*(v//4%2)
    return linear%64,linear//64+16*atom


coords=[rs(t,v,a) for t in range(128) for a in range(4) for v in range(8)]
assert len(coords)==4096 and len(set(coords))==4096
assert {new_A(r,k,q) for q in range(4) for r in range(64) for k in range(64)}==set(range(AEL))
assert {raw_B(n,k,q) for q in range(7) for n in range(32) for k in range(64)}==set(range(BEL))
geometry_checks=Counter()
physical_digests=[]
# Enumerate absolute bases across two SW128 periods and a larger shared offset.
for base in [0,1024,2048,8192,65536]:
    occupied=set()
    for bank in range(2):
        origin=base+bank*BANK
        for q in range(4):
            slot_base=origin+q*8192
            assert slot_base%1024==0 and slot_base%512==0
            assert sw(slot_base,3)==slot_base
            for r in range(64):
                for k in range(64):
                    cute=sw(origin+2*new_A(r,k,q),3)
                    # Independent TMA row/16B-chunk permutation for dense128B row.
                    hardware=slot_base+r*128+2*(k^((r&7)<<3))
                    assert cute==hardware and cute not in occupied
                    occupied.add(cute)
                    geometry_checks["all_A_element_TMA_CuTe_equal"]+=1
            for t in range(128):
                for atom in range(4):
                    for pair in range(4):
                        r,k=rs(t,2*pair,atom)
                        r1,k1=rs(t,2*pair+1,atom)
                        a=sw(origin+2*new_A(r,k,q),3)
                        b=sw(origin+2*new_A(r1,k1,q),3)
                        assert r==r1 and k1==k+1 and a%4==0 and b==a+2
                        assert a//128==b//128
                        geometry_checks["RS_packed_pair_aligned_not_cross_row"]+=1
        for q in range(7):
            for k in range(64):
                for n in range(0,32,8):
                    words=[sw(origin+2*AEL+2*raw_B(n+j,k,q),1) for j in range(8)]
                    assert words[0]%16==0 and words==list(range(words[0],words[0]+16,2))
                    assert not occupied.intersection(words)
                    occupied.update(words)
                    geometry_checks["B_unchanged_SIMT_vector8_absolute_geometry"]+=1
        # All operand halfwords are distinct and remain inside their own bank.
        assert set(a for a in occupied if origin<=a<origin+BANK)==set(range(origin,origin+BANK,2))
    assert occupied==set(range(base,base+2*BANK,2))
    physical_digests.append(hashlib.sha256(json.dumps(sorted(occupied)).encode()).hexdigest())
assert old_A(1,0,0)==16 and new_A(1,0,0)==64
assert 64*2>32
assert sw(512,3)==576 and sw(512,3)%128!=0
assert BANK==61440 and 2*BANK==122880 and TAIL==57344 and TAIL<2*BANK
assert all(x%1024==0 for x in [8192,32768,BANK])
# Source predicts natural members; actual offsetof/sizeof remains pending.
abi_estimate={"operand_union_bytes":122880,"typed_tail_bytes":57344,"pipeline_barrier_literal_member_bytes":32,
              "raw_done_literal_bytes":16,"natural_raw_done_offset_estimate":122912,
              "align1024_SharedStorage_estimate":123904,"actual_offsetof":None,"actual_sizeof":None,
              "typed_tail_within_union_before_barriers_and_raw_done":True,
              "estimate_not_compiler_ABI_or_resource_CAN":True}

# Host descriptor restrictions and native application domain.
host_cases=[]
for m,n,k in [(128,64,128),(1024,1024,1024),(32768,32768,32768)]:
    for lda in [k,k+8,(1<<39)-8]:
        assert m%128==0 and n%64==0 and k%128==0 and lda%8==0
        assert lda>=k and 2*lda<1<<40 and (2*lda)%16==0
        for row in [0,m//2-64]:
            for kk in [0,k//2-64]:
                for qr,qk in [(0,0),(0,k//2),(m//2,0),(m//2,k//2)]:
                    assert 0<=row+qr<=m-64 and 0<=kk+qk<=k-64
                    assert row+qr<(1<<31) and kk+qk<(1<<31)
                    assert ((row+qr)*lda+(kk+qk))*2%16==0
        host_cases.append({"m":m,"n":n,"k":k,"lda":lda,"byte_stride":2*lda,"all_four_box_bounds_and16B_offsets":True})
assert ((1<<39)*2)==1<<40
host_domain={"global_dims_max":32768,"common_metadata_limits_source_bound":True,
             "A_current_pointer_alignment_bytes":16,"A_byte_stride_multiple":16,"A_byte_stride_exclusive_max":1<<40,
             "new_native_guard":"uint64_t(lda)<2^39","B_ldb_and_pointer_guards_unchanged":True,
             "native_m128_n64_k128_moduli_unchanged":True,"ordinary_current_A_encode_count":1,
             "descriptor_dimensions":["k","m"],"descriptor_box":[64,64],"descriptor_step":[1,1],
             "dtype_codes":["FLOAT16","BFLOAT16"],"descriptor_interleave":"NONE","descriptor_swizzle":"128B",
             "Graph_captured_binding_fresh_device_content_source_semantics":True,
             "domain_narrowing":"original RSA legally allowed some lda>=2^39; TMA byte-stride bound narrows those rare cases and future Plan must select fallback",
             "oversized_stride_negative_lda":1<<39,"driver_encode_executed":False,"cases":host_cases}


class Blocked(Exception):
    pass


class Violation(Exception):
    pass


def need(ok,tag):
    if not ok: raise Violation(tag)


class State:
    """Atomic128-thread roles, four independent asynchronous TMA completion actors."""
    def __init__(self,n):
        self.n=n
        self.banks=[{"gen":None,"empty":0,"full":0,"rawphase":0,"expected":0,"issued":set(),
                     "complete":set(),"Bstores":False,"Bfences":False,"raw_waited":False,"releases":set()} for _ in range(2)]
        self.produced=0
        self.acquire=None
        self.waited=[None,None]
        self.released=[[],[]]
        self.issue={p:[] for p in range(7)}
        self.committed=[False,False]
        self.wgmma_waited=[False,False]
        self.tail_acquires=[]
        self.tail_done=False
        self.drains=set()
        self.bar6=set()
        self.counts=Counter()
        self.trace=[]

    def step(self,e):
        kind,actor,s,p,atom=e
        b=self.banks[s%2] if s is not None else None
        if kind in ("acquire","force_reuse"):
            if kind=="force_reuse":
                need(b["gen"] is None,"BANK_REUSE_BEFORE_TWO_WG_RELEASES")
            need(s==self.produced,"PRODUCER_STAGE_ORDER")
            free=b["gen"] is None and b["empty"]!=(1^((s//2)&1))
            if kind=="force_reuse": need(free,"BANK_REUSE_BEFORE_TWO_WG_RELEASES")
            if not free: raise Blocked()
            need(b["expected"]==0,"BANK_REUSE_WITH_RAW_TX_PENDING")
            self.acquire=s
            b.update(gen=s,issued=set(),complete=set(),Bstores=False,Bfences=False,raw_waited=False,releases=set())
        elif kind in ("expect","wrong_expect"):
            need(self.acquire==s,"EXPECT_WITHOUT_ACQUIRE")
            need(b["expected"]==0,"RAW_EXPECT_DOUBLE_ARM")
            need(kind=="expect","RAW_EXPECT_MUST_BE_ONLY32768_A_BYTES")
            b["expected"]=32768
        elif kind=="tma_issue":
            need(self.acquire==s and b["expected"]>0,"TMA_ISSUE_BEFORE_EXPECT")
            need(p not in b["issued"],"DUPLICATE_RAW_QUADRANT_ISSUE")
            b["issued"].add(p)
        elif kind=="tma_complete":
            if b["gen"]!=s or p not in b["issued"]: raise Blocked()
            need(p not in b["complete"],"DUPLICATE_RAW_QUADRANT_COMPLETION")
            b["complete"].add(p)
            b["expected"]-=8192
            if b["complete"]=={0,1,2,3}:
                need(b["expected"]==0,"RAW_TX_BYTE_COUNT_MISMATCH")
                b["rawphase"]^=1
        elif kind=="B128stores":
            need(self.acquire==s,"B_STORE_WITHOUT_ACQUIRE")
            b["Bstores"]=True
        elif kind=="B128proxyfences":
            need(b["Bstores"],"B_PROXY_FENCE_BEFORE_WRITES")
            b["Bfences"]=True
        elif kind in ("raw_wait","raw_wait_wrong_phase"):
            need(kind=="raw_wait","RAW_WAIT_MUST_COMPLEMENT_PRODUCER_PHASE")
            if b["expected"]!=0 or b["complete"]!={0,1,2,3} or b["rawphase"]==((s//2)&1):
                raise Blocked()
            b["raw_waited"]=True
        elif kind=="publish128":
            need(b["Bstores"] and b["Bfences"],"PUBLISH_BEFORE_ALL_B_WRITER_PROXY_FENCES")
            need(b["expected"]==0 and b["complete"]=={0,1,2,3} and b["raw_waited"],"PUBLISH_BEFORE_A_TX_COMPLETE_AND_RAW_WAIT")
            need(self.acquire==s,"PUBLISH_WITHOUT_STAGE_ACQUIRE")
            b["full"]^=1
            self.produced+=1
            self.acquire=None
        elif kind=="consumer_wait":
            if b["gen"]!=s or b["full"]==((s//2)&1): raise Blocked()
            need(s==len(self.released[actor]),"CONSUMER_STAGE_ORDER")
            self.waited[actor]=s
            self.committed[actor]=False
            self.wgmma_waited[actor]=False
        elif kind=="issue":
            need(self.waited[actor]==s and p in OWNED[actor],"ISSUE_OUTSIDE_OWNED_STAGE")
            need(len(self.issue[p])==4*s+atom,"WHOLEK_P_ISSUE_ORDER")
            need(atom in range(4),"K16_ATOM_DOMAIN")
            self.issue[p].append(4*s+atom)
        elif kind=="commit":
            need(all(len(self.issue[q])==4*(s+1) for q in OWNED[actor]),"COMMIT_BEFORE_ALL_WG_P_FOUR_K16")
            self.committed[actor]=True
        elif kind=="wgmma_wait0_postfences":
            need(self.committed[actor],"WGMMA_WAIT_WITHOUT_COMMIT")
            self.wgmma_waited[actor]=True
        elif kind=="release128":
            need(self.waited[actor]==s and self.wgmma_waited[actor],"BANK_RELEASE_BEFORE_WGMMA_DRAIN")
            need(actor not in b["releases"],"DUPLICATE_CONSUMER_RELEASE")
            b["releases"].add(actor)
            self.released[actor].append(s)
            self.waited[actor]=None
            if b["releases"]=={0,1}:
                b["gen"]=None
                b["empty"]^=1
        elif kind=="tail_acquire":
            need(self.produced==self.n,"TAIL_BEFORE_ALL_PRODUCTION")
            if b["gen"] is not None or b["empty"]==(1^((s//2)&1)): raise Blocked()
            need(b["expected"]==0,"TAIL_WITH_RAW_TRANSACTION_PENDING")
            self.tail_acquires.append(s%2)
        elif kind=="tail_done":
            need(self.produced==self.n and all(x["gen"] is None for x in self.banks),"TAIL_BEFORE_TWO_BANK_CONSUMER_DRAIN")
            need(self.tail_acquires==[self.n%2,(self.n+1)%2],"TAIL_WITHOUT_BOTH_BANK_ACQUIRES")
            self.tail_done=True
        elif kind=="consumer_drain":
            need(self.released[actor]==list(range(self.n)),"EARLY_CONSUMER_TAIL")
            self.drains.add(actor)
        elif kind=="bar6":
            need(self.tail_done if actor==2 else actor in self.drains,"BAR6_BEFORE_ROLE_DRAIN")
            self.bar6.add(actor)
        else: raise Violation("UNKNOWN_EVENT")
        self.counts[kind]+=1
        self.trace.append(e)

    def finish(self):
        need(self.bar6=={0,1,2} and self.drains=={0,1} and self.tail_done,"TYPED_TAIL_BEFORE_ALL384_DRAIN_BAR6")
        need(all(x==list(range(4*self.n)) for x in self.issue.values()),"ALL7P_ASCENDING_WHOLEK_INCOMPLETE")
        need(self.counts["tma_issue"]==self.counts["tma_complete"]==4*self.n,"FOUR_TMA_PER_STAGE")
        need(self.counts["publish128"]==self.n and self.counts["release128"]==2*self.n,"PIPELINE_COUNTS")


def ev(kind,actor,s=None,p=None,atom=None):
    return (kind,actor,s,p,atom)


def programs(n):
    producer=[]
    for s in range(n):
        producer += [ev("acquire",2,s),ev("expect",2,s)]
        producer += [ev("tma_issue",2,s,q) for q in range(4)]
        producer += [ev(k,2,s) for k in ("B128stores","B128proxyfences","raw_wait","publish128")]
    producer += [ev("tail_acquire",2,n),ev("tail_acquire",2,n+1),ev("tail_done",2),ev("bar6",2)]
    consumers=[]
    for wg in range(2):
        c=[]
        for s in range(n):
            c += [ev("consumer_wait",wg,s)]
            c += [ev("issue",wg,s,p,a) for p in OWNED[wg] for a in range(4)]
            c += [ev(k,wg,s) for k in ("commit","wgmma_wait0_postfences","release128")]
        c += [ev("consumer_drain",wg),ev("bar6",wg)]
        consumers.append(c)
    tmas=[[ev("tma_complete",3+q,s,q) for s in range(n)] for q in range(4)]
    return consumers+[producer]+tmas


def run(n,policy):
    queues=programs(n); indexes=[0]*7; st=State(n)
    rng=random.Random(policy) if isinstance(policy,int) else None
    while any(indexes[a]<len(queues[a]) for a in range(7)):
        order=list(range(7)) if rng else list(policy)
        if rng: rng.shuffle(order)
        advanced=False
        for a in order:
            if indexes[a]==len(queues[a]): continue
            try: st.step(queues[a][indexes[a]])
            except Blocked: continue
            indexes[a]+=1; advanced=True; break
        need(advanced,"LOGICAL_DEADLOCK")
    st.finish()
    return st


policies=[(2,3,4,5,6,0,1),(2,0,1,3,4,5,6),(0,1,2,3,4,5,6),
          (1,0,2,6,5,4,3),(3,4,5,6,2,0,1),(6,5,4,3,2,1,0)]+list(range(32))
state_results=[]
for n in DOMAINS:
    digests=[]
    for policy in policies:
        st=run(n,policy)
        digests.append(hashlib.sha256(json.dumps(st.trace).encode()).hexdigest())
    state_results.append({"stages":n,"policies":len(policies),"trace_sha256":digests,
                          "A_tma_copies":4*n,"A_transaction_bytes":32768*n,"SIMT_B_transaction_expected_bytes":0,
                          "WG_K16_issues":28*n,"per_WG_stage_release_count":n,"raw_wait_complements_producer_phase":True})


def rejected(trace,n,tag):
    st=State(n)
    try:
        for e in trace: st.step(e)
        st.finish()
    except Violation as exc:
        need(str(exc)==tag,"WRONG_NEGATIVE_CONTROL:"+str(exc)); return str(exc)
    except Blocked:
        raise Violation("NEGATIVE_BLOCKED_INSTEAD_OF_REQUIRED_INVARIANT")
    raise Violation("NEGATIVE_ACCEPTED")


negatives=[]
for n in [1,3,17,34]:
    trace=run(n,policies[0]).trace
    idx=trace.index(ev("B128proxyfences",2,0))+1
    m=trace[:idx]+[ev("publish128",2,0)]+trace[idx:]
    # This policy leaves TMA completions blocked until the producer's raw_wait.
    negatives.append({"stages":n,"control":"early_commit_before_rawA_complete",
                      "rejection":rejected(m,n,"PUBLISH_BEFORE_A_TX_COMPLETE_AND_RAW_WAIT")})
    m=[ev("raw_wait_wrong_phase",2,0) if e==ev("raw_wait",2,0) else e for e in trace]
    negatives.append({"stages":n,"control":"wrong_raw_barrier_phase",
                      "rejection":rejected(m,n,"RAW_WAIT_MUST_COMPLEMENT_PRODUCER_PHASE")})
    m=[e for e in trace if e!=ev("B128proxyfences",2,0)]
    negatives.append({"stages":n,"control":"omit_B_writer_proxy_fences",
                      "rejection":rejected(m,n,"PUBLISH_BEFORE_ALL_B_WRITER_PROXY_FENCES")})
    m=[ev("wrong_expect",2,0) if e==ev("expect",2,0) else e for e in trace]
    negatives.append({"stages":n,"control":"include_Bstore_in_expected_tx_bytes",
                      "rejection":rejected(m,n,"RAW_EXPECT_MUST_BE_ONLY32768_A_BYTES")})
    idx=trace.index(ev("publish128",2,n-1))+1
    m=trace[:idx]+[ev("tail_done",2)]+trace[idx:]
    negatives.append({"stages":n,"control":"tail_before_both_banks_consumer_drain",
                      "rejection":rejected(m,n,"TAIL_BEFORE_TWO_BANK_CONSUMER_DRAIN")})
for n in [3,17,34]:
    trace=run(n,(2,0,1,3,4,5,6)).trace
    idx=trace.index(ev("release128",0,0))+1
    m=trace[:idx]+[ev("force_reuse",2,2)]+trace[idx:]
    negatives.append({"stages":n,"control":"reuse_bank0_before_WG1_release",
                      "rejection":rejected(m,n,"BANK_REUSE_BEFORE_TWO_WG_RELEASES")})

# Original tail layout and whole CTA U ownership remain complete in typed epoch.
tail_words=set()
for p in range(7):
    for t in range(128):
        for v in range(16):
            x=128*(t%4)+(t//4%8)+16*(t//32%4)+64*(v%2)+8*(v//2%2)+512*(v//4)
            r,c=x%64,x//64
            word=p*2048+r*32+(c^((r&1)|((r&6)<<2)))
            assert word not in tail_words
            tail_words.add(word)
assert tail_words==set(range(14336))
owners={linear:tid for tid in range(384) for linear in range(tid,2048,384)}
assert len(owners)==2048 and set(owners)==set(range(2048))

pending={k:None for k in ["actual_compilation","actual_struct_sizeof","actual_member_offsetof",
         "driver_tensor_map_encode","driver_host_shared_optin","compiled_CAN","actual_initial_GPR",
         "producer_q64_fit","consumer_q192_fit","actual_GPR_per_role","actual_stack","actual_spill",
         "SASS","physical_residency","GPU_correctness","CUDA_precision","performance",
         "Graph_fresh_content_GPU_test","ordinary_launch_host_cost"]}
out={k:Path(str(PREFIX)+v) for k,v in [("candidate","candidate.cu"),("delta","source_delta.json"),("review","review.json")]}
assert all(not p.exists() for p in out.values())
out["candidate"].write_text(source)
out["delta"].write_text(json.dumps({"scope":"names/rawA SW128+required alignment/TMA raw stage+single-currentA host descriptor/necessary A stride bound only",
    "parent_path":str(PARENT),"parent_sha256":PARENT_SHA,"candidate_path":str(out["candidate"]),
    "candidate_sha256":sha(out["candidate"]),"ordered_exact_replacements":changes,
    "forward_inverse_exact":True,"normalized_unchanged_sections_sha256":sections},indent=2)+"\n")
assert {p:sha(p) for p in protected}==protected
report={
    "status":"PASS_CPU_SELECTIVE_TMA_A_GEOMETRY_STAGE_SOURCE_ONLY",
    "parent_sha256":PARENT_SHA,"candidate_sha256":sha(out["candidate"]),"model_sha256":sha(__file__),
    "source_protocol_hard_blocker_in_selected_new_geometry":False,
    "original_SW32_hard_blocker":{"BK64_box_inner_bytes":128,"SW32_limit":32,
        "row1k0_raw_old_element_offset":16,"row1k0_dense_element_offset":64},
    "512_only_base_hard_fail":{"base":512,"CuTe_swizzled_slot0_address":576,
        "zero_base_pattern_compatible":False,"candidate_required_alignment":1024},
    "geometry_checks":dict(geometry_checks),"absolute_bases":[0,1024,2048,8192,65536],
    "all_absolute_operand_halfwords_unique_complete":True,"physical_set_sha256":physical_digests,
    "RS_partition_coordinates":4096,"RS_geometry_scope":"128threads x4K16 x4pairs perslot/bank/base; all pairs aligned, same row, same A coordinate bits",
    "abi_estimate":abi_estimate,"host_descriptor_domain":host_domain,
    "state_domains":DOMAINS,"state_results":state_results,"valid_schedules":len(DOMAINS)*len(policies),
    "state_scope":"independent two-bank logical state/phase model with atomic128-thread role groups and4 independent TMA completion actors; bounded schedules, not hardware timing proof",
    "negative_controls":negatives,"negative_control_count":len(negatives),
    "unchanged_source_sections":sections,"tail_words":14336,"U_owners":2048,
    "stage_transaction_expect_bytes":32768,"stage_B_store_expected_transaction_bytes":0,
    "extra_costs":{"producer_raw_mbarrier_waits_per_thread_per_stage":1,"raw_barrier_storage_literal_bytes":16,
        "host_current_A_encode_per_ordinary_or_capture":1,"stronger_shared_alignment":1024,
        "additional_alignment_padding_estimate_vs_parent128":896,
        "SIMT_B_math_and_writes_unchanged":True,"consumer_wait_GMMA_issue_schedule_unchanged":True},
    "protected_before_after_sha256":protected,"frozen_future39_files_protected":17,
    "all_protected_old_bytes_unchanged":True,"pinned_header_sha256":{str(p):sha(p) for p in HEADERS},
    "pending_formal_evidence":pending,"uncompiled":True,"registered":False,
    "formal_results_build_GPU_bench_Git_network_mutation":False,"formal_iteration41_started":False,
    "owner40_work_blocked":False,"writes_restricted_to_prefix":str(PREFIX),
}
out["review"].write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"status":report["status"],"parent_sha256":PARENT_SHA,
    "candidate_sha256":sha(out["candidate"]),"delta_sha256":sha(out["delta"]),"review_sha256":sha(out["review"]),
    "valid_schedules":report["valid_schedules"],"negative_controls_rejected":len(negatives),
    "geometry_checks":dict(geometry_checks),"all_protected_old_bytes_unchanged":True,
    "actual_compilation":None,"GPU_correctness":None,"performance":None},indent=2))

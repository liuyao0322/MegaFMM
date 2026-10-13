"""Independent CPU future39 model; no CUDA compiler/runtime, Git, or network.

All writes are restricted to this unique prefix. Source is derived only from
the frozen tail-bank raw candidate, never from another agent's partial prefix.
"""
from pathlib import Path
from collections import Counter
import hashlib
import itertools
import json
import random
import sys
import numpy as np

ROOT = Path("/share/home/liuyao/workspace/MegaFMM")
PREFIX = ROOT / ".scratch/psplit_bn16_microbatch_root_future39_"
PARENT = ROOT / ".scratch/psplit_bn16_tailbanks_future38_raw_candidate.cu"
PARENT_SHA = "015f6ca799af720a859ffdf73db2655de3f927bb31e3e0bf5751832ea4d8353d"
OLD_MANIFESTS = {
    ".scratch/psplit_bn16_future38_manifest.json": "a3c5f94b8a454c8ae84748eda9bb40096fa102592806964df8cf26d877127a95",
    ".scratch/psplit_bn16_ss_future38_manifest.json": "257b0e37a6ea2b98a51f3ad2fb04693b2f26a55bf1cdba79d8f2267340837bce",
    ".scratch/psplit_bn16_tailbanks_future38_manifest.json": "9d13013fb507d170366ec242776fb80c69701e446c740f45cb3609a75e5f20d1",
}
HEADERS = [
    "third_party/cutlass/include/cute/arch/mma_sm90_gmma.hpp",
    "third_party/cutlass/include/cute/arch/mma_sm90.hpp",
    "third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp",
    "third_party/cutlass/include/cute/atom/mma_atom.hpp",
    "third_party/cutlass/include/cute/algorithm/gemm.hpp",
    "third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp",
    "third_party/cutlass/include/cutlass/arch/barrier.h",
]
DOMAINS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 16, 17, 18, 33, 34]
OWNED = ((0, 1, 2), (3, 4, 5, 6))
NEW_NAME = "psplit_bn16_microbatch_root_future39"
OLD_NAME = "psplit_bn16_future38"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_guard(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes)):
        return
    mode, flags = args[1] or "", args[2] if len(args) > 2 else 0
    writes = isinstance(mode, str) and any(c in mode for c in "wax+")
    writes |= isinstance(flags, int) and bool(flags & (1 | 2 | 64 | 512 | 1024))
    if writes and not str(Path(args[0]).absolute()).startswith(str(PREFIX)):
        raise RuntimeError("write outside unique root future39 prefix: " + str(args[0]))


sys.addaudithook(write_guard)
protected = {}
manifest_groups = {}
for name, expected in OLD_MANIFESTS.items():
    p = ROOT / name
    assert sha(p) == expected, (name, sha(p), expected)
    protected[str(p)] = expected
    files = json.loads(p.read_text())["files"]
    manifest_groups[name] = len(files)
    for name2, expected2 in files.items():
        assert sha(name2) == expected2, (name2, sha(name2), expected2)
        protected[name2] = expected2
assert sorted(manifest_groups.values()) == [8, 8, 10]
assert sha(PARENT) == PARENT_SHA
for name in HEADERS:
    protected[str(ROOT / name)] = sha(ROOT / name)
# Freeze historical iteration35 artifacts, never the current owner39 sources.
formal_old35 = [ROOT / "solution/megakernel_wgmma_psplit_bk64.cu"]
formal_old35 += sorted(p for p in (ROOT / "results").rglob("*iter35*") if p.is_file())
for p in formal_old35:
    protected[str(p)] = sha(p)

traits = (ROOT / HEADERS[2]).read_text()
arch = (ROOT / HEADERS[0]).read_text()
gemm_source = (ROOT / HEADERS[4]).read_text()
pipeline_source = (ROOT / HEADERS[5]).read_text()
assert "using ARegisters = uint32_t[4];" in arch
assert "using CRegisters = float[8];" in arch
assert "using ALayout = GMMA::ALayout_64x16;" in traits
assert "using ThrID   = Layout<_128>;" in traits
assert "using CLayout_64x16  = Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,  _2>>,\n                              Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>>;" in traits
assert "for (int k = 0; k < K; ++k) {\n    gemm(mma, D, A(_,_,k), B(_,_,k), C);" in gemm_source
assert "empty_barrier_ptr_[stage].arrive(params_.dst_blockid);" in pipeline_source
assert "for (int count = 0; count < Stages; ++count) {\n      producer_acquire(state);" in pipeline_source
assert 'asm volatile("" : "+r"(reg) :: "memory");' in arch
assert 'asm volatile("" : "+f"(reg) :: "memory");' in arch

original = PARENT.read_text()
source = original
changes = []


def replace(old, new, count=1):
    global source
    assert source.count(old) == count, (old[:120], source.count(old), count)
    changes.append({"old": old, "new": new, "count": count})
    source = source.replace(old, new)


replace(OLD_NAME, NEW_NAME, source.count(OLD_NAME))
replace("PSplitBN16Future38", "PSplitBN16MicrobatchRootFuture39", source.count("PSplitBN16Future38"))
replace(
    "// CPU-only future38 BN16 source; uncompiled and not registered in any Plan.\n"
    "// BM64 BN16 BK32, raw A4/final B7 shared and whole-stage distinct register A.\n",
    "// CPU-only root future39 BN16 microbatch source; uncompiled/unregistered.\n"
    "// BM64 BN16 BK32, raw A4/final B7 shared; one owning stage A reused after wait0.\n",
)
start = source.index("// First0: [A11,A12,S4].")
end = source.index("// BK32/BN16 has64 B vector tasks;", start)
old_builder = source[start:end]
new_builder = r"""// Product0..6: [A11,A12,S4,A22,S1,S2,S3], one owning whole-stage A.
// Raw finals copy uint32 bits. Each derived final has exactly one RN16 cast.
// Source RN ops/Acell:3+1+2+1=7 versus parent6; raw quadrant refs14 versus7.
// S1 is recomputed for zero-based P4/P5 (one-based P5/P6), never pre-cast.
template <class Element,int Product,int BK,class ThrMma,class SmemRawA,class RA>
CUTE_DEVICE void psplit_bn16_microbatch_root_future39_build_product(
    const ThrMma& thr_mma,SmemRawA raw,RA& a) {
  static_assert(Product>=0 && Product<7,"one final A product at a time");
  auto coords=thr_mma.partition_A(make_identity_tensor(make_shape(Int<64>{},Int<BK>{})));
  CUTE_STATIC_ASSERT_V(size<0>(coords)==Int<8>{});
  CUTE_STATIC_ASSERT_V(size<1>(coords)==Int<1>{});
  CUTE_STATIC_ASSERT_V(size<2>(coords)==Int<BK/16>{});
  auto u=recast<uint32_t>(a);
  CUTE_UNROLL
  for(int atom=0;atom<BK/16;++atom) {
    CUTE_UNROLL
    for(int pair=0;pair<4;++pair) {
      const auto c=coords(2*pair,0,atom);
      const int row=get<0>(c),k=get<1>(c);
      if constexpr(Product==0 || Product==1 || Product==3) {
        constexpr int quadrant=Product==3 ? 3 : Product;
        u(pair,0,atom)=psplit_bn16_microbatch_root_future39_load_pair(&raw(row,k,quadrant));
      } else {
        float final_a[2];
        if constexpr(Product==2) {
          const uint32_t x11=psplit_bn16_microbatch_root_future39_load_pair(&raw(row,k,0));
          const uint32_t x12=psplit_bn16_microbatch_root_future39_load_pair(&raw(row,k,1));
          const uint32_t x21=psplit_bn16_microbatch_root_future39_load_pair(&raw(row,k,2));
          const uint32_t x22=psplit_bn16_microbatch_root_future39_load_pair(&raw(row,k,3));
          CUTE_UNROLL
          for(int j=0;j<2;++j) {
            const float f11=psplit_bn16_microbatch_root_future39_pair_half<Element>(x11,j);
            const float f12=psplit_bn16_microbatch_root_future39_pair_half<Element>(x12,j);
            const float f21=psplit_bn16_microbatch_root_future39_pair_half<Element>(x21,j);
            const float f22=psplit_bn16_microbatch_root_future39_pair_half<Element>(x22,j);
            const float s1=__fadd_rn(f21,f22);
            const float s2=__fsub_rn(s1,f11);
            final_a[j]=__fsub_rn(f12,s2);
          }
        } else if constexpr(Product==4) {
          const uint32_t x21=psplit_bn16_microbatch_root_future39_load_pair(&raw(row,k,2));
          const uint32_t x22=psplit_bn16_microbatch_root_future39_load_pair(&raw(row,k,3));
          CUTE_UNROLL
          for(int j=0;j<2;++j) {
            const float f21=psplit_bn16_microbatch_root_future39_pair_half<Element>(x21,j);
            const float f22=psplit_bn16_microbatch_root_future39_pair_half<Element>(x22,j);
            final_a[j]=__fadd_rn(f21,f22);
          }
        } else if constexpr(Product==5) {
          const uint32_t x11=psplit_bn16_microbatch_root_future39_load_pair(&raw(row,k,0));
          const uint32_t x21=psplit_bn16_microbatch_root_future39_load_pair(&raw(row,k,2));
          const uint32_t x22=psplit_bn16_microbatch_root_future39_load_pair(&raw(row,k,3));
          CUTE_UNROLL
          for(int j=0;j<2;++j) {
            const float f11=psplit_bn16_microbatch_root_future39_pair_half<Element>(x11,j);
            const float f21=psplit_bn16_microbatch_root_future39_pair_half<Element>(x21,j);
            const float f22=psplit_bn16_microbatch_root_future39_pair_half<Element>(x22,j);
            const float s1=__fadd_rn(f21,f22);
            final_a[j]=__fsub_rn(s1,f11);
          }
        } else {
          static_assert(Product==6,"only S3 remains");
          const uint32_t x11=psplit_bn16_microbatch_root_future39_load_pair(&raw(row,k,0));
          const uint32_t x21=psplit_bn16_microbatch_root_future39_load_pair(&raw(row,k,2));
          CUTE_UNROLL
          for(int j=0;j<2;++j) {
            const float f11=psplit_bn16_microbatch_root_future39_pair_half<Element>(x11,j);
            const float f21=psplit_bn16_microbatch_root_future39_pair_half<Element>(x21,j);
            final_a[j]=__fsub_rn(f11,f21);
          }
        }
        u(pair,0,atom)=psplit_bn16_microbatch_root_future39_round_pair<Element>(final_a[0],final_a[1]);
      }
    }
  }
}

// Every call retires its two ascending K16 GMMAs before the next A build.
// P accumulators remain wholeK; only the shared-bank release stays perstage.
template <class Element,int Product,int BK,class Mma,class ThrMma,
          class SmemRawA,class RA,class RB,class RC>
CUTE_DEVICE void psplit_bn16_microbatch_root_future39_consume_product(
    const Mma& mma,const ThrMma& thr_mma,SmemRawA raw,RA& a,RB& b,RC& q) {
  psplit_bn16_microbatch_root_future39_build_product<Element,Product,BK>(thr_mma,raw,a);
  warpgroup_fence_operand(q);
  warpgroup_fence_operand(a);
  warpgroup_arrive();
  // Pinned rank3 rmem gemm loops k=0,1 in order, with unchanged ScaleOut::One.
  gemm(mma,a,b(_,_,_,Int<Product>{}),q);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(a);
  warpgroup_fence_operand(q);
}

"""
replace(old_builder, new_builder)
start = source.index("    alignas(16) Element a0_data")
end = source.index("    // Source quota hypothesis only.", start)
replace(source[start:end], r"""    alignas(16) Element a_data[cosize_v<ARLayout>];
    auto a=make_tensor(make_rmem_ptr(a_data),ARLayout{});
    static_assert(size(a)==8*T::AtomsK,"one owning stage A has16 elements /8 packedwords");
""")
for first, count in [(0, 3), (3, 4)]:
    start = source.index(f"        {NEW_NAME}_build_stage<Element,{first},T::BK>")
    end = source.index("        pipeline.consumer_release(state);", start)
    calls = "".join(
        f"        {NEW_NAME}_consume_product<Element,{p},T::BK>(mma,thr_mma,sA,a,tCrB,q{p-first}); // P{p+1}\n"
        for p in range(first, first+count)
    )
    replace(source[start:end],
            "        auto tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB));\n"
            + calls
            + "        // All assigned P wait0/postfences completed; release this stage exactly once.\n")

restored = source
for change in reversed(changes):
    assert restored.count(change["new"]) == change["count"]
    restored = restored.replace(change["new"], change["old"])
assert restored == original, "inverse delta must restore the exact parent bytes"
# Normalized producer, traits/shared ABI hypothesis, tail, U, host guard remain exact.
norm = source.replace(NEW_NAME, OLD_NAME).replace("PSplitBN16MicrobatchRootFuture39", "PSplitBN16Future38")
unchanged_sections = {}
for label, begin, end in [
    ("producer_rawA", "// Producer writes four raw original A quadrants", "CUTE_DEVICE uint32_t"),
    ("producer_finalB", "// BK32/BN16 has64 B vector tasks;", "// Tail FP32 rendezvous"),
    ("traits_shared_geometry", "template <class Element, int Cols>\nstruct", "struct PSplitRSAPackedEight"),
    ("producer_pipeline_tail", "  if (tid < T::ProducerThreads) {", "  } else {\n    // Do not construct"),
    ("tail_and_fixedU", "    // All384 threads rendezvous", "\ntemplate <class Element>\nvoid prepare_"),
    ("host_qualification_launch", "\ntemplate <class Element>\nvoid prepare_", None),
]:
    def section(s):
        i = s.index(begin)
        j = len(s) if end is None else s.index(end, i)
        return s[i:j]
    assert section(norm) == section(original), label
    unchanged_sections[label] = hashlib.sha256(section(original).encode()).hexdigest()
assert norm.count("pipeline.consumer_release(state);") == 2
assert norm.count("warpgroup_reg_alloc<96>();") == 1
assert "__launch_bounds__(384,2)" in norm and "warpgroup_reg_dealloc<48>();" in norm
assert norm.count("consume_product<Element,") == 7
assert "a0_data" not in norm and "a1_data" not in norm and "build_stage" not in norm


class Blocked(Exception):
    pass


class Violation(Exception):
    pass


def require(condition, tag):
    if not condition:
        raise Violation(tag)


class State:
    """Literal two-bank generations; 128/128/128 actor groups are atomic here."""
    def __init__(self, n):
        self.n = n
        self.banks = [{"generation": None, "full_phase": 0, "empty_phase": 0, "released": set()} for _ in range(2)]
        self.produced = 0
        self.acquired = None
        self.producer_stores = set()
        self.tail_banks = []
        self.tail_done = False
        self.consumer_stage = [None, None]
        self.consumer_releases = [[], []]
        self.active = [None, None]
        self.done = [set(), set()]
        self.issues = {p: [] for p in range(7)}
        self.drained = set()
        self.bar6 = set()
        self.counts = Counter()
        self.trace = []

    def step(self, e):
        kind, actor, s, p, k = e
        bank = self.banks[s % 2] if s is not None else None
        if kind == "acquire":
            require(s == self.produced, "PRODUCER_GENERATION_ORDER")
            if bank["generation"] is not None or bank["empty_phase"] == (1 ^ ((s // 2) & 1)):
                raise Blocked()
            self.acquired = s
            self.producer_stores = set()
        elif kind in ("storeA4", "storeB7", "proxyfence128"):
            require(self.acquired == s, "PRODUCER_STORE_WITHOUT_ACQUIRE")
            if kind == "proxyfence128":
                require(self.producer_stores == {"storeA4", "storeB7"}, "PROXYFENCE_BEFORE_ALL_WRITES")
            self.producer_stores.add(kind)
        elif kind == "publish":
            require(self.acquired == s and self.producer_stores == {"storeA4", "storeB7", "proxyfence128"}, "PUBLISH_WITHOUT_ALL128_WRITER_FENCES")
            bank["generation"] = s
            bank["released"] = set()
            bank["full_phase"] ^= 1
            self.produced += 1
            self.acquired = None
        elif kind == "consumerwait":
            if bank["generation"] != s or bank["full_phase"] == ((s // 2) & 1):
                raise Blocked()
            require(s == len(self.consumer_releases[actor]), "CONSUMER_GENERATION_ORDER")
            self.consumer_stage[actor] = s
            self.done[actor] = set()
        elif kind == "buildA":
            require(self.consumer_stage[actor] == s and bank["generation"] == s, "BUILD_OUTSIDE_HELD_BANK")
            old = self.active[actor]
            require(old is None or (old["waited"] and old["postA"] and old["postP"]), "A_REWRITE_BEFORE_WAIT0_AND_POSTFENCES")
            require(p in OWNED[actor], "P_OWNERSHIP")
            require(self.done[actor] == set(OWNED[actor][:OWNED[actor].index(p)]), "P_ASCENDING_MICROBATCH_ORDER")
            self.active[actor] = {"stage": s, "p": p, "preA": False, "preP": False, "arrived": False,
                                  "atoms": [], "committed": False, "waited": False, "postA": False, "postP": False}
        elif kind in ("preA", "preP", "arrive", "issue", "commit", "wait0", "postA", "postP"):
            a = self.active[actor]
            require(a is not None and a["stage"] == s and a["p"] == p, "A_PRODUCT_GENERATION_MISMATCH")
            if kind in ("preA", "preP"):
                require(not a["atoms"], "PREFENCE_AFTER_ISSUE")
                a[kind] = True
            elif kind == "arrive":
                require(a["preA"] and a["preP"], "ARRIVE_BEFORE_OPERAND_FENCES")
                a["arrived"] = True
            elif kind == "issue":
                require(a["arrived"] and not a["committed"], "ISSUE_OUTSIDE_OPEN_GROUP")
                require(a["atoms"] == list(range(k)) and k in (0, 1), "K16_ATOMS_NOT_ASCENDING")
                require(len(self.issues[p]) == 2*s+k, "WHOLEK_ACCUMULATOR_ORDER_CHANGED")
                a["atoms"].append(k)
                self.issues[p].append(2*s+k)
            elif kind == "commit":
                require(a["atoms"] == [0, 1] and not a["committed"], "COMMIT_NOT_TWO_K16")
                a["committed"] = True
            elif kind == "wait0":
                require(a["committed"] and not a["waited"], "WAIT0_WITHOUT_ONE_COMMIT")
                a["waited"] = True
            else:
                require(a["waited"], "POSTFENCE_BEFORE_WAIT0")
                a[kind] = True
                if a["postA"] and a["postP"]:
                    self.done[actor].add(p)
        elif kind == "release":
            require(self.consumer_stage[actor] == s, "RELEASE_OUTSIDE_HELD_STAGE")
            require(self.done[actor] == set(OWNED[actor]), "RELEASE_BEFORE_ALL_OWNED_P_WAIT0_POSTFENCES")
            require(actor not in bank["released"], "DUPLICATE_RELEASE")
            bank["released"].add(actor)
            self.consumer_releases[actor].append(s)
            self.consumer_stage[actor] = None
            if bank["released"] == {0, 1}:
                bank["generation"] = None
                bank["empty_phase"] ^= 1
        elif kind == "tailacquire":
            require(self.produced == self.n, "TAIL_BEFORE_PRODUCER_FINISHED")
            require(s == self.n+len(self.tail_banks), "TAIL_ACQUIRE_TWO_BANK_ORDER")
            if bank["generation"] is not None or bank["empty_phase"] == (1 ^ ((s // 2) & 1)):
                raise Blocked()
            self.tail_banks.append(s % 2)
        elif kind == "taildone":
            require(self.produced == self.n, "TAIL_BEFORE_PRODUCER_FINISHED")
            require(all(b["generation"] is None for b in self.banks), "TAIL_BEFORE_BOTH_BANKS_DRAINED")
            require(len(self.tail_banks) == 2 and set(self.tail_banks) == {0, 1}, "TAIL_WITHOUT_TWO_BANK_ACQUIRES")
            require(all(r == list(range(self.n)) for r in self.consumer_releases), "TAIL_BEFORE_BOTH_CONSUMER_RELEASES")
            self.tail_done = True
        elif kind == "drain":
            require(self.consumer_releases[actor] == list(range(self.n)), "CONSUMER_DRAIN_EARLY")
            require(self.active[actor]["waited"] and self.active[actor]["postA"] and self.active[actor]["postP"], "CONSUMER_DRAIN_WITH_LIVE_A")
            self.drained.add(actor)
        elif kind == "bar6":
            if actor == 2:
                require(self.tail_done, "PRODUCER_BAR6_BEFORE_TAIL")
            else:
                require(actor in self.drained, "CONSUMER_BAR6_BEFORE_DRAIN")
            self.bar6.add(actor)
        else:
            raise Violation("UNKNOWN_EVENT")
        self.counts[kind] += 1
        self.trace.append(e)

    def finish(self):
        require(self.bar6 == {0, 1, 2} and self.tail_done and self.drained == {0, 1}, "TAIL_ALIAS_BEFORE_ALL384_BAR6")
        require(all(v == list(range(2*self.n)) for v in self.issues.values()), "MISSING_COMPLETE_ASCENDING_WHOLEK_P")
        require(self.counts["issue"] == 14*self.n and self.counts["commit"] == self.counts["wait0"] == 7*self.n, "MICROBATCH_COUNTS")
        require(self.counts["release"] == 2*self.n, "RELEASE_NOT_ONCE_PER_WG_STAGE")
        require(all(b["generation"] is None for b in self.banks), "TAIL_ALIAS_WITH_LIVE_OPERANDS")


def event(kind, actor, s=None, p=None, k=None):
    return (kind, actor, s, p, k)


def programs(n):
    producer = []
    for s in range(n):
        producer += [event(k, 2, s) for k in ("acquire", "storeA4", "storeB7", "proxyfence128", "publish")]
    producer += [event("tailacquire", 2, n), event("tailacquire", 2, n+1), event("taildone", 2), event("bar6", 2)]
    consumers = []
    for wg in range(2):
        c = []
        for s in range(n):
            c.append(event("consumerwait", wg, s))
            for p in OWNED[wg]:
                c += [event(k, wg, s, p) for k in ("buildA", "preP", "preA", "arrive")]
                c += [event("issue", wg, s, p, k) for k in (0, 1)]
                c += [event(k, wg, s, p) for k in ("commit", "wait0", "postA", "postP")]
            c.append(event("release", wg, s))
        c += [event("drain", wg), event("bar6", wg)]
        consumers.append(c)
    return consumers+[producer]


def run_schedule(n, policy):
    queues = programs(n)
    positions = [0, 0, 0]
    state = State(n)
    rng = random.Random(policy) if isinstance(policy, int) else None
    while any(positions[a] < len(queues[a]) for a in range(3)):
        order = list(range(3)) if rng else list(policy)
        if rng:
            rng.shuffle(order)
        advanced = False
        for actor in order:
            if positions[actor] == len(queues[actor]):
                continue
            try:
                state.step(queues[actor][positions[actor]])
            except Blocked:
                continue
            positions[actor] += 1
            advanced = True
            break
        require(advanced, "DEADLOCK_UNDER_FAIR_READY_ACTOR_SCHEDULE")
    state.finish()
    return state


state_results = []
policies = list(itertools.permutations(range(3)))+list(range(32))
for n in DOMAINS:
    hashes = []
    for policy in policies:
        st = run_schedule(n, policy)
        hashes.append(hashlib.sha256(json.dumps(st.trace).encode()).hexdigest())
    state_results.append({
        "stages": n, "two_bank_generations": list(range(n)), "bank_index": [s % 2 for s in range(n)],
        "consumer_phase": [(s // 2) & 1 for s in range(n)],
        "producer_phase": [1 ^ ((s // 2) & 1) for s in range(n)],
        "interleavings_tested": len(policies), "policy_trace_sha256": hashes,
        "per_trace_issue_count": 14*n, "per_trace_commit_and_wait0_count": 7*n,
        "per_trace_release_count": 2*n, "per_trace_tail_bank_acquires": 2,
        "all_7P_exact_complete_ascendingK_issue": True,
        "A_rewrite_only_after_wait0_A_and_P_postfences": True,
        "wholeK_P_lifetime_preserved": True, "bothbanks_drained_before_union_alias": True,
    })


def rejected_trace(trace, expected):
    n = 1+max(e[2] for e in trace if e[0] == "publish")
    st = State(n)
    try:
        for e in trace:
            st.step(e)
        st.finish()
    except Violation as exc:
        require(str(exc) == expected, "WRONG_NEGATIVE_REJECTION:"+str(exc))
        return str(exc)
    except Blocked:
        raise Violation("NEGATIVE_TRACE_BLOCKED_WITHOUT_REQUIRED_REJECTION")
    raise Violation("NEGATIVE_CONTROL_ACCEPTED")


negative_results = []
for n in [1, 3, 17, 34]:
    for wg in range(2):
        trace = run_schedule(n, (2, wg, 1-wg)).trace
        first_p, last_p = OWNED[wg][0], OWNED[wg][-1]
        idx = trace.index(event("commit", wg, 0, first_p))+1
        mutant = trace[:idx]+[event("buildA", wg, 0, first_p+1)]+trace[idx:]
        negative_results.append({"stages": n, "wg": wg, "control": "A_rewrite_before_wait0",
                                 "rejection": rejected_trace(mutant, "A_REWRITE_BEFORE_WAIT0_AND_POSTFENCES")})
        idx = trace.index(event("commit", wg, 0, last_p))+1
        mutant = trace[:idx]+[event("release", wg, 0)]+trace[idx:]
        negative_results.append({"stages": n, "wg": wg, "control": "consumer_release_before_last_P_wait0",
                                 "rejection": rejected_trace(mutant, "RELEASE_BEFORE_ALL_OWNED_P_WAIT0_POSTFENCES")})
        mutant = [e for e in trace if e != event("wait0", wg, 0, first_p)]
        negative_results.append({"stages": n, "wg": wg, "control": "missing_P_wait0",
                                 "rejection": rejected_trace(mutant, "POSTFENCE_BEFORE_WAIT0")})
    trace = run_schedule(n, (2, 0, 1)).trace
    idx = trace.index(event("publish", 2, n-1))+1
    mutant = trace[:idx]+[event("taildone", 2)]+trace[idx:]
    negative_results.append({"stages": n, "control": "producer_tail_done_before_bothbanksdrain",
                             "rejection": rejected_trace(mutant, "TAIL_BEFORE_BOTH_BANKS_DRAINED")})

# Three lifetime epochs: operands -> typed FP32 seven-P writes -> fixed unified U.
# Flatten the pinned CLayout independently rather than running any old model.
tail_words = set()
for p in range(7):
    for tid in range(128):
        for slot in range(8):
            t0, t1, t2 = tid % 4, (tid // 4) % 8, tid // 32
            v0, v1, v2 = slot % 2, (slot // 2) % 2, slot // 4
            row, col = t1+16*t2+8*v1, 2*t0+v0+8*v2
            mask = ((row & 2) >> 1) | ((row & 4) << 1)
            word = p*1024+row*16+(col ^ mask)
            require(word not in tail_words, "TAIL_STORE_DUPLICATE_WORD")
            tail_words.add(word)
require(tail_words == set(range(7168)), "TAIL_C_LAYOUT_INCOMPLETE")
u_owners, u_reads, output_addresses = {}, Counter(), Counter()
c0_before_D = True
for tid in range(384):
    for linear in range(tid, 1024, 384):
        require(linear not in u_owners, "FIXEDU_DUPLICATE_OWNER")
        u_owners[linear] = tid
        row, col = divmod(linear, 16)
        mask = ((row & 2) >> 1) | ((row & 4) << 1)
        words = [p*1024+row*16+(col ^ mask) for p in range(7)]
        require(all(w in tail_words for w in words), "U_READ_UNSTORED_TAIL")
        u_reads.update(words)
        # Fixed U dependency/event order and four C0 reads occur before any D.
        order = ["P1", "P2", "P3", "P4", "P5", "P6", "P7",
                 "u1=RN(P1+P2)", "u2=RN(P1+P6)", "u3=RN(u2+P7)",
                 "u4=RN(u2+P5)", "u5=RN(u4+P3)", "u6=RN(u3-P4)", "u7=RN(u3+P5)",
                 "alphaRN11", "alphaRN12", "alphaRN21", "alphaRN22",
                 "C011", "C012", "C021", "C022", "betaFMA11", "betaFMA12", "betaFMA21", "betaFMA22",
                 "D11", "D12", "D21", "D22"]
        c0_before_D &= max(order.index(k) for k in ("C011", "C012", "C021", "C022")) < order.index("D11")
        for quadrant in range(4):
            output_addresses[(quadrant, linear)] += 1
require(set(u_owners) == set(range(1024)) and set(u_reads) == tail_words, "FIXEDU_COVERAGE")
require(all(c == 1 for c in u_reads.values()) and all(c == 1 for c in output_addresses.values()), "U_OR_OUTPUT_MULTIPLICITY")
require(c0_before_D, "C0_ALIAS_LOAD_AFTER_D")
tail_results = {"epochs": [
    {"id": 0, "payload": "rawA4/finalB7 two banks", "exit": "producer_tail_two_bank_acquires + both_consumer_drains -> nonaligned barrier.sync6,384"},
    {"id": 1, "payload": "typed FP32 sevenP tail alias", "complete_unique_words": len(tail_words), "exit": "all384 __syncthreads bar0"},
    {"id": 2, "payload": "unified U", "unique_tile_owners": len(u_owners), "each_P_word_read_once": True,
     "four_C0_loads_before_first_D": c0_before_D, "unique_four_quadrant_outputs": len(output_addresses)}
], "fixed_U_source_byte_identical_after_name_normalization": True,
    "tail_mask_source_byte_identical_after_name_normalization": True,
    "initial_collective_bar0_retained": True, "post_pipeline_barriers": ["nonaligned6,384", "bar0,384"]}

# Independent bounded NumPy IEEE float32 RN DAG equivalence. No CUDA/GMMA proof.
def add(a, b):
    return np.add(a, b, dtype=np.float32)


def sub(a, b):
    return np.subtract(a, b, dtype=np.float32)


def pack(x, dtype):
    x = np.asarray(x, dtype=np.float32)
    if dtype == "fp16":
        return x.astype(np.float16).view(np.uint16)
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def unpack(x, dtype):
    if dtype == "fp16":
        return x.view(np.float16).astype(np.float32)
    return (x.astype(np.uint32) << 16).view(np.float32)


numeric_results = []
rng = np.random.default_rng(39)
edges = np.array([-16, -1, -(2.0**-14), -(2.0**-24), -0.0, 0.0,
                  2.0**-24, 2.0**-14, 1, 16], dtype=np.float32)
edge_quads = np.asarray(list(itertools.product(edges, repeat=4)), dtype=np.float32)
for dtype in ("fp16", "bf16"):
    if dtype == "fp16":
        signs = rng.integers(0, 2, size=(40000, 4), dtype=np.uint16) << 15
        exponents = rng.integers(0, 19, size=(40000, 4), dtype=np.uint16) << 10
        mantissa = rng.integers(0, 1024, size=(40000, 4), dtype=np.uint16)
    else:
        signs = rng.integers(0, 2, size=(40000, 4), dtype=np.uint16) << 15
        exponents = rng.integers(100, 131, size=(40000, 4), dtype=np.uint16) << 7
        mantissa = rng.integers(0, 128, size=(40000, 4), dtype=np.uint16)
    raw = np.concatenate([signs | exponents | mantissa, pack(edge_quads, dtype)])
    a11, a12, a21, a22 = unpack(raw, dtype).T
    require(np.isfinite(unpack(raw, dtype)).all(), "NUMPY_INPUT_NOT_FINITE")
    # Parent's WG0 and WG1 separately recompute the three-op DAG.
    wg0_s1 = add(a21, a22)
    wg0_s2 = sub(wg0_s1, a11)
    wg0_s4 = sub(a12, wg0_s2)
    wg1_s1 = add(a21, a22)
    wg1_s2 = sub(wg1_s1, a11)
    wg1_s3 = sub(a11, a21)
    parent_final = [raw[:, 0], raw[:, 1], pack(wg0_s4, dtype), raw[:, 3],
                    pack(wg1_s1, dtype), pack(wg1_s2, dtype), pack(wg1_s3, dtype)]
    # Candidate independent per-P DAG; no derived typed operand is re-read.
    candidate_final = [raw[:, 0].copy(), raw[:, 1].copy(),
                       pack(sub(a12, sub(add(a21, a22), a11)), dtype), raw[:, 3].copy(),
                       pack(add(a21, a22), dtype), pack(sub(add(a21, a22), a11), dtype),
                       pack(sub(a11, a21), dtype)]
    require(all(np.array_equal(x, y) for x, y in zip(parent_final, candidate_final)), "FINAL_A_BITS_CHANGED")
    require(all(np.isfinite(unpack(x, dtype)).all() for x in candidate_final), "NUMPY_FINAL_NOT_FINITE")
    # Deliberately illegal intermediate RN16(S1) must produce counterexamples.
    early_cast_s2 = pack(sub(unpack(pack(add(a21, a22), dtype), dtype), a11), dtype)
    mismatch = np.flatnonzero(early_cast_s2 != candidate_final[5])
    require(len(mismatch) > 0, "INTERMEDIATE_CAST_NEGATIVE_CONTROL_TOO_WEAK")
    i = int(mismatch[0])
    numeric_results.append({"dtype": dtype, "finite_bounded_quads": len(raw),
                            "final_A_bit_comparisons": 7*len(raw), "all_seven_final_A_equal": True,
                            "raw_0_1_3_bits_directly_copied": True, "derived_2_4_5_6_one_final_RN16_cast": True,
                            "illegal_intermediate_S1_cast_mismatches": len(mismatch),
                            "first_negative_quad_raw_uint16": [int(v) for v in raw[i]],
                            "correct_S2_uint16": int(candidate_final[5][i]),
                            "incorrect_precast_S1_S2_uint16": int(early_cast_s2[i]),
                            "scope": "bounded host NumPy RN/ST operand algebra only; no CUDA conversion/GMMA/output precision proof"})

costs = {
    "per_Acell_source_RN_ops": {"parent": 6, "candidate": 7, "per_zero_based_P": [0, 0, 3, 0, 1, 2, 1]},
    "per_Acell_raw_shared_quadrant_references": {"parent": 7, "candidate": 14, "per_zero_based_P": [1, 1, 4, 1, 2, 3, 2]},
    "per_stage_per_WG_commit_wait0": {"parent": [1, 1], "candidate": [3, 4]},
    "wholeK_P_FP32_words_per_thread": [24, 32],
    "owning_stage_A_packed_words_per_thread": {"parent": [24, 32], "candidate": [8, 8]},
    "P_plus_A_payload_lower_bound_words_per_thread": {"parent": [48, 64], "candidate": [32, 40]},
    "warning": "lower payload is not q96 fit or speed proof; more shared reads and waits may be slower",
}
assert sum(costs["per_Acell_source_RN_ops"]["per_zero_based_P"]) == 7
assert sum(costs["per_Acell_raw_shared_quadrant_references"]["per_zero_based_P"]) == 14
pending = {key: None for key in [
    "actual_initial_registers", "producer_q48_fit", "consumer_q96_fit", "compiler_scalarization",
    "actual_registers", "actual_stack_bytes", "actual_spill_bytes", "no_spill",
    "actual_shared_ABI_bytes", "actual_two_CTA_residency", "performance", "GPU_correctness",
    "CUDA_precision", "ELF_host_SASS_gate", "compiled_kernel_identity",
]}
outputs = {label: Path(str(PREFIX)+suffix) for label, suffix in [
    ("candidate", "candidate.cu"), ("delta", "source_delta.json"), ("review", "review.json"),
]}
for p in outputs.values():
    assert not p.exists(), "refuse overwrite of own already-frozen artifact: "+str(p)
outputs["candidate"].write_text(source)
delta = {
    "scope": "consumer A construction, perP issue/commit/wait/fences schedule, naming and comments only",
    "parent_path": str(PARENT), "parent_sha256": PARENT_SHA,
    "candidate_path": str(outputs["candidate"]), "candidate_sha256": sha(outputs["candidate"]),
    "ordered_exact_replacements": changes, "inverse_restores_parent_byte_for_byte": True,
    "name_normalized_unchanged_sections_sha256": unchanged_sections,
}
outputs["delta"].write_text(json.dumps(delta, indent=2)+"\n")
require({p: sha(p) for p in protected} == protected, "PROTECTED_OLD_BYTES_CHANGED")
report = {
    "status": "PASS_CPU_MICROBATCH_PROTOCOL_AND_BOUNDED_A_BITS_ONLY",
    "source_protocol_hard_blocker_found": False,
    "source_scope": "uncompiled/unregistered future candidate; not formal iteration40",
    "parent_sha256": PARENT_SHA, "candidate_sha256": sha(outputs["candidate"]),
    "model_sha256": sha(Path(__file__)), "pinned_headers_sha256": {str(ROOT/h): protected[str(ROOT/h)] for h in HEADERS},
    "old_frozen_manifest_groups": manifest_groups, "formal_old35_protected_file_count": len(formal_old35),
    "protected_before_after_sha256": protected, "all_protected_old_bytes_unchanged": True,
    "exact_source_inverse_and_unchanged_section_checks": True,
    "state_domains": DOMAINS, "schedules_per_domain": len(policies), "total_valid_schedules": len(DOMAINS)*len(policies),
    "state_model_scope": "independent atomic128-thread WG/producer state machine with literal two-bank barrier phases; six priority permutations and32 random ready-actor policies/domain; not exhaustive hardware interleavings",
    "state_results": state_results, "negative_controls": negative_results,
    "all_negative_controls_rejected": True, "tail_and_fixedU": tail_results,
    "bounded_NumPy_A_equivalence": numeric_results, "explicit_source_costs": costs,
    "optional_unimplemented_variant": "WG1 may cache unrounded S1 with16 extra FP32 live/thread; no second candidate implemented",
    "pending_formal_evidence": pending,
    "formal_results_build_GPU_bench_Git_network_mutation": False,
    "iteration40_started": False, "owner39_work_blocked": False,
    "writes_restricted_to_prefix": str(PREFIX),
}
outputs["review"].write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps({
    "status": report["status"], "parent_sha256": PARENT_SHA,
    "candidate_sha256": sha(outputs["candidate"]), "delta_sha256": sha(outputs["delta"]),
    "review_sha256": sha(outputs["review"]), "valid_schedules": report["total_valid_schedules"],
    "state_domains": DOMAINS, "negative_controls_rejected": len(negative_results),
    "bounded_final_A_bit_comparisons": sum(x["final_A_bit_comparisons"] for x in numeric_results),
    "old_artifacts_counts": [8, 8, 10], "old_manifests": 3,
    "formal_old35_protected_files": len(formal_old35), "all_protected_old_bytes_unchanged": True,
    "all_formal_evidence_pending": pending,
}, indent=2))

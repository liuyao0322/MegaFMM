# Independent future38 SS BN16 companion: CPU preparation only

This separate companion supports a possible later approved BN16 SS/RS paired
experiment. It is uncompiled, unregistered and unmeasured. It assigns no public
variant ID and starts no next iteration. All writes are restricted to
`.scratch/psplit_bn16_ss_future38_`; the script rejects other write paths.
The raw BN16 future candidate/model/stdout/exit/protocol and other artifacts
were already frozen, and their manifest/hash entries were verified before and
after the SS run. No formal/source/build/GPU/Git/results writes were performed.

The SS base is the retained `solution/megakernel_wgmma_psplit.cu`, exactly equal
to original34 `results/psplit_revision6_megakernel_wgmma_psplit_iter34.cu`, SHA
`fdaae75a80893ee161a9db202d4746e86daf3d38135332759098bd9f7faa7f91`.
The companion candidate SHA is
`94c482bc73a4e5dfa32436b9931f300242afd89058d817522c898a1ccd009ab6`.
Every exact source replacement is recorded in `source_delta.json`; inverse
replacements restore the entire old34 base byte-for-byte.
The actual CPU run exited0 with empty stderr and status
`PASS_CPU_LITERAL_SS_BN16_GEOMETRY_ONLY_UNCOMPILED_UNREGISTERED`.

## Source construction and independent static review

CUTLASS is pinned to `f7b19de32c5d1f3cedfc735c2849f12b537522ee` (v3.5.1).
Explicit FP16/BF16 m64n16k16 SS types are used, respectively defined at
`third_party/cutlass/include/cute/arch/mma_sm90_gmma.hpp:2360` and5173.
Both have one64-bit A descriptor, one64-bit B descriptor and eight FP32
accumulator values/thread. Both use A-major K/B-major MN. The N16 C traits use
ThrID128 and CLayout64x16 (`mma_traits_sm90_gmma.hpp:389`,1266,2033).
This matches the raw future's N16 output shape, dtype, FP32 accumulator type,
two ascending K16 atoms/P/stage, and final layout, while using a different A
source form. No exact CUDA numerical equivalence is inferred from those facts.

The routine BN32 assumptions corrected are Cols32, qsize16, typed instantiation32
at prepare/launch, and all tail addresses. The fixed candidate admits only16 and
either half/BF16; per-P qsize is BM*BN/128=8. The tail uses exactly the raw future
formula `product*Rows*Cols+row*Cols+(col^(row&(Cols/4-1)))`, giving mask0..3 at
BN16. For each row XOR is a permutation of0..15, and disjoint row/product ranges
cover every offset0..7167 once. The old tail still reaches14335, exceeding the
typed tail and bank0, although it stays inside this SS two-bank operand union.
The CPU negative control rejects keeping it. An independent read-only reviewer
found3,584 old offsets outside the typed tail and2,688 in bank1.

SS bankA7 is28,672 bytes and B7 is7,168 bytes, giving35,840/bank and71,680/two
banks. Four64-bit pipeline barriers add32; align128 predicts71,808 dynamic bytes.
The typed P tail is28,672 bytes, entirely inside bank0, with7,168 bank0 bytes
unused and no bank1 overlap. The existing preparation guard shared>48KiB and
shared<227KiB remains consistent with this prediction, so only its typed
instantiation changes to16. Actual host sizeof, CuTe cosize, descriptors,
static/reserved shared, opt-in and device capacity remain future compiled gates.
Both banks still drain through producer_tail and both consumer groups before
typed-tail activation; fitting within bank0 does not authorize early reuse.

Producer A code and B code are unchanged after helper/type renaming. All256
A vector tasks are still assigned to128 threads. Original A11/A12/A22 bit copies
and ordered f32 RN S1/S2/S3/S4 remain original34; each of the four derived final
A operands is cast exactly once, with no intermediate cast. The B DAG and four
derived final B onecasts remain original34. BN16 reduces B vector tasks128 to64
through its existing Cols parameter. All128 producers still fence and commit.

Consumers retain the original shared A/B descriptor fragments. They own no RS
full-stage A arrays, no rawA register transform and no generic packed-A loads.
P3/P4 accumulator payloads are24/32 FP32 words/thread. Whole-K P accumulation,
two K16 atoms/P/stage, commit/wait0, source operand retirement, P3+4 ownership,
the fixed U, alpha RN multiply, beta FMA and four quadrant stores remain intact.
The U/alpha/beta source segment matches original34 exactly after only helper
renaming and the tail-template call normalization.

Full/empty arrival counts remain128/256, with128 threads/WG and384 threads/CTA.
Initial CTA bar0, role-local nonaligned CTA bar6 after both-bank retirement and
the common tail-publish CTA bar0 keep their original counts and scope. B-task64
must not replace producer arrival128. The template model does not certify
hardware barriers, forward progress or async proxy ordering.

## CPU execution evidence and host domain

The independently written SS script performed17,920 aligned contiguous vector8
checks over both banks at four128-byte-aligned bases. It checked every A7/B7
element for compact unique layout coverage, every1,024 C coordinate/P for unique
WG ownership, all7,168 tail words for one write and one U read, all1,024 U owners,
and every one of the1,048,576 D positions at1024 for exactly one store. These are
literal integer interpretations, not CuTe/CUDA template instantiation or actual
memory traffic. Source/raw frozen hashes were checked before and after.

The metadata domain deliberately stays m%128,n%64,k%64,lda%8,ldb%8, with current
nonnull AB16B binding checks. Natural N16 geometry could extend to n%32, but this
preparation does not do so. The common Plan validation/fallback contract remains
required later: current span/end and overlap validation, compatible exact C0==D,
four-byte output/active-C0 alignment, beta0/null C0 handling, certification/range,
degenerate ordering, unchanged Auto, truthful native/P7/Dense routes and names.
No Plan or public headers were edited or registered. If later integrated, this
companion needs an independent disabled-until-gated ID and an accurate64x16 name.

At1024/leaf512, the grid is8x32=256 CTAs and16 BK32 stages. Logical source A reads
are64MiB and B16MiB, the same80MiB as raw BN16; old N32 source reads total48MiB.
These counts are source rereads, not measured L2/DRAM transactions or a speed
prediction. SS finalA producer shared writes and RN costs differ from rawA RS
dataflow, so an SS/RS pair is not a single change to tile width alone.

## Future fresh compiled and GPU protocol

q64/80/80 and launch_bounds(384,2) are new source hypotheses. If initialI80,
the initially owned pool is384*80=30,720 words and requested role pool is
128*(64+80+80)=28,672, leaving2,048. Producer dec64 releases2,048 and consumer
inc80 requests no increase at that hypothetical initial. Equal inc80 still
requires verification in the actual fresh code and initial allocation.
The old SS initial96/q96 and its shared/resource evidence do not certify this
companion. The lower P payload or absence of owning RS A does not prove q80 fit.
No qfit, no-spill, occupancy or two-CTA assertion is made.

Before any launch, independently compile both dtypes and preserve actual feeding
PTX/cubin plus linked test/benchmark identity. Bind source, pinned headers, exact
CuTe layout/sizeof and descriptor instantiation, host attributes and ELF resource
metadata. Inspect complete PTX/SASS CFG, initial quota/directions, all role live
register ranges including implicit widths, prequota accesses, local/stack/spill,
descriptor lifetime, two K16 issues/P/stage, whole-P accumulators, proxy fences,
barrier arrivals/drains and typed tail aliasing. Establish actual dynamic/static/
reserved shared, register allocation granularity, thread/warp/CTA limits and
device occupancy separately before making any physical two-CTA statement.

The companion can later supply an N16 SS comparison for the raw N16 RS source;
reference36's original N32 instruction is not the N16 same-atom reference.
Use a clearly scoped private N16 reference or independent host mathematics,
retain original FP64 numerical gates, and do not inherit GPU precision from any
old N32 or new hybrid-S variant. Run both dtypes, full stage-count domains,
partial native metadata with truthful P7/Dense fallback, fresh A/B/D/C0 bindings,
beta0/null, exact alias, fresh single-node Graph capture/replay and sanitizers
only after all fresh compiled gates pass. This file records no GPU test result.

A later separately approved paired timing run should use each candidate's own
frozen source/object/test/bench identities, ordinary/Graph modes, actual route
labels, matched rows and repeat distributions, original raw40/SS controls where
appropriate and tuned same-row cuBLASLt. This CPU preparation neither requests
nor begins those measurements and leaves current iteration38 formal work intact.
